"""Baseline 모델 학습 스크립트 (No Domain Adaptation).

5채널 동기화 입력(preprocessed/)을 받는 단일 백본 모델 5종(tcn/cnn/transformer/
gru/mlp)을 학습한다. --model 인자로 모델을 선택하며, --multi_seed 로 여러 seed
반복 학습 후 평균·표준편차를 집계할 수 있다.
"""
import os
import sys
import argparse
import random
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import get_dataloaders
from baseline_model import AdvancedBaselineModel
from baseline_CNN import CNNBaselineModel
from baseline_Transformer import TransformerBaselineModel
from baseline_GRU import GRUBaselineModel
from baseline_MLP import MLPBaselineModel


def set_seed(seed=42):
    """난수 시드를 고정해 실험 재현성을 확보한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def parse_args():
    """커맨드라인 인자를 파싱한다."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi_seed", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])

    parser.add_argument(
        "--model",
        type=str,
        default="tcn",
        choices=["tcn", "cnn", "transformer", "gru", "mlp"]
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--data_dir", type=str, default="preprocessed",
                        help="preprocessed/, preprocessed_PCA/ 등 데이터 폴더 선택")
    parser.add_argument("--tag", type=str, default="",
                        help="결과/가중치 파일명에 붙일 식별자 (예: pca)")

    return parser.parse_args()


def build_model(model_name, in_channels=5, num_classes=10):
    """모델 이름에 해당하는 네트워크 인스턴스를 생성한다."""
    model_name = model_name.lower()

    if model_name == "tcn":
        return AdvancedBaselineModel(in_channels=in_channels, num_classes=num_classes)
    elif model_name == "cnn":
        return CNNBaselineModel(in_channels=in_channels, num_classes=num_classes)
    elif model_name == "transformer":
        return TransformerBaselineModel(in_channels=in_channels, num_classes=num_classes)
    elif model_name == "gru":
        return GRUBaselineModel(in_channels=in_channels, num_classes=num_classes)
    elif model_name == "mlp":
        return MLPBaselineModel(in_channels=in_channels, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def save_confusion_matrix(cm, class_names, title, save_path, show_plot=True):
    """혼동 행렬을 히트맵으로 그려 PNG 로 저장한다."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Oranges",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.close()


def train_baseline(args, seed=None):
    """단일 seed 로 baseline 모델을 학습하고 Source/Target 정확도를 반환한다."""
    if seed is None:
        seed = args.seed

    set_seed(seed)

    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.lr
    model_name = args.model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"학습 시작 | 디바이스: {device}")
    print(f"모델: {model_name}")
    print(f"Seed: {seed}")

    os.makedirs("weights", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    tag = f"_{args.tag}" if args.tag else ""
    save_path = f"weights/{model_name}_seed{seed}_baseline{tag}_best.pth"

    # ----------------------------------------------------------------------
    # 데이터 로더 (Source/Target 의 train/val 4분할)
    # ----------------------------------------------------------------------
    train_loader, val_loader, tgt_train_loader, tgt_val_loader, num_classes, le = get_dataloaders(
        batch_size=batch_size, data_dir=args.data_dir,
    )
    class_names = le.classes_

    model = build_model(
        model_name=model_name,
        in_channels=5,
        num_classes=num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    print("\n" + "=" * 50)
    print(f"[Phase 1] Source Domain Training - {model_name.upper()} | Seed {seed}")
    print("=" * 50)

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        running_loss = 0.0

        pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch [{epoch + 1:02d}/{epochs:02d}]")

        for batch_x, batch_y in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        scheduler.step()
        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        # 검증 (Source val)
        model.eval()
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()

        val_acc = 100.0 * val_correct / val_total

        # Source val 기준 best 모델 저장
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

        best_mark = "  (best)" if is_best else ""
        print(
            f"Epoch [{epoch + 1:02d}/{epochs:02d}] | "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}% | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
            f"{best_mark}"
        )

    print(f"\n학습 완료 | 최고 Val 정확도: {best_val_acc:.2f}%")

    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    # ----------------------------------------------------------------------
    # Phase 2: Source 도메인 평가
    # ----------------------------------------------------------------------
    print("\n" + "=" * 50)
    print(f"[Phase 2] Source Domain Evaluation - {model_name.upper()} | Seed {seed}")
    print("=" * 50)

    val_preds, val_targets = [], []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            _, predicted = outputs.max(1)
            val_preds.extend(predicted.cpu().numpy())
            val_targets.extend(batch_y.numpy())

    cm_val = confusion_matrix(val_targets, val_preds)
    source_cm_path = f"results/{model_name}_seed{seed}_baseline{tag}_source_confusion_matrix.png"
    save_confusion_matrix(
        cm=cm_val,
        class_names=class_names,
        title=f"{model_name.upper()} Baseline Source Prediction\n(Val Acc: {best_val_acc:.1f}%)",
        save_path=source_cm_path,
        show_plot=not args.no_plot
    )

    # ----------------------------------------------------------------------
    # Phase 3: Target 도메인 평가 (domain gap 측정)
    # ----------------------------------------------------------------------
    print("\n" + "=" * 50)
    print(f"[Phase 3] Target Domain Evaluation - {model_name.upper()} | Seed {seed}")
    print("=" * 50)

    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch_x, batch_y in tgt_val_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.numpy())

    tgt_acc = accuracy_score(all_targets, all_preds) * 100

    print(f"Target Domain(Val) 정확도: {tgt_acc:.2f}%")
    print(f"10개 운동 유형 도메인 격차(Shift): {best_val_acc - tgt_acc:.2f}%")

    cm_tgt = confusion_matrix(all_targets, all_preds)
    target_cm_path = f"results/{model_name}_seed{seed}_baseline{tag}_target_confusion_matrix.png"
    save_confusion_matrix(
        cm=cm_tgt,
        class_names=class_names,
        title=(
            f"{model_name.upper()} Baseline Target Prediction\n"
            f"(Val: {best_val_acc:.1f}% vs Target Val: {tgt_acc:.1f}%)"
        ),
        save_path=target_cm_path,
        show_plot=not args.no_plot
    )

    return best_val_acc, tgt_acc


def run_multi_seed(args):
    """여러 seed 로 반복 학습하고 평균·표준편차를 CSV 로 저장한다."""
    source_results = []
    target_results = []
    rows = []

    for seed in args.seeds:
        print("\n" + "#" * 60)
        print(f"Multi-seed run | Model: {args.model.upper()} | Seed: {seed}")
        print("#" * 60)

        source_acc, target_acc = train_baseline(args, seed=seed)
        source_results.append(source_acc)
        target_results.append(target_acc)
        rows.append([args.model, seed, source_acc, target_acc, source_acc - target_acc])

    source_mean = float(np.mean(source_results))
    source_std = float(np.std(source_results, ddof=1)) if len(source_results) > 1 else 0.0
    target_mean = float(np.mean(target_results))
    target_std = float(np.std(target_results, ddof=1)) if len(target_results) > 1 else 0.0

    os.makedirs("results", exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    csv_path = f"results/{args.model}_baseline{tag}_multiseed_results.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "seed", "source_val_acc", "target_val_acc", "domain_gap"])
        writer.writerows(rows)
        writer.writerow([])
        writer.writerow(["summary", "source_mean", "source_std", "target_mean", "target_std"])
        writer.writerow(["mean_std", source_mean, source_std, target_mean, target_std])

    print("\n" + "=" * 60)
    print("FINAL RESULT | MULTI-SEED")
    print("=" * 60)
    print(f"Model: {args.model.upper()}")
    print(f"Seeds: {args.seeds}")
    print(f"Source Val Acc: {source_mean:.2f} ± {source_std:.2f}")
    print(f"Target Val Acc: {target_mean:.2f} ± {target_std:.2f}")
    print(f"CSV saved: {csv_path}")


if __name__ == "__main__":
    args = parse_args()

    if args.multi_seed:
        run_multi_seed(args)
    else:
        train_baseline(args, seed=args.seed)
