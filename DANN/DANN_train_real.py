"""DANN 학습 스크립트 (단일 백본, 5채널 동기화 입력, multi-seed 지원).

Gradient Reversal Layer 기반 adversarial domain adaptation 으로 Source(Samsung1)
라벨만 사용해 Target(Samsung2) 운동 분류 성능을 끌어올린다. Source val 기준으로
best 모델을 저장해 model selection leakage 를 방지한다.

기존 source/target 분리 forward 구조를 source+target concat-forward 로 수정했다.
따라서 BatchNorm 이 source batch 와 target batch 를 각각 따로 정규화하지 않고,
하나의 mixed batch 를 기준으로 동작한다. 이 구조에서 domain discriminator 는 실제로
source/target 을 같은 forward pass 안에서 판별하며, GRL 이 feature extractor 로
적대적 그래디언트를 전달한다.

--multi_seed 옵션을 사용하면 여러 seed 를 반복 실행하고, 마지막에 평균과 표준편차를 출력한다.
"""
import os
import sys
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import get_dataloaders
from DANN_model_real import DANNModel


# ----------------------------------------------------------------------
# 난수 시드 고정
# ----------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_dann(seed=42, epochs=30, batch_size=64, lr=1e-3, domain_weight=1.0, save_cm=True):
    # ----------------------------------------------------------------------
    # 하이퍼파라미터 및 환경 설정
    # ----------------------------------------------------------------------
    set_seed(seed)

    BATCH_SIZE = batch_size
    EPOCHS = epochs
    LEARNING_RATE = lr
    DOMAIN_WEIGHT = domain_weight
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs('weights', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    save_path = f'weights/dann_real_seed{seed}_best_model.pth'

    # 데이터 로더 (Source/Target 의 train/val 4분할)
    train_loader, val_loader, tgt_train_loader, tgt_val_loader, num_classes, le = get_dataloaders(batch_size=BATCH_SIZE)
    class_names = le.classes_

    # 모델 선언 (EMG 2ch + IMU 3ch = 5채널 입력)
    model = DANNModel(in_channels=5, num_classes=num_classes).to(DEVICE)

    # 손실 함수
    criterion_class = nn.CrossEntropyLoss(label_smoothing=0.1)  # 운동 분류용
    criterion_domain = nn.CrossEntropyLoss()                    # 도메인 분류용 (Source vs Target)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print("\n" + "=" * 60)
    print(f"DANN Adversarial Training Start  |  seed={seed}")
    print("Mode: source+target concat-forward")
    print(f"Targeting {num_classes} classes on {DEVICE}")
    print("=" * 60)

    # ----------------------------------------------------------------------
    # 메인 학습 루프
    # ----------------------------------------------------------------------
    best_target_acc = 0.0
    best_val_acc = 0.0
    best_domain_loss = 0.0
    best_domain_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()

        # Source 와 Target(train) 데이터를 동시에 로드
        len_dataloader = min(len(train_loader), len(tgt_train_loader))
        data_zip = zip(train_loader, tgt_train_loader)

        total_loss, total_c_loss, total_d_loss, total_d_acc = 0, 0, 0, 0

        # GRL 의 alpha 스케줄링 (0 -> 1 로 점진 증가)
        p = float(epoch) / EPOCHS
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        pbar = tqdm(
            enumerate(data_zip),
            total=len_dataloader,
            desc=f"Seed {seed} | Epoch [{epoch+1:02d}/{EPOCHS:02d}]"
        )

        for i, ((src_x, src_y), (tgt_x, tgt_y)) in pbar:
            src_x, src_y = src_x.to(DEVICE), src_y.to(DEVICE)
            tgt_x = tgt_x.to(DEVICE)

            optimizer.zero_grad()

            # Step 1. Source 와 Target 을 하나의 batch 로 합쳐 한 번에 forward
            bs = src_x.size(0)
            bt = tgt_x.size(0)

            x = torch.cat([src_x, tgt_x], dim=0)

            domain_label = torch.cat([
                torch.zeros(bs, dtype=torch.long, device=DEVICE),
                torch.ones(bt, dtype=torch.long, device=DEVICE)
            ], dim=0)

            class_out, domain_out = model(x, alpha=alpha)

            # Step 2. 운동 분류 손실은 Source 에 대해서만 계산
            src_class_out = class_out[:bs]
            class_loss = criterion_class(src_class_out, src_y)

            # Step 3. 도메인 분류 손실은 Source+Target 전체에 대해 한 번에 계산
            domain_loss = criterion_domain(domain_out, domain_label)

            # Step 4. 통합 손실 계산 및 역전파
            loss = class_loss + DOMAIN_WEIGHT * domain_loss
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                domain_pred = domain_out.argmax(dim=1)
                domain_acc = (domain_pred == domain_label).float().mean().item()

            total_loss += loss.item()
            total_c_loss += class_loss.item()
            total_d_loss += domain_loss.item()
            total_d_acc += domain_acc

            pbar.set_postfix({
                'Total': f"{loss.item():.4f}",
                'Class': f"{class_loss.item():.4f}",
                'Domain': f"{domain_loss.item():.4f}",
                'DomAcc': f"{domain_acc * 100:.2f}%"
            })

        scheduler.step()

        avg_domain_loss = total_d_loss / len_dataloader
        avg_domain_acc = total_d_acc / len_dataloader * 100

        # Step 5. 검증 및 평가
        model.eval()

        # Source validation 평가
        val_preds, val_targets = [], []
        with torch.no_grad():
            for vx, vy in val_loader:
                vx = vx.to(DEVICE)
                out, _ = model(vx, alpha=0.0)
                val_preds.extend(out.max(1)[1].cpu().numpy())
                val_targets.extend(vy.numpy())
        val_acc = accuracy_score(val_targets, val_preds) * 100

        # Target validation 평가 (model selection leakage 방지를 위해 저장 기준에서 제외)
        tgt_preds, tgt_targets = [], []
        with torch.no_grad():
            for tx, ty in tgt_val_loader:
                tx = tx.to(DEVICE)
                out, _ = model(tx, alpha=0.0)
                tgt_preds.extend(out.max(1)[1].cpu().numpy())
                tgt_targets.extend(ty.numpy())
        tgt_acc = accuracy_score(tgt_targets, tgt_preds) * 100

        # Source val 기준 best 모델 저장
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_target_acc = tgt_acc
            best_domain_loss = avg_domain_loss
            best_domain_acc = avg_domain_acc
            torch.save(model.state_dict(), save_path)

        best_mark = "  (best)" if is_best else ""
        print(
            f"Epoch [{epoch+1:02d}/{EPOCHS:02d}] | "
            f"Class: {total_c_loss/len_dataloader:.4f} | "
            f"Domain: {avg_domain_loss:.4f} | "
            f"DomAcc: {avg_domain_acc:.2f}% | "
            f"Source Val: {val_acc:.2f}% | "
            f"Target Val: {tgt_acc:.2f}% | "
            f"Alpha: {alpha:.3f}"
            f"{best_mark}"
        )

    # ----------------------------------------------------------------------
    # 최종 결과 평가 및 시각화 저장
    # ----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Saving Final Results  |  seed={seed}")
    print("=" * 60)

    # 최고 성능 모델 로드
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    model.eval()

    print(
        f"\n최종 결과 | seed={seed} | "
        f"Best Source Val Acc: {best_val_acc:.2f}% | "
        f"Target Acc at Best Source: {best_target_acc:.2f}% | "
        f"Shift: {best_val_acc - best_target_acc:.2f}% | "
        f"Domain Loss at Best Source: {best_domain_loss:.4f} | "
        f"Domain Acc at Best Source: {best_domain_acc:.2f}%"
    )

    if save_cm:
        # Source 도메인 confusion matrix
        print("\nSaving Source Domain Confusion Matrix...")
        v_preds_final, v_true_final = [], []
        with torch.no_grad():
            for vx, vy in val_loader:
                vx = vx.to(DEVICE)
                out, _ = model(vx, alpha=0.0)
                v_preds_final.extend(out.max(1)[1].cpu().numpy())
                v_true_final.extend(vy.numpy())

        cm_val = confusion_matrix(v_true_final, v_preds_final)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'DANN Real Source Prediction\n(seed={seed}, Val Acc: {best_val_acc:.1f}%)', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'results/dann_real_seed{seed}_source_confusion_matrix.png', dpi=300)
        plt.close()

        # Target 도메인(unseen validation) confusion matrix
        print("Saving Target Domain Confusion Matrix...")
        t_preds_final, t_true_final = [], []
        with torch.no_grad():
            for tx, ty in tgt_val_loader:
                tx = tx.to(DEVICE)
                out, _ = model(tx, alpha=0.0)
                t_preds_final.extend(out.max(1)[1].cpu().numpy())
                t_true_final.extend(ty.numpy())

        cm_tgt = confusion_matrix(t_true_final, t_preds_final)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_tgt, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'DANN Real Target Prediction\n(seed={seed}, Source Val: {best_val_acc:.1f}% vs Target: {best_target_acc:.1f}%)', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'results/dann_real_seed{seed}_target_confusion_matrix.png', dpi=300)
        plt.close()
        print("혼동 행렬 시각화가 모두 저장되었습니다.")

    return {
        "seed": seed,
        "source_acc": best_val_acc,
        "target_acc": best_target_acc,
        "shift": best_val_acc - best_target_acc,
        "domain_loss": best_domain_loss,
        "domain_acc": best_domain_acc,
    }


# ----------------------------------------------------------------------
# Multi-seed 결과 요약
# ----------------------------------------------------------------------
def summarize_results(results):
    source = np.array([r["source_acc"] for r in results])
    target = np.array([r["target_acc"] for r in results])
    shift = np.array([r["shift"] for r in results])
    domain_loss = np.array([r["domain_loss"] for r in results])
    domain_acc = np.array([r["domain_acc"] for r in results])

    print("\n" + "=" * 72)
    print("Multi-seed Summary")
    print("=" * 72)
    print(f"{'Seed':>6} | {'Source':>8} | {'Target':>8} | {'Shift':>8} | {'DomLoss':>8} | {'DomAcc':>8}")
    print("-" * 72)

    for r in results:
        print(
            f"{r['seed']:>6} | "
            f"{r['source_acc']:>8.2f} | "
            f"{r['target_acc']:>8.2f} | "
            f"{r['shift']:>8.2f} | "
            f"{r['domain_loss']:>8.4f} | "
            f"{r['domain_acc']:>7.2f}%"
        )

    print("-" * 72)
    print(
        f"{'Mean':>6} | "
        f"{source.mean():>8.2f} | "
        f"{target.mean():>8.2f} | "
        f"{shift.mean():>8.2f} | "
        f"{domain_loss.mean():>8.4f} | "
        f"{domain_acc.mean():>7.2f}%"
    )
    print(
        f"{'Std':>6} | "
        f"{source.std(ddof=1):>8.2f} | "
        f"{target.std(ddof=1):>8.2f} | "
        f"{shift.std(ddof=1):>8.2f} | "
        f"{domain_loss.std(ddof=1):>8.4f} | "
        f"{domain_acc.std(ddof=1):>7.2f}%"
    )
    print("=" * 72)


# ----------------------------------------------------------------------
# 실행 옵션
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--multi_seed', action='store_true')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--domain_weight', type=float, default=1.0)
    parser.add_argument('--no_cm', action='store_true')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.multi_seed:
        results = []
        for seed in args.seeds:
            result = train_dann(
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                domain_weight=args.domain_weight,
                save_cm=not args.no_cm,
            )
            results.append(result)

        summarize_results(results)

    else:
        train_dann(
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            domain_weight=args.domain_weight,
            save_cm=not args.no_cm,
        )
