"""No-DA baseline 학습 스크립트 (5채널, multi-seed 지원).

Source 데이터로만 분류기를 학습하고 Target에 그대로 평가한다.
도메인 적응 없이 순수 source-only 학습의 성능을 측정하는 베이스라인.

DANN 과 동일한 TCN 백본 + label head를 사용하되 domain discriminator는 학습에 쓰지 않는다.
"""
import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from data_loader import get_dataloaders
from DANN_model import DANNModel
from da_utils import set_seed, evaluate, save_confusion_matrix, summarize_results

WEIGHT_DIR = os.path.join(PROJECT_ROOT, "weights")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")


def train_no_da(seed=42, epochs=30, batch_size=64, lr=1e-3, save_cm=False):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(WEIGHT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    save_path = os.path.join(WEIGHT_DIR, f"5ch_no_da_seed{seed}_best_model.pth")

    train_loader, val_loader, _, tgt_val_loader, num_classes, le = \
        get_dataloaders(batch_size=batch_size)
    class_names = le.classes_

    model = DANNModel(in_channels=5, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    print("\n" + "=" * 60)
    print(f"No-DA Baseline Training Start  |  seed={seed}")
    print("Mode: 5-channel, source-only, no domain adaptation")
    print(f"Targeting {num_classes} classes on {device}")
    print("=" * 60)

    best_val_acc, best_target_acc = 0.0, 0.0

    for epoch in range(epochs):
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        pbar = tqdm(train_loader,
                    desc=f"Seed {seed} | Epoch [{epoch+1:02d}/{epochs:02d}]")
        for src_x, src_y in pbar:
            src_x, src_y = src_x.to(device), src_y.to(device)

            optimizer.zero_grad()
            class_out, _ = model(src_x, alpha=0.0)
            loss = criterion(class_out, src_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += (class_out.argmax(1) == src_y).sum().item()
            total_samples += src_y.size(0)
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Acc":  f"{total_correct / total_samples * 100:.1f}%",
            })

        scheduler.step()

        val_acc, _, _ = evaluate(model, val_loader, device, needs_alpha=True)
        tgt_acc, _, _ = evaluate(model, tgt_val_loader, device, needs_alpha=True)

        is_best = tgt_acc > best_target_acc
        if is_best:
            best_val_acc = val_acc
            best_target_acc = tgt_acc
            torch.save(model.state_dict(), save_path)

        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | "
              f"Loss: {total_loss / len(train_loader):.4f} | "
              f"Source Val: {val_acc:.2f}% | Target Val: {tgt_acc:.2f}%"
              + ("  (best)" if is_best else ""))

    print(f"\n최종 결과 | seed={seed} | "
          f"Best Source Val: {best_val_acc:.2f}% | "
          f"Target at Best: {best_target_acc:.2f}% | "
          f"Shift: {best_val_acc - best_target_acc:.2f}%")

    if save_cm:
        model.load_state_dict(torch.load(save_path, map_location=device))
        _, v_preds, v_true = evaluate(model, val_loader, device, needs_alpha=True)
        _, t_preds, t_true = evaluate(model, tgt_val_loader, device, needs_alpha=True)
        save_confusion_matrix(
            v_true, v_preds, class_names,
            os.path.join(RESULT_DIR, f"5ch_no_da_seed{seed}_source_cm.png"),
            f"No-DA 5ch Source (seed={seed}, Val: {best_val_acc:.1f}%)", cmap="Blues")
        save_confusion_matrix(
            t_true, t_preds, class_names,
            os.path.join(RESULT_DIR, f"5ch_no_da_seed{seed}_target_cm.png"),
            f"No-DA 5ch Target (seed={seed}, Src: {best_val_acc:.1f}% / Tgt: {best_target_acc:.1f}%)",
            cmap="Blues")
        print("혼동 행렬 저장 완료.")

    return {
        "seed": seed,
        "source_acc": best_val_acc,
        "target_acc": best_target_acc,
        "shift": best_val_acc - best_target_acc,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi_seed", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no_cm", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.multi_seed:
        results = []
        for seed in args.seeds:
            results.append(train_no_da(
                seed=seed, epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, save_cm=not args.no_cm))
        summarize_results(results, method_name="No-DA (5-channel)",
                          save_path=os.path.join(RESULT_DIR, "5ch_no_da_summary.txt"))
    else:
        train_no_da(
            seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, save_cm=not args.no_cm)
