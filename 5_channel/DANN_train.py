"""DANN 학습 스크립트 (단일 백본, 5채널 동기화 입력, multi-seed 지원).

Gradient Reversal Layer 기반 adversarial domain adaptation 으로 Source(Samsung1)
라벨만 사용해 Target(Samsung2) 운동 분류 성능을 끌어올린다.

규약
  - 각 배치는 Source / Target 을 분리 forward 한다 (concat-forward 아님).
  - domain discriminator 내부에는 BatchNorm 을 두지 않는다 (DANN_model.py 참고).
  - Source val 기준으로 best 모델을 저장해 model selection leakage 를 방지한다.

--multi_seed 옵션으로 여러 seed 를 반복 실행하면, 마지막에 평균±표준편차를
콘솔과 텍스트 파일로 함께 보고한다.
"""
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# 프로젝트 루트를 import 경로에 추가 (5_channel/ → ../)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from data_loader import get_dataloaders
from DANN_model import DANNModel
from da_utils import set_seed, evaluate, save_confusion_matrix, summarize_results

WEIGHT_DIR = os.path.join(PROJECT_ROOT, "weights")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")


# ----------------------------------------------------------------------
# 단일 seed 학습
# ----------------------------------------------------------------------
def train_dann(seed=42, epochs=30, batch_size=64, lr=1e-3, domain_weight=1.0, save_cm=False):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(WEIGHT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    save_path = os.path.join(WEIGHT_DIR, f"5ch_dann_seed{seed}_best_model.pth")

    train_loader, val_loader, tgt_train_loader, tgt_val_loader, num_classes, le = \
        get_dataloaders(batch_size=batch_size)
    class_names = le.classes_

    model = DANNModel(in_channels=5, num_classes=num_classes).to(device)

    criterion_class = nn.CrossEntropyLoss(label_smoothing=0.1)  # 운동 분류용
    criterion_domain = nn.CrossEntropyLoss()                    # 도메인 분류용

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    print("\n" + "=" * 60)
    print(f"DANN Adversarial Training Start  |  seed={seed}")
    print("Mode: 5-channel input, separate-forward, no BN in discriminator")
    print(f"Targeting {num_classes} classes on {device}")
    print("=" * 60)

    best_val_acc, best_target_acc = 0.0, 0.0

    for epoch in range(epochs):
        model.train()

        len_loader = min(len(train_loader), len(tgt_train_loader))
        total_c_loss, total_d_loss, total_d_acc = 0.0, 0.0, 0.0

        # GRL 의 alpha 스케줄링 (0 -> 1 로 점진 증가)
        p = float(epoch) / epochs
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

        pbar = tqdm(enumerate(zip(train_loader, tgt_train_loader)),
                    total=len_loader, desc=f"Seed {seed} | Epoch [{epoch+1:02d}/{epochs:02d}]")

        for i, ((src_x, src_y), (tgt_x, tgt_y)) in pbar:
            src_x, src_y = src_x.to(device), src_y.to(device)
            tgt_x = tgt_x.to(device)

            optimizer.zero_grad()

            src_domain_label = torch.zeros(src_x.size(0), dtype=torch.long, device=device)
            tgt_domain_label = torch.ones(tgt_x.size(0), dtype=torch.long, device=device)

            # Step 1. Source 분리 forward (운동 분류 + 도메인 분류)
            src_class_out, src_domain_out = model(src_x, alpha=alpha)
            loss_s_label = criterion_class(src_class_out, src_y)
            loss_s_domain = criterion_domain(src_domain_out, src_domain_label)

            # Step 2. Target 분리 forward (도메인 분류만, UDA 라 라벨 미사용)
            _, tgt_domain_out = model(tgt_x, alpha=alpha)
            loss_t_domain = criterion_domain(tgt_domain_out, tgt_domain_label)

            # Step 3. 통합 손실 계산 및 역전파
            class_loss = loss_s_label
            domain_loss = loss_s_domain + loss_t_domain
            loss = class_loss + domain_weight * domain_loss

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                domain_pred = torch.cat([src_domain_out.argmax(1), tgt_domain_out.argmax(1)])
                domain_true = torch.cat([src_domain_label, tgt_domain_label])
                domain_acc = (domain_pred == domain_true).float().mean().item()

            total_c_loss += class_loss.item()
            total_d_loss += domain_loss.item()
            total_d_acc += domain_acc

            pbar.set_postfix({
                "Class": f"{class_loss.item():.4f}",
                "Domain": f"{domain_loss.item():.4f}",
                "DomAcc": f"{domain_acc*100:.2f}%",
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
              f"Class: {total_c_loss/len_loader:.4f} | "
              f"Domain: {total_d_loss/len_loader:.4f} | "
              f"DomAcc: {total_d_acc/len_loader*100:.2f}% | "
              f"Source Val: {val_acc:.2f}% | Target Val: {tgt_acc:.2f}% | "
              f"Alpha: {alpha:.3f}"
              f"{'  (best)' if is_best else ''}")

    print(f"\n최종 결과 | seed={seed} | "
          f"Best Source Val Acc: {best_val_acc:.2f}% | "
          f"Target Acc at Best Source: {best_target_acc:.2f}% | "
          f"Shift: {best_val_acc - best_target_acc:.2f}%")

    if save_cm:
        model.load_state_dict(torch.load(save_path, map_location=device))
        _, v_preds, v_true = evaluate(model, val_loader, device, needs_alpha=True)
        _, t_preds, t_true = evaluate(model, tgt_val_loader, device, needs_alpha=True)
        save_confusion_matrix(
            v_true, v_preds, class_names,
            os.path.join(RESULT_DIR, f"5ch_dann_seed{seed}_source_cm.png"),
            f"DANN 5ch Source (seed={seed}, Val Acc: {best_val_acc:.1f}%)", cmap="Blues")
        save_confusion_matrix(
            t_true, t_preds, class_names,
            os.path.join(RESULT_DIR, f"5ch_dann_seed{seed}_target_cm.png"),
            f"DANN 5ch Target (seed={seed}, Src: {best_val_acc:.1f}% vs Tgt: {best_target_acc:.1f}%)",
            cmap="Blues")
        print("혼동 행렬 시각화 저장 완료.")

    return {
        "seed": seed,
        "source_acc": best_val_acc,
        "target_acc": best_target_acc,
        "shift": best_val_acc - best_target_acc,
    }


# ----------------------------------------------------------------------
# 실행 옵션
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi_seed", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--domain_weight", type=float, default=1.0)
    parser.add_argument("--no_cm", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.multi_seed:
        results = []
        for seed in args.seeds:
            results.append(train_dann(
                seed=seed, epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, domain_weight=args.domain_weight, save_cm=not args.no_cm))
        summarize_results(results, method_name="DANN (5-channel)",
                          save_path=os.path.join(RESULT_DIR, "5ch_dann_summary.txt"))
    else:
        train_dann(
            seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, domain_weight=args.domain_weight, save_cm=not args.no_cm)
