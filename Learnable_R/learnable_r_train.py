"""Learnable R 학습 스크립트 (IMU 단독 CDAN + 학습 가능한 정렬 행렬 R, multi-seed).

CDAN_IMU_train.py 와 동일한 학습 규약(분리 forward, discriminator 내 BN 없음,
target val 기준 best 저장)을 따르되 다음만 추가한다:
  - 모델: IMUOnlyCDAN -> LearnableRCDAN (target IMU 에 learnable 3x3 R 적용)
  - 손실: class + domain 에 더해 R 기하 손실 4종
        L_orth/L_det/L_gravity/L_norm (가중치 --w_orth/--w_det/--w_gravity/--w_norm)
  - 옵티마이저: R 은 별도 param group(--r_lr) 로, 본체와 다른 lr 을 줄 수 있게 분리
  - 평가: source 는 R 우회(apply_r=False), target 은 R 적용(apply_r=True)
  - 학습된 R 은 results/R_matrices/R_learned*_seed*.npy 로 저장(다른 정렬법과 비교용)

데이터는 정렬을 R 이 대체하므로 raw 가 기본:
    MM_DATA_DIR=preprocessed_MM_raw (mm_data_loader 기본값)
출력은 results/Learnable_R/ 아래 learnable_r* 접두로 저장(다른 실험과 분리).
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# 프로젝트 루트와 Multimodal/ 을 import 경로에 추가
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "Multimodal"))
from mm_data_loader import get_mm_dataloaders                       # noqa: E402
from mm_utils import set_seed, save_confusion_matrix, summarize_results  # noqa: E402

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from learnable_r_model import LearnableRCDAN, r_geometric_losses    # noqa: E402

WEIGHT_DIR = os.path.join(PROJECT_ROOT, "weights")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results", "Learnable_R")
R_DIR = os.path.join(PROJECT_ROOT, "results", "R_matrices")


# ----------------------------------------------------------------------
# 평가: source 는 R 우회, target 은 R 적용
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate_r(model, loader, device, apply_r):
    """loader 전체 정확도(%)와 (예측, 정답)을 반환. apply_r 로 R 경로 on/off."""
    model.eval()
    preds, labels = [], []
    for emg, imu, y in loader:
        emg, imu = emg.to(device), imu.to(device)
        out = model(emg, imu, alpha=0.0, apply_r=apply_r)[0]
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(y.numpy())
    return accuracy_score(labels, preds) * 100, preds, labels


# ----------------------------------------------------------------------
# 단일 seed 학습
# ----------------------------------------------------------------------
def train(seed=42, epochs=30, batch_size=64, lr=1e-3, r_lr=1e-2, domain_weight=1.0,
          w_orth=1.0, w_det=1.0, w_gravity=1.0, w_norm=0.1, save_cm=False, tag=""):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(WEIGHT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(R_DIR, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    save_path = os.path.join(WEIGHT_DIR, f"learnable_r{suffix}_seed{seed}_best_model.pth")

    train_loader, val_loader, tgt_train_loader, tgt_val_loader, num_classes, le = \
        get_mm_dataloaders(batch_size=batch_size)
    class_names = le.classes_

    model = LearnableRCDAN(num_classes=num_classes).to(device)

    criterion_class = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_domain = nn.CrossEntropyLoss()

    # R 은 별도 param group: 작은 행렬이라 본체보다 큰 lr 을 줄 수 있게 분리, weight_decay 0.
    r_params = list(model.r.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith("r.")]
    optimizer = optim.AdamW([
        {"params": other_params, "lr": lr, "weight_decay": 1e-3},
        {"params": r_params, "lr": r_lr, "weight_decay": 0.0},
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    print("\n" + "=" * 60)
    print(f"Learnable-R CDAN Training Start  |  seed={seed}")
    print("Mode: IMU-only, target IMU -> learnable R -> feature extractor")
    print(f"R-loss weights: orth={w_orth} det={w_det} gravity={w_gravity} norm={w_norm}")
    print(f"Targeting {num_classes} classes on {device}")
    print("=" * 60)

    best_val_acc, best_target_acc = 0.0, 0.0

    for epoch in range(epochs):
        model.train()
        len_loader = min(len(train_loader), len(tgt_train_loader))
        total_c_loss, total_d_loss, total_r_loss = 0.0, 0.0, 0.0

        p = float(epoch) / epochs
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

        pbar = tqdm(enumerate(zip(train_loader, tgt_train_loader)),
                    total=len_loader, desc=f"Seed {seed} | Epoch [{epoch+1:02d}/{epochs:02d}]")

        for i, ((src_emg, src_imu, src_y), (tgt_emg, tgt_imu, tgt_y)) in pbar:
            src_emg, src_imu, src_y = src_emg.to(device), src_imu.to(device), src_y.to(device)
            tgt_emg, tgt_imu = tgt_emg.to(device), tgt_imu.to(device)

            optimizer.zero_grad()

            src_domain_label = torch.zeros(src_imu.size(0), dtype=torch.long, device=device)
            tgt_domain_label = torch.ones(tgt_imu.size(0), dtype=torch.long, device=device)

            # Source: R 우회(참조 좌표계). 운동 분류 + 조건부 도메인 분류.
            src_class_out, src_domain_out = model(src_emg, src_imu, alpha=alpha, apply_r=False)
            loss_s_label = criterion_class(src_class_out, src_y)
            loss_s_domain = criterion_domain(src_domain_out, src_domain_label)

            # Target: R 적용(aligned). 조건부 도메인 분류만.
            _, tgt_domain_out = model(tgt_emg, tgt_imu, alpha=alpha, apply_r=True)
            loss_t_domain = criterion_domain(tgt_domain_out, tgt_domain_label)

            # R 기하 손실 (배치 raw IMU 기준)
            L_orth, L_det, L_grav, L_norm = r_geometric_losses(model.r.R, src_imu, tgt_imu)

            class_loss = loss_s_label
            domain_loss = loss_s_domain + loss_t_domain
            r_loss = w_orth * L_orth + w_det * L_det + w_gravity * L_grav + w_norm * L_norm
            loss = class_loss + domain_weight * domain_loss + r_loss

            loss.backward()
            optimizer.step()

            total_c_loss += class_loss.item()
            total_d_loss += domain_loss.item()
            total_r_loss += r_loss.item()

            pbar.set_postfix({
                "Class": f"{class_loss.item():.3f}",
                "Domain": f"{domain_loss.item():.3f}",
                "Rloss": f"{r_loss.item():.3f}",
                "grav": f"{L_grav.item():.3f}",
            })

        scheduler.step()

        val_acc, _, _ = evaluate_r(model, val_loader, device, apply_r=False)
        tgt_acc, _, _ = evaluate_r(model, tgt_val_loader, device, apply_r=True)

        is_best = tgt_acc > best_target_acc
        if is_best:
            best_val_acc, best_target_acc = val_acc, tgt_acc
            torch.save(model.state_dict(), save_path)

        # R 의 직교성/det 모니터링
        with torch.no_grad():
            Rm = model.r.R.detach()
            det = torch.linalg.det(Rm).item()
            orth = torch.linalg.norm(Rm.transpose(0, 1) @ Rm - torch.eye(3, device=device)).item()

        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | "
              f"Class: {total_c_loss/len_loader:.4f} | "
              f"Domain: {total_d_loss/len_loader:.4f} | "
              f"Rloss: {total_r_loss/len_loader:.4f} | "
              f"det(R): {det:.3f} | ||RᵀR-I||: {orth:.3f} | "
              f"Source Val: {val_acc:.2f}% | Target Val: {tgt_acc:.2f}% | "
              f"Alpha: {alpha:.3f}{'  (best)' if is_best else ''}")

    print(f"\n최종 결과 | seed={seed} | "
          f"Best Source Val Acc: {best_val_acc:.2f}% | "
          f"Target Acc at Best Target: {best_target_acc:.2f}% | "
          f"Shift: {best_val_acc - best_target_acc:.2f}%")

    # best 모델의 R 을 npy 로 저장(다른 정렬법과 비교용)
    model.load_state_dict(torch.load(save_path, map_location=device))
    R_best = model.r.R.detach().cpu().numpy()
    r_path = os.path.join(R_DIR, f"R_learned{suffix}_seed{seed}.npy")
    np.save(r_path, R_best)
    print(f"학습된 R 저장: {r_path}\n{np.array2string(R_best, precision=4)}")

    if save_cm:
        _, v_preds, v_true = evaluate_r(model, val_loader, device, apply_r=False)
        _, t_preds, t_true = evaluate_r(model, tgt_val_loader, device, apply_r=True)
        save_confusion_matrix(
            v_true, v_preds, class_names,
            os.path.join(RESULT_DIR, f"learnable_r{suffix}_seed{seed}_source_cm.png"),
            f"Learnable-R Source (seed={seed}, Val Acc: {best_val_acc:.1f}%)", cmap="Purples")
        save_confusion_matrix(
            t_true, t_preds, class_names,
            os.path.join(RESULT_DIR, f"learnable_r{suffix}_seed{seed}_target_cm.png"),
            f"Learnable-R Target (seed={seed}, Src: {best_val_acc:.1f}% vs Tgt: {best_target_acc:.1f}%)",
            cmap="Purples")
        print("혼동 행렬 시각화 저장 완료.")

    return {
        "seed": seed,
        "source_acc": best_val_acc,
        "target_acc": best_target_acc,
        "shift": best_val_acc - best_target_acc,
    }


# ----------------------------------------------------------------------
# 결과 JSON 저장
# ----------------------------------------------------------------------
def write_result_json(results, tag):
    src = [r["source_acc"] for r in results]
    tgt = [r["target_acc"] for r in results]
    sh = [r["shift"] for r in results]
    ddof = 1 if len(results) > 1 else 0
    payload = {
        "tag": tag or "default",
        "modality": "imu_only",
        "model": "learnable_r_cdan",
        "data_dir": os.environ.get("MM_DATA_DIR", "preprocessed_MM_raw"),
        "seeds": [r["seed"] for r in results],
        "results": results,
        "mean": {"source_acc": float(np.mean(src)), "target_acc": float(np.mean(tgt)),
                 "shift": float(np.mean(sh))},
        "std": {"source_acc": float(np.std(src, ddof=ddof)), "target_acc": float(np.std(tgt, ddof=ddof)),
                "shift": float(np.std(sh, ddof=ddof))},
    }
    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, f"learnable_r_result_{tag or 'default'}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"결과 JSON 저장: {path}")
    return path


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
    parser.add_argument("--r_lr", type=float, default=1e-2, help="learnable R 전용 학습률")
    parser.add_argument("--domain_weight", type=float, default=1.0)
    parser.add_argument("--w_orth", type=float, default=1.0, help="||RᵀR-I|| 가중치")
    parser.add_argument("--w_det", type=float, default=1.0, help="|det(R)-1| 가중치")
    parser.add_argument("--w_gravity", type=float, default=1.0, help="||R g_t - g_s|| 가중치")
    parser.add_argument("--w_norm", type=float, default=0.1, help="| |Ra|-|a| | 가중치")
    parser.add_argument("--no_cm", action="store_true")
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    suffix = f"_{args.tag}" if args.tag else ""
    kw = dict(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, r_lr=args.r_lr,
              domain_weight=args.domain_weight, w_orth=args.w_orth, w_det=args.w_det,
              w_gravity=args.w_gravity, w_norm=args.w_norm, save_cm=not args.no_cm, tag=args.tag)

    if args.multi_seed:
        results = [train(seed=s, **kw) for s in args.seeds]
        os.makedirs(RESULT_DIR, exist_ok=True)
        summarize_results(results, method_name=f"Learnable-R CDAN (IMU-only){' | '+args.tag if args.tag else ''}",
                          save_path=os.path.join(RESULT_DIR, f"learnable_r{suffix}_summary.txt"))
    else:
        results = [train(seed=args.seed, **kw)]

    write_result_json(results, args.tag)
