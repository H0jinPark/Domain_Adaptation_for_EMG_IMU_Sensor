"""Learnable R 학습 스크립트 (IMU 단독 + feature MMD 정렬, multi-seed).

CDAN 적대 신호 대신, 인코더 출력 feature 공간에서 source 분포와 R-정렬된 target
분포 사이의 MMD 를 줄여 R(과 인코더)을 업데이트한다. R 에 매끄럽고 과제 관련성 있는
gradient 가 흘러, "target 을 source feature 분포에 맞추는 회전" 으로 학습된다.

흐름:
    target IMU --R--> aligned IMU --> 인코더 --> tgt_feat
    source IMU --------------------> 인코더 --> src_feat,  src label 분류
    loss = CE(src) + λ·MMD(src_feat, tgt_feat) + 기하 prior(L_orth/L_det[/gravity/norm])

R 은 L_orth=||RᵀR-I||, L_det=|det(R)-1| 로 회전(SO(3))에 묶는다. gravity 는 데이터가
채널별 표준화돼 중력 DC 가 사라졌으므로 기본 가중치 0(--w_gravity 로 켤 수 있음).
데이터는 preprocessed_MM_raw 기본, 출력은 results/Learnable_R/ 의 learnable_r_mmd*.
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "Multimodal"))
from mm_data_loader import get_mm_dataloaders                       # noqa: E402
from mm_utils import set_seed, save_confusion_matrix, summarize_results  # noqa: E402
from MMD_train import mmd_loss                                       # noqa: E402  (가우시안 커널 MMD 재사용)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from learnable_r_model import LearnableRMMD, r_geometric_losses     # noqa: E402

WEIGHT_DIR = os.path.join(PROJECT_ROOT, "weights")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results", "Learnable_R")
R_DIR = os.path.join(PROJECT_ROOT, "results", "R_matrices")


@torch.no_grad()
def evaluate_r(model, loader, device, apply_r):
    """source 는 apply_r=False, target 은 apply_r=True 로 정확도(%) 계산."""
    model.eval()
    preds, labels = [], []
    for emg, imu, y in loader:
        emg, imu = emg.to(device), imu.to(device)
        out = model(emg, imu, apply_r=apply_r)[0]
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(y.numpy())
    return accuracy_score(labels, preds) * 100, preds, labels


def train(seed=42, epochs=30, batch_size=64, lr=1e-3, r_lr=1e-2, lambda_mmd=1.0,
          w_orth=1.0, w_det=1.0, w_gravity=0.0, w_norm=0.0, save_cm=False, tag=""):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(WEIGHT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(R_DIR, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    save_path = os.path.join(WEIGHT_DIR, f"learnable_r_mmd{suffix}_seed{seed}_best_model.pth")

    train_loader, val_loader, tgt_train_loader, tgt_val_loader, num_classes, le = \
        get_mm_dataloaders(batch_size=batch_size)
    class_names = le.classes_

    model = LearnableRMMD(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # R 은 별도 param group(자체 lr, weight_decay 0)
    r_params = list(model.r.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith("r.")]
    optimizer = optim.AdamW([
        {"params": other_params, "lr": lr, "weight_decay": 1e-3},
        {"params": r_params, "lr": r_lr, "weight_decay": 0.0},
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    print("\n" + "=" * 60)
    print(f"Learnable-R MMD Training Start  |  seed={seed}")
    print("Mode: IMU-only, target IMU -> R -> encoder, align by feature MMD")
    print(f"lambda_mmd={lambda_mmd} | R-prior: orth={w_orth} det={w_det} "
          f"gravity={w_gravity} norm={w_norm}")
    print(f"Targeting {num_classes} classes on {device}")
    print("=" * 60)

    best_val_acc, best_target_acc = 0.0, 0.0

    for epoch in range(epochs):
        model.train()
        len_loader = min(len(train_loader), len(tgt_train_loader))
        total_c, total_m, total_r = 0.0, 0.0, 0.0

        pbar = tqdm(enumerate(zip(train_loader, tgt_train_loader)),
                    total=len_loader, desc=f"Seed {seed} | Epoch [{epoch+1:02d}/{epochs:02d}]")

        for i, ((src_emg, src_imu, src_y), (tgt_emg, tgt_imu, tgt_y)) in pbar:
            src_emg, src_imu, src_y = src_emg.to(device), src_imu.to(device), src_y.to(device)
            tgt_emg, tgt_imu = tgt_emg.to(device), tgt_imu.to(device)

            optimizer.zero_grad()

            # Source: R 우회, 분류 + feature. Target: R 적용, feature 만.
            src_out, src_feat = model(src_emg, src_imu, apply_r=False)
            _, tgt_feat = model(tgt_emg, tgt_imu, apply_r=True)

            loss_cls = criterion(src_out, src_y)
            loss_mmd = mmd_loss(src_feat, tgt_feat)
            L_orth, L_det, L_grav, L_norm = r_geometric_losses(model.r.R, src_imu, tgt_imu)
            r_prior = w_orth * L_orth + w_det * L_det + w_gravity * L_grav + w_norm * L_norm

            loss = loss_cls + lambda_mmd * loss_mmd + r_prior
            loss.backward()
            optimizer.step()

            total_c += loss_cls.item()
            total_m += loss_mmd.item()
            total_r += float(r_prior)
            pbar.set_postfix({"CE": f"{loss_cls.item():.3f}", "MMD": f"{loss_mmd.item():.4f}",
                              "Rprior": f"{float(r_prior):.3f}"})

        scheduler.step()

        val_acc, _, _ = evaluate_r(model, val_loader, device, apply_r=False)
        tgt_acc, _, _ = evaluate_r(model, tgt_val_loader, device, apply_r=True)

        is_best = tgt_acc > best_target_acc
        if is_best:
            best_val_acc, best_target_acc = val_acc, tgt_acc
            torch.save(model.state_dict(), save_path)

        with torch.no_grad():
            Rm = model.r.R.detach()
            det = torch.linalg.det(Rm).item()
            orth = torch.linalg.norm(Rm.transpose(0, 1) @ Rm - torch.eye(3, device=device)).item()

        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | CE: {total_c/len_loader:.4f} | "
              f"MMD: {total_m/len_loader:.4f} | Rprior: {total_r/len_loader:.4f} | "
              f"det(R): {det:.3f} | ||RᵀR-I||: {orth:.3f} | "
              f"Source Val: {val_acc:.2f}% | Target Val: {tgt_acc:.2f}%"
              f"{'  (best)' if is_best else ''}")

    print(f"\n최종 결과 | seed={seed} | Best Source Val: {best_val_acc:.2f}% | "
          f"Target at Best: {best_target_acc:.2f}% | Shift: {best_val_acc - best_target_acc:.2f}%")

    model.load_state_dict(torch.load(save_path, map_location=device))
    R_best = model.r.R.detach().cpu().numpy()
    r_path = os.path.join(R_DIR, f"R_learned_mmd{suffix}_seed{seed}.npy")
    np.save(r_path, R_best)
    print(f"학습된 R 저장: {r_path}\n{np.array2string(R_best, precision=4)}")

    if save_cm:
        _, v_preds, v_true = evaluate_r(model, val_loader, device, apply_r=False)
        _, t_preds, t_true = evaluate_r(model, tgt_val_loader, device, apply_r=True)
        save_confusion_matrix(
            v_true, v_preds, class_names,
            os.path.join(RESULT_DIR, f"learnable_r_mmd{suffix}_seed{seed}_source_cm.png"),
            f"Learnable-R MMD Source (seed={seed}, Val: {best_val_acc:.1f}%)", cmap="Purples")
        save_confusion_matrix(
            t_true, t_preds, class_names,
            os.path.join(RESULT_DIR, f"learnable_r_mmd{suffix}_seed{seed}_target_cm.png"),
            f"Learnable-R MMD Target (seed={seed}, Src: {best_val_acc:.1f}% vs Tgt: {best_target_acc:.1f}%)",
            cmap="Purples")
        print("혼동 행렬 시각화 저장 완료.")

    return {"seed": seed, "source_acc": best_val_acc, "target_acc": best_target_acc,
            "shift": best_val_acc - best_target_acc}


def write_result_json(results, tag):
    src = [r["source_acc"] for r in results]
    tgt = [r["target_acc"] for r in results]
    sh = [r["shift"] for r in results]
    ddof = 1 if len(results) > 1 else 0
    payload = {
        "tag": tag or "default", "modality": "imu_only", "model": "learnable_r_mmd",
        "data_dir": os.environ.get("MM_DATA_DIR", "preprocessed_MM_raw"),
        "seeds": [r["seed"] for r in results], "results": results,
        "mean": {"source_acc": float(np.mean(src)), "target_acc": float(np.mean(tgt)),
                 "shift": float(np.mean(sh))},
        "std": {"source_acc": float(np.std(src, ddof=ddof)), "target_acc": float(np.std(tgt, ddof=ddof)),
                "shift": float(np.std(sh, ddof=ddof))},
    }
    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, f"learnable_r_mmd_result_{tag or 'default'}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"결과 JSON 저장: {path}")
    return path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi_seed", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--r_lr", type=float, default=1e-2, help="learnable R 전용 학습률")
    parser.add_argument("--lambda_mmd", type=float, default=1.0, help="feature MMD 가중치")
    parser.add_argument("--w_orth", type=float, default=1.0, help="||RᵀR-I|| 가중치")
    parser.add_argument("--w_det", type=float, default=1.0, help="|det(R)-1| 가중치")
    parser.add_argument("--w_gravity", type=float, default=0.0,
                        help="||R g_t - g_s|| 가중치(데이터 표준화로 중력 제거됨 → 기본 0)")
    parser.add_argument("--w_norm", type=float, default=0.0, help="| |Ra|-|a| | 가중치")
    parser.add_argument("--no_cm", action="store_true")
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    suffix = f"_{args.tag}" if args.tag else ""
    kw = dict(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, r_lr=args.r_lr,
              lambda_mmd=args.lambda_mmd, w_orth=args.w_orth, w_det=args.w_det,
              w_gravity=args.w_gravity, w_norm=args.w_norm, save_cm=not args.no_cm, tag=args.tag)

    if args.multi_seed:
        results = [train(seed=s, **kw) for s in args.seeds]
        os.makedirs(RESULT_DIR, exist_ok=True)
        summarize_results(results, method_name=f"Learnable-R MMD (IMU-only){' | '+args.tag if args.tag else ''}",
                          save_path=os.path.join(RESULT_DIR, f"learnable_r_mmd{suffix}_summary.txt"))
    else:
        results = [train(seed=args.seed, **kw)]

    write_result_json(results, args.tag)
