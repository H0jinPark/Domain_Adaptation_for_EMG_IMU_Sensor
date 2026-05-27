"""Phase 2: CORAL 학습 (pretrained encoder 로드 후 fine-tune + CORAL 정렬).

ssl_train.py (Phase 1)로 저장된 EMG/IMU 인코더 가중치를 로드하고,
intermediate fusion backbone의 512차원 feature에 DeepCORAL loss를 건다.
"""
import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from pretrain_Multimodal.pm_data_loader import get_pm_dataloaders
from pretrain_Multimodal.pretrain_model import PretrainInterFusionClassifier
from pretrain_Multimodal.pm_utils import set_seed, evaluate, save_confusion_matrix, summarize_results

WEIGHT_DIR = os.path.join(PROJECT_ROOT, "weights")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")


# -----------------------------------------------------------------------
# CORAL 손실 (Multimodal/Coral_train.py와 동일)
# -----------------------------------------------------------------------
def coral_loss(source, target):
    """DeepCORAL 손실: L2 정규화된 feature의 공분산(상관) 행렬 차이.

    L2 normalize 후 계산하므로 공분산 행렬 원소가 [-1,1]로 bound되고
    loss 스케일이 [0,1]로 유지되어 CE loss와 비례가 맞는다.
    """
    d = source.size(1)
    n_s, n_t = source.size(0), target.size(0)

    src_mean = source.mean(0, keepdim=True)
    src_cov = (source - src_mean).t() @ (source - src_mean) / (n_s - 1)

    tgt_mean = target.mean(0, keepdim=True)
    tgt_cov = (target - tgt_mean).t() @ (target - tgt_mean) / (n_t - 1)

    return (src_cov - tgt_cov).pow(2).sum() / d


# -----------------------------------------------------------------------
# 단일 seed 학습
# -----------------------------------------------------------------------
def train_coral(seed=42, pretrain_seed=0, epochs=30, batch_size=64,
                lr=1e-3, lambda_coral=1.0, save_cm=True):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(WEIGHT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    save_path = os.path.join(WEIGHT_DIR, f"pt_coral_seed{seed}_best_model.pth")

    train_loader, val_loader, tgt_train_loader, tgt_val_loader, num_classes, le = \
        get_pm_dataloaders(batch_size=batch_size)
    class_names = le.classes_

    model = PretrainInterFusionClassifier(num_classes=num_classes)

    emg_path = os.path.join(WEIGHT_DIR, f"pt_emg_enc_seed{pretrain_seed}.pth")
    imu_path = os.path.join(WEIGHT_DIR, f"pt_imu_enc_seed{pretrain_seed}.pth")
    if os.path.exists(emg_path) and os.path.exists(imu_path):
        model.load_pretrained_encoders(emg_path, imu_path, device="cpu")
    else:
        print(f"  [WARNING] pretrained weights not found -> random init")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    print("\n" + "=" * 60)
    print(f"Phase 2 CORAL Training Start  |  seed={seed}  pretrain_seed={pretrain_seed}")
    print(f"Targeting {num_classes} classes on {device}")
    print("=" * 60)

    best_val_acc, best_target_acc = 0.0, 0.0

    for epoch in range(epochs):
        model.train()
        len_loader = min(len(train_loader), len(tgt_train_loader))
        running_loss, running_coral = 0.0, 0.0

        pbar = tqdm(enumerate(zip(train_loader, tgt_train_loader)),
                    total=len_loader,
                    desc=f"Seed {seed} | Epoch [{epoch+1:02d}/{epochs:02d}]")

        for i, ((src_emg, src_imu, src_y), (tgt_emg, tgt_imu, _)) in pbar:
            src_emg, src_imu, src_y = src_emg.to(device), src_imu.to(device), src_y.to(device)
            tgt_emg, tgt_imu = tgt_emg.to(device), tgt_imu.to(device)

            optimizer.zero_grad()

            src_out, src_feat = model(src_emg, src_imu)
            _, tgt_feat = model(tgt_emg, tgt_imu)

            loss_cls = criterion(src_out, src_y)
            loss_coral = coral_loss(src_feat, tgt_feat)
            loss = loss_cls + lambda_coral * loss_coral
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_coral += loss_coral.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "CORAL": f"{loss_coral.item():.4f}"})

        scheduler.step()

        val_acc, _, _ = evaluate(model, val_loader, device)
        tgt_acc, _, _ = evaluate(model, tgt_val_loader, device)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_target_acc = tgt_acc
            torch.save(model.state_dict(), save_path)

        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | "
              f"Loss: {running_loss/len_loader:.4f} | "
              f"CORAL: {running_coral/len_loader:.4f} | "
              f"Source Val: {val_acc:.2f}% | Target Val: {tgt_acc:.2f}%"
              + ("  (best)" if is_best else ""))

    print(f"\n최종 결과 | seed={seed} | "
          f"Best Source Val: {best_val_acc:.2f}% | "
          f"Target at Best: {best_target_acc:.2f}% | "
          f"Shift: {best_val_acc - best_target_acc:.2f}%")

    if save_cm:
        model.load_state_dict(torch.load(save_path, map_location=device))
        _, v_preds, v_true = evaluate(model, val_loader, device)
        _, t_preds, t_true = evaluate(model, tgt_val_loader, device)
        save_confusion_matrix(
            v_true, v_preds, class_names,
            os.path.join(RESULT_DIR, f"pt_coral_seed{seed}_source_cm.png"),
            f"PT-CORAL Source (seed={seed}, Val: {best_val_acc:.1f}%)", cmap="Greens")
        save_confusion_matrix(
            t_true, t_preds, class_names,
            os.path.join(RESULT_DIR, f"pt_coral_seed{seed}_target_cm.png"),
            f"PT-CORAL Target (seed={seed}, Src: {best_val_acc:.1f}% / Tgt: {best_target_acc:.1f}%)",
            cmap="Greens")
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
    parser.add_argument("--pretrain_seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_coral", type=float, default=1.0)
    parser.add_argument("--no_cm", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.multi_seed:
        results = []
        for seed in args.seeds:
            results.append(train_coral(
                seed=seed, pretrain_seed=args.pretrain_seed,
                epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, lambda_coral=args.lambda_coral,
                save_cm=not args.no_cm))
        summarize_results(results, method_name="CORAL (Pretrain-MM)",
                          save_path=os.path.join(RESULT_DIR, "pt_coral_summary.txt"))
    else:
        train_coral(
            seed=args.seed, pretrain_seed=args.pretrain_seed,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, lambda_coral=args.lambda_coral,
            save_cm=not args.no_cm)
