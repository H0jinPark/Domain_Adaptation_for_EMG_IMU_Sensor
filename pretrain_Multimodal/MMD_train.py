"""Phase 2: MMD 학습 (pretrained encoder 로드 후 fine-tune + MMD 정렬).

ssl_train.py (Phase 1)로 저장된 EMG/IMU 인코더 가중치를 로드하고,
intermediate fusion backbone의 512차원 feature에 gaussian-kernel MMD loss를 건다.
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
from pretrain_Multimodal.pm_data_loader import get_pm_dataloaders
from pretrain_Multimodal.pretrain_model import PretrainInterFusionClassifier
from pretrain_Multimodal.pm_utils import set_seed, evaluate, save_confusion_matrix, summarize_results

WEIGHT_DIR = os.path.join(PROJECT_ROOT, "weights")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")


# -----------------------------------------------------------------------
# MMD 손실 (Multimodal/MMD_train.py와 동일)
# -----------------------------------------------------------------------
def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size(0)) + int(target.size(0))
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    l2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(l2_distance.detach()) / (n_samples ** 2 - n_samples)
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    return sum(torch.exp(-l2_distance / bw) for bw in bandwidth_list)


def mmd_loss(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size(0))
    kernels = gaussian_kernel(source, target, kernel_mul, kernel_num, fix_sigma)
    xx = kernels[:batch_size, :batch_size]
    yy = kernels[batch_size:, batch_size:]
    xy = kernels[:batch_size, batch_size:]
    yx = kernels[batch_size:, :batch_size]
    return torch.mean(xx + yy - xy - yx)


# -----------------------------------------------------------------------
# 단일 seed 학습
# -----------------------------------------------------------------------
def train_mmd(seed=42, pretrain_seed=0, epochs=30, batch_size=64,
              lr=1e-3, lambda_mmd=0.5, save_cm=False):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(WEIGHT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    save_path = os.path.join(WEIGHT_DIR, f"pt_mmd_seed{seed}_best_model.pth")

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
    print(f"Phase 2 MMD Training Start  |  seed={seed}  pretrain_seed={pretrain_seed}")
    print(f"Targeting {num_classes} classes on {device}")
    print("=" * 60)

    best_val_acc, best_target_acc = 0.0, 0.0

    for epoch in range(epochs):
        model.train()
        len_loader = min(len(train_loader), len(tgt_train_loader))
        running_loss, running_mmd = 0.0, 0.0

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
            loss_mmd = mmd_loss(src_feat, tgt_feat)
            loss = loss_cls + lambda_mmd * loss_mmd
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_mmd += loss_mmd.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "MMD": f"{loss_mmd.item():.4f}"})

        scheduler.step()

        val_acc, _, _ = evaluate(model, val_loader, device)
        tgt_acc, _, _ = evaluate(model, tgt_val_loader, device)

        is_best = tgt_acc > best_target_acc
        if is_best:
            best_val_acc = val_acc
            best_target_acc = tgt_acc
            torch.save(model.state_dict(), save_path)

        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | "
              f"Loss: {running_loss/len_loader:.4f} | "
              f"MMD: {running_mmd/len_loader:.4f} | "
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
            os.path.join(RESULT_DIR, f"pt_mmd_seed{seed}_source_cm.png"),
            f"PT-MMD Source (seed={seed}, Val: {best_val_acc:.1f}%)", cmap="Blues")
        save_confusion_matrix(
            t_true, t_preds, class_names,
            os.path.join(RESULT_DIR, f"pt_mmd_seed{seed}_target_cm.png"),
            f"PT-MMD Target (seed={seed}, Src: {best_val_acc:.1f}% / Tgt: {best_target_acc:.1f}%)",
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
    parser.add_argument("--pretrain_seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_mmd", type=float, default=0.5)
    parser.add_argument("--no_cm", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.multi_seed:
        results = []
        for seed in args.seeds:
            results.append(train_mmd(
                seed=seed, pretrain_seed=args.pretrain_seed,
                epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, lambda_mmd=args.lambda_mmd,
                save_cm=not args.no_cm))
        summarize_results(results, method_name="MMD (Pretrain-MM)",
                          save_path=os.path.join(RESULT_DIR, "pt_mmd_summary.txt"))
    else:
        train_mmd(
            seed=args.seed, pretrain_seed=args.pretrain_seed,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, lambda_mmd=args.lambda_mmd,
            save_cm=not args.no_cm)
