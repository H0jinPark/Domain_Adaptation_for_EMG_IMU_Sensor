"""Phase 1 Hybrid: Source classification + Target contrastive learning.

Source (labeled)  : EMGEncoder -> emg_classifier -> CE loss  (EMG 독립 학습)
                    IMUEncoder -> imu_classifier -> CE loss  (IMU 독립 학습)
Target (unlabeled): per-modality augmented views x2 -> NT-Xent CL loss

학습 전략:
  - warmup_epochs 동안은 CL loss만 (도메인 불변 표현 선학습)
  - warmup 이후 modality별 cls loss 추가하여 joint 학습
  - encoder LR < head LR (encoder 과특화 방지)

total_loss = (emg_cls + imu_cls if joint) + cl_weight * cl_loss

학습 후 저장 (ssl_train.py와 동일한 파일명 -> Phase 2 코드 변경 없음):
  weights/pt_emg_enc_seed{seed}.pth
  weights/pt_imu_enc_seed{seed}.pth
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
from pretrain_Multimodal.pm_data_loader import get_hybrid_ssl_dataloaders
from pretrain_Multimodal.pm_utils import set_seed
from pretrain_Multimodal.pretrain_model import (
    HybridSSLModel, nt_xent_loss, augment_emg, augment_imu
)

WEIGHT_DIR = os.path.join(PROJECT_ROOT, "weights")


def train_hybrid_ssl(seed=0, epochs=50, warmup_epochs=10, batch_size=64,
                     encoder_lr=3e-4, head_lr=1e-3, temperature=0.5, cl_weight=1.0):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(WEIGHT_DIR, exist_ok=True)

    emg_save = os.path.join(WEIGHT_DIR, f"pt_emg_enc_seed{seed}.pth")
    imu_save = os.path.join(WEIGHT_DIR, f"pt_imu_enc_seed{seed}.pth")

    src_loader, tgt_loader, num_classes, _ = get_hybrid_ssl_dataloaders(batch_size=batch_size)

    model = HybridSSLModel(num_classes=num_classes).to(device)
    criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.1)

    # encoder와 head에 다른 LR 적용 (encoder 과특화 방지)
    encoder_params = (list(model.emg_encoder.parameters())
                      + list(model.imu_encoder.parameters()))
    head_params = (list(model.emg_projector.parameters())
                   + list(model.imu_projector.parameters())
                   + list(model.emg_classifier.parameters())
                   + list(model.imu_classifier.parameters()))
    optimizer = optim.AdamW([
        {"params": encoder_params, "lr": encoder_lr},
        {"params": head_params,    "lr": head_lr},
    ], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    print("\n" + "=" * 60)
    print(f"Hybrid SSL Pretraining Start  |  seed={seed}")
    print(f"Warmup: {warmup_epochs} epochs (CL only) -> Joint: {epochs - warmup_epochs} epochs")
    print(f"encoder_lr={encoder_lr}  head_lr={head_lr}  cl_weight={cl_weight}  tau={temperature}")
    print(f"Device: {device}")
    print("=" * 60)

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        is_warmup = epoch < warmup_epochs
        n_batches = min(len(src_loader), len(tgt_loader))

        total_cl = 0.0
        total_emg_cls, total_imu_cls = 0.0, 0.0
        emg_correct, imu_correct, total_samples = 0, 0, 0

        phase_tag = f"Warmup({epoch+1}/{warmup_epochs})" if is_warmup else "Joint"
        pbar = tqdm(range(n_batches),
                    desc=f"Epoch [{epoch+1:03d}/{epochs:03d}] [{phase_tag}]")

        src_iter = iter(src_loader)
        tgt_iter = iter(tgt_loader)

        for _ in pbar:
            src_emg, src_imu, src_y = next(src_iter)
            tgt_emg, tgt_imu = next(tgt_iter)

            src_emg = src_emg.to(device)
            src_imu = src_imu.to(device)
            src_y   = src_y.to(device)
            tgt_emg = tgt_emg.to(device)
            tgt_imu = tgt_imu.to(device)

            optimizer.zero_grad()

            # Target: CL loss (항상 계산)
            tgt_emg_v1, tgt_emg_v2 = augment_emg(tgt_emg), augment_emg(tgt_emg)
            tgt_imu_v1, tgt_imu_v2 = augment_imu(tgt_imu), augment_imu(tgt_imu)
            emg_z1, imu_z1 = model.forward_target_cl(tgt_emg_v1, tgt_imu_v1)
            emg_z2, imu_z2 = model.forward_target_cl(tgt_emg_v2, tgt_imu_v2)
            cl_loss = (nt_xent_loss(emg_z1, emg_z2, temperature)
                       + nt_xent_loss(imu_z1, imu_z2, temperature)) / 2

            loss = cl_weight * cl_loss

            # Source: modality별 cls loss (warmup 이후에만 추가)
            if not is_warmup:
                emg_logits = model.forward_source_emg(src_emg)
                imu_logits = model.forward_source_imu(src_imu)
                emg_cls_loss = criterion_cls(emg_logits, src_y)
                imu_cls_loss = criterion_cls(imu_logits, src_y)
                loss = loss + emg_cls_loss + imu_cls_loss

                total_emg_cls += emg_cls_loss.item()
                total_imu_cls += imu_cls_loss.item()
                emg_correct  += (emg_logits.argmax(1) == src_y).sum().item()
                imu_correct  += (imu_logits.argmax(1) == src_y).sum().item()
                total_samples += src_y.size(0)

            loss.backward()
            optimizer.step()
            total_cl += cl_loss.item()

            postfix = {"CL": f"{cl_loss.item():.4f}"}
            if not is_warmup:
                postfix["EMG_Acc"] = f"{emg_correct / total_samples * 100:.1f}%"
                postfix["IMU_Acc"] = f"{imu_correct / total_samples * 100:.1f}%"
            pbar.set_postfix(postfix)

        scheduler.step()

        avg_cl      = total_cl / n_batches
        avg_emg_cls = total_emg_cls / n_batches if not is_warmup else 0.0
        avg_imu_cls = total_imu_cls / n_batches if not is_warmup else 0.0
        avg_total   = cl_weight * avg_cl + avg_emg_cls + avg_imu_cls

        log = f"Epoch [{epoch+1:03d}/{epochs:03d}] [{phase_tag}] | CL: {avg_cl:.4f}"
        if not is_warmup:
            emg_acc = emg_correct / total_samples * 100
            imu_acc = imu_correct / total_samples * 100
            log += (f" | EMG_Cls: {avg_emg_cls:.4f} ({emg_acc:.1f}%)"
                    f" | IMU_Cls: {avg_imu_cls:.4f} ({imu_acc:.1f}%)")
        log += f" | Total: {avg_total:.4f}"

        is_best = avg_total < best_loss
        print(log + ("  (best)" if is_best else ""))

        if is_best:
            best_loss = avg_total
            model.save_encoders(emg_save, imu_save)

    print(f"\nHybrid Phase 1 완료 | seed={seed} | best_loss={best_loss:.4f}")
    print(f"  EMG encoder -> {emg_save}")
    print(f"  IMU encoder -> {imu_save}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--multi_seed", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup_epochs", type=int, default=10,
                        help="CL만 학습하는 초반 epoch 수")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--encoder_lr", type=float, default=3e-4,
                        help="EMG/IMU encoder learning rate")
    parser.add_argument("--head_lr", type=float, default=1e-3,
                        help="projector / classifier learning rate")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--cl_weight", type=float, default=1.0,
                        help="CL loss 가중치. total = (emg_cls + imu_cls) + cl_weight * cl_loss")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seeds = args.seeds if args.multi_seed else [args.seed]
    for seed in seeds:
        train_hybrid_ssl(
            seed=seed,
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            batch_size=args.batch_size,
            encoder_lr=args.encoder_lr,
            head_lr=args.head_lr,
            temperature=args.temperature,
            cl_weight=args.cl_weight,
        )
