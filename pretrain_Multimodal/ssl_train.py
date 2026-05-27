"""Phase 1: SimCLR 기반 EMG/IMU 인코더 SSL 사전학습.

Source train + Target train 데이터를 라벨 없이 사용하여
EMGEncoder / IMUEncoder 를 각각 SimCLR NT-Xent loss 로 독립 학습한다.

학습 후 저장:
  weights/pt_emg_enc_seed{seed}.pth  <- EMGEncoder.state_dict()
  weights/pt_imu_enc_seed{seed}.pth  <- IMUEncoder.state_dict()

저장된 가중치는 Phase 2 (DANN/MMD/Coral/CDAN_train.py)에서
PretrainInterFusionBackbone.load_pretrained_encoders()로 로드된다.

설계 요점
  - EMG/IMU 인코더를 같은 루프에서 학습하지만 그래디언트는 독립적이다.
  - IMU augmentation에 axis_sign_flip을 포함해 Source/Target 축 부호 불일치에
    대한 불변성을 학습시킨다.
  - projection head는 Phase 1에서만 사용하고 가중치를 버린다.
"""
import os
import sys
import argparse

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from pretrain_Multimodal.pm_data_loader import get_ssl_dataloaders
from pretrain_Multimodal.pm_utils import set_seed
from pretrain_Multimodal.pretrain_model import (
    EMGSSLModel, IMUSSLModel, nt_xent_loss, augment_emg, augment_imu
)

WEIGHT_DIR = os.path.join(PROJECT_ROOT, "weights")


# -----------------------------------------------------------------------
# 단일 seed SSL 학습
# -----------------------------------------------------------------------
def train_ssl(seed=0, epochs=50, batch_size=256, lr=1e-3, temperature=0.5):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(WEIGHT_DIR, exist_ok=True)

    emg_save = os.path.join(WEIGHT_DIR, f"pt_emg_enc_seed{seed}.pth")
    imu_save = os.path.join(WEIGHT_DIR, f"pt_imu_enc_seed{seed}.pth")

    ssl_loader, n_samples = get_ssl_dataloaders(batch_size=batch_size)

    emg_model = EMGSSLModel().to(device)
    imu_model = IMUSSLModel().to(device)

    optimizer = optim.AdamW(
        list(emg_model.parameters()) + list(imu_model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    print("\n" + "=" * 60)
    print(f"SSL Pretraining Start  |  seed={seed}")
    print(f"Samples: {n_samples}  |  Batch: {batch_size}  |  τ={temperature}")
    print(f"Device: {device}")
    print("=" * 60)

    best_loss = float("inf")

    for epoch in range(epochs):
        emg_model.train()
        imu_model.train()

        total_emg_loss, total_imu_loss = 0.0, 0.0

        pbar = tqdm(ssl_loader, desc=f"Epoch [{epoch+1:03d}/{epochs:03d}]")
        for emg, imu in pbar:
            emg, imu = emg.to(device), imu.to(device)

            # 두 augmented view 생성
            emg1, emg2 = augment_emg(emg), augment_emg(emg)
            imu1, imu2 = augment_imu(imu), augment_imu(imu)

            optimizer.zero_grad()

            # EMG NT-Xent
            z_emg1 = emg_model(emg1)
            z_emg2 = emg_model(emg2)
            loss_emg = nt_xent_loss(z_emg1, z_emg2, temperature=temperature)

            # IMU NT-Xent
            z_imu1 = imu_model(imu1)
            z_imu2 = imu_model(imu2)
            loss_imu = nt_xent_loss(z_imu1, z_imu2, temperature=temperature)

            loss = loss_emg + loss_imu
            loss.backward()
            optimizer.step()

            total_emg_loss += loss_emg.item()
            total_imu_loss += loss_imu.item()
            pbar.set_postfix({
                "EMG": f"{loss_emg.item():.4f}",
                "IMU": f"{loss_imu.item():.4f}",
            })

        scheduler.step()

        n_batches = len(ssl_loader)
        avg_total = (total_emg_loss + total_imu_loss) / n_batches
        print(f"Epoch [{epoch+1:03d}/{epochs:03d}] | "
              f"EMG: {total_emg_loss/n_batches:.4f} | "
              f"IMU: {total_imu_loss/n_batches:.4f} | "
              f"Total: {avg_total:.4f}"
              + ("  (best)" if avg_total < best_loss else ""))

        if avg_total < best_loss:
            best_loss = avg_total
            torch.save(emg_model.encoder.state_dict(), emg_save)
            torch.save(imu_model.encoder.state_dict(), imu_save)

    print(f"\nPhase 1 완료 | seed={seed} | best_loss={best_loss:.4f}")
    print(f"  EMG encoder -> {emg_save}")
    print(f"  IMU encoder -> {imu_save}")


# -----------------------------------------------------------------------
# 실행 옵션
# -----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--multi_seed", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seeds = args.seeds if args.multi_seed else [args.seed]
    for seed in seeds:
        train_ssl(
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            temperature=args.temperature,
        )
