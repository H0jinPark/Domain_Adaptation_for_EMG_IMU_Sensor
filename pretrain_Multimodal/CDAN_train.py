"""Phase 2: CDAN 학습 (pretrained encoder 로드 후 fine-tune + conditional discriminator).

ssl_train.py (Phase 1)로 저장된 EMG/IMU 인코더 가중치를 로드하고,
feature × class-prob 외적(multilinear conditioning)을 입력으로 받는
conditional domain discriminator + GRL로 end-to-end fine-tune한다.
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from pretrain_Multimodal.pm_data_loader import get_pm_dataloaders
from pretrain_Multimodal.pretrain_model import PretrainInterFusionCDAN
from pretrain_Multimodal.pm_utils import set_seed, evaluate, save_confusion_matrix, summarize_results

WEIGHT_DIR = os.path.join(PROJECT_ROOT, "weights")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")


# -----------------------------------------------------------------------
# 단일 seed 학습
# -----------------------------------------------------------------------
def train_cdan(seed=42, pretrain_seed=0, epochs=30, batch_size=64,
               lr=1e-3, domain_weight=1.0, save_cm=False):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(WEIGHT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    save_path = os.path.join(WEIGHT_DIR, f"pt_cdan_seed{seed}_best_model.pth")

    train_loader, val_loader, tgt_train_loader, tgt_val_loader, num_classes, le = \
        get_pm_dataloaders(batch_size=batch_size)
    class_names = le.classes_

    model = PretrainInterFusionCDAN(num_classes=num_classes)

    emg_path = os.path.join(WEIGHT_DIR, f"pt_emg_enc_seed{pretrain_seed}.pth")
    imu_path = os.path.join(WEIGHT_DIR, f"pt_imu_enc_seed{pretrain_seed}.pth")
    if os.path.exists(emg_path) and os.path.exists(imu_path):
        model.load_pretrained_encoders(emg_path, imu_path, device="cpu")
    else:
        print(f"  [WARNING] pretrained weights not found -> random init")

    model = model.to(device)

    criterion_class = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_domain = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    print("\n" + "=" * 60)
    print(f"Phase 2 CDAN Training Start  |  seed={seed}  pretrain_seed={pretrain_seed}")
    print("Mode: pretrained encoders + intermediate fusion + conditional discriminator")
    print(f"Targeting {num_classes} classes on {device}")
    print("=" * 60)

    best_val_acc, best_target_acc = 0.0, 0.0

    for epoch in range(epochs):
        model.train()
        len_loader = min(len(train_loader), len(tgt_train_loader))
        total_c_loss, total_d_loss, total_d_acc = 0.0, 0.0, 0.0

        p = float(epoch) / epochs
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

        pbar = tqdm(enumerate(zip(train_loader, tgt_train_loader)),
                    total=len_loader,
                    desc=f"Seed {seed} | Epoch [{epoch+1:02d}/{epochs:02d}]")

        for i, ((src_emg, src_imu, src_y), (tgt_emg, tgt_imu, _)) in pbar:
            src_emg, src_imu, src_y = src_emg.to(device), src_imu.to(device), src_y.to(device)
            tgt_emg, tgt_imu = tgt_emg.to(device), tgt_imu.to(device)

            optimizer.zero_grad()

            src_domain_label = torch.zeros(src_emg.size(0), dtype=torch.long, device=device)
            tgt_domain_label = torch.ones(tgt_emg.size(0), dtype=torch.long, device=device)

            src_class_out, src_domain_out = model(src_emg, src_imu, alpha=alpha)
            loss_s_label = criterion_class(src_class_out, src_y)
            loss_s_domain = criterion_domain(src_domain_out, src_domain_label)

            _, tgt_domain_out = model(tgt_emg, tgt_imu, alpha=alpha)
            loss_t_domain = criterion_domain(tgt_domain_out, tgt_domain_label)

            loss = loss_s_label + domain_weight * (loss_s_domain + loss_t_domain)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                domain_pred = torch.cat([src_domain_out.argmax(1), tgt_domain_out.argmax(1)])
                domain_true = torch.cat([src_domain_label, tgt_domain_label])
                d_acc = (domain_pred == domain_true).float().mean().item()

            total_c_loss += loss_s_label.item()
            total_d_loss += (loss_s_domain + loss_t_domain).item()
            total_d_acc += d_acc
            pbar.set_postfix({
                "Class": f"{loss_s_label.item():.4f}",
                "Domain": f"{(loss_s_domain+loss_t_domain).item():.4f}",
                "DomAcc": f"{d_acc*100:.1f}%",
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
            os.path.join(RESULT_DIR, f"pt_cdan_seed{seed}_source_cm.png"),
            f"PT-CDAN Source (seed={seed}, Val: {best_val_acc:.1f}%)", cmap="Blues")
        save_confusion_matrix(
            t_true, t_preds, class_names,
            os.path.join(RESULT_DIR, f"pt_cdan_seed{seed}_target_cm.png"),
            f"PT-CDAN Target (seed={seed}, Src: {best_val_acc:.1f}% / Tgt: {best_target_acc:.1f}%)",
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
    parser.add_argument("--domain_weight", type=float, default=1.0)
    parser.add_argument("--no_cm", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.multi_seed:
        results = []
        for seed in args.seeds:
            results.append(train_cdan(
                seed=seed, pretrain_seed=args.pretrain_seed,
                epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, domain_weight=args.domain_weight,
                save_cm=not args.no_cm))
        summarize_results(results, method_name="CDAN (Pretrain-MM)",
                          save_path=os.path.join(RESULT_DIR, "pt_cdan_summary.txt"))
    else:
        train_cdan(
            seed=args.seed, pretrain_seed=args.pretrain_seed,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, domain_weight=args.domain_weight,
            save_cm=not args.no_cm)
