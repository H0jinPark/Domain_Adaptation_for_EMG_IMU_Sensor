import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import get_dataloaders
from RAINCOAT_model import RAINCOATModel

from geomloss import SamplesLoss
import torch.nn.functional as F

sinkhorn_loss = SamplesLoss(
    loss="sinkhorn",
    p=2,
    blur=0.05,
    scaling=0.8,
    debias=True
)

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

    kernel_val = [torch.exp(-l2_distance / bw) for bw in bandwidth_list]
    return sum(kernel_val)


def mmd_loss(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size(0))

    kernels = gaussian_kernel(
        source,
        target,
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
        fix_sigma=fix_sigma
    )

    xx = kernels[:batch_size, :batch_size]
    yy = kernels[batch_size:, batch_size:]
    xy = kernels[:batch_size, batch_size:]
    yx = kernels[batch_size:, :batch_size]

    loss = torch.mean(xx + yy - xy - yx)
    return loss


def train_raincoat():
    BATCH_SIZE = 64
    EPOCHS = 30
    LEARNING_RATE = 1e-3

    LAMBDA_FUSED = 0.5
    LAMBDA_TIME = 0.2
    LAMBDA_FREQ = 0.5

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("weights", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    save_path = "weights/raincoat_best_model.pth"

    train_loader, val_loader, tgt_train_loader, tgt_val_loader, num_classes, le = get_dataloaders(
        batch_size=BATCH_SIZE
    )
    class_names = le.classes_

    model = RAINCOATModel(in_channels=5, num_classes=num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print("\n" + "=" * 50)
    print("🌧️ RAINCOAT-lite Training Start!")
    print(f"Targeting {num_classes} classes on {DEVICE}")
    print("=" * 50)

    best_val_acc = 0.0
    best_target_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()

        len_dataloader = min(len(train_loader), len(tgt_train_loader))
        data_zip = zip(train_loader, tgt_train_loader)

        total_loss = 0.0
        total_cls = 0.0
        total_align = 0.0

        pbar = tqdm(
            enumerate(data_zip),
            total=len_dataloader,
            desc=f"Epoch [{epoch+1:02d}/{EPOCHS}]"
        )

        for i, ((src_x, src_y), (tgt_x, tgt_y)) in pbar:
            src_x, src_y = src_x.to(DEVICE), src_y.to(DEVICE)
            tgt_x = tgt_x.to(DEVICE)

            optimizer.zero_grad()

            src_out, src_feat, src_time, src_freq = model(src_x)
            tgt_out, tgt_feat, tgt_time, tgt_freq = model(tgt_x)

            loss_cls = criterion(src_out, src_y)

            src_feat_n = F.normalize(src_feat, dim=1)
            tgt_feat_n = F.normalize(tgt_feat, dim=1)

            src_time_n = F.normalize(src_time, dim=1)
            tgt_time_n = F.normalize(tgt_time, dim=1)

            src_freq_n = F.normalize(src_freq, dim=1)
            tgt_freq_n = F.normalize(tgt_freq, dim=1)

            loss_fused = sinkhorn_loss(src_feat_n, tgt_feat_n)
            loss_time = sinkhorn_loss(src_time_n, tgt_time_n)
            loss_freq = sinkhorn_loss(src_freq_n, tgt_freq_n)

            loss_align = (
                LAMBDA_FUSED * loss_fused
                + LAMBDA_TIME * loss_time
                + LAMBDA_FREQ * loss_freq
            )

            loss = loss_cls + loss_align

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_cls += loss_cls.item()
            total_align += loss_align.item()

            pbar.set_postfix({
                "Total": f"{loss.item():.4f}",
                "Cls": f"{loss_cls.item():.4f}",
                "Align": f"{loss_align.item():.4f}"
            })

        scheduler.step()

        model.eval()

        val_preds, val_targets = [], []
        with torch.no_grad():
            for vx, vy in val_loader:
                vx = vx.to(DEVICE)
                out, _, _, _ = model(vx)
                val_preds.extend(out.max(1)[1].cpu().numpy())
                val_targets.extend(vy.numpy())

        val_acc = accuracy_score(val_targets, val_preds) * 100

        tgt_preds, tgt_targets = [], []
        with torch.no_grad():
            for tx, ty in tgt_val_loader:
                tx = tx.to(DEVICE)
                out, _, _, _ = model(tx)
                tgt_preds.extend(out.max(1)[1].cpu().numpy())
                tgt_targets.extend(ty.numpy())

        tgt_acc = accuracy_score(tgt_targets, tgt_preds) * 100

        print(
            f"Epoch [{epoch+1:02d}/{EPOCHS}] | "
            f"Loss: {total_loss / len_dataloader:.4f} | "
            f"Cls: {total_cls / len_dataloader:.4f} | "
            f"Align: {total_align / len_dataloader:.4f} | "
            f"Source Val: {val_acc:.2f}% | "
            f"Target Val: {tgt_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_target_acc = tgt_acc
            torch.save(model.state_dict(), save_path)
            print(f"   -> 🌟 Best Source Val Model Saved! (Source Val: {best_val_acc:.2f}%)")

    print("\n" + "=" * 50)
    print("📂 Saving Final Results...")
    print("=" * 50)

    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    model.eval()

    print(f"🚨 RAINCOAT-lite Target Domain 정확도: {best_target_acc:.2f}%")
    print(f">> 최종 격차(Shift): {best_val_acc - best_target_acc:.2f}%")

    print("\n📊 Saving Source Domain Confusion Matrix...")

    v_preds_final, v_true_final = [], []
    with torch.no_grad():
        for vx, vy in val_loader:
            vx = vx.to(DEVICE)
            out, _, _, _ = model(vx)
            v_preds_final.extend(out.max(1)[1].cpu().numpy())
            v_true_final.extend(vy.numpy())

    cm_val = confusion_matrix(v_true_final, v_preds_final)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_val,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f"RAINCOAT-lite Source Prediction\n(Val Acc: {best_val_acc:.1f}%)", fontsize=16)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/raincoat_source_confusion_matrix.png", dpi=300)
    plt.close()

    print("🔍 Saving Target Domain Confusion Matrix...")

    t_preds_final, t_true_final = [], []
    with torch.no_grad():
        for tx, ty in tgt_val_loader:
            tx = tx.to(DEVICE)
            out, _, _, _ = model(tx)
            t_preds_final.extend(out.max(1)[1].cpu().numpy())
            t_true_final.extend(ty.numpy())

    cm_tgt = confusion_matrix(t_true_final, t_preds_final)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_tgt,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(
        f"RAINCOAT-lite Target Prediction\n"
        f"(Source Val: {best_val_acc:.1f}% vs Target: {best_target_acc:.1f}%)",
        fontsize=16
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/raincoat_target_confusion_matrix.png", dpi=300)
    print("📊 혼동 행렬 시각화가 모두 저장되었습니다.")
    plt.show()


if __name__ == "__main__":
    train_raincoat()