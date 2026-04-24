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
from baseline.baseline_model import AdvancedBaselineModel


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


class MMDWrapper(nn.Module):
    def __init__(self, base_model):
        super(MMDWrapper, self).__init__()
        self.backbone = base_model

    def forward(self, x):
        x = self.backbone.stem(x)
        features = self.backbone.layers(x)
        features = features.squeeze(-1)
        out = self.backbone.classifier(features)
        return out, features


def train_mmd():
    BATCH_SIZE = 64
    EPOCHS = 30
    LEARNING_RATE = 1e-3
    LAMBDA_MMD = 0.5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🔥 MMD Training Start! Using device: {DEVICE}")

    os.makedirs("weights", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    save_path = "weights/mmd_best_model.pth"

    train_loader, val_loader, tgt_train_loader, tgt_val_loader, num_classes, le = get_dataloaders(
        batch_size=BATCH_SIZE
    )
    class_names = le.classes_

    base_model = AdvancedBaselineModel(in_channels=5, num_classes=num_classes)
    model = MMDWrapper(base_model).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print("\n" + "=" * 50)
    print("📏 MMD Alignment Training Start!")
    print("=" * 50)

    best_target_acc = 0.0
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()

        data_zip = zip(train_loader, tgt_train_loader)
        len_loader = min(len(train_loader), len(tgt_train_loader))

        running_loss, running_mmd = 0.0, 0.0

        pbar = tqdm(
            enumerate(data_zip),
            total=len_loader,
            desc=f"Epoch [{epoch+1:02d}/{EPOCHS}]"
        )

        for i, ((src_x, src_y), (tgt_x, tgt_y)) in pbar:
            src_x, src_y = src_x.to(DEVICE), src_y.to(DEVICE)
            tgt_x = tgt_x.to(DEVICE)

            optimizer.zero_grad()

            src_out, src_feat = model(src_x)
            tgt_out, tgt_feat = model(tgt_x)

            loss_cls = criterion(src_out, src_y)
            loss_mmd = mmd_loss(src_feat, tgt_feat)
            total_loss = loss_cls + LAMBDA_MMD * loss_mmd

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            running_mmd += loss_mmd.item()

            pbar.set_postfix({
                "Loss": f"{total_loss.item():.4f}",
                "MMD": f"{loss_mmd.item():.4f}"
            })

        scheduler.step()

        model.eval()

        v_preds, v_targets = [], []
        with torch.no_grad():
            for vx, vy in val_loader:
                vx = vx.to(DEVICE)
                v_out, _ = model(vx)
                v_preds.extend(v_out.max(1)[1].cpu().numpy())
                v_targets.extend(vy.numpy())

        val_acc = accuracy_score(v_targets, v_preds) * 100

        t_preds, t_targets = [], []
        with torch.no_grad():
            for tx, ty in tgt_val_loader:
                tx = tx.to(DEVICE)
                t_out, _ = model(tx)
                t_preds.extend(t_out.max(1)[1].cpu().numpy())
                t_targets.extend(ty.numpy())

        tgt_acc = accuracy_score(t_targets, t_preds) * 100

        print(
            f"Epoch [{epoch+1:02d}] | "
            f"Loss: {running_loss / len_loader:.4f} | "
            f"MMD: {running_mmd / len_loader:.4f} | "
            f"Source Val: {val_acc:.2f}% | "
            f"Target Val: {tgt_acc:.2f}%"
        )

        if tgt_acc > best_target_acc:
            best_target_acc = tgt_acc
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print("   -> 🌟 Best Target Model Saved!")

    print(
        f"\n✅ 최종 결과 | "
        f"Best Target Acc: {best_target_acc:.2f}% | "
        f"Shift: {best_val_acc - best_target_acc:.2f}%"
    )

    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    model.eval()

    print("\n📊 Saving Source Domain Confusion Matrix...")

    v_preds_final, v_true_final = [], []

    with torch.no_grad():
        for vx, vy in val_loader:
            vx = vx.to(DEVICE)
            v_out, _ = model(vx)
            v_preds_final.extend(v_out.max(1)[1].cpu().numpy())
            v_true_final.extend(vy.numpy())

    cm_val = confusion_matrix(v_true_final, v_preds_final)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_val,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f"MMD (UDA) Source Prediction\n(Val Acc: {best_val_acc:.1f}%)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("results/mmd_source_confusion_matrix.png", dpi=300)
    plt.close()

    print("🔍 Saving Target Domain Confusion Matrix...")

    t_preds_final, t_true_final = [], []

    with torch.no_grad():
        for tx, ty in tgt_val_loader:
            tx = tx.to(DEVICE)
            t_out, _ = model(tx)
            t_preds_final.extend(t_out.max(1)[1].cpu().numpy())
            t_true_final.extend(ty.numpy())

    cm_tgt = confusion_matrix(t_true_final, t_preds_final)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_tgt,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(
        f"MMD (UDA) Target Prediction\n"
        f"(Source Val: {best_val_acc:.1f}% vs Target Val: {best_target_acc:.1f}%)"
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("results/mmd_target_confusion_matrix.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    train_mmd()