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
from CoDATS_model import CoDATSModel


def train_codats():
    BATCH_SIZE = 64
    EPOCHS = 30
    LEARNING_RATE = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("weights", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    save_path = "weights/codats_best_model.pth"

    train_loader, val_loader, tgt_train_loader, tgt_val_loader, num_classes, le = get_dataloaders(
        batch_size=BATCH_SIZE
    )
    class_names = le.classes_

    model = CoDATSModel(in_channels=5, num_classes=num_classes).to(DEVICE)

    criterion_class = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_domain = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print("\n" + "=" * 50)
    print("🚀 CoDATS Adversarial Training Start!")
    print(f"Targeting {num_classes} classes on {DEVICE}")
    print("=" * 50)

    best_target_acc = 0.0
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()

        len_dataloader = min(len(train_loader), len(tgt_train_loader))
        data_zip = zip(train_loader, tgt_train_loader)

        total_loss, total_c_loss, total_d_loss = 0.0, 0.0, 0.0

        p = float(epoch) / EPOCHS
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

        pbar = tqdm(
            enumerate(data_zip),
            total=len_dataloader,
            desc=f"Epoch [{epoch+1:02d}/{EPOCHS}]"
        )

        for i, ((src_x, src_y), (tgt_x, tgt_y)) in pbar:
            src_x, src_y = src_x.to(DEVICE), src_y.to(DEVICE)
            tgt_x = tgt_x.to(DEVICE)

            optimizer.zero_grad()

            src_domain_label = torch.zeros(src_x.size(0), dtype=torch.long).to(DEVICE)
            tgt_domain_label = torch.ones(tgt_x.size(0), dtype=torch.long).to(DEVICE)

            src_class_out, src_domain_out = model(src_x, alpha=alpha)
            tgt_class_out, tgt_domain_out = model(tgt_x, alpha=alpha)

            loss_s_label = criterion_class(src_class_out, src_y)

            loss_s_domain = criterion_domain(src_domain_out, src_domain_label)
            loss_t_domain = criterion_domain(tgt_domain_out, tgt_domain_label)

            class_loss = loss_s_label
            domain_loss = loss_s_domain + loss_t_domain

            loss = class_loss + domain_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_c_loss += class_loss.item()
            total_d_loss += domain_loss.item()

            pbar.set_postfix({
                "Total": f"{loss.item():.4f}",
                "Class": f"{class_loss.item():.4f}",
                "Domain": f"{domain_loss.item():.4f}"
            })

        scheduler.step()

        model.eval()

        val_preds, val_targets = [], []
        with torch.no_grad():
            for vx, vy in val_loader:
                vx = vx.to(DEVICE)
                out, _ = model(vx, alpha=alpha)
                val_preds.extend(out.max(1)[1].cpu().numpy())
                val_targets.extend(vy.numpy())

        val_acc = accuracy_score(val_targets, val_preds) * 100

        tgt_preds, tgt_targets = [], []
        with torch.no_grad():
            for tx, ty in tgt_val_loader:
                tx = tx.to(DEVICE)
                out, _ = model(tx, alpha=alpha)
                tgt_preds.extend(out.max(1)[1].cpu().numpy())
                tgt_targets.extend(ty.numpy())

        tgt_acc = accuracy_score(tgt_targets, tgt_preds) * 100

        print(
            f"Epoch [{epoch+1:02d}/{EPOCHS}] | "
            f"Loss: {total_loss / len_dataloader:.4f} | "
            f"Class: {total_c_loss / len_dataloader:.4f} | "
            f"Domain: {total_d_loss / len_dataloader:.4f} | "
            f"Source Val: {val_acc:.2f}% | "
            f"Target Val: {tgt_acc:.2f}% | "
            f"Alpha: {alpha:.3f}"
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

    print(f"🚨 CoDATS Target Domain 정확도: {best_target_acc:.2f}%")
    print(f">> 최종 격차(Shift): {best_val_acc - best_target_acc:.2f}%")

    print("\n📊 Saving Source Domain Confusion Matrix...")

    v_preds_final, v_true_final = [], []
    with torch.no_grad():
        for vx, vy in val_loader:
            vx = vx.to(DEVICE)
            out, _ = model(vx, alpha=1.0)
            v_preds_final.extend(out.max(1)[1].cpu().numpy())
            v_true_final.extend(vy.numpy())

    cm_val = confusion_matrix(v_true_final, v_preds_final)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_val,
        annot=True,
        fmt="d",
        cmap="Greys",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f"CoDATS (UDA) Source Prediction\n(Val Acc: {best_val_acc:.1f}%)", fontsize=16)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/codats_source_confusion_matrix.png", dpi=300)
    plt.close()

    print("🔍 Saving Target Domain Confusion Matrix...")

    t_preds_final, t_true_final = [], []
    with torch.no_grad():
        for tx, ty in tgt_val_loader:
            tx = tx.to(DEVICE)
            out, _ = model(tx, alpha=1.0)
            t_preds_final.extend(out.max(1)[1].cpu().numpy())
            t_true_final.extend(ty.numpy())

    cm_tgt = confusion_matrix(t_true_final, t_preds_final)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_tgt,
        annot=True,
        fmt="d",
        cmap="Greys",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(
        f"CoDATS (UDA) Target Prediction\n"
        f"(Source Val: {best_val_acc:.1f}% vs Target: {best_target_acc:.1f}%)",
        fontsize=16
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/codats_target_confusion_matrix.png", dpi=300)
    print("📊 혼동 행렬 시각화가 모두 저장되었습니다.")
    plt.show()


if __name__ == "__main__":
    train_codats()