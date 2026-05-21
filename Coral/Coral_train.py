"""CORAL 학습 스크립트 (단일 백본, 5채널 동기화 입력).

TCN baseline 백본의 feature 에 CORAL 손실(source/target covariance alignment)을
추가해 Source 라벨만으로 Target 운동 분류 성능을 끌어올린다.
"""
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


def coral_loss(source, target):
    """Source/Target feature 의 공분산 행렬 차이를 Frobenius norm 으로 측정한다."""
    d = source.size(1)
    n_s = source.size(0)
    n_t = target.size(0)

    src_mean = torch.mean(source, 0, keepdim=True)
    src_cov = (source - src_mean).t() @ (source - src_mean) / (n_s - 1)

    tgt_mean = torch.mean(target, 0, keepdim=True)
    tgt_cov = (target - tgt_mean).t() @ (target - tgt_mean) / (n_t - 1)

    loss = torch.norm(src_cov - tgt_cov, p="fro")
    loss = loss / (4 * d * d)
    return loss


class CORALWrapper(nn.Module):
    """baseline 백본을 감싸 (분류 logits, feature) 를 함께 반환한다."""

    def __init__(self, base_model):
        super(CORALWrapper, self).__init__()
        self.backbone = base_model

    def forward(self, x):
        x = self.backbone.stem(x)
        features = self.backbone.layers(x)
        features = features.squeeze(-1)
        out = self.backbone.classifier(features)
        return out, features


def train_coral():
    # ----------------------------------------------------------------------
    # 하이퍼파라미터 및 환경 설정
    # ----------------------------------------------------------------------
    BATCH_SIZE = 64
    EPOCHS = 30
    LEARNING_RATE = 1e-3
    LAMBDA_CORAL = 0.5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"CORAL Training Start | 디바이스: {DEVICE}")

    os.makedirs("weights", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    save_path = "weights/coral_best_source_model.pth"

    train_loader, val_loader, tgt_train_loader, tgt_val_loader, num_classes, le = get_dataloaders(
        batch_size=BATCH_SIZE
    )
    class_names = le.classes_

    base_model = AdvancedBaselineModel(in_channels=5, num_classes=num_classes)
    model = CORALWrapper(base_model).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print("\n" + "=" * 50)
    print("CORAL Alignment Training Start")
    print("=" * 50)

    # ----------------------------------------------------------------------
    # 메인 학습 루프
    # ----------------------------------------------------------------------
    best_source_acc = 0.0
    best_target_acc = 0.0
    best_epoch = 0

    for epoch in range(EPOCHS):
        model.train()

        data_zip = zip(train_loader, tgt_train_loader)
        len_loader = min(len(train_loader), len(tgt_train_loader))

        running_loss, running_coral = 0.0, 0.0

        pbar = tqdm(
            enumerate(data_zip),
            total=len_loader,
            desc=f"Epoch [{epoch + 1:02d}/{EPOCHS:02d}]"
        )

        for i, ((src_x, src_y), (tgt_x, tgt_y)) in pbar:
            src_x, src_y = src_x.to(DEVICE), src_y.to(DEVICE)
            tgt_x = tgt_x.to(DEVICE)

            optimizer.zero_grad()

            src_out, src_feat = model(src_x)
            tgt_out, tgt_feat = model(tgt_x)

            loss_cls = criterion(src_out, src_y)
            loss_coral = coral_loss(src_feat, tgt_feat)
            total_loss = loss_cls + LAMBDA_CORAL * loss_coral

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            running_coral += loss_coral.item()

            pbar.set_postfix({
                "Loss": f"{total_loss.item():.4f}",
                "CORAL": f"{loss_coral.item():.4f}"
            })

        scheduler.step()

        # 검증 및 평가
        model.eval()

        # Source validation 평가
        v_preds, v_targets = [], []
        with torch.no_grad():
            for vx, vy in val_loader:
                vx = vx.to(DEVICE)
                v_out, _ = model(vx)
                v_preds.extend(v_out.max(1)[1].cpu().numpy())
                v_targets.extend(vy.numpy())

        val_acc = accuracy_score(v_targets, v_preds) * 100

        # Target validation 평가
        t_preds, t_targets = [], []
        with torch.no_grad():
            for tx, ty in tgt_val_loader:
                tx = tx.to(DEVICE)
                t_out, _ = model(tx)
                t_preds.extend(t_out.max(1)[1].cpu().numpy())
                t_targets.extend(ty.numpy())

        tgt_acc = accuracy_score(t_targets, t_preds) * 100

        # Source val 기준 best 모델 저장
        is_best = val_acc > best_source_acc
        if is_best:
            best_source_acc = val_acc
            best_target_acc = tgt_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), save_path)

        best_mark = "  (best)" if is_best else ""
        print(
            f"Epoch [{epoch + 1:02d}/{EPOCHS:02d}] | "
            f"Loss: {running_loss / len_loader:.4f} | "
            f"CORAL: {running_coral / len_loader:.4f} | "
            f"Source Val: {val_acc:.2f}% | "
            f"Target Val: {tgt_acc:.2f}%"
            f"{best_mark}"
        )

    print(
        f"\n최종 결과 | "
        f"Best Source Val Acc: {best_source_acc:.2f}% | "
        f"Target Acc at Best Source: {best_target_acc:.2f}% | "
        f"Shift: {best_source_acc - best_target_acc:.2f}%"
    )

    # ----------------------------------------------------------------------
    # 최종 결과 평가 및 시각화 저장
    # ----------------------------------------------------------------------
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    model.eval()

    # Source 도메인 confusion matrix
    print("\nSaving Source Domain Confusion Matrix...")

    v_preds_final, v_true_final = [], []
    with torch.no_grad():
        for vx, vy in val_loader:
            vx = vx.to(DEVICE)
            v_out, _ = model(vx)
            v_preds_final.extend(v_out.max(1)[1].cpu().numpy())
            v_true_final.extend(vy.numpy())

    final_source_acc = accuracy_score(v_true_final, v_preds_final) * 100
    cm_val = confusion_matrix(v_true_final, v_preds_final)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_val,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f"CORAL (UDA) Source Prediction\n(Val Acc: {final_source_acc:.1f}%)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("results/coral_source_confusion_matrix.png", dpi=300)
    plt.close()

    # Target 도메인 confusion matrix
    print("Saving Target Domain Confusion Matrix...")

    t_preds_final, t_true_final = [], []
    with torch.no_grad():
        for tx, ty in tgt_val_loader:
            tx = tx.to(DEVICE)
            t_out, _ = model(tx)
            t_preds_final.extend(t_out.max(1)[1].cpu().numpy())
            t_true_final.extend(ty.numpy())

    final_target_acc = accuracy_score(t_true_final, t_preds_final) * 100
    cm_tgt = confusion_matrix(t_true_final, t_preds_final)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_tgt,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(
        f"CORAL (UDA) Target Prediction\n"
        f"(Source Val: {final_source_acc:.1f}% vs Target Val: {final_target_acc:.1f}%)"
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("results/coral_target_confusion_matrix.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    train_coral()
