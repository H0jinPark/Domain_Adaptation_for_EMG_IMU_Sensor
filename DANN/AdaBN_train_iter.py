"""AdaBN 학습 스크립트 (단일 백본, 5채널 동기화 입력).

Source(Samsung1) 라벨만 사용해 운동 분류기를 학습한다.
동시에 매 iteration마다 Target(Samsung2) train 데이터를 라벨 없이 forward 하여
BatchNorm running statistics 를 target 분포에 지속적으로 적응시킨다.

이 스크립트는 DANN 에서 domain discriminator 와 GRL 을 제거한 비교군이다.
기존 DANN 의 source/target 분리 forward 구조 중 target forward 에 의한
implicit AdaBN 효과만 남겨, target 성능 향상이 BatchNorm adaptation 만으로
얼마나 발생하는지 확인한다.
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

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import get_dataloaders
from AdaBN_model_iter import AdaBNModel


def train_adabn():
    # ----------------------------------------------------------------------
    # 하이퍼파라미터 및 환경 설정
    # ----------------------------------------------------------------------
    BATCH_SIZE = 64
    EPOCHS = 30
    LEARNING_RATE = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs('weights', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    save_path = 'weights/adabn_iter_best_model.pth'

    # 데이터 로더 (Source/Target 의 train/val 4분할)
    train_loader, val_loader, tgt_train_loader, tgt_val_loader, num_classes, le = get_dataloaders(batch_size=BATCH_SIZE)
    class_names = le.classes_

    # 모델 선언 (EMG 2ch + IMU 3ch = 5채널 입력)
    model = AdaBNModel(in_channels=5, num_classes=num_classes).to(DEVICE)

    # 손실 함수
    criterion_class = nn.CrossEntropyLoss(label_smoothing=0.1)  # 운동 분류용

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print("\n" + "=" * 50)
    print("AdaBN Iteration-wise Training Start")
    print(f"Targeting {num_classes} classes on {DEVICE}")
    print("=" * 50)

    # ----------------------------------------------------------------------
    # 메인 학습 루프
    # ----------------------------------------------------------------------
    best_target_acc = 0.0
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()

        # Source 와 Target(train) 데이터를 동시에 로드
        len_dataloader = min(len(train_loader), len(tgt_train_loader))
        data_zip = zip(train_loader, tgt_train_loader)

        total_loss = 0.0

        pbar = tqdm(enumerate(data_zip), total=len_dataloader, desc=f"Epoch [{epoch+1:02d}/{EPOCHS:02d}]")

        for i, ((src_x, src_y), (tgt_x, tgt_y)) in pbar:
            src_x, src_y = src_x.to(DEVICE), src_y.to(DEVICE)
            tgt_x = tgt_x.to(DEVICE)

            optimizer.zero_grad()

            # Step 1. Source 데이터로 운동 분류 학습
            src_class_out = model(src_x)
            class_loss = criterion_class(src_class_out, src_y)

            # Step 2. Target 데이터는 라벨 없이 forward 만 수행
            # model.train() 상태이므로 BatchNorm running statistics 가 target 분포를 반영한다.
            with torch.no_grad():
                _ = model(tgt_x)

            # Step 3. Source classification loss 만 역전파
            class_loss.backward()
            optimizer.step()

            total_loss += class_loss.item()

            pbar.set_postfix({
                'Class': f"{class_loss.item():.4f}"
            })

        scheduler.step()

        # Step 4. 검증 및 평가
        model.eval()

        # Source validation 평가
        val_preds, val_targets = [], []
        with torch.no_grad():
            for vx, vy in val_loader:
                vx = vx.to(DEVICE)
                out = model(vx)
                val_preds.extend(out.max(1)[1].cpu().numpy())
                val_targets.extend(vy.numpy())
        val_acc = accuracy_score(val_targets, val_preds) * 100

        # Target validation 평가
        tgt_preds, tgt_targets = [], []
        with torch.no_grad():
            for tx, ty in tgt_val_loader:
                tx = tx.to(DEVICE)
                out = model(tx)
                tgt_preds.extend(out.max(1)[1].cpu().numpy())
                tgt_targets.extend(ty.numpy())
        tgt_acc = accuracy_score(tgt_targets, tgt_preds) * 100

        # Source val 기준 best 모델 저장
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_target_acc = tgt_acc
            torch.save(model.state_dict(), save_path)

        best_mark = "  (best)" if is_best else ""
        print(
            f"Epoch [{epoch+1:02d}/{EPOCHS:02d}] | "
            f"Class: {total_loss/len_dataloader:.4f} | "
            f"Source Val: {val_acc:.2f}% | "
            f"Target Val: {tgt_acc:.2f}%"
            f"{best_mark}"
        )

    # ----------------------------------------------------------------------
    # 최종 결과 평가 및 시각화 저장
    # ----------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Saving Final Results")
    print("=" * 50)

    # 최고 성능 모델 로드
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    model.eval()

    print(
        f"\n최종 결과 | "
        f"Best Source Val Acc: {best_val_acc:.2f}% | "
        f"Target Acc at Best Source: {best_target_acc:.2f}% | "
        f"Shift: {best_val_acc - best_target_acc:.2f}%"
    )

    # Source 도메인 confusion matrix
    print("\nSaving Source Domain Confusion Matrix...")
    v_preds_final, v_true_final = [], []
    with torch.no_grad():
        for vx, vy in val_loader:
            vx = vx.to(DEVICE)
            out = model(vx)
            v_preds_final.extend(out.max(1)[1].cpu().numpy())
            v_true_final.extend(vy.numpy())

    cm_val = confusion_matrix(v_true_final, v_preds_final)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'AdaBN Iter Source Prediction\n(Val Acc: {best_val_acc:.1f}%)', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/adabn_iter_source_confusion_matrix.png', dpi=300)
    plt.close()

    # Target 도메인(unseen validation) confusion matrix
    print("Saving Target Domain Confusion Matrix...")
    t_preds_final, t_true_final = [], []
    with torch.no_grad():
        for tx, ty in tgt_val_loader:
            tx = tx.to(DEVICE)
            out = model(tx)
            t_preds_final.extend(out.max(1)[1].cpu().numpy())
            t_true_final.extend(ty.numpy())

    cm_tgt = confusion_matrix(t_true_final, t_preds_final)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_tgt, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'AdaBN Iter Target Prediction\n(Source Val: {best_val_acc:.1f}% vs Target: {best_target_acc:.1f}%)', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/adabn_iter_target_confusion_matrix.png', dpi=300)
    print("혼동 행렬 시각화가 모두 저장되었습니다.")
    plt.show()


if __name__ == "__main__":
    train_adabn()
