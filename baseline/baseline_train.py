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

# 상위 폴더의 data_loader와 baseline_model 임포트
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import get_dataloaders
from baseline_model import AdvancedBaselineModel

def train_baseline():
    # --- 하이퍼파라미터 ---
    BATCH_SIZE = 64
    EPOCHS = 20
    LEARNING_RATE = 1e-3
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 학습 시작! 사용 디바이스: {device}")
    
    os.makedirs('weights', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    save_path = 'weights/baseline_best.pth'

    # --- 데이터 로더 (4분할 구조로 수정) ---
    # get_dataloaders()가 train, val, tgt_train, tgt_val 4개의 로더를 반환하도록 맞춰져 있다고 가정합니다.
    train_loader, val_loader, tgt_train_loader, tgt_val_loader, num_classes, le = get_dataloaders(batch_size=BATCH_SIZE)
    class_names = le.classes_

    # --- 모델 및 최적화 (5채널 원복) ---
    model = AdvancedBaselineModel(in_channels=5, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print("\n" + "="*50)
    print("🚀 [Phase 1] Source Domain Training (Session-wise Split)")
    print("="*50)
    
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        # 1. Training
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        # tqdm 프로그레스 바 적용
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch [{epoch+1:02d}/{EPOCHS}]")
        
        for batch_x, batch_y in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_x.size(0)
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
            
            # 실시간 Loss 업데이트
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        scheduler.step()
        train_acc = 100. * correct / total
        
        # 2. Validation (Source Val)
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()
                
        val_acc = 100. * val_correct / val_total
        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Best 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"   -> 🌟 Best model saved! (Val Acc: {best_val_acc:.2f}%)")
            
    print(f"\n✅ 학습 완료! 최고 Val 정확도: {best_val_acc:.2f}%")

    # --- 3. Source Domain (Validation) 평가 및 시각화 ---
    print("\n" + "="*50)
    print("📊 [Phase 2] Source Domain (Validation) Evaluation")
    print("="*50)
    
    model.load_state_dict(torch.load(save_path))
    model.eval()
    val_preds, val_targets = [], []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            _, predicted = outputs.max(1)
            val_preds.extend(predicted.cpu().numpy())
            val_targets.extend(batch_y.numpy())

    # Source Domain Confusion Matrix 시각화
    cm_val = confusion_matrix(val_targets, val_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_val, annot=True, fmt='d', cmap='Oranges',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Baseline Source Prediction\n(Val Acc: {best_val_acc:.1f}%)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('results/baseline_source_confusion_matrix.png', dpi=300)
    plt.show()

    # --- 4. Target Domain 평가 및 시각화 ---
    print("\n" + "="*50)
    print("🔍 [Phase 3] Target Domain Evaluation (Unseen Target Val Check)")
    print("="*50)
    
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        # Baseline은 tgt_train_loader를 학습에 쓰지 않았으므로, 최종 평가는 엄밀하게 분리된 tgt_val_loader로 수행
        for batch_x, batch_y in tgt_val_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.numpy())
            
    tgt_acc = accuracy_score(all_targets, all_preds) * 100
    print(f"🚨 Target Domain(Val) 정확도: {tgt_acc:.2f}%")
    print(f">> 10개 운동 유형 도메인 격차(Shift): {best_val_acc - tgt_acc:.2f}%")

    # Target Domain Confusion Matrix 시각화
    cm_tgt = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_tgt, annot=True, fmt='d', cmap='Oranges',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Baseline Target Prediction\n(Val: {best_val_acc:.1f}% vs Target Val: {tgt_acc:.1f}%)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('results/baseline_target_confusion_matrix.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    train_baseline()