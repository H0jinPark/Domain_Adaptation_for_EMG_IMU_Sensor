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
from DANN_model import DANNModel

def train_dann():
    # =====================================================================
    # 1. 하이퍼파라미터 및 환경 설정
    # =====================================================================
    BATCH_SIZE = 64
    EPOCHS = 30 # DANN은 적대적 학습이라 조금 더 오래 돌리는게 좋습니다.
    LEARNING_RATE = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs('weights', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    save_path = 'weights/dann_best_model.pth'

    # 🌟 데이터 로더 (4분할 구조로 호출)
    train_loader, val_loader, tgt_train_loader, tgt_val_loader, num_classes, le = get_dataloaders(batch_size=BATCH_SIZE)
    class_names = le.classes_

    # 모델 선언 (🚨 5채널 원복 🚨)
    model = DANNModel(in_channels=5, num_classes=num_classes).to(DEVICE)
    
    # 손실 함수
    criterion_class = nn.CrossEntropyLoss(label_smoothing=0.1) # 운동 분류용
    criterion_domain = nn.CrossEntropyLoss() # 도메인 분류용 (Source vs Target)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print("\n" + "="*50)
    print("🚀 DANN Adversarial Training Start!")
    print(f"Targeting {num_classes} classes on {DEVICE}")
    print("="*50)

    # =====================================================================
    # 2. 메인 학습 루프
    # =====================================================================
    best_target_acc = 0.0
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        
        # Source와 Target(Train) 데이터를 동시에 로드
        len_dataloader = min(len(train_loader), len(tgt_train_loader))
        data_zip = zip(train_loader, tgt_train_loader)
        
        total_loss, total_c_loss, total_d_loss = 0, 0, 0
        
        # GRL의 Alpha 스케줄링
        p = float(epoch) / EPOCHS
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # 🌟 tqdm 프로그레스 바 적용
        pbar = tqdm(enumerate(data_zip), total=len_dataloader, desc=f"Epoch [{epoch+1:02d}/{EPOCHS}]")

        for i, ((src_x, src_y), (tgt_x, tgt_y)) in pbar:
            src_x, src_y = src_x.to(DEVICE), src_y.to(DEVICE)
            tgt_x, tgt_y = tgt_x.to(DEVICE), tgt_y.to(DEVICE)
            
            optimizer.zero_grad()
            
            # --- [Step 1] Source 데이터 학습 (운동 분류 + 도메인 분류) ---
            src_domain_label = torch.zeros(src_x.size(0), dtype=torch.long).to(DEVICE)
            src_class_out, src_domain_out = model(src_x, alpha=alpha)
            
            loss_s_label = criterion_class(src_class_out, src_y)
            loss_s_domain = criterion_domain(src_domain_out, src_domain_label)
            
            # --- [Step 2] Target 데이터 학습 (도메인 분류 + SDA 라벨 학습) ---
            tgt_domain_label = torch.ones(tgt_x.size(0), dtype=torch.long).to(DEVICE)
            tgt_class_out, tgt_domain_out = model(tgt_x, alpha=alpha)
            
            loss_t_domain = criterion_domain(tgt_domain_out, tgt_domain_label)
            #SDA
            # loss_t_label = criterion_class(tgt_class_out, tgt_y) 
            
            # --- [Step 3] 통합 손실 계산 및 역전파 ---
            domain_loss = loss_s_domain + loss_t_domain
            # class_loss = loss_s_label + loss_t_label
            class_loss = loss_s_label
            
            loss = class_loss + domain_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_c_loss += class_loss.item()
            total_d_loss += domain_loss.item()
            
            # 실시간 손실 업데이트
            pbar.set_postfix({
                'Total': f"{loss.item():.4f}", 
                'Class': f"{class_loss.item():.4f}",
                'Domain': f"{domain_loss.item():.4f}"
            })

        scheduler.step()
        
        # --- [Step 4] 검증 및 테스트 ---
        model.eval()
        
        # Source Validation 평가
        val_preds, val_targets = [], []
        with torch.no_grad():
            for vx, vy in val_loader:
                vx = vx.to(DEVICE)
                out, _ = model(vx, alpha=alpha)
                val_preds.extend(out.max(1)[1].cpu().numpy())
                val_targets.extend(vy.numpy())
        val_acc = accuracy_score(val_targets, val_preds) * 100
        
        # Unseen Target Validation 평가 (리키지 방지)
        tgt_preds, tgt_targets = [], []
        with torch.no_grad():
            for tx, ty in tgt_val_loader:
                tx = tx.to(DEVICE)
                out, _ = model(tx, alpha=alpha)
                tgt_preds.extend(out.max(1)[1].cpu().numpy())
                tgt_targets.extend(ty.numpy())
        tgt_acc = accuracy_score(tgt_targets, tgt_preds) * 100
        
        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | Loss: {total_loss/len_dataloader:.4f} | Source Val: {val_acc:.2f}% | Target Val: {tgt_acc:.2f}% | Alpha: {alpha:.3f}")

        # ✅ Model Selection Leakage 방지: Source Val 기준으로 최고 성능 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_target_acc = tgt_acc
            torch.save(model.state_dict(), save_path)
            print(f"   -> 🌟 Best Source Val Model Saved! (Source Val: {best_val_acc:.2f}%)")

    # =====================================================================
    # 3. 최종 결과 평가 및 시각화 저장
    # =====================================================================
    print("\n" + "="*50)
    print("📂 Saving Final Results...")
    print("="*50)
    
    # 최고 성능 모델 로드
    model.load_state_dict(torch.load(save_path))
    model.eval()
    
    print(
        f"\n✅ 최종 결과 | "
        f"Best Source Val Acc: {best_val_acc:.2f}% | "
        f"Target Acc at Best Source: {best_target_acc:.2f}% | "
        f"Shift: {best_val_acc - best_target_acc:.2f}%"
    )
    # --- Source Domain Confusion Matrix ---
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
    sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'DANN (UDA) Source Prediction\n(Val Acc: {best_val_acc:.1f}%)', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/dann_source_confusion_matrix.png', dpi=300)
    plt.close()

    # --- Target Domain (Unseen Validation) Confusion Matrix ---
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
    sns.heatmap(cm_tgt, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'DANN (UDA) Target Prediction\n(Source Val: {best_val_acc:.1f}% vs Target: {best_target_acc:.1f}%)', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/dann_target_confusion_matrix.png', dpi=300)
    print("📊 혼동 행렬 시각화가 모두 저장되었습니다.")
    plt.show()

if __name__ == "__main__":
    train_dann()