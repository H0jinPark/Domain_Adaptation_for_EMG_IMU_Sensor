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
from baseline.baseline_model import AdvancedBaselineModel

# =====================================================================
# 1. CORAL Loss 정의
# =====================================================================
def coral_loss(source, target):
    """
    소스와 타겟의 공분산 행렬(Covariance Matrix) 사이의 거리를 계산합니다.
    """
    d = source.size(1)  # feature dim (512)
    n_s = source.size(0)
    n_t = target.size(0)

    # Covariance 계산
    src_mean = torch.mean(source, 0, keepdim=True)
    src_cov = (source - src_mean).t() @ (source - src_mean) / (n_s - 1)

    tgt_mean = torch.mean(target, 0, keepdim=True)
    tgt_cov = (target - tgt_mean).t() @ (target - tgt_mean) / (n_t - 1)

    # Frobenius norm
    loss = torch.norm(src_cov - tgt_cov, p='fro')
    loss = loss / (4 * d * d)
    return loss

# =====================================================================
# 2. 특징 추출을 위한 모델 래퍼 (Wrapper)
# =====================================================================
class CORALWrapper(nn.Module):
    def __init__(self, base_model):
        super(CORALWrapper, self).__init__()
        self.backbone = base_model # AdvancedBaselineModel
        
    def forward(self, x):
        # 기존 모델의 구조를 활용해 특징과 분류 결과를 분리해서 반환
        x = self.backbone.stem(x)
        features = self.backbone.layers(x)
        features = features.squeeze(-1) # (Batch, 512) - 이게 CORAL이 쓰일 특징
        out = self.backbone.classifier(features)
        return out, features

# =====================================================================
# 3. 메인 학습 루프
# =====================================================================
def train_coral():
    BATCH_SIZE = 64
    EPOCHS = 30
    LEARNING_RATE = 1e-3
    LAMBDA_CORAL = 0.5 # CORAL 손실의 가중치 (실험하며 조정 필요)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🔥 CORAL Training Start! Using device: {DEVICE}")
    os.makedirs('weights', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    save_path = 'weights/coral_best_model.pth'

    # 🌟 데이터 로더 (4분할 구조로 호출)
    train_loader, val_loader, tgt_train_loader, tgt_val_loader, num_classes, le = get_dataloaders(batch_size=BATCH_SIZE)
    class_names = le.classes_

    # 모델 준비 (5채널 원복)
    base_model = AdvancedBaselineModel(in_channels=5, num_classes=num_classes)
    model = CORALWrapper(base_model).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print("\n" + "="*50)
    print("📏 CORAL Alignment Training Start!")
    print("="*50)

    best_target_acc = 0.0
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        # Source와 Target(Train) 데이터를 동시에 로드
        data_zip = zip(train_loader, tgt_train_loader)
        len_loader = min(len(train_loader), len(tgt_train_loader))
        
        running_loss, running_coral = 0, 0

        # 🌟 tqdm 프로그레스 바 적용
        pbar = tqdm(enumerate(data_zip), total=len_loader, desc=f"Epoch [{epoch+1:02d}/{EPOCHS}]")

        for i, ((src_x, src_y), (tgt_x, tgt_y)) in pbar:
            src_x, src_y = src_x.to(DEVICE), src_y.to(DEVICE)
            tgt_x, tgt_y = tgt_x.to(DEVICE), tgt_y.to(DEVICE)

            optimizer.zero_grad()

            # 1. 모델 피드포워드 (특징 추출 및 분류)
            src_out, src_feat = model(src_x)
            tgt_out, tgt_feat = model(tgt_x)
                        
            loss_cls_src = criterion(src_out, src_y)
            loss_cls = loss_cls_src
            loss_coral = coral_loss(src_feat, tgt_feat)
            total_loss = loss_cls + LAMBDA_CORAL * loss_coral
            
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            running_coral += loss_coral.item()

            # 실시간 업데이트
            pbar.set_postfix({
                'Loss': f"{total_loss.item():.4f}", 
                'CORAL': f"{loss_coral.item():.4f}"
            })

        scheduler.step()

        # --- 검증 및 테스트 ---
        model.eval()
        
        # Source Validation 평가
        v_preds, v_targets = [], []
        with torch.no_grad():
            for vx, vy in val_loader:
                vx = vx.to(DEVICE)
                v_out, _ = model(vx)
                v_preds.extend(v_out.max(1)[1].cpu().numpy())
                v_targets.extend(vy.numpy())
        val_acc = accuracy_score(v_targets, v_preds) * 100

        # Unseen Target Validation 평가 (리키지 방지)
        t_preds, t_targets = [], []
        with torch.no_grad():
            for tx, ty in tgt_val_loader:
                tx = tx.to(DEVICE)
                t_out, _ = model(tx)
                t_preds.extend(t_out.max(1)[1].cpu().numpy())
                t_targets.extend(ty.numpy())
        tgt_acc = accuracy_score(t_targets, t_preds) * 100

        print(f"Epoch [{epoch+1:02d}] | Loss: {running_loss/len_loader:.4f} | CORAL: {running_coral/len_loader:.4f} | Source Val: {val_acc:.2f}% | Target Val: {tgt_acc:.2f}%")

        # Target Validation 기준으로 Best 모델 저장
        if tgt_acc > best_target_acc:
            best_target_acc = tgt_acc
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"   -> 🌟 Best Target Model Saved!")

    # --- 최종 결과 및 시각화 ---
    print(f"\n✅ 최종 결과 | Best Target Acc: {best_target_acc:.2f}% | Shift: {best_val_acc - best_target_acc:.2f}%")
    
    model.load_state_dict(torch.load(save_path))
    model.eval()

    # 1. Source Domain (Validation) 시각화
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
    sns.heatmap(cm_val, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'CORAL (UDA) Source Prediction\n(Val Acc: {best_val_acc:.1f}%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('results/coral_source_confusion_matrix.png', dpi=300)
    plt.close() # 메모리 관리를 위해 창 닫기

    # 2. Target Domain (Unseen Validation) 시각화
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
    sns.heatmap(cm_tgt, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'CORAL (UDA) Target Prediction\n(Source Val: {best_val_acc:.1f}% vs Target Val: {best_target_acc:.1f}%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('results/coral_target_confusion_matrix.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    train_coral()