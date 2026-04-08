import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# 모듈 임포트
from data_preprocess import preprocess_single_file 
from data_loader import get_dataloaders
from CL_model import CrossModalCLModel, cl_distillation_loss
from utils.visualizer import save_history_plot

def train_cl_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 학습 장치: {device}")
    
    # 1. 데이터 로드 및 3단계 분할 (S1-7: Train / S8: Val / S9-10: Test)
    target_parquet = 'data/sensor_data.parquet'
    train_df, val_df, test_df = preprocess_single_file(target_parquet)

    # 2. DataLoader 생성
    emg_channels = 7
    imu_channels = 21
    
    # 2-1. 학습 및 검증 로더
    train_loader, val_loader, le = get_dataloaders(
        train_df, val_df, 
        window_size=2048, step_size=1024, batch_size=256, mode='both'
    )
    # 2-2. 최종 평가용 테스트 로더
    _, test_loader, _ = get_dataloaders(
        train_df, test_df, 
        window_size=2048, step_size=1024, batch_size=256, mode='both'
    )
    
    num_classes = len(le.classes_)

    # 3. 모델 및 옵티마이저 초기화
    model = CrossModalCLModel(
        imu_channels=imu_channels, emg_channels=emg_channels, num_classes=num_classes
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)
    
    # 4. 학습 루프 및 전략적 제어 변수
    num_epochs = 300 
    history = {
        'train_loss': [], 'val_loss': [], 
        'train_emg_acc': [], 'val_emg_acc': [],
        'train_imu_acc': [], 'val_imu_acc': []
    }

    # IMU 뇌사(Freeze) 및 EMG 얼리스토핑 설정
    imu_patience_limit = 20
    imu_patience_counter = 0
    best_val_imu_acc = 0.0
    is_imu_frozen = False 

    emg_patience_limit = 20
    emg_patience_counter = 0
    best_val_emg_acc = 0.0
    best_model_path = "best_cl_model_cross_subject.pth"

    for epoch in range(num_epochs):
        # --- [Phase 1. Training] ---
        model.train()
        train_loss, train_emg_correct, train_imu_correct = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            emg_inputs = inputs[:, :emg_channels, :]
            imu_inputs = inputs[:, emg_channels:, :]
            
            optimizer.zero_grad()
            imu_z, emg_z, imu_pred, emg_pred = model(imu_inputs, emg_inputs)
            
            total_loss, l_imu, l_emg, l_align = cl_distillation_loss(
                imu_z, emg_z, imu_pred, emg_pred, labels, alpha=1
            )
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item() * inputs.size(0)
            train_emg_correct += torch.sum(emg_pred.argmax(dim=1) == labels.data)
            train_imu_correct += torch.sum(imu_pred.argmax(dim=1) == labels.data)
            pbar.set_postfix({'loss': f"{total_loss.item():.4f}"})

        # --- [Phase 2. Validation (S8)] ---
        model.eval()
        val_loss, val_emg_correct, val_imu_correct = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                emg_inputs = inputs[:, :emg_channels, :]
                imu_inputs = inputs[:, emg_channels:, :]
                
                imu_z, emg_z, imu_pred, emg_pred = model(imu_inputs, emg_inputs)
                total_loss, _, _, _ = cl_distillation_loss(imu_z, emg_z, imu_pred, emg_pred, labels)
                
                val_loss += total_loss.item() * inputs.size(0)
                val_emg_correct += torch.sum(emg_pred.argmax(dim=1) == labels.data)
                val_imu_correct += torch.sum(imu_pred.argmax(dim=1) == labels.data)

        # 결과 기록
        n_train, n_val = len(train_loader.dataset), len(val_loader.dataset)
        history['train_loss'].append(train_loss / n_train)
        history['val_loss'].append(val_loss / n_val)
        history['train_emg_acc'].append((train_emg_correct / n_train).item())
        history['val_emg_acc'].append((val_emg_correct / n_val).item())
        history['train_imu_acc'].append((train_imu_correct / n_train).item())
        history['val_imu_acc'].append((val_imu_correct / n_val).item())

        curr_imu_acc = history['val_imu_acc'][-1]
        curr_emg_acc = history['val_emg_acc'][-1]

        print(f"\n📊 Epoch [{epoch+1}/{num_epochs}] Summary (Val: S8):")
        print(f"   [Train] EMG Acc: {history['train_emg_acc'][-1]*100:.1f}% | IMU Acc: {history['train_imu_acc'][-1]*100:.1f}%")
        print(f"   [Val]   EMG Acc: {curr_emg_acc*100:.1f}% | IMU Acc: {curr_imu_acc*100:.1f}%")

        # --- [Phase 3. 전략적 제어 로직] ---
        # 3-1. IMU Freeze 체크
        if not is_imu_frozen:
            if curr_imu_acc > best_val_imu_acc:
                best_val_imu_acc = curr_imu_acc
                imu_patience_counter = 0
            else:
                imu_patience_counter += 1
            if imu_patience_counter >= imu_patience_limit:
                print(f"🚨 [IMU 뇌사] S8 데이터에서 IMU 성능 정체 -> IM우 인코더 동결!")
                for param in model.imu_encoder.parameters(): param.requires_grad = False
                is_imu_frozen = True
        
        # 3-2. EMG Early Stopping 및 모델 저장
        if curr_emg_acc > best_val_emg_acc:
            best_val_emg_acc = curr_emg_acc
            emg_patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"🔥 Best Model 갱신! (Val EMG Acc: {best_val_emg_acc*100:.2f}%)")
        else:
            emg_patience_counter += 1
            print(f"⚠️ EMG 정체 (카운트: {emg_patience_counter}/{emg_patience_limit})")

        if emg_patience_counter >= emg_patience_limit:
            print(f"🛑 [조기 종료] {epoch+1} 에폭에서 학습을 마칩니다.")
            break
        print("-" * 50)

    # 5. 최종 평가 (Test Set: S9-10)
    print("\n" + "="*50)
    print("🏆 학습 완료! 격리된 Test Set으로 최종 평가를 시작합니다...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            emg_inputs, imu_inputs = inputs[:, :emg_channels, :], inputs[:, emg_channels:, :]
            _, _, _, emg_pred = model(imu_inputs, emg_inputs)
            all_preds.extend(emg_pred.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 결과 리포트 및 시각화
    print("\n📝 [최종 논문용 CL 분류 보고서 (Test Set)]")
    print(classification_report(all_labels, all_preds, target_names=le.classes_))

    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
    plt.title('CL Model - Cross Subject Test Result', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted (EMG Only)', fontsize=12); plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/cl_cross_subject_cm.png', dpi=300)
    
    save_history_plot({'train_loss': history['train_loss'], 'val_loss': history['val_loss'], 
                       'train_acc': history['train_emg_acc'], 'val_acc': history['val_emg_acc']}, 
                      save_path='results/cl_cross_subject_history.png')
    print("✅ 모든 실험 결과 저장 완료!")

if __name__ == "__main__":
    train_cl_model()