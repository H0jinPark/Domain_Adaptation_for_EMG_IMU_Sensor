import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# 우리가 만든 모듈들 임포트
from data_preprocess import preprocess_single_file 
from data_loader import get_dataloaders
from CL_model import CrossModalCLModel, cl_distillation_loss
from utils.visualizer import save_history_plot

def train_cl_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 학습 장치: {device} (3070의 힘을 보여주세요!)")
    
    # 1. 데이터 로드 및 전처리
    target_parquet = 'data/sensor_data.parquet'
    train_df, test_df = preprocess_single_file(target_parquet)

    # 2. DataLoader 생성 (🔥 mode='both' 로 설정하여 두 센서 데이터를 모두 가져옴)
    emg_channels = 7
    imu_channels = 21
    
    train_loader, val_loader, le = get_dataloaders(
        train_df, 
        test_df, 
        window_size=2048, 
        step_size=1024,  
        batch_size=64,
        mode='both'  # EMG와 IMU 결합 데이터 (총 28채널)
    )
    num_classes = len(le.classes_)

    # 3. 모델 초기화
    model = CrossModalCLModel(
        imu_channels=imu_channels, 
        emg_channels=emg_channels, 
        num_classes=num_classes
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. 학습 루프 (지표 추적을 세분화합니다)
    num_epochs = 30
    history = {
        'train_loss': [], 'val_loss': [], 
        'train_emg_acc': [], 'val_emg_acc': [],
        'train_imu_acc': [], 'val_imu_acc': []
    }

    for epoch in range(num_epochs):
        # --- [1. Training Phase] ---
        model.train()
        train_loss, train_emg_correct, train_imu_correct = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 🔪 데이터 분리 (결합된 28채널을 다시 7채널, 21채널로 쪼갭니다)
            # data_loader에서 [emg_data, imu_data] 순서로 합쳤으므로
            emg_inputs = inputs[:, :emg_channels, :]
            imu_inputs = inputs[:, emg_channels:, :]
            
            optimizer.zero_grad()
            
            # 모델 통과 (Feature 추출 및 분류)
            imu_z, emg_z, imu_pred, emg_pred = model(imu_inputs, emg_inputs)
            
            # Loss 계산 (alpha=1.0 으로 Alignment Loss 비중을 동일하게 둠)
            total_loss, l_imu, l_emg, l_align = cl_distillation_loss(
                imu_z, emg_z, imu_pred, emg_pred, labels, alpha=2.0
            )
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item() * inputs.size(0)
            train_emg_correct += torch.sum(emg_pred.argmax(dim=1) == labels.data)
            train_imu_correct += torch.sum(imu_pred.argmax(dim=1) == labels.data)
            
            pbar.set_postfix({'loss': f"{total_loss.item():.4f}", 'align': f"{l_align.item():.4f}"})

        # --- [2. Validation Phase] ---
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

        # --- [3. 결과 계산 및 기록] ---
        n_train = len(train_loader.dataset)
        n_val = len(val_loader.dataset)
        
        history['train_loss'].append(train_loss / n_train)
        history['val_loss'].append(val_loss / n_val)
        history['train_emg_acc'].append((train_emg_correct / n_train).item())
        history['val_emg_acc'].append((val_emg_correct / n_val).item())
        history['train_imu_acc'].append((train_imu_correct / n_train).item())
        history['val_imu_acc'].append((val_imu_correct / n_val).item())

        print(f"\n📊 Epoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"   [Train] Loss: {history['train_loss'][-1]:.4f} | EMG Acc: {history['train_emg_acc'][-1]*100:.1f}% | IMU Acc: {history['train_imu_acc'][-1]*100:.1f}%")
        print(f"   [Val]   Loss: {history['val_loss'][-1]:.4f}   | EMG Acc: {history['val_emg_acc'][-1]*100:.1f}% | IMU Acc: {history['val_imu_acc'][-1]*100:.1f}%")
        print("-" * 50)

    print("\n🏆 지식 증류 학습 완료!")
    torch.save(model.state_dict(), "best_cl_distilled_model.pth")
    print("💾 모델 저장 완료: best_cl_distilled_model.pth")

if __name__ == "__main__":
    train_cl_model()