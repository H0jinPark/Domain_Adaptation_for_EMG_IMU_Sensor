import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from utils.visualizer import save_history_plot
from data_preprocess import preprocess_single_file 
from data_loader import get_dataloaders
from model import Simple1DCNN

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 학습 장치: {device} (메모리 직결 모드)")
    
    # 1. 파일 저장 없이 메모리에서 바로 전처리 결과 받기
    target_parquet = 'data/sensor_data.parquet'
    try:
        train_df, test_df = preprocess_single_file(target_parquet)
    except Exception as e:
        print(f"❌ 전처리 중 에러 발생: {e}")
        return

    # 2. DataLoader 생성 (🔥 Baseline 측정을 위해 'emg_only' 모드 추가!)
    train_loader, val_loader, le = get_dataloaders(
        train_df, 
        test_df, 
        window_size=2048, 
        step_size=1024,  # 50% Overlap
        batch_size=64,   # 3070 VRAM 최적화
        mode='imu_only'  # <--- 중요: IMU 데이터만 가져옵니다.
    )
    num_classes = len(le.classes_)

    # 3. 모델 동적 초기화 (데이터에서 채널 수를 자동으로 읽어옴)
    sample_inputs, _ = next(iter(train_loader))
    input_channels = sample_inputs.shape[1] # (Batch, Channel, Length)에서 Channel 추출
    print(f"💡 동적 할당: 입력 채널 수 {input_channels}개, 분류 클래스 {num_classes}개")

    model = Simple1DCNN(num_classes=num_classes, input_channels=input_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. 학습 루프
    num_epochs = 30
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        # --- [1. Training Phase] ---
        model.train()
        train_loss, train_correct = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # --- [2. Validation Phase] ---
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)

        # --- [3. 에포크 결과 계산 및 기록] ---
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = train_correct.cpu().item() / len(train_loader.dataset)
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_correct.cpu().item() / len(val_loader.dataset)

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        # --- [4. 에포크마다 정확도 출력] ---
        print(f"\n📊 Epoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"   [Train] Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc*100:.2f}%")
        print(f"   [Val]   Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc*100:.2f}%")
        print("-" * 40)

    print("\n" + "="*30)    
    print("🏆 모든 학습 완료! 최종 평가를 시작합니다.")
    
    # 5. 모델 저장 (Baseline임을 명시)
    model_save_path = "best_emg_baseline_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"💾 모델 저장 완료: {model_save_path}")

    # 6. 최종 Inference (Test Set 평가)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 7. 결과 요약 출력 및 시각화
    print("\n📝 [최종 분류 보고서]")
    print(classification_report(all_labels, all_preds, target_names=le.classes_))

    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
    plt.title('EMG Baseline Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    print("🎨 혼동 행렬 저장 완료: confusion_matrix.png")

    save_history_plot(history)

if __name__ == "__main__":
    train_model()
    print("✅ 모든 학습이 완료되었습니다! 수고하셨습니다!")