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
    
    # 1. 파일 로드 및 3단계 분할 (Train: S1-7, Val: S8, Test: S9-10)
    target_parquet = 'data/sensor_data.parquet'
    try:
        # 수정된 preprocess_single_file은 3개의 DF를 반환합니다.
        train_df, val_df, test_df = preprocess_single_file(target_parquet)
    except Exception as e:
        print(f"❌ 데이터 로드 중 에러 발생: {e}")
        return

    # 2. DataLoader 생성 (통제 변인: Batch Size 256)
    # 2-1. 학습 및 검증용 로더
    train_loader, val_loader, le = get_dataloaders(
        train_df, 
        val_df, 
        window_size=2048, 
        step_size=1024,  
        batch_size=256,  
        mode='emg_only'  
    )
    # 2-2. 최종 평가용 테스트 로더 (Shuffle 없이 생성)
    _, test_loader, _ = get_dataloaders(
        train_df, 
        test_df, 
        window_size=2048, 
        step_size=1024,  
        batch_size=256,  
        mode='emg_only'
    )
    
    num_classes = len(le.classes_)

    # 3. 모델 초기화
    sample_inputs, _ = next(iter(train_loader))
    input_channels = sample_inputs.shape[1] 
    print(f"💡 동적 할당: 입력 채널 수 {input_channels}개, 분류 클래스 {num_classes}개")

    model = Simple1DCNN(num_classes=num_classes, input_channels=input_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)
    
    # 4. 학습 루프 세팅
    num_epochs = 300
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    # 얼리스토핑 설정 (타겟: Validation 피험자 S8)
    patience_limit = 20
    patience_counter = 0
    best_val_acc = 0.0
    model_save_path = "best_emg_baseline_model.pth"

    for epoch in range(num_epochs):
        # --- [Phase 1. Training] ---
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

        # --- [Phase 2. Validation (모의고사)] ---
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

        # 결과 기록
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = train_correct.cpu().item() / len(train_loader.dataset)
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_correct.cpu().item() / len(val_loader.dataset)

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        print(f"\n📊 Epoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"   [Train] Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc*100:.2f}%")
        print(f"   [Val]   Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc*100:.2f}%")

        # --- [Phase 3. Early Stopping 체크] ---
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            patience_counter = 0 
            torch.save(model.state_dict(), model_save_path)
            print(f"🔥 Best Model 갱신! (Val Acc: {best_val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            print(f"⚠️ 최고점 갱신 실패 (카운트: {patience_counter}/{patience_limit})")

        if patience_counter >= patience_limit:
            print(f"\n🛑 얼리스토핑 발동! {epoch+1} 에폭에서 종료합니다.")
            break
        print("-" * 40)

    print("\n" + "="*40)    
    print(f"🏆 학습 종료! 최고 검증 정확도(S8): {best_val_acc*100:.2f}%")
    
    # 5. 최적 모델 로드 후 최종 본시험(Test Set) 진행
    print("🚀 격리된 Test Set(S9-10)으로 최종 일반화 성능을 측정합니다...")
    model.load_state_dict(torch.load(model_save_path))

    # 6. 최종 Inference (Test Set 평가)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader: # 🚨 반드시 test_loader 사용!
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 7. 결과 출력 및 저장
    print("\n📝 [최종 논문용 분류 보고서 (Test Set)]")
    print(classification_report(all_labels, all_preds, target_names=le.classes_))

    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
    plt.title('EMG Baseline - Cross Subject Test Result', fontsize=14)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('baseline_cross_subject_cm.png', dpi=300)
    
    save_history_plot(history, save_path='baseline_training_history.png')
    print("🎨 혼동 행렬 및 학습 곡선 저장 완료!")

if __name__ == "__main__":
    train_model()
    print("✅ 베이스라인 Cross-Subject 실험이 완료되었습니다!")