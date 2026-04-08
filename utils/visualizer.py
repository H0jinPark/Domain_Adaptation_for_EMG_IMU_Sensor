import matplotlib.pyplot as plt
import os

def save_history_plot(history, save_path='learning_history.png'):
    """
    학습 과정의 Loss와 Accuracy 변화를 시각화하여 저장합니다.
    
    Args:
        history (dict): 'train_loss', 'val_loss', 'train_acc', 'val_acc' 리스트를 포함한 딕셔너리
        save_path (str): 그래프 이미지를 저장할 경로
    """
    epochs = range(1, len(history['train_loss']) + 1)

    # 그래프 스타일 설정 (Seaborn 스타일과 유사하게)
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- 1. Loss Plot ---
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', markersize=4, alpha=0.7)
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Val Loss', markersize=4, alpha=0.7)
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- 2. Accuracy Plot ---
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Train Acc', markersize=4, alpha=0.7)
    ax2.plot(epochs, history['val_acc'], 'r-s', label='Val Acc', markersize=4, alpha=0.7)
    ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_ylim(0, 1.05) # 정확도는 0~1 사이이므로 범위를 고정
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # 레이아웃 조정 및 저장
    plt.tight_layout()
    
    # 저장 경로가 존재하지 않으면 폴더 생성
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    plt.savefig(save_path, dpi=300) # 논문용으로 고해상도 저장
    plt.close() # 메모리 해제
    print(f"📈 학습 히스토리 그래프 저장 완료: {save_path}")

if __name__ == "__main__":
    # 테스트용 가짜 데이터
    test_history = {
        'train_loss': [0.9, 0.7, 0.5],
        'val_loss': [1.0, 0.8, 0.6],
        'train_acc': [0.5, 0.7, 0.85],
        'val_acc': [0.45, 0.65, 0.8]
    }
    save_history_plot(test_history)