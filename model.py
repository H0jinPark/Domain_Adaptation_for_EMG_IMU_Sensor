import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple1DCNN(nn.Module):
    def __init__(self, num_classes, input_channels=8): # EMG 전용일 때 기본값 8
        super(Simple1DCNN, self).__init__()
        
        # 1. Feature Extractor (특징 추출기)
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # 2048 -> 1024
            
            # Layer 2
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # 1024 -> 512
            
            # Layer 3
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # 512 -> 256
        )
        
        # 2. Adaptive Pooling (입력 길이에 상관없이 출력 크기를 고정)
        # 256이라는 길이를 강제로 8 정도로 압축하거나, 아예 1로 만들 수 있습니다.
        self.adaptive_pool = nn.AdaptiveAvgPool1d(8) 
        
        # 3. Classifier (분류기)
        # 256(채널) * 8(압축된 길이) = 2048
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch, Channels, Length)
        x = self.features(x)
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        x = self.fc_layers(x)
        return x

if __name__ == "__main__":
    # --- 실험 1: EMG Only (채널 8개 가정) ---
    model_emg = Simple1DCNN(num_classes=10, input_channels=8)
    sample_emg = torch.randn(8, 8, 2048)
    print(f"✅ EMG 전용 출력: {model_emg(sample_emg).shape}")

    # --- 실험 2: Both (EMG 8 + IMU 21 = 29개 가정) ---
    model_both = Simple1DCNN(num_classes=10, input_channels=29)
    sample_both = torch.randn(8, 29, 2048)
    print(f"✅ 통합 모델 출력: {model_both(sample_both).shape}")