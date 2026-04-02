import torch
import torch.nn as nn
import torch.nn.functional as F


class Simple1DCNN(nn.Module):
    def __init__(self, num_classes, input_channels=28):
        super(Simple1DCNN, self, ).__init__()
        
        # 1. 첫 번째 컨볼루션 레이어
        # 입력: (Batch, 28, 2048) -> 출력: (Batch, 64, 2048)
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        
        # 2. 두 번째 컨볼루션 레이어
        # 출력: (Batch, 128, 1024) (Maxpool 이후)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        
        # 3. 세 번째 컨볼루션 레이어
        # 출력: (Batch, 256, 512) (Maxpool 이후)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.5)
        
        # 4. Fully Connected Layers
        # 2048 -> (pool 3번) -> 2048/8 = 256
        # 최종 특징 맵 크기: 256(채널) * 256(길이) = 65536
        self.fc1 = nn.Linear(256 * 256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (Batch, 28, 2048)
        
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # (Batch, 64, 1024)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # (Batch, 128, 512)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # (Batch, 256, 256)
        
        # Flatten: (Batch, 256 * 256)
        x = x.view(x.size(0), -1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

if __name__ == "__main__":
    # 모델 요약 확인용 (가상 데이터 투입)
    model = Simple1DCNN(num_classes=10) # 클래스 10개 가정
    
    # 🚨 테스트용 입력 데이터 길이도 2048로 변경했습니다!
    sample_input = torch.randn(8, 28, 2048) # (Batch, Channels, Length)
    output = model(sample_input)
    print(f"✅ 모델 출력 형태: {output.shape}") # (8, 10) 나와야 함