"""TCN 기반 baseline 모델 정의 (AdvancedBaselineModel).

SE block 과 Residual TCN block 으로 구성된 시계열 분류 백본. 5채널 동기화 입력
(EMG 2ch + IMU 3ch)을 받아 운동 클래스를 예측한다. ResidualTCNBlock / SEBlock1D
는 CORAL/MMD/DANN 등 다른 모델에서도 재사용된다.
"""
import torch
import torch.nn as nn


class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation block (채널 어텐션)."""

    def __init__(self, channels, reduction=16):
        super(SEBlock1D, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)


class ResidualTCNBlock(nn.Module):
    """Dilated Conv 2개 + SE block + residual connection 으로 구성된 TCN block."""

    def __init__(self, in_channels, out_channels, kernel_size=5, dilation=1, dropout=0.5):
        super(ResidualTCNBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.se = SEBlock1D(out_channels)
        # 입출력 채널 수가 다르면 1x1 Conv 로 residual 경로를 맞춘다.
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.net(x)
        out = self.se(out)
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class AdvancedBaselineModel(nn.Module):
    """TCN baseline 분류기. 입력 (B, in_channels, 5000) -> (B, num_classes)."""

    def __init__(self, in_channels=5, num_classes=10):
        super(AdvancedBaselineModel, self).__init__()

        # Stem: 초기 Conv1d 로 채널 확장 및 시퀀스 길이 축소 (5000 -> 1000)
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=11, stride=5, padding=5),
            nn.BatchNorm1d(64),
            nn.GELU()
        )

        # TCN layers: dilation 을 1,2,4,8,16,32 로 키워 넓은 시간 범위를 모델링
        self.layers = nn.Sequential(
            ResidualTCNBlock(64, 64, dilation=1),
            ResidualTCNBlock(64, 128, dilation=2),
            nn.MaxPool1d(2),  # 1000 -> 500

            ResidualTCNBlock(128, 128, dilation=4),
            ResidualTCNBlock(128, 256, dilation=8),
            nn.MaxPool1d(2),  # 500 -> 250

            ResidualTCNBlock(256, 256, dilation=16),
            ResidualTCNBlock(256, 512, dilation=32),
            nn.AdaptiveAvgPool1d(1)  # 전역 평균 풀링으로 시퀀스를 요약
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = x.squeeze(-1)
        return self.classifier(x)


# ----------------------------------------------------------------------
# 단독 실행 테스트 (모델 파라미터 수 및 출력 shape 검증용)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    model = AdvancedBaselineModel(in_channels=5, num_classes=10)

    # 더미 데이터 생성 (배치 32, 채널 5, 길이 5000)
    dummy_input = torch.randn(32, 5, 5000)
    output = model(dummy_input)

    # 모델 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model successfully built.")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape} (Expected: 32, 10)")
    print(f"Total Trainable Parameters: {total_params:,}")
