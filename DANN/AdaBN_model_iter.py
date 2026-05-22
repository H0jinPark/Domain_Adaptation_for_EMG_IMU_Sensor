"""AdaBN 모델 정의 (단일 백본, 5채널 동기화 입력).

TCN feature extractor + label classifier 구조.
Domain discriminator 와 GRL 은 제거하고, target train 데이터를 매 iteration마다
forward 하여 BatchNorm running statistics 를 target 분포에 적응시키는 구조.
"""
import torch
import torch.nn as nn
from baseline.baseline_model import ResidualTCNBlock, SEBlock1D


# ----------------------------------------------------------------------
# AdaBN Model
# ----------------------------------------------------------------------
class AdaBNModel(nn.Module):
    """AdaBN 분류기. forward(x) -> class_logits."""

    def __init__(self, in_channels=5, num_classes=10):
        super(AdaBNModel, self).__init__()

        # [1] Feature Extractor: 공통 특징 추출 (TCN 백본)
        self.feature_extractor = nn.Sequential(
            # Stem (5000 -> 1000)
            nn.Conv1d(in_channels, 64, kernel_size=11, stride=5, padding=5),
            nn.BatchNorm1d(64),
            nn.GELU(),

            # Residual TCN blocks
            ResidualTCNBlock(64, 64, dilation=1),
            ResidualTCNBlock(64, 128, dilation=2),
            nn.MaxPool1d(2),  # 1000 -> 500

            ResidualTCNBlock(128, 128, dilation=4),
            ResidualTCNBlock(128, 256, dilation=8),
            nn.MaxPool1d(2),  # 500 -> 250

            ResidualTCNBlock(256, 256, dilation=16),
            ResidualTCNBlock(256, 512, dilation=32),
            nn.AdaptiveAvgPool1d(1)  # (B, 512, 1)
        )

        # [2] Label Classifier: 운동 종목 분류
        self.label_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # 1. 공통 특징 추출
        features = self.feature_extractor(x)
        features = features.squeeze(-1)  # (B, 512)

        # 2. 라벨 분류 (운동 종목 예측)
        class_output = self.label_classifier(features)

        return class_output


# ----------------------------------------------------------------------
# 단독 실행 테스트 (출력 shape 검증용)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    model = AdaBNModel(in_channels=5, num_classes=10)
    dummy_input = torch.randn(32, 5, 5000)

    # 순전파 테스트
    c_out = model(dummy_input)

    print(f"Label Classifier Output: {c_out.shape}  (expected [32, 10])")
