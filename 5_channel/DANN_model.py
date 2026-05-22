"""DANN 모델 정의 (단일 백본, 5채널 동기화 입력, discriminator BN 제거).

TCN feature extractor + label classifier + domain discriminator(GRL) 구조.
Gradient Reversal Layer 로 도메인 판별기에 적대적 그래디언트를 흘려보내
domain-invariant feature 를 학습한다.

실험 규약상 source/target 을 분리 forward 하므로, domain discriminator 내부에는
BatchNorm 을 두지 않는다. 분리 forward 시 discriminator 의 BatchNorm 이 도메인별
batch statistics 로 다시 정규화돼 적대 학습이 왜곡되는 것을 막기 위함이다.
Feature extractor 의 BatchNorm 은 그대로 유지한다.
"""
import os
import sys

import torch
import torch.nn as nn
from torch.autograd import Function

# 프로젝트 루트를 import 경로에 추가 (5_channel/ → ../)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline.baseline_model import ResidualTCNBlock  # noqa: E402


# ----------------------------------------------------------------------
# Gradient Reversal Layer (GRL)
# 순전파에서는 입력을 그대로 통과시키고, 역전파에서는 그래디언트에 -alpha 를 곱한다.
# ----------------------------------------------------------------------
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


# ----------------------------------------------------------------------
# DANN Model
# ----------------------------------------------------------------------
class DANNModel(nn.Module):
    """DANN 분류기. forward(x, alpha) -> (class_logits, domain_logits)."""

    def __init__(self, in_channels=5, num_classes=10):
        super(DANNModel, self).__init__()

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

        # [3] Domain Discriminator: Source(0) vs Target(1) 판별
        #     분리 forward 규약에 맞춰 내부 BatchNorm 을 두지 않는다.
        self.domain_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x, alpha=1.0):
        # 1. 공통 특징 추출
        features = self.feature_extractor(x).squeeze(-1)  # (B, 512)

        # 2. 라벨 분류 (운동 종목 예측)
        class_output = self.label_classifier(features)

        # 3. 도메인 분류 (GRL 을 거쳐 적대적 학습; alpha 는 학습 중 점진 증가)
        domain_output = self.domain_classifier(ReverseLayerF.apply(features, alpha))

        return class_output, domain_output


# ----------------------------------------------------------------------
# 단독 실행 테스트 (출력 shape 검증용)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    model = DANNModel(in_channels=5, num_classes=10)
    dummy_input = torch.randn(32, 5, 5000)

    c_out, d_out = model(dummy_input, alpha=0.1)

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Label Classifier Output    : {tuple(c_out.shape)}  (expected (32, 10))")
    print(f"Domain Discriminator Output: {tuple(d_out.shape)}  (expected (32, 2))")
    print(f"Total Trainable Parameters : {total:,}")
