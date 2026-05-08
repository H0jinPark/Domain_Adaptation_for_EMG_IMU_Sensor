import torch
import torch.nn as nn
from torch.autograd import Function

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline.baseline_model import ResidualTCNBlock


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class SplitDualEncoderDANN(nn.Module):
    """
    Ablation: 스케일만 맞춘 독립 인코더 (shared stem 없음)

    입력: (B, 5, 5000) — preprocessed/ 데이터 (1000Hz 동기화)
    모델 내부에서 채널을 분리:
        EMG: x[:, :2, :]  → 독립 인코더 → 256-dim
        IMU: x[:, 2:, :]  → 독립 인코더 → 128-dim
    두 인코더는 처음부터 끝까지 서로 정보를 공유하지 않음.

    DANN_MM_sync_model과 비교:
        sync  → shared Conv1d(5→64) 후 분기  (cross-channel interaction O)
        split → 채널 분리 후 각자 독립 처리  (cross-channel interaction X)
    """
    def __init__(self, num_classes=10):
        super().__init__()

        # EMG 전용 인코더: ch 0-1, 1000Hz 5초 = 5000 points
        self.emg_encoder = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=11, stride=5, padding=5),  # 5000 → 1000
            nn.BatchNorm1d(64),
            nn.GELU(),
            ResidualTCNBlock(64,  128, dilation=1),
            ResidualTCNBlock(128, 256, dilation=2),
            nn.MaxPool1d(2),                                         # 1000 → 500
            ResidualTCNBlock(256, 256, dilation=4),
            ResidualTCNBlock(256, 256, dilation=8),
            nn.AdaptiveAvgPool1d(1),
        )

        # IMU 전용 인코더: ch 2-4, 동일 5000 points (1000Hz 업샘플)
        self.imu_encoder = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=11, stride=5, padding=5),  # 5000 → 1000
            nn.BatchNorm1d(32),
            nn.GELU(),
            ResidualTCNBlock(32,  64,  dilation=1),
            ResidualTCNBlock(64,  128, dilation=2),
            nn.MaxPool1d(2),                                         # 1000 → 500
            ResidualTCNBlock(128, 128, dilation=4),
            nn.AdaptiveAvgPool1d(1),
        )

        # fused dim = 256 + 128 = 384
        self.label_classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2),
        )

    def get_features(self, x):
        f_emg = self.emg_encoder(x[:, :2, :]).squeeze(-1)  # (B, 256)
        f_imu = self.imu_encoder(x[:, 2:, :]).squeeze(-1)  # (B, 128)
        return torch.cat([f_emg, f_imu], dim=1)            # (B, 384)

    def forward(self, x, alpha=1.0):
        features      = self.get_features(x)
        class_output  = self.label_classifier(features)
        domain_output = self.domain_classifier(ReverseLayerF.apply(features, alpha))
        return class_output, domain_output


if __name__ == "__main__":
    model = SplitDualEncoderDANN(num_classes=10)
    x = torch.randn(32, 5, 5000)
    cls_out, dom_out = model(x, alpha=0.5)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"class output : {tuple(cls_out.shape)}")
    print(f"domain output: {tuple(dom_out.shape)}")
    print(f"Total params : {total:,}")
