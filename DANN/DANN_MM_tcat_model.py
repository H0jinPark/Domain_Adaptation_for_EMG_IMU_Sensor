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


class EMGTemporalEncoder(nn.Module):
    """(B, 2, 5000) → (B, 32, 256)  feature map (temporal 정보 보존)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=11, stride=5, padding=5),  # 5000 → 1000
            nn.BatchNorm1d(32),
            nn.GELU(),
            ResidualTCNBlock(32, 32, dilation=1),
            nn.MaxPool1d(2),                                         # 1000 → 500
            ResidualTCNBlock(32, 32, dilation=2),
            nn.AdaptiveAvgPool1d(256),                               # → 256
        )

    def forward(self, x):
        return self.net(x)  # (B, 32, 256)


class IMUTemporalEncoder(nn.Module):
    """(B, 3, 500) → (B, 32, 256)  feature map (temporal 정보 보존)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, stride=2, padding=2),  # 500 → 250
            nn.BatchNorm1d(32),
            nn.GELU(),
            ResidualTCNBlock(32, 32, dilation=1),
            nn.AdaptiveAvgPool1d(256),                              # → 256
        )

    def forward(self, x):
        return self.net(x)  # (B, 32, 256)


class TemporalConcatDANN(nn.Module):
    """
    두 인코더가 temporal feature map을 출력한 뒤 channel-concat → 공유 TCN stack

    [1] EMGTemporalEncoder: (B, 2, 5000) → (B, 32, 256)
    [2] IMUTemporalEncoder: (B, 3,  500) → (B, 32, 256)
    [3] concat(channel dim): → (B, 64, 256)
    [4] Conv1d(64→64) → baseline TCN stack → (B, 512)
    [5] Label Classifier / Domain Discriminator(GRL)
    """
    def __init__(self, num_classes=10):
        super().__init__()

        self.emg_encoder = EMGTemporalEncoder()
        self.imu_encoder = IMUTemporalEncoder()

        # (B, 64, 256) 입력을 받아 baseline 구조로 처리
        self.fusion_tcn = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),

            ResidualTCNBlock(64,  64,  dilation=1),
            ResidualTCNBlock(64,  128, dilation=2),
            nn.MaxPool1d(2),                         # 256 → 128

            ResidualTCNBlock(128, 128, dilation=4),
            ResidualTCNBlock(128, 256, dilation=8),
            nn.MaxPool1d(2),                         # 128 → 64

            ResidualTCNBlock(256, 256, dilation=16),
            ResidualTCNBlock(256, 512, dilation=32),
            nn.AdaptiveAvgPool1d(1),
        )

        self.label_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2),
        )

    def get_features(self, x_emg, x_imu):
        f_emg = self.emg_encoder(x_emg)                    # (B, 32, 256)
        f_imu = self.imu_encoder(x_imu)                    # (B, 32, 256)
        fused = torch.cat([f_emg, f_imu], dim=1)           # (B, 64, 256)
        return self.fusion_tcn(fused).squeeze(-1)           # (B, 512)

    def forward(self, x_emg, x_imu, alpha=1.0):
        features      = self.get_features(x_emg, x_imu)
        class_output  = self.label_classifier(features)
        domain_output = self.domain_classifier(ReverseLayerF.apply(features, alpha))
        return class_output, domain_output


if __name__ == "__main__":
    model = TemporalConcatDANN(num_classes=10)
    emg = torch.randn(32, 2, 5000)
    imu = torch.randn(32, 3, 500)
    cls_out, dom_out = model(emg, imu, alpha=0.5)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"EMG input    : {tuple(emg.shape)}")
    print(f"IMU input    : {tuple(imu.shape)}")
    print(f"class output : {tuple(cls_out.shape)}")
    print(f"domain output: {tuple(dom_out.shape)}")
    print(f"Total params : {total:,}")
