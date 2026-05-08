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


class SyncDualEncoderDANN(nn.Module):
    """
    입력: (B, 5, 5000) — preprocessed/ 데이터 그대로 사용 (EMG+IMU 1000Hz 동기화)

    [1] 공유 Stem: Conv1d(5→64) — 5채널 전부 한 번에 처리 → cross-modal 상호작용
    [2] 이후 분리:
        EMG branch (고주파 특성) → 256-dim
        IMU branch (저주파 특성) → 128-dim
    [3] concat(384) → Label Classifier / Domain Discriminator(GRL)
    """
    def __init__(self, num_classes=10):
        super().__init__()

        # 공유 Stem: 모든 채널을 함께 처리해 cross-channel 상호작용 확보
        self.shared_stem = nn.Sequential(
            nn.Conv1d(5, 64, kernel_size=11, stride=5, padding=5),  # 5000 → 1000
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

        # EMG branch: 고주파 근전도 특성 전담, 더 깊고 넓게
        self.emg_branch = nn.Sequential(
            ResidualTCNBlock(64,  128, dilation=1),
            ResidualTCNBlock(128, 256, dilation=2),
            nn.MaxPool1d(2),                        # 1000 → 500
            ResidualTCNBlock(256, 256, dilation=4),
            ResidualTCNBlock(256, 256, dilation=8),
            nn.AdaptiveAvgPool1d(1),
        )

        # IMU branch: 저주파 모션 특성 전담, 가볍게
        self.imu_branch = nn.Sequential(
            ResidualTCNBlock(64,  64,  dilation=1),
            ResidualTCNBlock(64,  128, dilation=2),
            nn.MaxPool1d(2),                        # 1000 → 500
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
        shared = self.shared_stem(x)
        f_emg  = self.emg_branch(shared).squeeze(-1)  # (B, 256)
        f_imu  = self.imu_branch(shared).squeeze(-1)  # (B, 128)
        return torch.cat([f_emg, f_imu], dim=1)       # (B, 384)

    def forward(self, x, alpha=1.0):
        features     = self.get_features(x)
        class_output = self.label_classifier(features)
        domain_output = self.domain_classifier(ReverseLayerF.apply(features, alpha))
        return class_output, domain_output


if __name__ == "__main__":
    model = SyncDualEncoderDANN(num_classes=10)
    x = torch.randn(32, 5, 5000)
    cls_out, dom_out = model(x, alpha=0.5)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"class output : {tuple(cls_out.shape)}")
    print(f"domain output: {tuple(dom_out.shape)}")
    print(f"Total params : {total:,}")
