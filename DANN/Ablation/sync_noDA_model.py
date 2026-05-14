import torch
import torch.nn as nn

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from baseline.baseline_model import ResidualTCNBlock


class SyncDualEncoder(nn.Module):
    """
    Sync 구조에서 DANN(GRL + domain classifier)만 제거한 ablation 모델.
    입력: (B, 5, 5000) — EMG 2ch + IMU 3ch, 1000Hz 동기화

    [1] 공유 Stem: Conv1d(5→64) — cross-channel 상호작용
    [2] EMG branch → 256-dim
        IMU branch → 128-dim
    [3] concat(384) → Label Classifier
    """
    def __init__(self, num_classes=10):
        super().__init__()

        self.shared_stem = nn.Sequential(
            nn.Conv1d(5, 64, kernel_size=11, stride=5, padding=5),  # 5000 → 1000
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

        self.emg_branch = nn.Sequential(
            ResidualTCNBlock(64,  128, dilation=1),
            ResidualTCNBlock(128, 256, dilation=2),
            nn.MaxPool1d(2),                        # 1000 → 500
            ResidualTCNBlock(256, 256, dilation=4),
            ResidualTCNBlock(256, 256, dilation=8),
            nn.AdaptiveAvgPool1d(1),
        )

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

    def forward(self, x):
        shared = self.shared_stem(x)
        f_emg  = self.emg_branch(shared).squeeze(-1)   # (B, 256)
        f_imu  = self.imu_branch(shared).squeeze(-1)   # (B, 128)
        feat   = torch.cat([f_emg, f_imu], dim=1)      # (B, 384)
        return self.label_classifier(feat)


if __name__ == "__main__":
    model = SyncDualEncoder(num_classes=10)
    x = torch.randn(32, 5, 5000)
    out = model(x)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"output shape : {tuple(out.shape)}")
    print(f"Total params : {total:,}")
