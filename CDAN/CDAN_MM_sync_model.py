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


class SyncCDAN(nn.Module):
    """
    Sync 구조 + CDAN (Conditional Domain Adversarial Network)

    입력: (B, 5, 5000) — EMG 2ch + IMU 3ch, 1000Hz 동기화

    [1] 공유 Stem: Conv1d(5→64) — cross-channel 상호작용
    [2] EMG branch → 256-dim / IMU branch → 128-dim
    [3] concat(384) → Label Classifier → class prob (K)
    [4] CDAN 조건화 (각 branch 독립):
          EMG disc 입력: cat[GRL(f_emg)(256), softmax(K)] = 256+K
          IMU disc 입력: cat[GRL(f_imu)(128), softmax(K)] = 128+K

    forward(x, alpha) → (class_logits, dom_emg_logits, dom_imu_logits)
    get_features(x)   → (B, 384)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

        # Sync 공유 Stem
        self.shared_stem = nn.Sequential(
            nn.Conv1d(5, 64, kernel_size=11, stride=5, padding=5),  # 5000 → 1000
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

        # EMG branch
        self.emg_branch = nn.Sequential(
            ResidualTCNBlock(64,  128, dilation=1),
            ResidualTCNBlock(128, 256, dilation=2),
            nn.MaxPool1d(2),                        # 1000 → 500
            ResidualTCNBlock(256, 256, dilation=4),
            ResidualTCNBlock(256, 256, dilation=8),
            nn.AdaptiveAvgPool1d(1),
        )

        # IMU branch
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

        # EMG discriminator: cat[GRL(f_emg)(256), softmax(K)] = 256+K
        self.domain_classifier_emg = nn.Sequential(
            nn.Linear(256 + num_classes, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

        # IMU discriminator: cat[GRL(f_imu)(128), softmax(K)] = 128+K
        self.domain_classifier_imu = nn.Sequential(
            nn.Linear(128 + num_classes, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2),
        )

    def get_features(self, x):
        shared = self.shared_stem(x)
        f_emg  = self.emg_branch(shared).squeeze(-1)  # (B, 256)
        f_imu  = self.imu_branch(shared).squeeze(-1)  # (B, 128)
        return torch.cat([f_emg, f_imu], dim=1)       # (B, 384)

    def forward(self, x, alpha=1.0):
        shared = self.shared_stem(x)
        f_emg  = self.emg_branch(shared).squeeze(-1)  # (B, 256)
        f_imu  = self.imu_branch(shared).squeeze(-1)  # (B, 128)

        feat    = torch.cat([f_emg, f_imu], dim=1)        # (B, 384)
        cls_out = self.label_classifier(feat)               # (B, K)
        p       = torch.softmax(cls_out.detach(), dim=1)   # (B, K) — detach

        # EMG branch: CDAN 조건화
        f_emg_rev  = ReverseLayerF.apply(f_emg, alpha)
        dom_emg    = self.domain_classifier_emg(
            torch.cat([f_emg_rev, p], dim=1)              # (B, 256+K)
        )

        # IMU branch: CDAN 조건화
        f_imu_rev  = ReverseLayerF.apply(f_imu, alpha)
        dom_imu    = self.domain_classifier_imu(
            torch.cat([f_imu_rev, p], dim=1)              # (B, 128+K)
        )

        return cls_out, dom_emg, dom_imu


if __name__ == "__main__":
    model = SyncCDAN(num_classes=10)
    x = torch.randn(32, 5, 5000)
    cls_out, dom_emg, dom_imu = model(x, alpha=0.5)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"class output   : {tuple(cls_out.shape)}")
    print(f"dom_emg output : {tuple(dom_emg.shape)}")
    print(f"dom_imu output : {tuple(dom_imu.shape)}")
    print(f"Total params   : {total:,}")
