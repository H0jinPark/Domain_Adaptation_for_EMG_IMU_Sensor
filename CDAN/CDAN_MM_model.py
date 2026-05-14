import torch
import torch.nn as nn
from torch.autograd import Function

import sys
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_ROOT)
sys.path.append(os.path.join(_ROOT, 'baseline'))
from baseline.baseline_DualEncoder import EMGEncoder, IMUEncoder


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class CDAN_MM(nn.Module):
    """
    Conditional Domain Adversarial Network — Multimodal (Dual Discriminator)

    EMG / IMU 각 모달리티 feature에 독립적인 discriminator를 배치.
    두 discriminator 모두 class 예측값으로 조건화(CDAN 방식).

    EMG discriminator 입력: cat[GRL(f_emg)(256), softmax(cls)(K)] = 256+K
    IMU discriminator 입력: cat[GRL(f_imu)(128), softmax(cls)(K)] = 128+K

    forward(x_emg, x_imu, alpha) → (class_logits, dom_emg_logits, dom_imu_logits)
    get_features(x_emg, x_imu)  → (B, 384)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.emg_encoder = EMGEncoder()   # (B, 2, 5000) → (B, 256)
        self.imu_encoder = IMUEncoder()   # (B, 3,  500) → (B, 128)

        self.label_classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        # EMG discriminator: cat[f_emg(256), softmax(K)] = 256+K
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

        # IMU discriminator: cat[f_imu(128), softmax(K)] = 128+K
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

        self.num_classes = num_classes

    def get_features(self, x_emg, x_imu):
        f_emg = self.emg_encoder(x_emg)   # (B, 256)
        f_imu = self.imu_encoder(x_imu)   # (B, 128)
        return torch.cat([f_emg, f_imu], dim=1)  # (B, 384)

    def forward(self, x_emg, x_imu, alpha=1.0):
        f_emg = self.emg_encoder(x_emg)   # (B, 256)
        f_imu = self.imu_encoder(x_imu)   # (B, 128)

        f       = torch.cat([f_emg, f_imu], dim=1)  # (B, 384)
        cls_out = self.label_classifier(f)            # (B, num_classes)
        p       = torch.softmax(cls_out.detach(), dim=1)  # (B, K) — detach

        # EMG branch
        f_emg_rev  = ReverseLayerF.apply(f_emg, alpha)
        dom_emg_in = torch.cat([f_emg_rev, p], dim=1)        # (B, 256+K)
        dom_emg    = self.domain_classifier_emg(dom_emg_in)  # (B, 2)

        # IMU branch
        f_imu_rev  = ReverseLayerF.apply(f_imu, alpha)
        dom_imu_in = torch.cat([f_imu_rev, p], dim=1)        # (B, 128+K)
        dom_imu    = self.domain_classifier_imu(dom_imu_in)  # (B, 2)

        return cls_out, dom_emg, dom_imu


if __name__ == "__main__":
    model = CDAN_MM(num_classes=10)
    emg = torch.randn(32, 2, 5000)
    imu = torch.randn(32, 3,  500)

    cls_out, dom_emg, dom_imu = model(emg, imu, alpha=0.5)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"class output   : {tuple(cls_out.shape)}")
    print(f"dom_emg output : {tuple(dom_emg.shape)}")
    print(f"dom_imu output : {tuple(dom_imu.shape)}")
    print(f"Total params   : {total:,}")
