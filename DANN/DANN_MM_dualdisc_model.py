"""DANN-MM-DualDisc 모델 정의 (멀티모달, 모달리티별 도메인 판별기).

DANN-MM 과 달리 EMG/IMU 각 모달리티에 별도의 GRL + Domain Discriminator 를 두어
도메인 손실을 2개 둔다. 라벨 분류는 두 모달리티 feature 를 concat 해 수행한다.
"""
import torch
import torch.nn as nn
from torch.autograd import Function

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline.baseline_DualEncoder import EMGEncoder, IMUEncoder


class ReverseLayerF(Function):
    """Gradient Reversal Layer: 역전파 시 그래디언트에 -alpha 를 곱한다."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class DualEncoderDualDiscDANN(nn.Module):
    """모달리티별 도메인 판별기를 갖는 멀티모달 DANN.

    forward(x_emg, x_imu, alpha) -> (class_logits, dom_emg_logits, dom_imu_logits)
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.emg_encoder = EMGEncoder()   # (B, 2, 5000) -> (B, 256)
        self.imu_encoder = IMUEncoder()   # (B, 3,  500) -> (B, 256)

        # 라벨 분류기는 concat 된 융합 feature(256+256=512)를 사용
        self.label_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        # 모달리티별 Domain Discriminator (각 모달리티 feature 는 256차원)
        self.domain_emg = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2),
        )
        self.domain_imu = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2),
        )

    def get_features(self, x_emg, x_imu):
        """EMG/IMU feature 를 각각 (B, 256) 으로 반환한다."""
        f_emg = self.emg_encoder(x_emg)   # (B, 256)
        f_imu = self.imu_encoder(x_imu)   # (B, 256)
        return f_emg, f_imu

    def forward(self, x_emg, x_imu, alpha=1.0):
        f_emg, f_imu = self.get_features(x_emg, x_imu)
        class_output = self.label_classifier(torch.cat([f_emg, f_imu], dim=1))
        # 모달리티별 GRL + 도메인 판별
        dom_emg = self.domain_emg(ReverseLayerF.apply(f_emg, alpha))
        dom_imu = self.domain_imu(ReverseLayerF.apply(f_imu, alpha))
        return class_output, dom_emg, dom_imu


# ----------------------------------------------------------------------
# 단독 실행 테스트 (출력 shape 검증용)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    model = DualEncoderDualDiscDANN(num_classes=10)
    emg = torch.randn(32, 2, 5000)
    imu = torch.randn(32, 3, 500)
    cls_out, dom_e, dom_i = model(emg, imu, alpha=0.5)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"class output : {tuple(cls_out.shape)}")
    print(f"dom EMG out  : {tuple(dom_e.shape)}")
    print(f"dom IMU out  : {tuple(dom_i.shape)}")
    print(f"Total params : {total:,}")
