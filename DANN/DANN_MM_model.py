"""DANN-MM 모델 정의 (멀티모달, EMG/IMU 분리 입력).

DualEncoder(EMG/IMU 독립 인코더) 위에 label classifier 와 domain
discriminator(GRL)를 얹은 멀티모달 DANN. EMG/IMU feature 를 concat 한
512차원 융합 feature 에 도메인 적대 학습을 적용한다.
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


class DualEncoderDANN(nn.Module):
    """멀티모달 DANN 분류기.

    forward(x_emg, x_imu, alpha) -> (class_logits, domain_logits)
    get_features(x_emg, x_imu)   -> (B, 512)
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.emg_encoder = EMGEncoder()  # (B, 2, 5000) -> (B, 256)
        self.imu_encoder = IMUEncoder()  # (B, 3,  500) -> (B, 256)
        # 융합 차원 = 256 + 256 = 512

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
        """EMG/IMU feature 를 concat 해 (B, 512) 융합 feature 를 반환한다."""
        return torch.cat([self.emg_encoder(x_emg), self.imu_encoder(x_imu)], dim=1)

    def forward(self, x_emg, x_imu, alpha=1.0):
        features = self.get_features(x_emg, x_imu)
        class_output = self.label_classifier(features)
        # GRL 을 거쳐 도메인 판별 (적대적 학습)
        domain_output = self.domain_classifier(ReverseLayerF.apply(features, alpha))
        return class_output, domain_output


# ----------------------------------------------------------------------
# 단독 실행 테스트 (출력 shape 검증용)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    model = DualEncoderDANN(num_classes=10)
    emg = torch.randn(32, 2, 5000)
    imu = torch.randn(32, 3, 500)
    cls_out, dom_out = model(emg, imu, alpha=0.5)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"class output : {tuple(cls_out.shape)}")
    print(f"domain output: {tuple(dom_out.shape)}")
    print(f"Total params : {total:,}")
