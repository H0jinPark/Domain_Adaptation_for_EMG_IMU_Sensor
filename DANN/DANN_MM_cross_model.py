import torch
import torch.nn as nn
from torch.autograd import Function

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline.baseline_DualEncoder import EMGEncoder, IMUEncoder


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class DualEncoderDANN_Cross(nn.Module):
    """
    Label Classifier  : concat(f_emg, f_imu) → 384-dim  (모달리티 전체 정보 유지)
    Domain Discriminator: proj_emg(f_emg) ⊙ proj_imu(f_imu) → cross_dim  (cross-modal 상호작용만 입력)

    도메인 판별기가 단일 모달리티 특성 대신
    두 모달리티 간 곱(내적) 패턴으로 도메인을 구분하도록 강제.
    GRL은 이 cross-modal 패턴을 도메인-불변하게 만들도록 학습.
    """
    def __init__(self, num_classes=10, cross_dim=128):
        super().__init__()
        self.emg_encoder = EMGEncoder()          # (B, 2, 5000) → (B, 256)
        self.imu_encoder = IMUEncoder()          # (B, 3,  500) → (B, 128)

        self.emg_proj = nn.Linear(256, cross_dim)  # 내적을 위한 공통 차원으로 투영
        self.imu_proj = nn.Linear(128, cross_dim)

        self.label_classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(cross_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2),
        )

    def get_features(self, x_emg, x_imu):
        return torch.cat([self.emg_encoder(x_emg), self.imu_encoder(x_imu)], dim=1)

    def forward(self, x_emg, x_imu, alpha=1.0):
        f_emg = self.emg_encoder(x_emg)   # (B, 256)
        f_imu = self.imu_encoder(x_imu)   # (B, 128)

        class_output = self.label_classifier(torch.cat([f_emg, f_imu], dim=1))

        f_cross = self.emg_proj(f_emg) * self.imu_proj(f_imu)  # (B, cross_dim)
        domain_output = self.domain_classifier(ReverseLayerF.apply(f_cross, alpha))

        return class_output, domain_output


if __name__ == "__main__":
    model = DualEncoderDANN_Cross(num_classes=10, cross_dim=128)
    emg = torch.randn(32, 2, 5000)
    imu = torch.randn(32, 3, 500)
    cls_out, dom_out = model(emg, imu, alpha=0.5)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"class output : {tuple(cls_out.shape)}")
    print(f"domain output: {tuple(dom_out.shape)}")
    print(f"Total params : {total:,}")
