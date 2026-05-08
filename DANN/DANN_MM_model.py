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


class DualEncoderDANN(nn.Module):
    """
    forward(x_emg, x_imu, alpha) → (class_logits, domain_logits)
    get_features(x_emg, x_imu)  → (B, 384)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.emg_encoder = EMGEncoder()  # (B, 2, 5000) → (B, 256)
        self.imu_encoder = IMUEncoder()  # (B, 3,  500) → (B, 128)
        # fused dim = 384

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

    def get_features(self, x_emg, x_imu):
        return torch.cat([self.emg_encoder(x_emg), self.imu_encoder(x_imu)], dim=1)

    def forward(self, x_emg, x_imu, alpha=1.0):
        features = self.get_features(x_emg, x_imu)
        class_output = self.label_classifier(features)
        domain_output = self.domain_classifier(ReverseLayerF.apply(features, alpha))
        return class_output, domain_output


if __name__ == "__main__":
    model = DualEncoderDANN(num_classes=10)
    emg = torch.randn(32, 2, 5000)
    imu = torch.randn(32, 3, 500)
    cls_out, dom_out = model(emg, imu, alpha=0.5)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"class output : {tuple(cls_out.shape)}")
    print(f"domain output: {tuple(dom_out.shape)}")
    print(f"Total params : {total:,}")
