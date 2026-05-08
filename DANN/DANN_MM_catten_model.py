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


class CrossModalAttnStem(nn.Module):
    """
    Step 1. Depthwise temporal conv (채널별 독립 압축, 정보 혼합 없음)
            (B, 5, 5000) → (B*T, 5, d)   T=1000

    Step 2. Cross-channel Self-Attention
            5개 채널이 서로 정보를 주고받음 (EMG ↔ IMU interaction)

    Step 3. 모달리티별로 다시 분리
            EMG tokens [0:2] → (B, 2*d, T)   EMG 인코더 입력
            IMU tokens [2:5] → (B, 3*d, T)   IMU 인코더 입력
    """
    def __init__(self, in_ch=5, d_per_ch=32, nhead=4):
        super().__init__()
        self.in_ch    = in_ch
        self.d_per_ch = d_per_ch

        # 채널별 독립 temporal 압축: 5000 → 1000
        self.depthwise = nn.Conv1d(
            in_ch, in_ch * d_per_ch,
            kernel_size=11, stride=5, padding=5, groups=in_ch,
        )
        self.bn = nn.BatchNorm1d(in_ch * d_per_ch)

        # 5채널 token 간 Cross-modal Self-Attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_per_ch, nhead=nhead,
            dim_feedforward=d_per_ch * 4,
            dropout=0.1, batch_first=True,
        )
        self.channel_attn = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x):
        B = x.size(0)

        # (B, 5, 5000) → (B, 5*d, T)
        x = self.bn(self.depthwise(x))
        T = x.size(-1)  # 1000

        # (B, 5*d, T) → (B*T, 5, d)
        x = x.view(B, self.in_ch, self.d_per_ch, T)   # (B, 5, d, T)
        x = x.permute(0, 3, 1, 2).reshape(B * T, self.in_ch, self.d_per_ch)

        # Self-Attention: 5채널이 서로 정보 교환
        x = self.channel_attn(x)                       # (B*T, 5, d)

        # (B*T, 5, d) → (B, 5, d, T)
        x = x.reshape(B, T, self.in_ch, self.d_per_ch).permute(0, 2, 3, 1)

        # 모달리티별 분리
        # EMG: ch 0-1 → (B, 2*d, T)
        # IMU: ch 2-4 → (B, 3*d, T)
        f_emg = x[:, :2,  :, :].reshape(B, 2 * self.d_per_ch, T)
        f_imu = x[:, 2:5, :, :].reshape(B, 3 * self.d_per_ch, T)
        return f_emg, f_imu


class ModalAttnDANN(nn.Module):
    """
    진짜 멀티모달 구조:
    - CrossModalAttnStem이 EMG↔IMU 채널 간 interaction 수행
    - Attention 이후 각 모달리티가 독립 인코더로 분기
    - 최종 concat(384) → Label Classifier / Domain Discriminator(GRL)

    d_per_ch=32 기준:
        EMG encoder 입력: (B, 2*32=64,  1000)  → 256-dim
        IMU encoder 입력: (B, 3*32=96,  1000)  → 128-dim
    """
    def __init__(self, num_classes=10, d_per_ch=32, nhead=4):
        super().__init__()

        self.attn_stem = CrossModalAttnStem(in_ch=5, d_per_ch=d_per_ch, nhead=nhead)

        emg_in = 2 * d_per_ch   # 64
        imu_in = 3 * d_per_ch   # 96

        self.emg_encoder = nn.Sequential(
            ResidualTCNBlock(emg_in, 128, dilation=1),
            ResidualTCNBlock(128,    256, dilation=2),
            nn.MaxPool1d(2),
            ResidualTCNBlock(256, 256, dilation=4),
            ResidualTCNBlock(256, 256, dilation=8),
            nn.AdaptiveAvgPool1d(1),
        )

        self.imu_encoder = nn.Sequential(
            ResidualTCNBlock(imu_in, 64,  dilation=1),
            ResidualTCNBlock(64,     128, dilation=2),
            nn.MaxPool1d(2),
            ResidualTCNBlock(128, 128, dilation=4),
            nn.AdaptiveAvgPool1d(1),
        )

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
        f_emg_in, f_imu_in = self.attn_stem(x)
        f_emg = self.emg_encoder(f_emg_in).squeeze(-1)   # (B, 256)
        f_imu = self.imu_encoder(f_imu_in).squeeze(-1)   # (B, 128)
        return torch.cat([f_emg, f_imu], dim=1)          # (B, 384)

    def forward(self, x, alpha=1.0):
        features      = self.get_features(x)
        class_output  = self.label_classifier(features)
        domain_output = self.domain_classifier(ReverseLayerF.apply(features, alpha))
        return class_output, domain_output


if __name__ == "__main__":
    model = ModalAttnDANN(num_classes=10, d_per_ch=32, nhead=4)
    x = torch.randn(32, 5, 5000)
    cls_out, dom_out = model(x, alpha=0.5)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"class output : {tuple(cls_out.shape)}")
    print(f"domain output: {tuple(dom_out.shape)}")
    print(f"Total params : {total:,}")
