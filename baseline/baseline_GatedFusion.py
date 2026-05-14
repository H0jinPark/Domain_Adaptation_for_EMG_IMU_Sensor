import torch
import torch.nn as nn

# 입력 규약
#   x_emg : (B, 2, 5000)  — EMG 1000Hz, 5초
#   x_imu : (B, 3,  500)  — IMU  100Hz, 5초

from baseline_model import ResidualTCNBlock


class EMGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=11, stride=5, padding=5),  # 5000 → 1000
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResidualTCNBlock(64,  64,  dilation=1),
            ResidualTCNBlock(64,  128, dilation=2),
            nn.MaxPool1d(2),                          # 1000 → 500
            ResidualTCNBlock(128, 128, dilation=4),
            ResidualTCNBlock(128, 256, dilation=8),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        return self.blocks(self.stem(x)).squeeze(-1)  # (B, 256)


class IMUEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, stride=1, padding=2),  # 500 → 500
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResidualTCNBlock(32,  64,  dilation=1),
            ResidualTCNBlock(64,  128, dilation=2),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        return self.blocks(self.stem(x)).squeeze(-1)  # (B, 128)


class GatedFusionModel(nn.Module):
    """EMG + IMU를 element-wise gate로 융합 → 분류

    gate 는 두 모달리티 feature를 입력으로 받아
    각 feature 차원마다 EMG vs IMU 기여도를 학습함.

    forward(x_emg, x_imu) → (B, num_classes)
    get_features(x_emg, x_imu) → (B, 256)  ← DA 방법에서 사용
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.emg_encoder = EMGEncoder()            # → 256
        self.imu_encoder = IMUEncoder()            # → 128
        self.imu_proj    = nn.Linear(128, 256)     # IMU를 EMG와 같은 차원으로 투영

        # gate: concat(f_emg, f_imu_proj) → (B, 512) → (B, 256) ∈ (0, 1)
        # 각 feature 차원별로 EMG / IMU 중 어느 쪽을 더 신뢰할지 학습
        self.gate = nn.Sequential(
            nn.Linear(512, 256),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def get_features(self, x_emg, x_imu):
        f_emg      = self.emg_encoder(x_emg)           # (B, 256)
        f_imu_proj = self.imu_proj(self.imu_encoder(x_imu))  # (B, 256)

        g      = self.gate(torch.cat([f_emg, f_imu_proj], dim=1))  # (B, 256)
        fused  = g * f_emg + (1 - g) * f_imu_proj                  # (B, 256)
        return fused

    def forward(self, x_emg, x_imu):
        return self.classifier(self.get_features(x_emg, x_imu))


# =====================================================================
if __name__ == "__main__":
    model = GatedFusionModel(num_classes=10)
    emg = torch.randn(32, 2, 5000)
    imu = torch.randn(32, 3,  500)

    out  = model(emg, imu)
    feat = model.get_features(emg, imu)

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"EMG input : {tuple(emg.shape)}")
    print(f"IMU input : {tuple(imu.shape)}")
    print(f"Output    : {tuple(out.shape)}   (expected [32, 10])")
    print(f"Features  : {tuple(feat.shape)}  (expected [32, 256])")
    print(f"Total params: {total:,}")
