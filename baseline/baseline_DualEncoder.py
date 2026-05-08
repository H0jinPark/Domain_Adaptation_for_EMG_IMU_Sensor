import torch
import torch.nn as nn

# 입력 규약
#   x_emg : (B, 2, 5000)  — EMG 1000Hz, 5초
#   x_imu : (B, 3,  500)  — IMU  100Hz, 5초

try:
    from baseline_model import ResidualTCNBlock
except ModuleNotFoundError:
    from baseline.baseline_model import ResidualTCNBlock


class EMGEncoder(nn.Module):
    """EMG 전용 인코더 (B, 2, 5000) → (B, 256)
    - 고주파 근전도 신호(20~450Hz) 처리에 맞게 설계
    - Baseline TCN과 동일한 dilation 구조(1→2→4→8→16→32)
    """
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
            nn.MaxPool1d(2),                           # 1000 → 500
            ResidualTCNBlock(128, 128, dilation=4),
            ResidualTCNBlock(128, 256, dilation=8),
            nn.MaxPool1d(2),                           # 500 → 250
            ResidualTCNBlock(256, 256, dilation=16),
            ResidualTCNBlock(256, 256, dilation=32),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        return self.blocks(self.stem(x)).squeeze(-1)  # (B, 256)


class IMUEncoder(nn.Module):
    """IMU 전용 인코더 (B, 3, 500) → (B, 128)
    - 저주파 가속도/자이로 신호(100Hz, 5초) 처리
    - 가벼운 TCN으로 모션 패턴 포착
    """
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, stride=2, padding=2),  # 500 → 250
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResidualTCNBlock(32,  64,  dilation=1),
            ResidualTCNBlock(64,  128, dilation=2),
            nn.MaxPool1d(2),                           # 250 → 125
            ResidualTCNBlock(128, 128, dilation=4),
            ResidualTCNBlock(128, 128, dilation=8),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        return self.blocks(self.stem(x)).squeeze(-1)  # (B, 128)


class DualEncoderModel(nn.Module):
    """EMG + IMU 독립 인코딩 후 피처 융합 → 분류

    forward(x_emg, x_imu) → (B, num_classes)
    get_features(x_emg, x_imu) → (B, 384)  ← DA 방법에서 사용
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.emg_encoder = EMGEncoder()   # → 256
        self.imu_encoder = IMUEncoder()   # → 128
        # fused dim = 256 + 128 = 384
        self.classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def get_features(self, x_emg, x_imu):
        f_emg = self.emg_encoder(x_emg)
        f_imu = self.imu_encoder(x_imu)
        return torch.cat([f_emg, f_imu], dim=1)  # (B, 384)

    def forward(self, x_emg, x_imu):
        return self.classifier(self.get_features(x_emg, x_imu))


# =====================================================================
if __name__ == "__main__":
    model = DualEncoderModel(num_classes=10)
    emg = torch.randn(32, 2, 5000)
    imu = torch.randn(32, 3, 500)

    out  = model(emg, imu)
    feat = model.get_features(emg, imu)

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"EMG input : {tuple(emg.shape)}")
    print(f"IMU input : {tuple(imu.shape)}")
    print(f"Output    : {tuple(out.shape)}   (expected [32, 10])")
    print(f"Features  : {tuple(feat.shape)}  (expected [32, 384])")
    print(f"Total params: {total:,}")
