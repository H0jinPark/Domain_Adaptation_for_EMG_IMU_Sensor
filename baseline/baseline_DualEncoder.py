"""멀티모달 DualEncoder 모델 정의.

EMG/IMU 를 각각 전용 인코더로 처리한 뒤 feature 를 concat 해 분류한다.
EMGEncoder / IMUEncoder 는 TCN baseline(AdvancedBaselineModel)과 같은
ResidualTCNBlock 으로 구성된 4층 인코더이며, 입력 채널 수만 다르고 출력은
256차원으로 같다. 이 두 인코더는 DANN-MM 등 멀티모달 DA 모델에서도 재사용된다.

입력 규약 (preprocessed_MM/)
  x_emg : (B, 2, 5000)  -- EMG 1000Hz, 5초
  x_imu : (B, 3,  500)  -- IMU  100Hz, 5초
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline.baseline_model import ResidualTCNBlock


class _TCNEncoder(nn.Module):
    """ResidualTCNBlock 4층으로 구성된 feature extractor.

    TCN baseline 과 동일한 stem + Residual TCN block 패턴을 따르되, 블록 4개
    (dilation 1,2,4,8)로 256차원까지만 키운다. 입력 채널 수만 인자로 받으며,
    출력은 시퀀스 길이와 무관하게 (B, 256) 이다 (AdaptiveAvgPool 로 요약).
    """

    def __init__(self, in_channels):
        super().__init__()
        # Stem: 초기 Conv1d 로 채널 확장 및 시퀀스 길이 축소
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=11, stride=5, padding=5),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        # Residual TCN layers: 4개 블록, dilation 1,2,4,8 로 시간 범위 확장
        self.layers = nn.Sequential(
            ResidualTCNBlock(64, 64, dilation=1),
            ResidualTCNBlock(64, 128, dilation=2),
            nn.MaxPool1d(2),
            ResidualTCNBlock(128, 128, dilation=4),
            ResidualTCNBlock(128, 256, dilation=8),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        return self.layers(self.stem(x)).squeeze(-1)  # (B, 256)


class EMGEncoder(_TCNEncoder):
    """EMG 전용 인코더.  (B, 2, 5000) -> (B, 256)"""

    def __init__(self):
        super().__init__(in_channels=2)


class IMUEncoder(_TCNEncoder):
    """IMU 전용 인코더.  (B, 3, 500) -> (B, 256)"""

    def __init__(self):
        super().__init__(in_channels=3)


class DualEncoderModel(nn.Module):
    """EMG/IMU 독립 인코딩 후 feature 융합 분류기.

    forward(x_emg, x_imu)      -> (B, num_classes)
    get_features(x_emg, x_imu) -> (B, 512)   # DA 방법에서 feature 로 사용
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.emg_encoder = EMGEncoder()   # -> 256
        self.imu_encoder = IMUEncoder()   # -> 256
        # 융합 차원 = 256 + 256 = 512
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def get_features(self, x_emg, x_imu):
        """EMG/IMU feature 를 concat 해 (B, 512) 융합 feature 를 반환한다."""
        f_emg = self.emg_encoder(x_emg)
        f_imu = self.imu_encoder(x_imu)
        return torch.cat([f_emg, f_imu], dim=1)  # (B, 512)

    def forward(self, x_emg, x_imu):
        return self.classifier(self.get_features(x_emg, x_imu))


# ----------------------------------------------------------------------
# 단독 실행 테스트 (출력 shape 검증용)
# ----------------------------------------------------------------------
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
    print(f"Features  : {tuple(feat.shape)}  (expected [32, 512])")
    print(f"Total params: {total:,}")
