"""DANN-MM-Stack 모델 정의 (모달별 CL 사전학습 + (B,2,256) stack 융합 DANN).

Phase 1(모달별 contrastive 사전학습)은 DANN_MM_ssl 과 정확히 동일하다.
다른 점은 Phase 2 융합 방식뿐:
  - Sep/SSL : EMG/IMU feature 를 옆으로 concat -> (B, 512) -> Linear 분류
  - Stack   : EMG/IMU feature 를 (B, 2, 256) 으로 stack -> 기본 DANN 모델
              (DANN_model.py)의 첫 1D conv 와 같은 형식의 conv 로 두 modality 를
              섞어 처리

2개 modality 를 채널로, 256 feature 를 길이축으로 보고 1D conv 를 적용한다.
EMG/IMU 인코더는 따로 학습돼 256개 feature 차원이 modality 간 대응되지 않으므로,
fusion conv 가 그 위에서 두 modality 를 결합하는 법을 학습한다.
"""
import torch
import torch.nn as nn
from torch.autograd import Function

import sys
import os
# 프로젝트 루트를 import 경로에 추가 (DANN/DANN_experiment/ → ../../)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from baseline.baseline_DualEncoder import EMGEncoder, IMUEncoder
from baseline.baseline_model import ResidualTCNBlock


class ReverseLayerF(Function):
    """역전파 시 그래디언트에 -alpha 를 곱하는 Gradient Reversal Layer."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class DualEncoderDANN_Stack(nn.Module):
    """모달별 인코더 + (B,2,256) stack 융합 DANN.

    forward(x_emg, x_imu, alpha) -> (class_logits, domain_logits)
    get_features(x_emg, x_imu)   -> (B, 256)  융합 feature
    """

    def __init__(self, emg_encoder: EMGEncoder, imu_encoder: IMUEncoder, num_classes=10):
        super().__init__()
        self.emg_encoder = emg_encoder
        self.imu_encoder = imu_encoder

        # (B,2,256) 융합: 기본 DANN 모델(DANN_model.py)의 첫 1D conv 와 같은 형식.
        # 2개 modality 를 채널로, 256 feature 를 길이축으로 본다.
        self.fusion = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=11, stride=5, padding=5),   # 256 -> 52
            nn.BatchNorm1d(64),
            nn.GELU(),
            ResidualTCNBlock(64, 128, dilation=1),
            ResidualTCNBlock(128, 256, dilation=2),
            nn.AdaptiveAvgPool1d(1),                                 # (B, 256, 1)
        )

        self.label_classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2),
        )

    @classmethod
    def from_pretrained(cls, emg_path, imu_path, num_classes=10):
        """CL 사전학습된 인코더 가중치를 로드해 DualEncoderDANN_Stack 를 생성한다."""
        emg_enc = EMGEncoder()
        imu_enc = IMUEncoder()
        emg_enc.load_state_dict(torch.load(emg_path, map_location='cpu'))
        imu_enc.load_state_dict(torch.load(imu_path, map_location='cpu'))
        return cls(emg_enc, imu_enc, num_classes)

    def get_features(self, x_emg, x_imu):
        """EMG/IMU feature 를 (B,2,256) 으로 stack 한 뒤 conv 융합해 (B,256) 반환."""
        f_emg = self.emg_encoder(x_emg)               # (B, 256)
        f_imu = self.imu_encoder(x_imu)               # (B, 256)
        stacked = torch.stack([f_emg, f_imu], dim=1)  # (B, 2, 256)
        return self.fusion(stacked).squeeze(-1)       # (B, 256)

    def forward(self, x_emg, x_imu, alpha=1.0):
        features = self.get_features(x_emg, x_imu)
        class_output = self.label_classifier(features)
        domain_output = self.domain_classifier(ReverseLayerF.apply(features, alpha))
        return class_output, domain_output


# ----------------------------------------------------------------------
# 단독 실행 테스트 (출력 shape 검증용)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    emg_enc = EMGEncoder()
    imu_enc = IMUEncoder()
    model = DualEncoderDANN_Stack(emg_enc, imu_enc, num_classes=10)
    emg = torch.randn(16, 2, 5000)
    imu = torch.randn(16, 3,  500)

    feat = model.get_features(emg, imu)
    cls_out, dom_out = model(emg, imu, alpha=0.5)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"fused feature : {tuple(feat.shape)}  (expected [16, 256])")
    print(f"class output  : {tuple(cls_out.shape)}  (expected [16, 10])")
    print(f"domain output : {tuple(dom_out.shape)}  (expected [16, 2])")
    print(f"Total params  : {total:,}")
