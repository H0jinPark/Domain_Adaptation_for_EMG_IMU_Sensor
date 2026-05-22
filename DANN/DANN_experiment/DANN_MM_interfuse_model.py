"""DANN-MM-InterFuse 모델 정의 (모달별 CL 사전학습 + intermediate fusion DANN).

Phase 1 (모달별 contrastive 사전학습)은 DANN_MM_ssl 과 동일하다.
Phase 2 융합을 인코더의 AdaptiveAvgPool **이전**으로 옮긴다:
  - 각 인코더의 pool 직전 시간 feature map 을 꺼내고 (EMG (B,256,500), IMU (B,256,50))
  - 시간축을 정렬해 (IMU 를 EMG 길이로 업샘플)
  - 채널축으로 stack -> (B, 512, T)  매 시각마다 EMG+IMU 가 함께
  - joint conv 가 시간축을 따라 두 modality 를 처리 -> cross-modal 시간 상관 학습
  - 그 다음 pool -> label/domain head

인코더는 modality 별로 완전히 분리되어 있다 (각자 자기 modality 만 입력받음).
융합은 두 인코더가 각자 시간 feature map 을 낸 *다음* 단계에서 일어난다.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class DualEncoderDANN_InterFuse(nn.Module):
    """모달별 인코더 + intermediate fusion DANN.

    forward(x_emg, x_imu, alpha) -> (class_logits, domain_logits)
    get_features(x_emg, x_imu)   -> (B, 512)  시간 융합 feature
    """

    def __init__(self, emg_encoder: EMGEncoder, imu_encoder: IMUEncoder, num_classes=10):
        super().__init__()
        self.emg_encoder = emg_encoder
        self.imu_encoder = imu_encoder

        # joint fusion: 채널 stack 된 (B,512,T) 시간 feature map 을 6층 TCN 으로 처리.
        # stem conv 가 매 시각마다 512채널(EMG256+IMU256)을 함께 보고, 6층 Residual
        # TCN 이 시간 범위를 넓혀 cross-modal 시간 상관(동시 발생·시간 차)을 학습한다.
        self.fusion = nn.Sequential(
            # Stem: concat 된 512채널 입력 + 가벼운 시퀀스 축소 (T=500 -> 250)
            nn.Conv1d(512, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            # 6층 Residual TCN (dilation 1,2,4,8,16,32) — 기본 DANN 모델과 동일한
            # dilation 패턴. 단 MaxPool 은 생략: 융합 입력 길이가 500(원본 5000의
            # 1/10)이라 풀링하면 dilation 16/32 의 시간 범위가 시퀀스를 넘어
            # 무의미해지므로 길이(250)를 유지한다.
            ResidualTCNBlock(64,  64,  dilation=1),
            ResidualTCNBlock(64,  128, dilation=2),
            ResidualTCNBlock(128, 128, dilation=4),
            ResidualTCNBlock(128, 256, dilation=8),
            ResidualTCNBlock(256, 256, dilation=16),
            ResidualTCNBlock(256, 512, dilation=32),
            nn.AdaptiveAvgPool1d(1),                                   # (B, 512, 1)
        )

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

    @classmethod
    def from_pretrained(cls, emg_path, imu_path, num_classes=10):
        """CL 사전학습된 인코더 가중치를 로드해 DualEncoderDANN_InterFuse 를 생성한다."""
        emg_enc = EMGEncoder()
        imu_enc = IMUEncoder()
        emg_enc.load_state_dict(torch.load(emg_path, map_location='cpu'))
        imu_enc.load_state_dict(torch.load(imu_path, map_location='cpu'))
        return cls(emg_enc, imu_enc, num_classes)

    @staticmethod
    def _temporal_map(encoder, x):
        """encoder 의 AdaptiveAvgPool 직전 시간 feature map 을 반환한다.  (B, 256, T)

        _TCNEncoder 는 stem -> layers(마지막 원소가 AdaptiveAvgPool) 구조이므로
        layers[:-1] 로 pool 직전까지만 통과시킨다.
        """
        return encoder.layers[:-1](encoder.stem(x))

    def get_features(self, x_emg, x_imu):
        """pool 직전 시간 feature map 을 정렬·stack·6층 TCN 으로 (B,512) 융합 feature 반환."""
        emg_map = self._temporal_map(self.emg_encoder, x_emg)   # (B, 256, 500)
        imu_map = self._temporal_map(self.imu_encoder, x_imu)   # (B, 256,  50)
        # 시간축 정렬: IMU 를 EMG 길이에 맞춰 업샘플 (정보 손실 없음)
        imu_map = F.interpolate(imu_map, size=emg_map.size(-1),
                                mode='linear', align_corners=True)
        # 채널 stack: 매 시각마다 EMG+IMU 가 함께 (B, 512, T)
        fused = torch.cat([emg_map, imu_map], dim=1)
        return self.fusion(fused).squeeze(-1)                   # (B, 512)

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
    model = DualEncoderDANN_InterFuse(emg_enc, imu_enc, num_classes=10)
    emg = torch.randn(16, 2, 5000)
    imu = torch.randn(16, 3,  500)

    em = model._temporal_map(emg_enc, emg)
    im = model._temporal_map(imu_enc, imu)
    print(f"EMG temporal map : {tuple(em.shape)}  (expected [16, 256, 500])")
    print(f"IMU temporal map : {tuple(im.shape)}  (expected [16, 256,  50])")

    feat = model.get_features(emg, imu)
    cls_out, dom_out = model(emg, imu, alpha=0.5)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"fused feature : {tuple(feat.shape)}  (expected [16, 512])")
    print(f"class output  : {tuple(cls_out.shape)}  (expected [16, 10])")
    print(f"domain output : {tuple(dom_out.shape)}  (expected [16, 2])")
    print(f"Total params  : {total:,}")
