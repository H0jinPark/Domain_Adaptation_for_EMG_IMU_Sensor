"""DANN-MM-Sep 모델 정의 (분리 사전학습 + 융합 DANN).

Phase 1: EMGEncoderWithClassifier / IMUEncoderWithClassifier 를 source data 로 독립 학습.
Phase 2: 사전학습된 인코더를 DualEncoderDANN_Sep 에 이식 후 GRL 도메인 적대 학습.
두 페이즈 모두 인코더 파라미터를 업데이트한다 (동결 없음).
"""
import torch
import torch.nn as nn
from torch.autograd import Function

import sys
import os
# 프로젝트 루트를 import 경로에 추가 (DANN/DANN_experiment/ → ../../)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from baseline.baseline_DualEncoder import EMGEncoder, IMUEncoder


class ReverseLayerF(Function):
    """역전파 시 그래디언트에 -alpha 를 곱하는 Gradient Reversal Layer."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class EMGEncoderWithClassifier(nn.Module):
    """Phase 1 EMG 단독 분류기.  (B, 2, 5000) -> (B, num_classes)"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.encoder = EMGEncoder()   # (B, 2, 5000) -> (B, 256)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x_emg):
        return self.classifier(self.encoder(x_emg))


class IMUEncoderWithClassifier(nn.Module):
    """Phase 1 IMU 단독 분류기.  (B, 3, 500) -> (B, num_classes)"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.encoder = IMUEncoder()   # (B, 3, 500) -> (B, 256)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x_imu):
        return self.classifier(self.encoder(x_imu))


class DualEncoderDANN_Sep(nn.Module):
    """Phase 2: 사전학습 인코더 기반 멀티모달 DANN.

    forward(x_emg, x_imu, alpha) -> (class_logits, domain_logits)
    get_features(x_emg, x_imu)   -> (B, 512)
    """

    def __init__(self, emg_encoder: EMGEncoder, imu_encoder: IMUEncoder, num_classes=10):
        super().__init__()
        self.emg_encoder = emg_encoder
        self.imu_encoder = imu_encoder
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

    @classmethod
    def from_pretrained(cls, emg_path, imu_path, num_classes=10):
        """저장된 인코더 가중치를 로드해 DualEncoderDANN_Sep 를 생성한다."""
        emg_enc = EMGEncoder()
        imu_enc = IMUEncoder()
        emg_enc.load_state_dict(torch.load(emg_path, map_location='cpu'))
        imu_enc.load_state_dict(torch.load(imu_path, map_location='cpu'))
        return cls(emg_enc, imu_enc, num_classes)

    def get_features(self, x_emg, x_imu):
        """EMG/IMU feature 를 concat 해 (B, 512) 융합 feature 를 반환한다."""
        return torch.cat([self.emg_encoder(x_emg), self.imu_encoder(x_imu)], dim=1)

    def forward(self, x_emg, x_imu, alpha=1.0):
        features = self.get_features(x_emg, x_imu)
        class_output = self.label_classifier(features)
        domain_output = self.domain_classifier(ReverseLayerF.apply(features, alpha))
        return class_output, domain_output


# ----------------------------------------------------------------------
# 단독 실행 테스트 (출력 shape 검증용)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    emg_model = EMGEncoderWithClassifier(num_classes=10)
    imu_model = IMUEncoderWithClassifier(num_classes=10)
    emg = torch.randn(32, 2, 5000)
    imu = torch.randn(32, 3,  500)

    print("=== Phase 1 ===")
    print(f"EMG output : {tuple(emg_model(emg).shape)}  (expected [32, 10])")
    print(f"IMU output : {tuple(imu_model(imu).shape)}  (expected [32, 10])")

    print("\n=== Phase 2 ===")
    model = DualEncoderDANN_Sep(emg_model.encoder, imu_model.encoder, num_classes=10)
    cls_out, dom_out = model(emg, imu, alpha=0.5)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"class output : {tuple(cls_out.shape)}  (expected [32, 10])")
    print(f"domain output: {tuple(dom_out.shape)}  (expected [32, 2])")
    print(f"Total params : {total:,}")
