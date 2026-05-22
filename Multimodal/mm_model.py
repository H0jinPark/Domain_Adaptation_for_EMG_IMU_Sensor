"""멀티모달 intermediate fusion concat 모델 정의 (EMG/IMU 분리 입력, 단일 phase).

두 모달리티를 각자 전용 TCN 인코더로 처리하되, 융합을 인코더의 AdaptiveAvgPool
**이전** 단계로 옮긴 intermediate fusion 구조:
  - 각 인코더의 pool 직전 시간 feature map 을 꺼낸다 (EMG (B,256,500), IMU (B,256,50))
  - IMU 를 EMG 길이로 업샘플해 시간축을 정렬한다
  - 채널축으로 concat -> (B, 512, T)  매 시각마다 EMG+IMU 가 함께
  - joint TCN 이 시간축을 따라 두 modality 를 처리해 cross-modal 시간 상관을 학습
  - 그 다음 pool -> label / domain head

사전학습(SSL) 없이 end-to-end 단일 phase 로 학습한다. 이 파일은 4개 멀티모달
실험(MMD/Coral/DANN/CDAN)의 공용 백본을 정의한다. MMD/Coral 은 정렬 손실만
다르므로 동일한 InterFusionClassifier 를 쓰고, DANN/CDAN 은 여기에 domain
discriminator 를 더한다.

규약상 source/target 을 분리 forward 하므로 domain discriminator 내부에는
BatchNorm 을 두지 않는다 (백본/인코더의 BatchNorm 은 유지).
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# 프로젝트 루트를 import 경로에 추가 (Multimodal/ → ../)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline.baseline_DualEncoder import EMGEncoder, IMUEncoder  # noqa: E402
from baseline.baseline_model import ResidualTCNBlock              # noqa: E402


# ----------------------------------------------------------------------
# Gradient Reversal Layer (GRL)
# ----------------------------------------------------------------------
class ReverseLayerF(Function):
    """역전파 시 그래디언트에 -alpha 를 곱하는 Gradient Reversal Layer."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


# ----------------------------------------------------------------------
# 공용 빌더
# ----------------------------------------------------------------------
def _build_fusion():
    """concat 된 (B,512,T) 시간 feature map 을 처리하는 joint TCN 을 만든다.

    stem conv 가 매 시각마다 512채널(EMG256+IMU256)을 함께 보고, 6층 Residual TCN
    (dilation 1,2,4,8,16,32)이 시간 범위를 넓혀 cross-modal 시간 상관을 학습한다.
    융합 입력 길이가 500이라 MaxPool 은 생략하고 길이를 유지한다.
    """
    return nn.Sequential(
        nn.Conv1d(512, 64, kernel_size=7, stride=2, padding=3),  # T=500 -> 250
        nn.BatchNorm1d(64),
        nn.GELU(),
        ResidualTCNBlock(64, 64, dilation=1),
        ResidualTCNBlock(64, 128, dilation=2),
        ResidualTCNBlock(128, 128, dilation=4),
        ResidualTCNBlock(128, 256, dilation=8),
        ResidualTCNBlock(256, 256, dilation=16),
        ResidualTCNBlock(256, 512, dilation=32),
        nn.AdaptiveAvgPool1d(1),  # (B, 512, 1)
    )


def _build_label_classifier(num_classes):
    """512차원 융합 feature 를 받는 운동 분류 head."""
    return nn.Sequential(
        nn.Linear(512, 256),
        nn.GELU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )


def _build_domain_classifier():
    """Source/Target 도메인 판별기. 분리 forward 규약에 맞춰 BatchNorm 을 두지 않는다."""
    return nn.Sequential(
        nn.Linear(512, 256),
        nn.GELU(),
        nn.Dropout(0.5),
        nn.Linear(256, 128),
        nn.GELU(),
        nn.Dropout(0.5),
        nn.Linear(128, 2),
    )


# ----------------------------------------------------------------------
# 공용 백본: intermediate fusion concat
# ----------------------------------------------------------------------
class InterFusionBackbone(nn.Module):
    """EMG/IMU TCN 인코더 + intermediate fusion concat + joint TCN.

    forward(x_emg, x_imu) -> (B, 512)  시간 융합 feature
    """

    def __init__(self):
        super().__init__()
        self.emg_encoder = EMGEncoder()  # (B, 2, 5000) -> temporal map (B, 256, 500)
        self.imu_encoder = IMUEncoder()  # (B, 3,  500) -> temporal map (B, 256,  50)
        self.fusion = _build_fusion()

    @staticmethod
    def _temporal_map(encoder, x):
        """encoder 의 AdaptiveAvgPool 직전 시간 feature map 을 반환한다.  (B, 256, T)

        _TCNEncoder 는 stem -> layers(마지막 원소가 AdaptiveAvgPool) 구조이므로
        layers[:-1] 로 pool 직전까지만 통과시킨다.
        """
        return encoder.layers[:-1](encoder.stem(x))

    def forward(self, x_emg, x_imu):
        emg_map = self._temporal_map(self.emg_encoder, x_emg)  # (B, 256, 500)
        imu_map = self._temporal_map(self.imu_encoder, x_imu)  # (B, 256,  50)
        # 시간축 정렬: IMU 를 EMG 길이에 맞춰 업샘플
        imu_map = F.interpolate(imu_map, size=emg_map.size(-1),
                                mode="linear", align_corners=True)
        # 채널 concat: 매 시각마다 EMG+IMU 가 함께 (B, 512, T)
        fused = torch.cat([emg_map, imu_map], dim=1)
        return self.fusion(fused).squeeze(-1)  # (B, 512)


# ----------------------------------------------------------------------
# MMD / CORAL 용 모델 (discriminator 없음)
# ----------------------------------------------------------------------
class InterFusionClassifier(nn.Module):
    """intermediate fusion 백본 + 운동 분류 head.

    forward(x_emg, x_imu) -> (class_logits, fused_features)
    MMD/CORAL 은 fused_features 에 정렬 손실을 건다.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = InterFusionBackbone()
        self.label_classifier = _build_label_classifier(num_classes)

    def forward(self, x_emg, x_imu):
        features = self.backbone(x_emg, x_imu)
        return self.label_classifier(features), features


# ----------------------------------------------------------------------
# DANN 용 모델 (domain discriminator + GRL)
# ----------------------------------------------------------------------
class InterFusionDANN(nn.Module):
    """intermediate fusion 백본 + 운동 분류 head + domain discriminator(GRL).

    forward(x_emg, x_imu, alpha) -> (class_logits, domain_logits)
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = InterFusionBackbone()
        self.label_classifier = _build_label_classifier(num_classes)
        self.domain_classifier = _build_domain_classifier()

    def forward(self, x_emg, x_imu, alpha=1.0):
        features = self.backbone(x_emg, x_imu)
        class_output = self.label_classifier(features)
        domain_output = self.domain_classifier(ReverseLayerF.apply(features, alpha))
        return class_output, domain_output


# ----------------------------------------------------------------------
# CDAN 용 모델 (conditional domain discriminator + GRL)
# ----------------------------------------------------------------------
class InterFusionCDAN(nn.Module):
    """intermediate fusion 백본 + 운동 분류 head + conditional domain discriminator.

    feature 와 분류 확률의 외적(multilinear conditioning)을 도메인 판별기 입력으로
    쓴다. forward(x_emg, x_imu, alpha) -> (class_logits, domain_logits)
    """

    def __init__(self, num_classes=10, feature_dim=512):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        self.backbone = InterFusionBackbone()
        self.label_classifier = _build_label_classifier(num_classes)

        # 도메인 판별기 입력 = feature 와 분류 확률의 외적 (feature_dim x num_classes)
        # 분리 forward 규약에 맞춰 내부 BatchNorm 을 두지 않는다.
        cdan_input_dim = feature_dim * num_classes
        self.domain_classifier = nn.Sequential(
            nn.Linear(cdan_input_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
        )

    def conditional_feature(self, features, class_logits):
        """feature 와 분류 확률의 외적(multilinear map)을 평탄화해 반환한다."""
        class_probs = F.softmax(class_logits, dim=1)
        conditional = torch.bmm(class_probs.unsqueeze(2), features.unsqueeze(1))
        return conditional.view(features.size(0), -1)

    def forward(self, x_emg, x_imu, alpha=1.0):
        features = self.backbone(x_emg, x_imu)
        class_output = self.label_classifier(features)

        conditional_features = self.conditional_feature(features, class_output)
        domain_output = self.domain_classifier(
            ReverseLayerF.apply(conditional_features, alpha)
        )
        return class_output, domain_output


# ----------------------------------------------------------------------
# 단독 실행 테스트 (출력 shape 검증용)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    emg = torch.randn(8, 2, 5000)
    imu = torch.randn(8, 3, 500)

    clf = InterFusionClassifier(num_classes=10)
    logits, feat = clf(emg, imu)
    print(f"[Classifier] logits: {tuple(logits.shape)}  feat: {tuple(feat.shape)}"
          f"  (expected (8, 10) / (8, 512))")

    dann = InterFusionDANN(num_classes=10)
    c_out, d_out = dann(emg, imu, alpha=0.5)
    print(f"[DANN]  class: {tuple(c_out.shape)}  domain: {tuple(d_out.shape)}"
          f"  (expected (8, 10) / (8, 2))")

    cdan = InterFusionCDAN(num_classes=10)
    c_out, d_out = cdan(emg, imu, alpha=0.5)
    print(f"[CDAN]  class: {tuple(c_out.shape)}  domain: {tuple(d_out.shape)}"
          f"  (expected (8, 10) / (8, 2))")

    for name, m in [("Classifier", clf), ("DANN", dann), ("CDAN", cdan)]:
        total = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"  {name:11s} params: {total:,}")
