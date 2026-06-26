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
from baseline.baseline_DualEncoder import EMGEncoder, IMUEncoder, _TCNEncoder  # noqa: E402
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


def _build_fusion_imu():
    """IMU 단독 (B,256,T) 시간 feature map 을 처리하는 joint TCN.

    멀티모달 _build_fusion 과 깊이·채널 스케줄을 동일하게 맞춰 공정 비교가 되도록
    하되, EMG concat 이 없으므로 stem 입력만 512 -> 256 으로 바꾼다. 나머지 6층
    Residual TCN(dilation 1,2,4,8,16,32)과 최종 pool 은 그대로다.
    """
    return nn.Sequential(
        nn.Conv1d(256, 64, kernel_size=7, stride=2, padding=3),  # T=500 -> 250
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
# IMU 단독 백본 (EMG 미사용, 멀티모달 fusion 설계를 단일 브랜치로 미러링)
# ----------------------------------------------------------------------
class IMUOnlyBackbone(nn.Module):
    """IMU TCN 인코더 + joint TCN (EMG 브랜치 없음).

    멀티모달 InterFusionBackbone 과 동일하게 인코더의 pool 직전 시간 feature map
    (B,256,50) 을 꺼내 길이 500 으로 업샘플한 뒤 joint TCN 으로 처리한다. concat
    상대(EMG)가 없을 뿐 깊이·시간 처리 방식은 같아 공정 비교가 된다.

    forward(x_imu) -> (B, 512)
    """

    def __init__(self):
        super().__init__()
        self.imu_encoder = IMUEncoder()  # (B, 3, 500) -> temporal map (B, 256, 50)
        self.fusion = _build_fusion_imu()

    def forward(self, x_imu):
        # 인코더의 AdaptiveAvgPool 직전 시간 feature map (B, 256, 50)
        imu_map = self.imu_encoder.layers[:-1](self.imu_encoder.stem(x_imu))
        # 멀티모달과 동일한 시간 길이(500)로 업샘플해 joint TCN 수용야를 맞춘다
        imu_map = F.interpolate(imu_map, size=500, mode="linear", align_corners=True)
        return self.fusion(imu_map).squeeze(-1)  # (B, 512)


class IMUOnlyCDAN(nn.Module):
    """IMU 단독 백본 + 운동 분류 head + conditional domain discriminator.

    InterFusionCDAN 의 IMU 전용판. mm_utils.evaluate / CDAN 학습 루프가 EMG·IMU
    분리 배치를 넘기는 규약을 그대로 쓰도록 forward 시그니처를 (x_emg, x_imu, alpha)
    로 유지하되 x_emg 는 무시한다.
    """

    def __init__(self, num_classes=10, feature_dim=512):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        self.backbone = IMUOnlyBackbone()
        self.label_classifier = _build_label_classifier(num_classes)

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
        class_probs = F.softmax(class_logits, dim=1)
        conditional = torch.bmm(class_probs.unsqueeze(2), features.unsqueeze(1))
        return conditional.view(features.size(0), -1)

    def forward(self, x_emg, x_imu, alpha=1.0):  # x_emg 는 규약 호환용, 미사용
        features = self.backbone(x_imu)
        class_output = self.label_classifier(features)

        conditional_features = self.conditional_feature(features, class_output)
        domain_output = self.domain_classifier(
            ReverseLayerF.apply(conditional_features, alpha)
        )
        return class_output, domain_output


# ----------------------------------------------------------------------
# 방향정보 입력 증강 (회전 없이 중력방향·PCA 이동방향·관계를 채널로 덧붙임)
# ----------------------------------------------------------------------
class DirectionFeatures(nn.Module):
    """raw IMU (B,3,T) 에서 중력방향·PCA 이동방향·둘의 관계를 계산해 입력 채널로 덧붙인다.

    회전(정준화) 없이 sensor frame 그대로. 학습 파라미터 없음. 윈도우마다:
      · 중력방향 ĝ = 평균가속(DC≈중력) 단위벡터 (3)
      · PCA 이동방향 v1..v_k = centering 후 공분산 고유벡터 top-k (3k)
        - 부호 모호성: 모션 왜도(skewness) 부호로 v 방향 고정
      · 관계 불변 스칼라 (invariant=True):
        - 고윳값 비율 λ1:λ2:λ3 (3)  — 모션이 1D/평면/등방인지
        - 중력방향 모션분산 비율 (ĝᵀCĝ)/trace(C) (1) — 움직임이 중력과 평행/수직인지 직접
        - |cos(ĝ, v1)| (1)
    이 스칼라들은 회전·부호 불변 → 크로스 디바이스에서 안정적인 판별신호.

    출력: (B, 3 + extra_dim, T)  (방향 feature 는 시간축으로 broadcast)
    """

    def __init__(self, gravity=True, n_pca=2, invariant=True):
        super().__init__()
        self.gravity, self.n_pca, self.invariant = gravity, n_pca, invariant

    def extra_dim(self):
        k = 0
        if self.gravity:
            k += 3
        k += 3 * self.n_pca
        if self.invariant:
            k += 3 + 1 + (1 if self.n_pca > 0 else 0)
        return k

    def forward(self, x):  # x: (B,3,T)
        B, C, T = x.shape
        m = x.mean(dim=2, keepdim=True)                       # (B,3,1) DC≈중력
        g = m.squeeze(2)
        gn = g / (g.norm(dim=1, keepdim=True) + 1e-6)         # (B,3)
        xc = x - m                                            # 중력 제거한 모션
        cov = torch.bmm(xc, xc.transpose(1, 2)) / T           # (B,3,3)
        evals, evecs = torch.linalg.eigh(cov)                 # 오름차순 고윳값

        extra = []
        if self.gravity:
            extra.append(gn)
        v1 = None
        for k in range(self.n_pca):
            v = evecs[:, :, -1 - k]                           # (B,3) top-k 축
            proj = torch.einsum('bct,bc->bt', xc, v)          # 모션의 v 투영
            s = torch.sign((proj ** 3).sum(1)).unsqueeze(1)   # 왜도 부호로 방향 고정
            s = torch.where(s == 0, torch.ones_like(s), s)
            v = v * s
            if k == 0:
                v1 = v
            extra.append(v)
        if self.invariant:
            lam = evals.flip(1)                               # 내림차순 λ1,λ2,λ3
            tot = lam.sum(1, keepdim=True) + 1e-6
            extra.append(lam / tot)                           # (B,3) 고윳값 비율
            vg = torch.einsum('bc,bcd,bd->b', gn, cov, gn).unsqueeze(1) / tot
            extra.append(vg)                                  # (B,1) 중력방향 모션분산 비율
            if self.n_pca > 0:
                extra.append((gn * v1).sum(1, keepdim=True).abs())  # (B,1) |cos(ĝ,v1)|

        feats = torch.cat(extra, dim=1)                       # (B, extra_dim)
        return torch.cat([x, feats.unsqueeze(2).expand(-1, -1, T)], dim=1)


class IMUDirBackbone(nn.Module):
    """방향정보 증강 + IMU TCN 인코더 + joint TCN (IMUOnlyBackbone 의 증강판).

    forward(x_imu) -> (B, 512)
    """

    def __init__(self, gravity=True, n_pca=2, invariant=True):
        super().__init__()
        self.dir = DirectionFeatures(gravity, n_pca, invariant)
        self.imu_encoder = _TCNEncoder(in_channels=3 + self.dir.extra_dim())
        self.fusion = _build_fusion_imu()

    def forward(self, x_imu):
        x = self.dir(x_imu)                                   # (B, 3+extra, 500)
        imu_map = self.imu_encoder.layers[:-1](self.imu_encoder.stem(x))  # (B,256,50)
        imu_map = F.interpolate(imu_map, size=500, mode="linear", align_corners=True)
        return self.fusion(imu_map).squeeze(-1)               # (B,512)


class IMUDirCDAN(nn.Module):
    """방향정보 증강 IMU 단독 + 운동분류 head + conditional domain discriminator.

    IMUOnlyCDAN 의 입력증강판 — 회전 없이 raw IMU 에 중력·PCA·관계 채널을 더해 학습.
    forward(x_emg, x_imu, alpha) (x_emg 미사용).
    """

    def __init__(self, num_classes=10, feature_dim=512,
                 gravity=True, n_pca=2, invariant=True):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.backbone = IMUDirBackbone(gravity, n_pca, invariant)
        self.label_classifier = _build_label_classifier(num_classes)

        cdan_input_dim = feature_dim * num_classes
        self.domain_classifier = nn.Sequential(
            nn.Linear(cdan_input_dim, 1024), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(1024, 512), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(512, 2),
        )

    def conditional_feature(self, features, class_logits):
        class_probs = F.softmax(class_logits, dim=1)
        conditional = torch.bmm(class_probs.unsqueeze(2), features.unsqueeze(1))
        return conditional.view(features.size(0), -1)

    def forward(self, x_emg, x_imu, alpha=1.0):  # x_emg 규약 호환용, 미사용
        features = self.backbone(x_imu)
        class_output = self.label_classifier(features)
        conditional_features = self.conditional_feature(features, class_output)
        domain_output = self.domain_classifier(
            ReverseLayerF.apply(conditional_features, alpha))
        return class_output, domain_output


# ----------------------------------------------------------------------
# EMG 단독 백본/분류기 (source-only baseline; IMUOnly 와 구조 대칭, IMU 미사용)
# ----------------------------------------------------------------------
class EMGOnlyBackbone(nn.Module):
    """EMG TCN 인코더 + joint TCN (IMU 브랜치 없음).

    IMUOnlyBackbone 과 대칭. EMG 인코더의 pool 직전 시간 feature map (B,256,500) 을
    꺼내 joint TCN 으로 처리한다. EMG map 은 이미 길이 500 이라 업샘플이 필요 없다.

    forward(x_emg) -> (B, 512)
    """

    def __init__(self):
        super().__init__()
        self.emg_encoder = EMGEncoder()      # (B, 2, 5000) -> temporal map (B, 256, 500)
        self.fusion = _build_fusion_imu()    # 256ch 입력 단일브랜치 joint TCN (모달 무관)

    def forward(self, x_emg):
        emg_map = self.emg_encoder.layers[:-1](self.emg_encoder.stem(x_emg))  # (B,256,500)
        return self.fusion(emg_map).squeeze(-1)  # (B, 512)


class EMGOnlyClassifier(nn.Module):
    """EMG 단독 백본 + 운동 분류 head (도메인 판별기 없음 = source-only baseline).

    InterFusionClassifier 의 EMG 전용판. evaluate(needs_alpha=False) 와 데이터로더의
    (emg, imu, y) 규약에 맞춰 forward(x_emg, x_imu) 시그니처를 유지하되 x_imu 는 무시한다.
    forward -> (class_logits, features)
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = EMGOnlyBackbone()
        self.label_classifier = _build_label_classifier(num_classes)

    def forward(self, x_emg, x_imu=None):  # x_imu 는 규약 호환용, 미사용
        features = self.backbone(x_emg)
        return self.label_classifier(features), features


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

    imu_cdan = IMUOnlyCDAN(num_classes=10)
    c_out, d_out = imu_cdan(emg, imu, alpha=0.5)
    print(f"[IMU-CDAN]  class: {tuple(c_out.shape)}  domain: {tuple(d_out.shape)}"
          f"  (expected (8, 10) / (8, 2))")

    emg_clf = EMGOnlyClassifier(num_classes=10)
    c_out, feat = emg_clf(emg, imu)
    print(f"[EMG-only]  class: {tuple(c_out.shape)}  feat: {tuple(feat.shape)}"
          f"  (expected (8, 10) / (8, 512))")

    imu_dir = IMUDirCDAN(num_classes=10)
    in_ch = 3 + imu_dir.backbone.dir.extra_dim()
    c_out, d_out = imu_dir(emg, imu, alpha=0.5)
    print(f"[IMU-Dir]  class: {tuple(c_out.shape)}  domain: {tuple(d_out.shape)}"
          f"  | 증강 입력 채널={in_ch}  (expected (8, 10) / (8, 2))")

    for name, m in [("Classifier", clf), ("DANN", dann), ("CDAN", cdan),
                    ("IMU-CDAN", imu_cdan), ("EMG-only", emg_clf), ("IMU-Dir", imu_dir)]:
        total = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"  {name:11s} params: {total:,}")
