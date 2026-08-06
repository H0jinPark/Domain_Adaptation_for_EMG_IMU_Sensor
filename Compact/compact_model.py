"""경량 멀티모달 intermediate fusion 모델 (Multimodal/mm_model.py 의 축소판).

원본 백본이 과대하다는 판단(2026-07-29 회의)에 따라 TCN 블록 수만 줄인 변형이다.
**클래스 이름·forward 시그니처·출력 차원을 원본과 동일하게 유지**하므로, 학습
스크립트는 import 경로만 바꾸면 그대로 돈다.

    from Multimodal.mm_model import InterFusionCDAN   # 원본
    from Compact.compact_model import InterFusionCDAN # 경량

블록 수 변경 (그 외 stem/pool/head/판별기는 원본 그대로)
  · 모달 인코더  4블록 -> 2블록   (dilation 1,2,4,8      -> 1,4)
  · joint fusion 6블록 -> 3블록   (dilation 1,2,4,8,16,32 -> 1,4,16)

채널 스케줄은 양끝을 고정한 채 중간 계단만 걷어냈다. 인코더 출력은 256ch,
fusion 출력은 512ch 로 원본과 같아서 concat 차원(512)과 label/domain head 를
손대지 않아도 된다.

수용야(receptive field) 영향
  ResidualTCNBlock 은 kernel=5 conv 2개라 블록당 8*dilation 스텝을 더한다.
  · 인코더: MaxPool(2) 뒤의 dilation 4 는 pool 이전 기준으로 8 에 해당하므로
    **최대 유효 dilation 8 은 원본과 동일**하다. 누적 RF 만 217 -> 73 스텝으로
    줄어든다(EMG 0.36초 / IMU 3.6초 상당).
  · fusion: 8*(1+4+16)=168 스텝 x 20ms = 3.36초. 원본은 504 스텝(10.1초)로 5초
    윈도우를 두 번 덮었다. 끝단 AdaptiveAvgPool 이 시간축 전역을 평균하므로
    윈도우 전체 맥락 자체는 남지만, 깊은 층의 지역 RF 는 윈도우보다 짧아진다.
    윈도우 전체(5초)를 RF 로 덮고 싶으면 FUSION_DILATIONS=(1, 8, 32) 로 두면
    6.56초가 되어 원본의 커버리지를 되찾는다(블록 수는 3 그대로).

원본과 마찬가지로 source/target 을 분리 forward 하므로 domain discriminator 내부에는
BatchNorm 을 두지 않는다.
"""
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# 프로젝트 루트를 import 경로에 추가 (Compact/ → ../)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline.baseline_model import ResidualTCNBlock              # noqa: E402
from Multimodal.mm_model import DirectionFeatures                 # noqa: E402

# 이 변형을 식별하는 이름 (결과 json/체크포인트 태깅용)
VARIANT = "compact"


def _dilations_from_env(name, default):
    """환경변수로 dilation 사다리를 갈아끼운다 (예: COMPACT_FUSION_DILATIONS=1,8,32).

    모듈 상수를 나중에 바꿔도 함수 기본인자는 정의 시점에 묶이므로 반영되지 않는다.
    스윕에서 한 프로세스당 한 구성을 쓰는 방식이라 import 시점의 환경변수로 정한다.
    실제 사용된 값은 VARIANT_INFO 로 노출해 결과 json 에 기록한다.
    """
    raw = os.environ.get(name)
    if not raw:
        return default
    vals = tuple(int(x) for x in raw.replace(" ", "").split(",") if x)
    if len(vals) != len(default):
        raise ValueError(f"{name} 은 {len(default)}개 값이어야 한다 (받음: {vals})")
    return vals


# 블록 구성 — 환경변수로 덮어쓸 수 있다
ENCODER_DILATIONS = _dilations_from_env("COMPACT_ENCODER_DILATIONS", (1, 4))
FUSION_DILATIONS = _dilations_from_env("COMPACT_FUSION_DILATIONS", (1, 4, 16))


def variant_info():
    """이 프로세스가 실제로 쓰는 구성 (결과 json 에 그대로 박아 provenance 로 쓴다)."""
    # fusion 스텝은 20ms (EMG map 500 = 5초 -> stem stride 2 -> 250)
    rf_steps = 8 * sum(FUSION_DILATIONS)
    return {
        "variant": VARIANT,
        "encoder_dilations": list(ENCODER_DILATIONS),
        "fusion_dilations": list(FUSION_DILATIONS),
        "fusion_rf_sec": round(rf_steps * 0.02, 2),
    }


# ----------------------------------------------------------------------
# Gradient Reversal Layer (GRL) — 원본과 동일
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
# 경량 모달 인코더 (4블록 -> 2블록)
# ----------------------------------------------------------------------
class _TCNEncoder(nn.Module):
    """ResidualTCNBlock 2층으로 구성된 feature extractor (원본 4층의 축소판).

    stem 과 MaxPool 위치, 출력 채널(256)은 원본과 같다. 따라서 pool 직전 시간
    feature map 의 shape 도 원본과 동일하다 — EMG (B,256,500), IMU (B,256,50).
    intermediate fusion 이 이 길이에 의존하므로 이 부분은 바꾸지 않았다.

    채널: 64 -(blk1)-> 128 -(pool)-> -(blk2)-> 256
    """

    def __init__(self, in_channels, dilations=ENCODER_DILATIONS):
        super().__init__()
        if len(dilations) != 2:
            raise ValueError(f"인코더는 2블록 구성이다 (받은 dilations={dilations})")
        d1, d2 = dilations

        # Stem: 초기 Conv1d 로 채널 확장 및 시퀀스 길이 축소 (원본과 동일)
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=11, stride=5, padding=5),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        # Residual TCN layers: 2개 블록. MaxPool 뒤 dilation 은 pool 이전 기준
        # 2배로 작용하므로 (1, 4) 는 원본의 최대 유효 dilation 8 을 유지한다.
        self.layers = nn.Sequential(
            ResidualTCNBlock(64, 128, dilation=d1),
            nn.MaxPool1d(2),
            ResidualTCNBlock(128, 256, dilation=d2),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        return self.layers(self.stem(x)).squeeze(-1)  # (B, 256)


class EMGEncoder(_TCNEncoder):
    """EMG 전용 경량 인코더.  (B, 2, 5000) -> (B, 256)"""

    def __init__(self):
        super().__init__(in_channels=2)


class IMUEncoder(_TCNEncoder):
    """IMU 전용 경량 인코더.  (B, 3, 500) -> (B, 256)"""

    def __init__(self):
        super().__init__(in_channels=3)


# ----------------------------------------------------------------------
# 경량 joint fusion (6블록 -> 3블록)
# ----------------------------------------------------------------------
def _build_fusion(in_channels=512, dilations=FUSION_DILATIONS):
    """concat 된 (B,in_ch,T) 시간 feature map 을 처리하는 joint TCN.

    in_channels=512 : 멀티모달(EMG 256 + IMU 256 concat)
    in_channels=256 : 단일 모달(IMU 단독 / EMG 단독) — 원본의 _build_fusion_imu 역할

    stem conv 는 원본 그대로 두고(T=500 -> 250) Residual TCN 만 6층에서 3층으로
    줄인다. 출력은 원본과 같은 512ch 라 label/domain head 를 재사용한다.
    채널: 64 -> 128 -> 256 -> 512
    """
    if len(dilations) != 3:
        raise ValueError(f"fusion 은 3블록 구성이다 (받은 dilations={dilations})")
    d1, d2, d3 = dilations
    return nn.Sequential(
        nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),  # T=500 -> 250
        nn.BatchNorm1d(64),
        nn.GELU(),
        ResidualTCNBlock(64, 128, dilation=d1),
        ResidualTCNBlock(128, 256, dilation=d2),
        ResidualTCNBlock(256, 512, dilation=d3),
        nn.AdaptiveAvgPool1d(1),  # (B, 512, 1)
    )


def _build_fusion_imu():
    """단일 모달(256ch) 입력용 joint TCN. 원본과 이름을 맞춰 둔 얇은 별칭."""
    return _build_fusion(in_channels=256)


# ----------------------------------------------------------------------
# head / discriminator — 원본과 완전히 동일 (축소 대상이 아님)
# ----------------------------------------------------------------------
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


class RandomMultilinearMap(nn.Module):
    """CDAN 랜덤 multilinear map — 원논문(Long et al., NeurIPS 2018) 4.2절.

    조건부 판별기의 입력을 feature 와 분류확률의 **외적**으로 만들면 차원이
    feature_dim x num_classes 로 곱해져 폭발한다. 원논문은 이 값이 4096 을 넘으면
    외적 대신 고정 랜덤 사영의 원소곱을 쓰라고 규정한다:

        T⊙(f, g) = (1/√d) · (R_f f) ⊙ (R_g g)

    여기 설정은 512 x 10 = 5120 > 4096 이라 원논문 기준 RP 를 써야 하는 구간이다.
    5120 -> 1024 로 줄면 판별기 첫 Linear 가 5,243,904 -> 1,048,576 params 가 된다.

    R_f, R_g 는 학습하지 않는 고정 행렬이라 Parameter 가 아닌 buffer 로 둔다.
    buffer 는 state_dict 에 함께 저장되므로, 학습 중 best 체크포인트를 다시 불러
    test 를 재는 이 파이프라인에서 같은 사영이 그대로 복원된다. 생성 시점의 난수는
    학습 스크립트가 모델을 만들기 전에 부르는 set_seed(seed) 에 종속돼 재현된다.
    """

    def __init__(self, feature_dim, num_classes, output_dim=1024):
        super().__init__()
        self.output_dim = output_dim
        self.register_buffer("random_f", torch.randn(feature_dim, output_dim))
        self.register_buffer("random_g", torch.randn(num_classes, output_dim))

    def forward(self, features, class_probs):
        f = features @ self.random_f          # (B, output_dim)
        g = class_probs @ self.random_g       # (B, output_dim)
        # 원논문/공식 구현의 스케일: 입력 2개이므로 d^(1/2) 로 나눈다
        return f * g / math.sqrt(self.output_dim)


def _build_cdan_domain_classifier(feature_dim, num_classes, cdan_rp=False, rp_dim=1024):
    """CDAN 조건부 판별기와 조건화 모듈을 함께 만든다.

    cdan_rp=False : 입력 = feature x class-prob 외적 (feature_dim * num_classes)
    cdan_rp=True  : 입력 = 랜덤 multilinear map 출력 (rp_dim)

    반환: (조건화 모듈 또는 None, 판별기 MLP)
    """
    rp = RandomMultilinearMap(feature_dim, num_classes, rp_dim) if cdan_rp else None
    in_dim = rp_dim if cdan_rp else feature_dim * num_classes
    mlp = nn.Sequential(
        nn.Linear(in_dim, 1024),
        nn.GELU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.GELU(),
        nn.Dropout(0.5),
        nn.Linear(512, 2),
    )
    return rp, mlp


def _conditional_feature(features, class_logits, rp=None):
    """판별기에 넣을 조건부 feature 를 만든다.

    rp 가 주어지면 랜덤 multilinear map, 아니면 원래대로 외적을 평탄화한다.
    """
    class_probs = F.softmax(class_logits, dim=1)
    if rp is not None:
        return rp(features, class_probs)
    conditional = torch.bmm(class_probs.unsqueeze(2), features.unsqueeze(1))
    return conditional.view(features.size(0), -1)


# ----------------------------------------------------------------------
# 공용 백본: intermediate fusion concat
# ----------------------------------------------------------------------
class InterFusionBackbone(nn.Module):
    """EMG/IMU 경량 인코더 + intermediate fusion concat + 경량 joint TCN.

    forward(x_emg, x_imu) -> (B, 512)  시간 융합 feature
    """

    def __init__(self):
        super().__init__()
        self.emg_encoder = EMGEncoder()  # (B, 2, 5000) -> temporal map (B, 256, 500)
        self.imu_encoder = IMUEncoder()  # (B, 3,  500) -> temporal map (B, 256,  50)
        self.fusion = _build_fusion(in_channels=512)

    @staticmethod
    def _temporal_map(encoder, x):
        """encoder 의 AdaptiveAvgPool 직전 시간 feature map 을 반환한다.  (B, 256, T)"""
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

    forward(x_emg, x_imu, alpha) -> (class_logits, domain_logits)
    """

    def __init__(self, num_classes=10, feature_dim=512, cdan_rp=False, rp_dim=1024):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.cdan_rp = cdan_rp

        self.backbone = InterFusionBackbone()
        self.label_classifier = _build_label_classifier(num_classes)
        self.rp, self.domain_classifier = _build_cdan_domain_classifier(
            feature_dim, num_classes, cdan_rp, rp_dim)

    def conditional_feature(self, features, class_logits):
        return _conditional_feature(features, class_logits, self.rp)

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
    """IMU 경량 인코더 + 경량 joint TCN (EMG 브랜치 없음).

    forward(x_imu) -> (B, 512)
    """

    def __init__(self):
        super().__init__()
        self.imu_encoder = IMUEncoder()  # (B, 3, 500) -> temporal map (B, 256, 50)
        self.fusion = _build_fusion(in_channels=256)

    def forward(self, x_imu):
        imu_map = self.imu_encoder.layers[:-1](self.imu_encoder.stem(x_imu))
        # 멀티모달과 동일한 시간 길이(500)로 업샘플해 joint TCN 수용야를 맞춘다
        imu_map = F.interpolate(imu_map, size=500, mode="linear", align_corners=True)
        return self.fusion(imu_map).squeeze(-1)  # (B, 512)


class IMUOnlyCDAN(nn.Module):
    """IMU 단독 경량 백본 + 운동 분류 head + conditional domain discriminator.

    forward 시그니처는 (x_emg, x_imu, alpha) 규약을 유지하되 x_emg 는 무시한다.
    """

    def __init__(self, num_classes=10, feature_dim=512, cdan_rp=False, rp_dim=1024):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.cdan_rp = cdan_rp

        self.backbone = IMUOnlyBackbone()
        self.label_classifier = _build_label_classifier(num_classes)
        self.rp, self.domain_classifier = _build_cdan_domain_classifier(
            feature_dim, num_classes, cdan_rp, rp_dim)

    def conditional_feature(self, features, class_logits):
        return _conditional_feature(features, class_logits, self.rp)

    def forward(self, x_emg, x_imu, alpha=1.0):  # x_emg 는 규약 호환용, 미사용
        features = self.backbone(x_imu)
        class_output = self.label_classifier(features)

        conditional_features = self.conditional_feature(features, class_output)
        domain_output = self.domain_classifier(
            ReverseLayerF.apply(conditional_features, alpha)
        )
        return class_output, domain_output


# ----------------------------------------------------------------------
# 방향정보 입력 증강판 (DirectionFeatures 는 원본 모듈에서 그대로 재사용)
# ----------------------------------------------------------------------
class IMUDirBackbone(nn.Module):
    """방향정보 증강 + IMU 경량 인코더 + 경량 joint TCN.

    forward(x_imu) -> (B, 512)
    """

    def __init__(self, gravity=True, n_pca=2, invariant=True):
        super().__init__()
        self.dir = DirectionFeatures(gravity, n_pca, invariant)
        self.imu_encoder = _TCNEncoder(in_channels=3 + self.dir.extra_dim())
        self.fusion = _build_fusion(in_channels=256)

    def forward(self, x_imu):
        x = self.dir(x_imu)                                   # (B, 3+extra, 500)
        imu_map = self.imu_encoder.layers[:-1](self.imu_encoder.stem(x))  # (B,256,50)
        imu_map = F.interpolate(imu_map, size=500, mode="linear", align_corners=True)
        return self.fusion(imu_map).squeeze(-1)               # (B,512)


class IMUDirCDAN(nn.Module):
    """방향정보 증강 IMU 단독 경량 모델 + conditional domain discriminator."""

    def __init__(self, num_classes=10, feature_dim=512,
                 gravity=True, n_pca=2, invariant=True, cdan_rp=False, rp_dim=1024):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.cdan_rp = cdan_rp
        self.backbone = IMUDirBackbone(gravity, n_pca, invariant)
        self.label_classifier = _build_label_classifier(num_classes)
        self.rp, self.domain_classifier = _build_cdan_domain_classifier(
            feature_dim, num_classes, cdan_rp, rp_dim)

    def conditional_feature(self, features, class_logits):
        return _conditional_feature(features, class_logits, self.rp)

    def forward(self, x_emg, x_imu, alpha=1.0):  # x_emg 규약 호환용, 미사용
        features = self.backbone(x_imu)
        class_output = self.label_classifier(features)
        conditional_features = self.conditional_feature(features, class_output)
        domain_output = self.domain_classifier(
            ReverseLayerF.apply(conditional_features, alpha))
        return class_output, domain_output


# ----------------------------------------------------------------------
# EMG 단독 백본/분류기 (source-only baseline; IMUOnly 와 구조 대칭)
# ----------------------------------------------------------------------
class EMGOnlyBackbone(nn.Module):
    """EMG 경량 인코더 + 경량 joint TCN (IMU 브랜치 없음).

    EMG map 은 이미 길이 500 이라 업샘플이 필요 없다.  forward(x_emg) -> (B, 512)
    """

    def __init__(self):
        super().__init__()
        self.emg_encoder = EMGEncoder()          # (B,2,5000) -> (B,256,500)
        self.fusion = _build_fusion(in_channels=256)

    def forward(self, x_emg):
        emg_map = self.emg_encoder.layers[:-1](self.emg_encoder.stem(x_emg))
        return self.fusion(emg_map).squeeze(-1)  # (B, 512)


class EMGOnlyClassifier(nn.Module):
    """EMG 단독 경량 백본 + 운동 분류 head (도메인 판별기 없음 = source-only baseline).

    forward(x_emg, x_imu) -> (class_logits, features)   (x_imu 는 규약 호환용, 미사용)
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = EMGOnlyBackbone()
        self.label_classifier = _build_label_classifier(num_classes)

    def forward(self, x_emg, x_imu=None):
        features = self.backbone(x_emg)
        return self.label_classifier(features), features


# ----------------------------------------------------------------------
# 단독 실행 테스트 (shape 검증 + 원본 대비 파라미터 감소량)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from Multimodal import mm_model as orig

    emg = torch.randn(8, 2, 5000)
    imu = torch.randn(8, 3, 500)

    def n_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print(f"=== Compact 변형 (VARIANT={VARIANT}) ===")
    print(f"  인코더 dilations {ENCODER_DILATIONS} (2블록) "
          f"| fusion dilations {FUSION_DILATIONS} (3블록)\n")

    pairs = [
        ("Classifier", InterFusionClassifier(), orig.InterFusionClassifier()),
        ("DANN",       InterFusionDANN(),       orig.InterFusionDANN()),
        ("CDAN",       InterFusionCDAN(),       orig.InterFusionCDAN()),
        ("IMU-CDAN",   IMUOnlyCDAN(),           orig.IMUOnlyCDAN()),
        ("IMU-Dir",    IMUDirCDAN(),            orig.IMUDirCDAN()),
        ("EMG-only",   EMGOnlyClassifier(),     orig.EMGOnlyClassifier()),
    ]

    # shape 검증
    for name, new, _ in pairs:
        with torch.no_grad():
            if isinstance(new, (InterFusionClassifier, EMGOnlyClassifier)):
                a, b = new(emg, imu)
            else:
                a, b = new(emg, imu, alpha=0.5)
        print(f"  [{name:10s}] out1 {tuple(a.shape)}  out2 {tuple(b.shape)}")

    print(f"\n  {'모델':<12}{'원본':>14}{'경량':>14}{'감소':>10}")
    for name, new, old in pairs:
        p_new, p_old = n_params(new), n_params(old)
        print(f"  {name:<12}{p_old:>14,}{p_new:>14,}{1 - p_new / p_old:>9.1%}")

    # --- CDAN-RP (랜덤 multilinear map) 확인 -----------------------------
    print("\n=== CDAN-RP: 조건부 판별기 입력을 외적(5120) 대신 랜덤사영(1024) 으로 ===")
    for name, mk in [("CDAN", InterFusionCDAN), ("IMU-CDAN", IMUOnlyCDAN)]:
        full, rp = mk(), mk(cdan_rp=True)
        with torch.no_grad():
            a, b = rp(emg, imu, alpha=0.5)
            cond = rp.conditional_feature(rp.backbone(emg, imu) if name == "CDAN"
                                          else rp.backbone(imu),
                                          torch.randn(8, 10))
        print(f"  [{name:9s}] out {tuple(a.shape)} / {tuple(b.shape)}"
              f" | 조건부 feature {tuple(cond.shape)}"
              f" | {n_params(full):,} -> {n_params(rp):,}"
              f" ({1 - n_params(rp) / n_params(full):.1%})")
    # 랜덤 행렬이 학습 대상이 아니고 state_dict 에는 남는지 확인
    rp = InterFusionCDAN(cdan_rp=True)
    n_buf = sum(1 for k in rp.state_dict() if k.startswith("rp."))
    n_par = sum(1 for k, _ in rp.named_parameters() if k.startswith("rp."))
    print(f"  랜덤 행렬: state_dict 항목 {n_buf}개 / 학습 파라미터 {n_par}개 (0 이어야 정상)")
