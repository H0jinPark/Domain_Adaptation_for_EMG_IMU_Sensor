"""보간 없는 intermediate fusion 모델 (Multimodal/mm_model.py 의 변형).

동기
  원본은 IMU 시간 feature map 을 (B,256,50) 으로 만든 뒤 **F.interpolate 로 10배
  늘려** EMG 의 500 에 맞춰 concat 한다. 10배 선형 업샘플은 없는 정보를 만들어내지
  않으면서 시간축만 늘리는 것이라, 융합 단계가 보는 IMU 는 사실상 10샘플마다 한 번씩만
  갱신되는 계단이다. 여기서는 **IMU 인코더가 처음부터 길이 500 을 내도록** 바꿔서
  보간을 없앤다.

바뀐 것은 IMU 인코더뿐이다 (EMG 인코더·fusion·head·판별기는 원본 그대로).
  · stem stride 5 -> 1   : (B,3,500) -> (B,64,500)   ← 사용자가 지정한 지점
  · 블록 사이 MaxPool(2) 제거                        ← 길이 500 유지
  · 결과 시간 map (B,256,500) 이 EMG 와 **같은 100Hz** 라 그대로 concat 된다

시간 해상도가 맞는다는 점이 핵심이다.
  EMG: 5000샘플/5초 = 1000Hz -> stem(/5) 200Hz -> MaxPool(/2) 100Hz -> 500
  IMU:  500샘플/5초 =  100Hz -> stem(/1) 100Hz -> pool 없음      -> 500
  둘 다 100Hz 격자 위에 있으므로 concat 시 시각이 실제로 대응한다.

수용야 주의 (이 변형의 유일한 대가)
  원본 IMU 인코더는 뒤쪽 두 블록이 10Hz 격자에서 돌아 누적 RF 가 약 10초였다.
  길이를 유지하면 같은 dilation 이 100Hz 격자에서 돌므로 RF 가 8*(1+2+4+8)=120샘플
  = **1.2초**로 줄어든다. 그래서 dilation 사다리를 환경변수로 갈아끼울 수 있게 했다.
      NOINTERP_IMU_DILATIONS=4,8,16,32   -> 8*60=480샘플 = 4.8초
  기본값은 원본과 같은 (1,2,4,8) 이고, 보상판은 스윕에서 별도 태그로 돌린다.

클래스 이름·forward 시그니처·출력 차원은 원본과 동일하므로 학습 스크립트는 import 만
갈아끼우면 된다.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# 프로젝트 루트를 import 경로에 추가 (Compact/ → ../)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline.baseline_DualEncoder import EMGEncoder                 # noqa: E402
from baseline.baseline_model import ResidualTCNBlock                 # noqa: E402
from Multimodal.mm_model import (                                    # noqa: E402
    ReverseLayerF, _build_fusion, _build_fusion_imu,
    _build_label_classifier, _build_domain_classifier, DirectionFeatures)

VARIANT = "nointerp"


def _dilations_from_env(name, default):
    """환경변수로 dilation 사다리를 갈아끼운다 (예: NOINTERP_IMU_DILATIONS=4,8,16,32).

    모듈 상수를 나중에 바꿔도 함수 기본인자는 정의 시점에 묶이므로 import 시점에 정한다.
    실제 사용된 값은 variant_info() 로 노출해 결과 json 에 provenance 로 남긴다.
    """
    raw = os.environ.get(name)
    if not raw:
        return default
    vals = tuple(int(x) for x in raw.replace(" ", "").split(",") if x)
    if len(vals) != len(default):
        raise ValueError(f"{name} 은 {len(default)}개 값이어야 한다 (받음: {vals})")
    return vals


IMU_DILATIONS = _dilations_from_env("NOINTERP_IMU_DILATIONS", (1, 2, 4, 8))


def variant_info():
    """이 프로세스가 실제로 쓰는 구성 (결과 json 에 그대로 박는다)."""
    return {
        "variant": VARIANT,
        "imu_dilations": list(IMU_DILATIONS),
        # IMU map 은 100Hz(=10ms) 격자
        "imu_rf_sec": round(8 * sum(IMU_DILATIONS) * 0.01, 2),
        "interpolation": "none (IMU encoder outputs T=500 natively)",
    }


# ----------------------------------------------------------------------
# 길이 보존 IMU 인코더 — 이 파일의 핵심
# ----------------------------------------------------------------------
class IMUEncoderNoInterp(nn.Module):
    """(B,3,500) -> 시간 map (B,256,500).  stride 1 stem + MaxPool 없음.

    원본 _TCNEncoder 와 블록 수(4)·채널 스케줄(64→64→128→128→256)은 같고,
    길이를 줄이는 두 지점(stem stride, MaxPool)만 없앴다.
    """

    def __init__(self, in_channels=3, dilations=IMU_DILATIONS):
        super().__init__()
        if len(dilations) != 4:
            raise ValueError(f"IMU 인코더는 4블록 구성이다 (받은 dilations={dilations})")
        d1, d2, d3, d4 = dilations
        # stride=1 이라 길이가 유지된다. padding=5, kernel=11 → 출력 길이 = 입력 길이.
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        # 원본의 MaxPool(2) 자리를 Identity 로 둬서 블록 순서·인덱스를 원본과 맞춘다
        # (temporal_map 이 layers[:-1] 로 뒤 pool 만 떼어내는 규약을 그대로 쓰기 위함).
        self.layers = nn.Sequential(
            ResidualTCNBlock(64, 64, dilation=d1),
            ResidualTCNBlock(64, 128, dilation=d2),
            nn.Identity(),                      # 원본: nn.MaxPool1d(2)
            ResidualTCNBlock(128, 128, dilation=d3),
            ResidualTCNBlock(128, 256, dilation=d4),
            nn.AdaptiveAvgPool1d(1),
        )

    def temporal_map(self, x):
        """AdaptiveAvgPool 직전 시간 feature map. (B, 256, 500)"""
        return self.layers[:-1](self.stem(x))

    def forward(self, x):
        return self.layers(self.stem(x)).squeeze(-1)  # (B, 256)


# ----------------------------------------------------------------------
# 백본: 보간 없이 채널 concat
# ----------------------------------------------------------------------
class InterFusionBackbone(nn.Module):
    """EMG 인코더(원본) + 길이보존 IMU 인코더 + 원본 joint TCN.

    forward(x_emg, x_imu) -> (B, 512)
    """

    def __init__(self):
        super().__init__()
        self.emg_encoder = EMGEncoder()          # (B,2,5000) -> map (B,256,500)
        self.imu_encoder = IMUEncoderNoInterp()  # (B,3, 500) -> map (B,256,500)
        self.fusion = _build_fusion()

    @staticmethod
    def _emg_map(encoder, x):
        """원본 EMG 인코더의 AdaptiveAvgPool 직전 map. (B,256,500)"""
        return encoder.layers[:-1](encoder.stem(x))

    def forward(self, x_emg, x_imu):
        emg_map = self._emg_map(self.emg_encoder, x_emg)   # (B, 256, 500)
        imu_map = self.imu_encoder.temporal_map(x_imu)     # (B, 256, 500)  ← 보간 없음
        if emg_map.size(-1) != imu_map.size(-1):
            raise RuntimeError(
                f"보간 없이 concat 하려면 길이가 같아야 한다: "
                f"EMG {emg_map.size(-1)} vs IMU {imu_map.size(-1)}")
        fused = torch.cat([emg_map, imu_map], dim=1)       # (B, 512, 500)
        return self.fusion(fused).squeeze(-1)              # (B, 512)


# ----------------------------------------------------------------------
# 아래는 원본과 동일 (백본만 갈아끼운 래퍼)
# ----------------------------------------------------------------------
class InterFusionClassifier(nn.Module):
    """intermediate fusion 백본 + 운동 분류 head."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = InterFusionBackbone()
        self.label_classifier = _build_label_classifier(num_classes)

    def forward(self, x_emg, x_imu):
        features = self.backbone(x_emg, x_imu)
        return self.label_classifier(features), features


class InterFusionDANN(nn.Module):
    """백본 + 분류 head + domain discriminator(GRL)."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = InterFusionBackbone()
        self.label_classifier = _build_label_classifier(num_classes)
        self.domain_classifier = _build_domain_classifier()

    def forward(self, x_emg, x_imu, alpha=1.0):
        features = self.backbone(x_emg, x_imu)
        return (self.label_classifier(features),
                self.domain_classifier(ReverseLayerF.apply(features, alpha)))


class InterFusionCDAN(nn.Module):
    """백본 + 분류 head + conditional domain discriminator (원본 CDAN 과 동일 규약)."""

    def __init__(self, num_classes=10, feature_dim=512):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.backbone = InterFusionBackbone()
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

    def forward(self, x_emg, x_imu, alpha=1.0):
        features = self.backbone(x_emg, x_imu)
        class_output = self.label_classifier(features)
        cond = self.conditional_feature(features, class_output)
        return class_output, self.domain_classifier(ReverseLayerF.apply(cond, alpha))


# ----------------------------------------------------------------------
# 단독 실행 검증: 길이·파라미터·보간 부재 확인
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from Multimodal import mm_model as orig

    emg, imu = torch.randn(4, 2, 5000), torch.randn(4, 3, 500)

    def n_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print(f"=== NoInterp 변형 (IMU dilations {IMU_DILATIONS}) ===")
    print(f"  {variant_info()}\n")

    bb, ob = InterFusionBackbone(), orig.InterFusionBackbone()
    with torch.no_grad():
        e = bb._emg_map(bb.emg_encoder, emg)
        i = bb.imu_encoder.temporal_map(imu)
        oi = ob._temporal_map(ob.imu_encoder, imu)
    print(f"  EMG map          {tuple(e.shape)}")
    print(f"  IMU map (신규)   {tuple(i.shape)}   ← 보간 없이 EMG 와 동일 길이")
    print(f"  IMU map (원본)   {tuple(oi.shape)}   → F.interpolate 로 500 까지 10배 확대")
    assert e.shape[-1] == i.shape[-1] == 500

    print(f"\n  {'모델':<12}{'원본':>14}{'NoInterp':>14}{'증감':>10}")
    for name, new, old in [("Classifier", InterFusionClassifier(), orig.InterFusionClassifier()),
                           ("DANN", InterFusionDANN(), orig.InterFusionDANN()),
                           ("CDAN", InterFusionCDAN(), orig.InterFusionCDAN())]:
        a, b = n_params(new), n_params(old)
        print(f"  {name:<12}{b:>14,}{a:>14,}{(a-b)/b:>9.1%}")

    with torch.no_grad():
        c, d = InterFusionCDAN()(emg, imu, alpha=0.5)
    print(f"\n  CDAN forward: class {tuple(c.shape)}  domain {tuple(d.shape)}")

    import inspect
    src = inspect.getsource(InterFusionBackbone.forward)
    print(f"  forward 안에 interpolate 호출: {'있음 (!!)' if 'interpolate' in src else '없음 ✓'}")
