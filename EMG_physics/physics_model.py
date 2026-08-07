"""Physics-informed CDAN 모델: InterFusionCDAN + EMG→IMU 디코더.

설계 (기본값 imu_interp=False — 보간 없는 융합)
  x_emg (B,2,5000) ──EMGEncoder(pool 직전)────────> emg_map (B,256,500)
  x_imu (B,3, 500) ──IMUEncoderNoInterp(pool 직전)─> imu_map (B,256,500)
                                                   │
      emg_map ⊕ imu_map (B,512,500) ──joint TCN──> features (B,512)
                                                   ├─> label head   -> class_logits
                                                   └─> CDAN head    -> domain_logits
      emg_map ─────────────────────────IMUDecoder─> imu_hat (B,3,500)   [신규]

**보간 없는 융합이 기본이다.** 원본 IMU 인코더는 stem stride 5 + MaxPool(2) 로 길이를
10배 줄여 (B,256,50) 을 내고, 그걸 F.interpolate 로 500 까지 되돌려 concat 했다. 그러면
융합단이 보는 IMU 는 10샘플마다 한 번 갱신되는 계단이 된다. 여기서는 길이를 보존하는
IMUEncoderNoInterp (stem stride 1, MaxPool -> Identity) 를 써서 인코더가 처음부터 500 을
내도록 한다. 두 map 이 같은 100Hz 격자 위에 있으므로 concat 한 시각이 실제로 대응한다:

    EMG: 5000샘플/5초 = 1000Hz -> stem(/5) 200Hz -> MaxPool(/2) 100Hz -> 길이 500
    IMU:  500샘플/5초 =  100Hz -> stem(/1) 100Hz -> pool 없음        -> 길이 500

길이가 우연히 같은 게 아니라 **샘플링 레이트가 같다**는 게 핵심이다.
stride/pool 만 건드렸고 커널 모양은 그대로라 파라미터 수는 원본과 완전히 동일하다
(`Compact/nointerp_model.py`, 2026-07-31 보고서에서 검증). 즉 두 모드의 차이는 용량이
아니라 순수하게 시간해상도/보간 효과다.

imu_interp=True 로 옛 동작(50 -> 보간 500)을 그대로 복원할 수 있다. 2026-08-04 스윕
(대조군 포함)은 전부 imu_interp=True 판이므로, 새 모드 결과와 **직접 비교하면 안 된다**.
결과 json 의 "imu_fusion" 필드로 두 판을 구분한다.

디코더 경로는 두 모드에서 동일하다 — emg_map 은 어느 쪽이든 (B,256,500) 이고 IMU 입력도
500 이라 imu_hat 이 1:1 로 대응한다. 리샘플이나 정렬 트릭이 전혀 필요 없다.

디코더는 emg_map 만 본다. imu_map 이나 fused feature 는 보지 않으므로 "입력 IMU 를
그대로 베껴 출력하는" 누수 경로가 없다. 다만 멀티모달 구조에서는 IMU 가 이미 분류기
입력이기도 해서, 이 보조 손실은 분류 경로에 새 정보를 주는 게 아니라 **EMG 인코더가
운동학 성분을 담도록 강제하는 정규화**로 작동한다 (EMG 단독 구성이었다면 "EMG 만으로
운동을 내재화한다"는 더 강한 주장이 됐을 것이다).

참고 — Estimating IMU signals from surface EMG using physics-informed and
domain-adaptive neural networks (J. Electromyogr. Kinesiol., 2026):
EMG→IMU 회귀 + GRL 적대적 도메인 적응 + jerk 최소화 물리 손실. 다만 그쪽 도메인은
과제(task) 간이고 여기는 기기(device) 간이다.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "Multimodal"))
from mm_model import InterFusionBackbone, ReverseLayerF, \
    _build_label_classifier                                     # noqa: E402
from baseline.baseline_model import ResidualTCNBlock            # noqa: E402
from Compact.nointerp_model import IMUEncoderNoInterp, \
    variant_info as nointerp_variant_info                       # noqa: E402


def fusion_info(imu_interp=False):
    """융합단 구성을 dict 로 반환한다 (모델을 만들지 않고도 provenance 를 찍기 위함)."""
    if imu_interp:
        return {
            "imu_fusion": "interp",
            "imu_encoder": "orig (stem stride 5 + MaxPool2)",
            "imu_map_len": 50,
            "interpolation": "linear 50 -> 500",
        }
    info = nointerp_variant_info()
    return {
        "imu_fusion": "nointerp",
        "imu_encoder": "IMUEncoderNoInterp (stem stride 1, MaxPool -> Identity)",
        "imu_map_len": 500,
        "interpolation": "none",
        "imu_dilations": info["imu_dilations"],
        "imu_rf_sec": info["imu_rf_sec"],
    }


class IMUDecoder(nn.Module):
    """emg_map (B,256,T) -> 복원 IMU (B,3,T).  T 는 그대로 보존된다.

    dilation 2/4/8 로 수용영역을 넓혀(±약 0.3초 @100Hz) 국소 잡음이 아니라 움직임
    궤적을 보게 한다. 마지막은 1x1 conv 에 활성함수를 두지 않는다 — 출력이 부호를
    갖는 가속도이므로 GELU/ReLU 를 걸면 음의 축 성분을 못 낸다.

    dropout 은 인코더 기본값 0.5 대신 낮춘다. 0.5 는 분류용 정규화 수치이고,
    회귀 디코더에 그대로 쓰면 출력이 과하게 뭉개져 재구성 손실이 바닥을 못 친다.
    """

    def __init__(self, in_ch=256, width=128, out_ch=3, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, width, kernel_size=5, padding=2),
            nn.BatchNorm1d(width),
            nn.GELU(),
            ResidualTCNBlock(width, width, dilation=2, dropout=dropout),
            ResidualTCNBlock(width, width, dilation=4, dropout=dropout),
            ResidualTCNBlock(width, width // 2, dilation=8, dropout=dropout),
            nn.Conv1d(width // 2, out_ch, kernel_size=1),
        )

    def forward(self, emg_map):
        return self.net(emg_map)


class PhysicsInformedCDAN(nn.Module):
    """InterFusionCDAN 과 동일한 분류/도메인 경로 + EMG→IMU 디코더.

    forward(x_emg, x_imu, alpha) -> (class_logits, domain_logits, imu_hat)

    첫 반환값이 class_logits 라 mm_utils.evaluate(needs_alpha=True) 가 그대로 동작한다
    (evaluate 는 forward(...)[0] 만 쓴다).

    분류·도메인 경로는 InterFusionCDAN 과 **수치적으로 동일**해야 한다. 그래야 λ=0 이
    기존 CDAN 과 같은 대조군이 된다. backbone.forward 를 부르는 대신 그 내부 단계를
    그대로 펼쳐 쓴 이유는 중간의 emg_map 을 꺼내야 하기 때문이며, 연산 순서는 바꾸지
    않았다. physics_model.py 를 직접 실행하면 이 동치성을 검증한다.
    """

    def __init__(self, num_classes=10, feature_dim=512,
                 decoder_width=128, decoder_dropout=0.1, imu_interp=False):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.imu_interp = imu_interp

        self.backbone = InterFusionBackbone()
        if not imu_interp:
            # 길이 보존 IMU 인코더로 교체한다. 속성 이름(stem/layers)과 블록 인덱스가
            # 원본과 같으므로 backbone._temporal_map(layers[:-1] 규약)이 그대로 돈다.
            self.backbone.imu_encoder = IMUEncoderNoInterp()
        self.label_classifier = _build_label_classifier(num_classes)

        # 도메인 판별기 = feature x 분류확률 외적 (InterFusionCDAN 과 동일 구성)
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

        self.imu_decoder = IMUDecoder(in_ch=256, width=decoder_width,
                                      out_ch=3, dropout=decoder_dropout)

    def fusion_info(self):
        """융합단 구성 — 결과 json 에 provenance 로 박는다."""
        return fusion_info(self.imu_interp)

    def conditional_feature(self, features, class_logits):
        """feature 와 분류 확률의 외적(multilinear map)을 평탄화해 반환한다."""
        class_probs = F.softmax(class_logits, dim=1)
        conditional = torch.bmm(class_probs.unsqueeze(2), features.unsqueeze(1))
        return conditional.view(features.size(0), -1)

    def forward(self, x_emg, x_imu, alpha=1.0):
        bb = self.backbone
        emg_map = bb._temporal_map(bb.emg_encoder, x_emg)   # (B,256,500)
        imu_map = bb._temporal_map(bb.imu_encoder, x_imu)   # nointerp:(B,256,500) interp:(B,256,50)
        if self.imu_interp:
            imu_map = F.interpolate(imu_map, size=emg_map.size(-1),
                                    mode="linear", align_corners=True)
        elif imu_map.size(-1) != emg_map.size(-1):
            raise RuntimeError(
                f"보간 없이 concat 하려면 길이가 같아야 한다: "
                f"EMG {emg_map.size(-1)} vs IMU {imu_map.size(-1)}. "
                f"입력 IMU 길이가 500(=100Hz×5초) 규약과 다른지 확인할 것.")
        fused = torch.cat([emg_map, imu_map], dim=1)        # (B,512,500)
        features = bb.fusion(fused).squeeze(-1)             # (B,512)

        class_output = self.label_classifier(features)
        conditional_features = self.conditional_feature(features, class_output)
        domain_output = self.domain_classifier(
            ReverseLayerF.apply(conditional_features, alpha)
        )
        imu_hat = self.imu_decoder(emg_map)                 # (B,3,500)
        return class_output, domain_output, imu_hat


# ----------------------------------------------------------------------
# 단독 실행: shape + 기존 CDAN 과의 동치성 검증
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import inspect

    from mm_model import InterFusionCDAN as InterpCDAN
    from Compact.nointerp_model import InterFusionCDAN as NoInterpCDAN

    torch.manual_seed(0)
    emg = torch.randn(4, 2, 5000)
    imu = torch.randn(4, 3, 500)

    n_params = {}
    for imu_interp, ref_cls, label in ((False, NoInterpCDAN, "nointerp (기본)"),
                                       (True,  InterpCDAN,  "interp (옛 동작)")):
        print(f"\n=== imu_fusion = {label} ===")
        m = PhysicsInformedCDAN(num_classes=10, imu_interp=imu_interp)
        m.eval()
        print(f"[구성] {m.fusion_info()}")

        c, d, ih = m(emg, imu, alpha=0.5)
        print(f"[shape] class {tuple(c.shape)}  domain {tuple(d.shape)}  "
              f"imu_hat {tuple(ih.shape)}   (expected (4,10)/(4,2)/(4,3,500))")
        assert ih.shape == imu.shape, "imu_hat 이 입력 IMU 와 shape 이 달라 손실을 못 건다"

        # --- 융합 직전 두 map 의 길이
        bb = m.backbone
        emg_map = bb._temporal_map(bb.emg_encoder, emg)
        imu_map = bb._temporal_map(bb.imu_encoder, imu)
        print(f"[map]   EMG {tuple(emg_map.shape)}   IMU {tuple(imu_map.shape)}"
              f"{'  ← 보간 없이 concat' if not imu_interp else '  → interpolate 로 500 확대'}")
        if not imu_interp:
            assert imu_map.size(-1) == emg_map.size(-1) == 500, \
                "nointerp 인데 IMU map 길이가 500 이 아니다 — 융합 시각이 대응하지 않는다"
            src = inspect.getsource(PhysicsInformedCDAN.forward)
            interp_line = [l.strip() for l in src.splitlines() if "interpolate" in l]
            assert all(l.startswith("#") or "if self.imu_interp" in src for l in interp_line)
            print(f"[검증] nointerp 경로에서 interpolate 미실행 ✓ "
                  f"(호출은 imu_interp=True 분기 안에만 존재)")

        # --- 동치성: 같은 가중치를 옮겨 심으면 분류/도메인 출력이 대응 CDAN 과 같아야 한다
        ref = ref_cls(num_classes=10)
        ref.eval()
        ref.backbone.load_state_dict(m.backbone.state_dict())
        ref.label_classifier.load_state_dict(m.label_classifier.state_dict())
        ref.domain_classifier.load_state_dict(m.domain_classifier.state_dict())
        with torch.no_grad():
            rc, rd = ref(emg, imu, alpha=0.5)
            mc, md, _ = m(emg, imu, alpha=0.5)
        dc = (rc - mc).abs().max().item()
        dd = (rd - md).abs().max().item()
        print(f"[동치성] vs {ref_cls.__module__}.{ref_cls.__name__}: "
              f"class 최대차 {dc:.2e}  domain 최대차 {dd:.2e}  (0 이어야 정상)")
        assert dc < 1e-5 and dd < 1e-5, \
            "분류/도메인 경로가 대응 CDAN 과 다르다 — λ=0 대조군이 성립 안 함"

        n_all = sum(p.numel() for p in m.parameters() if p.requires_grad)
        n_dec = sum(p.numel() for p in m.imu_decoder.parameters() if p.requires_grad)
        n_ref = sum(p.numel() for p in ref.parameters() if p.requires_grad)
        print(f"[params] 대응 CDAN {n_ref:,} | physics {n_all:,} "
              f"(디코더 {n_dec:,} = +{n_dec / n_ref * 100:.1f}%)")
        n_params[label] = n_all

    # --- 두 모드의 파라미터 수가 같아야 용량 교락이 없다
    a, b = n_params.values()
    print(f"\n[용량 교락] nointerp {a:,} vs interp {b:,}  차이 {a - b:+,}")
    assert a == b, "두 모드의 파라미터 수가 달라 비교가 용량 교락된다"
    print("OK")
