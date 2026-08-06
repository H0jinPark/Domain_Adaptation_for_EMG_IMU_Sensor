"""멀티모달(EMG+IMU) learnable-R 모델 정의 (새 파일 — 기존 모듈은 건드리지 않음).

IMU 단독 learnable_r_model.LearnableRCDAN 의 멀티모달판. 예전에 사전학습 R
(PCA/Kabsch/gravity 등)로 target IMU 를 정렬한 뒤 EMG+IMU 를 함께 넣어 멀티모달
추론하던 구조를, 그 R 을 **학습 가능한 3x3 SO(3) 회전**으로 대체한다. 흐름:

    target IMU --R--> aligned IMU ┐
                                  ├─ InterFusion(EMG+IMU concat) --> label / domain head
    EMG (회전 없음) ───────────────┘

재사용(수정 없이 import 만):
  · learnable_r_model : build_learnable_r(SO(3)/quat/matrix R), post_r_normalize,
                        gravity_loss / pca_alignment_loss (학습 스크립트에서 사용).
  · Multimodal/mm_model : InterFusionBackbone(EMG/IMU TCN + intermediate fusion),
                          ReverseLayerF(GRL), _build_label_classifier.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# Multimodal/ 의 멀티모달 백본·헤드·GRL 재사용
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "Multimodal"))
from mm_model import (  # noqa: E402
    InterFusionBackbone, ReverseLayerF, _build_label_classifier)

# 같은 폴더의 learnable_r_model 에서 R 파라미터화·정규화 헬퍼 재사용
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from learnable_r_model import build_learnable_r, post_r_normalize  # noqa: E402


class LearnableRMMCDAN(nn.Module):
    """InterFusion(EMG+IMU) 멀티모달 CDAN + target IMU 정렬용 learnable R.

    · R 은 target IMU 3축에만 적용된다 — EMG 는 회전 개념이 없어 절대 건드리지 않는다.
    · source 는 apply_r=False (IMU 정렬 안 함 = 참조 좌표계), target 은 apply_r=True.
    · gravity/pca 기하 prior 는 학습 루프에서 raw IMU + R 로 계산(인코더 무관), CDAN 은
      융합 feature(EMG+IMU)에 조건부 적대 정렬. 헤드·판별기 구조는 InterFusionCDAN 과 동일.

    forward(x_emg, x_imu, alpha, apply_r) -> (class_logits, domain_logits, fused_features)
    """

    def __init__(self, num_classes=10, feature_dim=512, R_init=None, post_r_norm="none",
                 r_param="so3", backbone="orig", cdan_rp=False):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.post_r_norm = post_r_norm   # "none" | "instance" | "batchnorm" (IMU 에만 적용)
        self.r_param = r_param            # "so3"(기본) | "quat" | "matrix"
        self.backbone_variant = backbone  # "orig" | "compact"
        self.cdan_rp = cdan_rp

        self.r = build_learnable_r(r_param, R_init)
        if post_r_norm == "batchnorm":
            self.input_bn = nn.BatchNorm1d(3)  # R 직후 IMU 축별 정규화 (회전→정규화)

        # 백본/헤드/판별기는 백본 변형에 따라 갈아끼운다. compact 는 TCN 블록을 줄인
        # 버전이며 출력 차원(512)이 같아 나머지 로직은 그대로다.
        if backbone == "compact":
            from Compact.compact_model import (
                InterFusionBackbone as CompactBackbone,
                _build_label_classifier as compact_label_head,
                _build_cdan_domain_classifier as compact_disc)
            self.backbone = CompactBackbone()
            self.label_classifier = compact_label_head(num_classes)
            self.rp, self.domain_classifier = compact_disc(
                feature_dim, num_classes, cdan_rp)
        elif backbone == "nointerp":
            # IMU 인코더가 길이 500 을 그대로 내므로 fusion 앞의 F.interpolate 가 없다.
            # 헤드/판별기는 원본과 동일 → 차이는 오직 IMU 시간해상도.
            if cdan_rp:
                raise ValueError("cdan_rp 는 backbone='compact' 에서만 지원한다")
            from Compact.nointerp_model import InterFusionBackbone as NoInterpBackbone
            self.rp = None
            self.backbone = NoInterpBackbone()
            self.label_classifier = _build_label_classifier(num_classes)
            cdan_input_dim = feature_dim * num_classes
            self.domain_classifier = nn.Sequential(
                nn.Linear(cdan_input_dim, 1024), nn.GELU(), nn.Dropout(0.5),
                nn.Linear(1024, 512), nn.GELU(), nn.Dropout(0.5),
                nn.Linear(512, 2),
            )
        elif backbone == "orig":
            if cdan_rp:
                raise ValueError("cdan_rp 는 backbone='compact' 에서만 지원한다")
            self.rp = None
            self.backbone = InterFusionBackbone()
            self.label_classifier = _build_label_classifier(num_classes)
            cdan_input_dim = feature_dim * num_classes
            self.domain_classifier = nn.Sequential(
                nn.Linear(cdan_input_dim, 1024), nn.GELU(), nn.Dropout(0.5),
                nn.Linear(1024, 512), nn.GELU(), nn.Dropout(0.5),
                nn.Linear(512, 2),
            )
        else:
            raise ValueError(f"알 수 없는 backbone: {backbone!r} (orig|compact|nointerp)")

    def conditional_feature(self, features, class_logits):
        class_probs = F.softmax(class_logits, dim=1)
        if self.rp is not None:
            return self.rp(features, class_probs)
        conditional = torch.bmm(class_probs.unsqueeze(2), features.unsqueeze(1))
        return conditional.view(features.size(0), -1)

    def forward(self, x_emg, x_imu, alpha=1.0, apply_r=False):
        if apply_r:
            x_imu = self.r(x_imu)                       # IMU 만 회전 (EMG 는 그대로)
        x_imu = post_r_normalize(x_imu, self.post_r_norm,
                                 getattr(self, "input_bn", None))
        features = self.backbone(x_emg, x_imu)          # EMG+IMU intermediate fusion
        class_output = self.label_classifier(features)

        conditional_features = self.conditional_feature(features, class_output)
        domain_output = self.domain_classifier(
            ReverseLayerF.apply(conditional_features, alpha))
        return class_output, domain_output, features


# ----------------------------------------------------------------------
# 단독 실행 테스트 (shape 검증)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from learnable_r_model import gravity_loss, pca_alignment_loss

    emg = torch.randn(8, 2, 5000)
    src_imu = torch.randn(8, 3, 500)
    tgt_imu = torch.randn(8, 3, 500)

    model = LearnableRMMCDAN(num_classes=10, post_r_norm="batchnorm")
    c_src, d_src, f_src = model(emg, src_imu, alpha=0.5, apply_r=False)
    c_tgt, d_tgt, f_tgt = model(emg, tgt_imu, alpha=0.5, apply_r=True)
    print(f"[LearnableR-MM-CDAN] src class/domain/feat: "
          f"{tuple(c_src.shape)}/{tuple(d_src.shape)}/{tuple(f_src.shape)}  "
          f"tgt: {tuple(c_tgt.shape)}/{tuple(d_tgt.shape)}/{tuple(f_tgt.shape)}  "
          f"(expected (8,10)/(8,2)/(8,512))")

    L_grav = gravity_loss(model.r.R, src_imu, tgt_imu)
    L_pca = pca_alignment_loss(model.r.R, src_imu, tgt_imu)
    print(f"  gravity @ R=I -> {L_grav:.4f}   pca @ R=I -> {L_pca:.4f}")

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  LearnableR-MM-CDAN trainable params: {total:,}  (R adds {model.r.R.numel()})")
