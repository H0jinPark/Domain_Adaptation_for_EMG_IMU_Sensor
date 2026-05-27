"""2-Phase 멀티모달 모델 정의.

Phase 1 (ssl_train.py에서 사용):
  EMGSSLModel / IMUSSLModel: encoder + SimCLR projection head
  nt_xent_loss: NT-Xent contrastive loss

Phase 2 (DANN/MMD/Coral/CDAN_train.py에서 사용):
  PretrainInterFusionBackbone: mm_model.py의 InterFusionBackbone과 동일한
    구조이나 load_pretrained_encoders()로 Phase 1 가중치를 로드할 수 있다.
  PretrainInterFusionClassifier / DANN / CDAN: 각 DA 방법의 Phase 2 모델.

규약
  - discriminator 내부에는 BatchNorm 을 두지 않는다 (분리 forward 규약).
  - load_pretrained_encoders()는 backbone 이 아직 device 에 옮겨지기 전에
    (cpu에서) 호출한 뒤 .to(device)를 수행한다.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline.baseline_DualEncoder import EMGEncoder, IMUEncoder
from baseline.baseline_model import ResidualTCNBlock


# -----------------------------------------------------------------------
# Gradient Reversal Layer
# -----------------------------------------------------------------------
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


# -----------------------------------------------------------------------
# Phase 1: 시계열 augmentation
# -----------------------------------------------------------------------
def ts_jitter(x, sigma=0.05):
    return x + torch.randn_like(x) * sigma


def ts_scaling(x, sigma=0.1):
    factor = 1.0 + torch.randn(x.size(0), 1, 1, device=x.device) * sigma
    return x * factor


def ts_time_shift(x, max_shift_ratio=0.1):
    shift = torch.randint(0, max(1, int(x.size(-1) * max_shift_ratio)), (1,)).item()
    return torch.roll(x, shifts=shift, dims=-1)


def ts_axis_sign_flip(x, flip_prob=0.5):
    """각 채널을 독립적으로 flip_prob 확률로 부호 반전한다 (IMU 축 불일치 대응)."""
    sign = torch.where(
        torch.rand(x.size(0), x.size(1), 1, device=x.device) < flip_prob,
        torch.full_like(x[:, :, :1], -1.0),
        torch.ones_like(x[:, :, :1]),
    )
    return x * sign


def augment_emg(x):
    x = ts_jitter(x)
    x = ts_scaling(x)
    x = ts_time_shift(x)
    return x


def augment_imu(x):
    x = ts_jitter(x)
    x = ts_scaling(x)
    x = ts_time_shift(x)
    x = ts_axis_sign_flip(x)
    return x


# -----------------------------------------------------------------------
# Phase 1: SimCLR projection head + SSL model
# -----------------------------------------------------------------------
class _ProjectionHead(nn.Module):
    """SimCLR projection head: encoder_dim -> 128 -> 64 (L2 normalized)."""

    def __init__(self, in_dim=256, hidden_dim=128, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class EMGSSLModel(nn.Module):
    """EMG encoder + projection head. Phase 1 학습 후 encoder 만 Phase 2에 넘긴다."""

    def __init__(self):
        super().__init__()
        self.encoder = EMGEncoder()
        self.projector = _ProjectionHead()

    def forward(self, x):
        return self.projector(self.encoder(x))


class IMUSSLModel(nn.Module):
    """IMU encoder + projection head. Phase 1 학습 후 encoder 만 Phase 2에 넘긴다."""

    def __init__(self):
        super().__init__()
        self.encoder = IMUEncoder()
        self.projector = _ProjectionHead()

    def forward(self, x):
        return self.projector(self.encoder(x))


def nt_xent_loss(z1, z2, temperature=0.5):
    """NT-Xent loss (SimCLR). z shape: (N, D), L2 normalized."""
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)           # (2N, D)
    sim = torch.mm(z, z.t()) / temperature   # (2N, 2N)
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float("-inf"))
    labels = torch.cat([torch.arange(N, device=z.device) + N,
                        torch.arange(N, device=z.device)])   # (2N,)
    return F.cross_entropy(sim, labels)


# -----------------------------------------------------------------------
# Phase 2: 공용 빌더 (Multimodal/mm_model.py와 동일)
# -----------------------------------------------------------------------
def _build_fusion():
    return nn.Sequential(
        nn.Conv1d(512, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm1d(64),
        nn.GELU(),
        ResidualTCNBlock(64, 64, dilation=1),
        ResidualTCNBlock(64, 128, dilation=2),
        ResidualTCNBlock(128, 128, dilation=4),
        ResidualTCNBlock(128, 256, dilation=8),
        ResidualTCNBlock(256, 256, dilation=16),
        ResidualTCNBlock(256, 512, dilation=32),
        nn.AdaptiveAvgPool1d(1),
    )


def _build_label_classifier(num_classes):
    return nn.Sequential(
        nn.Linear(512, 256),
        nn.GELU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )


def _build_domain_classifier():
    return nn.Sequential(
        nn.Linear(512, 256),
        nn.GELU(),
        nn.Dropout(0.5),
        nn.Linear(256, 128),
        nn.GELU(),
        nn.Dropout(0.5),
        nn.Linear(128, 2),
    )


# -----------------------------------------------------------------------
# Phase 2: Backbone (pretrained encoder 로딩 지원)
# -----------------------------------------------------------------------
class PretrainInterFusionBackbone(nn.Module):
    """InterFusionBackbone + load_pretrained_encoders().

    구조는 Multimodal/mm_model.py의 InterFusionBackbone과 동일하다.
    Phase 1 가중치 로드 후 Phase 2 end-to-end fine-tune.
    """

    def __init__(self):
        super().__init__()
        self.emg_encoder = EMGEncoder()
        self.imu_encoder = IMUEncoder()
        self.fusion = _build_fusion()

    def load_pretrained_encoders(self, emg_path, imu_path, device="cpu"):
        self.emg_encoder.load_state_dict(torch.load(emg_path, map_location=device))
        self.imu_encoder.load_state_dict(torch.load(imu_path, map_location=device))
        print(f"  [Phase2] EMG encoder  <- {os.path.basename(emg_path)}")
        print(f"  [Phase2] IMU encoder  <- {os.path.basename(imu_path)}")

    @staticmethod
    def _temporal_map(encoder, x):
        return encoder.layers[:-1](encoder.stem(x))

    def forward(self, x_emg, x_imu):
        emg_map = self._temporal_map(self.emg_encoder, x_emg)   # (B, 256, 500)
        imu_map = self._temporal_map(self.imu_encoder, x_imu)   # (B, 256,  50)
        imu_map = F.interpolate(imu_map, size=emg_map.size(-1),
                                mode="linear", align_corners=True)
        fused = torch.cat([emg_map, imu_map], dim=1)             # (B, 512, T)
        return self.fusion(fused).squeeze(-1)                    # (B, 512)


# -----------------------------------------------------------------------
# Phase 2: MMD/Coral 용
# -----------------------------------------------------------------------
class PretrainInterFusionClassifier(nn.Module):
    """backbone + label head. forward -> (class_logits, features)."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = PretrainInterFusionBackbone()
        self.label_classifier = _build_label_classifier(num_classes)

    def load_pretrained_encoders(self, emg_path, imu_path, device="cpu"):
        self.backbone.load_pretrained_encoders(emg_path, imu_path, device)

    def forward(self, x_emg, x_imu):
        features = self.backbone(x_emg, x_imu)
        return self.label_classifier(features), features


# -----------------------------------------------------------------------
# Phase 2: DANN 용
# -----------------------------------------------------------------------
class PretrainInterFusionDANN(nn.Module):
    """backbone + label head + domain discriminator(GRL).
    forward(x_emg, x_imu, alpha) -> (class_logits, domain_logits)
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = PretrainInterFusionBackbone()
        self.label_classifier = _build_label_classifier(num_classes)
        self.domain_classifier = _build_domain_classifier()

    def load_pretrained_encoders(self, emg_path, imu_path, device="cpu"):
        self.backbone.load_pretrained_encoders(emg_path, imu_path, device)

    def forward(self, x_emg, x_imu, alpha=1.0):
        features = self.backbone(x_emg, x_imu)
        class_output = self.label_classifier(features)
        domain_output = self.domain_classifier(ReverseLayerF.apply(features, alpha))
        return class_output, domain_output


# -----------------------------------------------------------------------
# Phase 2: CDAN 용
# -----------------------------------------------------------------------
class PretrainInterFusionCDAN(nn.Module):
    """backbone + label head + conditional domain discriminator(GRL).
    forward(x_emg, x_imu, alpha) -> (class_logits, domain_logits)
    """

    def __init__(self, num_classes=10, feature_dim=512):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.backbone = PretrainInterFusionBackbone()
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

    def load_pretrained_encoders(self, emg_path, imu_path, device="cpu"):
        self.backbone.load_pretrained_encoders(emg_path, imu_path, device)

    def conditional_feature(self, features, class_logits):
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


# -----------------------------------------------------------------------
# Phase 1 Hybrid: Source classification + Target contrastive learning
# -----------------------------------------------------------------------
class HybridSSLModel(nn.Module):
    """Phase 1 Hybrid pretraining.

    Source (labeled)  : EMGEncoder -> emg_classifier (CE loss)
                        IMUEncoder -> imu_classifier (CE loss)   <- modality별 독립 학습
    Target (unlabeled): per-modality augmented views -> NT-Xent (CL loss)

    각 encoder가 독립적인 classification signal을 받아 한쪽으로 쏠리지 않게 한다.
    Phase 2와 동일한 파일명으로 encoder weights를 저장하므로
    Phase 2 (DANN/MMD/Coral/CDAN_train.py) 코드 변경 없음.
    """

    def __init__(self, num_classes):
        super().__init__()
        self.emg_encoder = EMGEncoder()
        self.imu_encoder = IMUEncoder()
        self.emg_projector = _ProjectionHead(in_dim=256)   # CL head (target only)
        self.imu_projector = _ProjectionHead(in_dim=256)   # CL head (target only)
        self.emg_classifier = nn.Sequential(               # EMG-only cls head
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        self.imu_classifier = nn.Sequential(               # IMU-only cls head
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward_source_emg(self, x_emg):
        """EMG encoder classification. Returns (B, num_classes) logits."""
        return self.emg_classifier(self.emg_encoder(x_emg))

    def forward_source_imu(self, x_imu):
        """IMU encoder classification. Returns (B, num_classes) logits."""
        return self.imu_classifier(self.imu_encoder(x_imu))

    def forward_target_cl(self, x_emg, x_imu):
        """Target CL projection. Returns (emg_z, imu_z), each (B, 64) L2-normalized."""
        emg_z = self.emg_projector(self.emg_encoder(x_emg))
        imu_z = self.imu_projector(self.imu_encoder(x_imu))
        return emg_z, imu_z

    def save_encoders(self, emg_path, imu_path):
        torch.save(self.emg_encoder.state_dict(), emg_path)
        torch.save(self.imu_encoder.state_dict(), imu_path)


# -----------------------------------------------------------------------
# 단독 실행 테스트
# -----------------------------------------------------------------------
if __name__ == "__main__":
    emg = torch.randn(8, 2, 5000)
    imu = torch.randn(8, 3, 500)

    # Phase 1 shape 확인
    emg_ssl = EMGSSLModel()
    imu_ssl = IMUSSLModel()
    print(f"[EMGSSLModel] {tuple(emg_ssl(emg).shape)}  (expected (8, 64))")
    print(f"[IMUSSLModel] {tuple(imu_ssl(imu).shape)}  (expected (8, 64))")

    # Phase 2 shape 확인
    clf = PretrainInterFusionClassifier(10)
    dann = PretrainInterFusionDANN(10)
    cdan = PretrainInterFusionCDAN(10)

    logits, feat = clf(emg, imu)
    c, d = dann(emg, imu, alpha=0.5)
    cc, dd = cdan(emg, imu, alpha=0.5)

    print(f"[Classifier] logits={tuple(logits.shape)} feat={tuple(feat.shape)}")
    print(f"[DANN]  class={tuple(c.shape)} domain={tuple(d.shape)}")
    print(f"[CDAN]  class={tuple(cc.shape)} domain={tuple(dd.shape)}")

    for name, m in [("Classifier", clf), ("DANN", dann), ("CDAN", cdan)]:
        total = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"  {name:11s} params: {total:,}")
