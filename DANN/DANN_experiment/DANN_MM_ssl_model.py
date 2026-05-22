"""DANN-MM-SSL 모델 정의 (모달별 contrastive 사전학습 + 융합 DANN).

Phase 1: EMG/IMU 인코더를 라벨 없이 SimCLR 식 contrastive 로 각각 사전학습.
         source_train + target_train 을 합쳐 쓰므로 target 라벨을 보지 않고도
         target 분포가 사전학습에 반영된다 (source bias 제거가 목적).
Phase 2: 사전학습된 인코더를 DualEncoderDANN_Sep 에 이식 후 GRL 도메인 적대 학습.

ContrastiveModel = encoder + projection head. 사전학습이 끝나면 projection head 는
버리고 encoder 만 Phase 2 로 넘긴다. Phase 2 모델(DualEncoderDANN_Sep)은
DANN_MM_sep_model 의 것을 그대로 재사용한다 (Phase 1 만 다른 ablation).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
# 프로젝트 루트를 import 경로에 추가 (DANN/DANN_experiment/ → ../../)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from baseline.baseline_DualEncoder import EMGEncoder, IMUEncoder


class ProjectionHead(nn.Module):
    """contrastive 용 projection head.  (B, feat_dim) -> (B, proj_dim)

    SimCLR 와 같이 contrastive loss 는 이 projection 공간에서 계산하고,
    사전학습이 끝나면 head 는 버리고 encoder 만 보존한다.
    """

    def __init__(self, feat_dim=256, hidden_dim=256, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x):
        return self.net(x)


class ContrastiveModel(nn.Module):
    """encoder + projection head.  Phase 1 SSL 사전학습용.

    forward(x) -> (B, proj_dim)  contrastive embedding
    학습 후 self.encoder 만 state_dict 로 저장해 Phase 2 에서 재사용한다.
    """

    def __init__(self, encoder, feat_dim=256, proj_dim=128):
        super().__init__()
        self.encoder = encoder
        self.projector = ProjectionHead(feat_dim, feat_dim, proj_dim)

    def forward(self, x):
        return self.projector(self.encoder(x))


def nt_xent_loss(z1, z2, temperature=0.5):
    """SimCLR NT-Xent loss.

    z1, z2 : (B, D)  같은 윈도우의 두 augmentation 뷰 임베딩.
    같은 인덱스 i 의 (z1_i, z2_i) 가 양성쌍, 배치 내 나머지 2B-2 개가 음성쌍.
    라벨을 전혀 쓰지 않는다 (instance discrimination).
    """
    B = z1.size(0)
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)        # (2B, D)
    sim = torch.mm(z, z.t()) / temperature                    # (2B, 2B)
    self_mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, float('-inf'))           # 자기 자신 제외
    # i in [0,B) → 양성 i+B,  i in [B,2B) → 양성 i-B
    targets = (torch.arange(2 * B, device=z.device) + B) % (2 * B)
    return F.cross_entropy(sim, targets)


# ----------------------------------------------------------------------
# 단독 실행 테스트 (출력 shape 검증용)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    emg_ssl = ContrastiveModel(EMGEncoder())
    imu_ssl = ContrastiveModel(IMUEncoder())
    emg = torch.randn(16, 2, 5000)
    imu = torch.randn(16, 3,  500)

    z1, z2 = emg_ssl(emg), emg_ssl(emg)
    print(f"EMG projection : {tuple(z1.shape)}  (expected [16, 128])")
    print(f"IMU projection : {tuple(imu_ssl(imu).shape)}  (expected [16, 128])")
    print(f"NT-Xent (EMG)  : {nt_xent_loss(z1, z2).item():.4f}")

    enc_params = sum(p.numel() for p in emg_ssl.encoder.parameters())
    print(f"EMG encoder params : {enc_params:,}")
