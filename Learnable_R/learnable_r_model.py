"""Learnable R 정렬 모델 정의 (IMU 단독 CDAN + 학습 가능한 3x3 정렬 행렬).

기존 IMUOnlyCDAN 백본을 그대로 두고, target IMU 앞단에 학습 가능한 3x3 행렬 R 을
하나 끼운다. 흐름은 다이어그램과 동일하다:

    target IMU --R--> aligned IMU --> (IMU feature extractor) --> label / domain head

source IMU 는 참조 좌표계로 보고 R 을 거치지 않는다(apply_r=False). 즉 R 은 target
센서 좌표계를 source 좌표계로 회전시켜 맞추는 역할만 한다. 전처리 단계의 손수
구한 정렬(pca/kabsch/gravity/permutation)을 학습으로 대체하는 셈이라, 데이터는
정렬 안 한 preprocessed_MM_raw 를 쓰는 것이 자연스럽다.

R 은 4개의 기하 손실로 정칙화한다 (proper rotation + 물리 제약):
    L_orth    = ||RᵀR - I||_F        R 을 직교(회전)로
    L_det     = |det(R) - 1|         반사 없는 proper rotation 으로
    L_gravity = ||R g_t - g_s||      target 중력방향을 source 중력방향에 정렬
    L_norm    = mean| |R a| - |a| |  가속도 크기 보존 (직교성과 중복되나 명시적 제약)

R 의 직교성을 부드럽게 유도하므로 hard 한 SO(3) 재매개화 없이 일반 행렬 파라미터로
둔다. L_orth/L_det 가 회전으로 수렴시키고, L_gravity 가 yaw 를 제외한 방향을, L_norm
이 스케일을 잡아준다.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# Multimodal/ 의 백본·헤드·GRL 을 재사용한다.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "Multimodal"))
from mm_model import (  # noqa: E402
    IMUOnlyBackbone, ReverseLayerF, _build_label_classifier, _build_domain_classifier)


# ----------------------------------------------------------------------
# 학습 가능한 3x3 정렬 행렬
# ----------------------------------------------------------------------
def _hat(w):
    """so(3) 벡터 (3,) → 반대칭행렬 (3,3)."""
    O = torch.zeros((), device=w.device, dtype=w.dtype)
    return torch.stack([O, -w[2], w[1], w[2], O, -w[0], -w[1], w[0], O]).reshape(3, 3)


def so3_exp(w):
    """so(3) 벡터 (3,) → SO(3) 회전행렬 (Rodrigues). det=+1·RᵀR=I 항상 보장."""
    theta = torch.sqrt((w * w).sum() + 1e-12)
    K = _hat(w / theta)
    I = torch.eye(3, device=w.device, dtype=w.dtype)
    return I + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)


def so3_log(R):
    """SO(3) → so(3) 벡터 (행렬 init 을 w 로 변환할 때만 사용)."""
    R = torch.as_tensor(R, dtype=torch.float32)
    cos = ((R.trace() - 1.0) / 2.0).clamp(-1.0, 1.0)
    theta = torch.arccos(cos)
    v = torch.stack([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-4:
        return 0.5 * v
    return theta / (2.0 * torch.sin(theta)) * v


class LearnableR(nn.Module):
    """target IMU (B,3,T) 의 센서축을 회전시키는 학습 가능한 회전 R.

    mode="so3"(기본): R 을 so(3) 3-파라미터 w 의 행렬지수로 둔다 → **R 이 항상 proper
    rotation**(det=+1, RᵀR=I). det 붕괴·NaN·MSE shrinkage 가 구조적으로 불가능하고,
    gravity/pca/적대 손실은 R 을 회전만 시킨다. L_rotation 항은 이제 구조적으로 0(불필요).
    init=None → w=0 → R=I 에서 출발. mode="matrix": 기존 자유 3x3(하위호환).
    """

    def __init__(self, init=None, mode="so3"):
        super().__init__()
        self.mode = mode
        if mode == "so3":
            w0 = torch.zeros(3) if init is None else so3_log(init)
            self.w = nn.Parameter(w0.float())
        else:
            R0 = torch.eye(3) if init is None else torch.as_tensor(init, dtype=torch.float32)
            self._Rmat = nn.Parameter(R0.clone().float())

    @property
    def R(self):
        return so3_exp(self.w) if self.mode == "so3" else self._Rmat

    def forward(self, x):  # x: (B,3,T)
        # aligned[b,i,t] = sum_j R[i,j] x[b,j,t]
        return torch.einsum("ij,bjt->bit", self.R, x)


# ----------------------------------------------------------------------
# 기하 손실
# ----------------------------------------------------------------------
def gravity_dir(x):
    """raw IMU 배치 (B,3,T) 에서 배치 평균 중력 방향 벡터 (3,) 를 추정한다.

    IMU 가 중력 포함 raw accel 이라 시간평균 DC 성분이 ≈ 중력. 샘플별 DC 를 구한 뒤
    배치 평균한다. (정규화는 호출부에서 필요 시.)
    """
    return x.mean(dim=2).mean(dim=0)  # (B,3,T) -> (B,3) -> (3,)


def r_geometric_losses(R, x_src, x_tgt):
    """R 의 기하 제약 손실 4종을 계산해 (L_orth, L_det, L_gravity, L_norm) 로 반환.

    x_src / x_tgt : 같은 배치의 raw IMU (B,3,T). 중력 정렬은 두 도메인의 배치 평균
    중력 방향을 비교한다.
    """
    I = torch.eye(3, device=R.device, dtype=R.dtype)
    L_orth = torch.linalg.norm(R.transpose(0, 1) @ R - I)        # ||RᵀR - I||_F
    L_det = (torch.linalg.det(R) - 1.0).abs()                    # |det(R) - 1|

    g_src = gravity_dir(x_src)                                   # (3,)
    g_tgt = gravity_dir(x_tgt)                                   # (3,)
    L_gravity = torch.linalg.norm(R @ g_tgt - g_src)            # ||R g_t - g_s||

    Ra = torch.einsum("ij,bjt->bit", R, x_tgt)                  # (B,3,T)
    L_norm = (Ra.norm(dim=1) - x_tgt.norm(dim=1)).abs().mean()  # mean| |Ra| - |a| |
    return L_orth, L_det, L_gravity, L_norm


def r_unified_losses(R, x_src, x_tgt):
    """통합 손실용 제곱형 기하 손실 (L_rotation, L_gravity) 를 반환한다.

    다이어그램 식 그대로:
        L_rotation = ||RᵀR - I||²_F + |det(R) - 1|²    (proper rotation)
        L_gravity  = ||R g_t - g_s||²                   (중력 정렬, pitch/roll 2 DOF)
    r_geometric_losses 의 비제곱형과 달리 제곱이라 0 근처에서 더 매끄럽고 큰 위반에
    강한 페널티를 준다. g 는 배치 평균 중력 방향(raw accel DC).
    """
    I = torch.eye(3, device=R.device, dtype=R.dtype)
    L_rotation = (R.transpose(0, 1) @ R - I).pow(2).sum() \
        + (torch.linalg.det(R) - 1.0).pow(2)
    g_src, g_tgt = gravity_dir(x_src), gravity_dir(x_tgt)
    L_gravity = (R @ g_tgt - g_src).pow(2).sum()
    return L_rotation, L_gravity


def pca_frame(x):
    """배치 평균 모션 공분산의 주축 프레임 (3,3) 을 부호고정해 반환 (physics-informed, 매 배치).

    x: (B,3,T) raw IMU. DC(중력) 제거 후 모션 공분산 → eigh → 고윳값 내림차순 정렬.
    부호 모호성은 모션 투영 왜도(skewness) 부호로 고정([[research-direction-feature-augment]]
    DirectionFeatures 와 동일 규약). 열 = 주축(col0=최대분산). **계산된 R 주입 아님** — 매
    배치 신호에서 주축을 새로 구하는 물리량이라 캘리브레이션이 아니다.
    """
    m = x.mean(dim=2, keepdim=True)                   # (B,3,1) DC≈중력
    xc = x - m                                        # 모션 (중력 제거)
    T = xc.size(2)
    cov = torch.einsum("bct,bdt->bcd", xc, xc) / T    # (B,3,3) 샘플별 공분산
    cov = cov.mean(dim=0)                             # (3,3) 배치평균
    cov = cov + 1e-6 * torch.eye(3, device=x.device, dtype=x.dtype)  # eigh 안정 jitter
    _, evecs = torch.linalg.eigh(cov)                 # 고윳값 오름차순
    F = evecs.flip(1)                                 # 내림차순 (col0=최대분산축)
    proj = torch.einsum("bct,ck->bkt", xc, F)         # (B,3,T) 각 주축 투영
    sign = torch.sign((proj ** 3).sum(dim=(0, 2)))    # (3,) 왜도 부호
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    F = F * sign.unsqueeze(0)                         # (3,3) 열별 부호고정
    # proper rotation 보장(det=+1): 부호고정이 핸디드니스를 뒤집으면 최소분산축 반전.
    # source/target 둘 다 det +1 이라야 정렬 R=Fs Ftᵀ 가 회전이 되어 rotation reg 와 안 싸움.
    if torch.det(F) < 0:
        F = torch.cat([F[:, :2], -F[:, 2:3]], dim=1)
    return F


def pca_alignment_loss(R, x_src, x_tgt):
    """on-the-fly PCA 정렬 손실: target 주축 프레임을 source 프레임에 R 로 맞춤.

    L_pca = ||R Ft - Fs||²_F   (Fs/Ft = 매 배치 계산한 주축, gravity loss 와 동형 규약).
    gravity 가 못 보는 yaw(수평 모션구조)를 결정적으로 잡는다. 부호고정으로 ± 모호성 완화,
    단 수평 모션이 등방적이거나 고윳값 근접 시 yaw under-determined(한계).
    """
    Fs, Ft = pca_frame(x_src), pca_frame(x_tgt)
    return ((R @ Ft) - Fs).pow(2).sum()


def random_rotation(batch, device, max_angle_deg=15.0):
    """배치마다 서로 다른 작은 무작위 회전행렬 (batch,3,3) 을 생성한다.

    축은 균등 무작위 단위벡터, 각도는 [0, max_angle] 균등. target IMU 일관성
    증강용 — 센서 좌표계의 작은 회전은 물리적으로 타당한 교란이라, 정렬·예측이
    이에 불변하도록 유도하면 target 예측이 안정된다(Rodrigues 공식).
    """
    axis = F.normalize(torch.randn(batch, 3, device=device), dim=1)        # (B,3)
    angle = torch.rand(batch, device=device) * (max_angle_deg * 3.14159265 / 180.0)
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    zero = torch.zeros_like(x)
    K = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=1).view(batch, 3, 3)
    I = torch.eye(3, device=device).expand(batch, 3, 3)
    s = torch.sin(angle).view(batch, 1, 1)
    c = (1.0 - torch.cos(angle)).view(batch, 1, 1)
    return I + s * K + c * (K @ K)                                          # (B,3,3)


def augment_imu(x, max_angle_deg=15.0):
    """raw IMU (B,3,T) 에 샘플별 작은 무작위 회전을 적용한 증강 뷰를 반환한다."""
    Rb = random_rotation(x.size(0), x.device, max_angle_deg)               # (B,3,3)
    return torch.einsum("bij,bjt->bit", Rb, x)


def post_r_normalize(x, mode, bn=None):
    """R 직후 인코더 입력 정규화 (회전→정규화 순서, 정규화 딜레마 분리책).

    gravity/rotation 손실은 학습 루프에서 raw isotropic 배치로 계산하므로 R 은 중력
    살아있는 pre-norm 데이터를 본다(식별성·gravity 유지). 여기 정규화는 R 뒤 공통(source)
    프레임에서 source·target 동일하게 걸어 인코더가 축별 표준화 입력을 받게 한다.
      "none"      : 그대로
      "instance"  : 윈도우별·축별 (x-mean)/std (per-sample, 도메인 불변)
      "batchnorm" : 채널별 BatchNorm (running 통계≈데이터셋 zscore). bn 모듈 필요.
    """
    if mode == "instance":
        m = x.mean(dim=2, keepdim=True)
        s = x.std(dim=2, keepdim=True)
        return (x - m) / (s + 1e-5)
    if mode == "batchnorm":
        return bn(x)
    return x


# ----------------------------------------------------------------------
# Learnable R + IMU 단독 CDAN
# ----------------------------------------------------------------------
class LearnableRCDAN(nn.Module):
    """IMUOnlyCDAN 에 target IMU 정렬용 learnable R 만 추가한 모델.

    forward(x_emg, x_imu, alpha, apply_r) — x_emg 는 규약 호환용(미사용).
    apply_r=True 면 입력 IMU 를 R 로 정렬한 뒤 백본에 넣는다(=target 경로). source 는
    apply_r=False 로 호출해 R 을 우회한다. 헤드·판별기 구조는 IMUOnlyCDAN 과 동일.
    """

    def __init__(self, num_classes=10, feature_dim=512, R_init=None, post_r_norm="none"):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.post_r_norm = post_r_norm   # "none" | "instance" | "batchnorm"

        self.r = LearnableR(R_init)
        if post_r_norm == "batchnorm":
            self.input_bn = nn.BatchNorm1d(3)  # R 직후 축별 정규화 (회전→정규화)
        self.backbone = IMUOnlyBackbone()
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

    def forward(self, x_emg, x_imu, alpha=1.0, apply_r=False):  # x_emg 미사용
        if apply_r:
            x_imu = self.r(x_imu)
        x_imu = post_r_normalize(x_imu, self.post_r_norm,
                                 getattr(self, "input_bn", None))
        features = self.backbone(x_imu)
        class_output = self.label_classifier(features)

        conditional_features = self.conditional_feature(features, class_output)
        domain_output = self.domain_classifier(
            ReverseLayerF.apply(conditional_features, alpha))
        return class_output, domain_output, features


# ----------------------------------------------------------------------
# Learnable R + IMU 단독 + MMD 정렬 (CDAN 대신 feature MMD 로 R 을 업데이트)
# ----------------------------------------------------------------------
class LearnableRMMD(nn.Module):
    """IMU 단독 백본 + label head + target IMU 앞 learnable R. 정렬은 MMD 로 한다.

    동기: CDAN 의 적대적 신호(3-DOF R 로 판별기를 속이기엔 약함)나 죽은 gravity 손실
    대신, 인코더 출력 feature 공간에서 source 분포와 R-정렬된 target 분포 사이의 MMD 를
    직접 줄여 R(과 인코더)을 업데이트한다. R 에 매끄럽고 과제 관련성 있는 gradient 가
    흐른다.

    forward(x_emg, x_imu, apply_r) -> (class_logits, features)   (x_emg 미사용)
      · source 는 apply_r=False (참조 좌표계)
      · target 은 apply_r=True  (R 로 정렬 후 인코더)
    학습 루프가 src_feat / tgt_feat 사이 MMD 를 계산하고, R 은 L_orth/L_det 로 회전에
    묶인다(gravity 가 데이터에서 표준화로 제거됐으므로 L_gravity 는 기본 off).
    """

    def __init__(self, num_classes=10, R_init=None):
        super().__init__()
        self.num_classes = num_classes
        self.r = LearnableR(R_init)
        self.backbone = IMUOnlyBackbone()
        self.label_classifier = _build_label_classifier(num_classes)

    def forward(self, x_emg, x_imu, apply_r=False):  # x_emg 미사용
        if apply_r:
            x_imu = self.r(x_imu)
        features = self.backbone(x_imu)
        return self.label_classifier(features), features


# ----------------------------------------------------------------------
# Learnable R + IMU 단독 + DANN 정렬 (통합 손실)
# ----------------------------------------------------------------------
class LearnableRDANN(nn.Module):
    """IMU 단독 백본 + label head + plain DANN domain discriminator + target IMU 앞 R.

    CDAN(조건부 외적)이 아닌 plain DANN: GRL 을 통과한 512차원 feature 를 그대로
    domain 판별기(512→2)에 넣어 source vs R-정렬 target 을 적대적으로 정렬한다.
    통합 손실에서 L_domain 역할을 한다.

    forward(x_emg, x_imu, alpha, apply_r) -> (class_logits, domain_logits, features)
      · source 는 apply_r=False (참조 좌표계)
      · target 은 apply_r=True  (R 로 정렬 후 인코더)
    consistency 손실은 학습 루프가 augment_imu(target) 의 class_logits 을 따로 뽑아
    원본 target 예측과 비교한다.
    """

    def __init__(self, num_classes=10, R_init=None, post_r_norm="none"):
        super().__init__()
        self.num_classes = num_classes
        self.post_r_norm = post_r_norm   # "none" | "instance" | "batchnorm"
        self.r = LearnableR(R_init)
        if post_r_norm == "batchnorm":
            self.input_bn = nn.BatchNorm1d(3)  # R 직후 축별 정규화 (데이터셋 zscore 근사)
        self.backbone = IMUOnlyBackbone()
        self.label_classifier = _build_label_classifier(num_classes)
        self.domain_classifier = _build_domain_classifier()  # 512 -> 2 (plain DANN)

    def _norm(self, x):
        """R 직후 인코더 입력 정규화 (정규화 딜레마 분리책).

        gravity/rotation 손실은 학습 루프에서 raw isotropic 배치로 계산하므로 R 은
        중력 살아있는 pre-norm 데이터를 본다(식별성·gravity 유지). 여기 정규화는 R
        뒤 공통(source) 프레임에서 source·target 동일하게 걸어, 인코더가 축별 표준화된
        입력(zscore 품질)을 받게 한다 → 축별 진폭/DC 도메인차 제거. R 은 축배정·방향만,
        스케일은 norm 이 담당하도록 역할 분리. 순서가 핵심: 회전(R) → 정규화.

        "instance"  : 윈도우별·축별 표준화 (x-mean)/std. 도메인 불변, per-sample(노이즈↑).
        "batchnorm" : 채널별 BatchNorm. running 통계가 데이터셋 zscore 를 근사(75% 레시피의
                      "pca 회전 → zscore" 와 동형). P1 에서 source 통계 학습, P2 는 eval 로
                      고정 → R 이 target 을 source 프레임으로 돌리므로 source 통계가 맞다.
        """
        if self.post_r_norm == "instance":
            m = x.mean(dim=2, keepdim=True)
            s = x.std(dim=2, keepdim=True)
            return (x - m) / (s + 1e-5)
        if self.post_r_norm == "batchnorm":
            return self.input_bn(x)
        return x

    def forward(self, x_emg, x_imu, alpha=1.0, apply_r=False):  # x_emg 미사용
        if apply_r:
            x_imu = self.r(x_imu)
        x_imu = self._norm(x_imu)
        features = self.backbone(x_imu)
        class_output = self.label_classifier(features)
        domain_output = self.domain_classifier(ReverseLayerF.apply(features, alpha))
        return class_output, domain_output, features


# ----------------------------------------------------------------------
# 단독 실행 테스트 (shape / 손실 동작 검증)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    emg = torch.randn(8, 2, 5000)
    src_imu = torch.randn(8, 3, 500)
    tgt_imu = torch.randn(8, 3, 500)

    model = LearnableRCDAN(num_classes=10, post_r_norm="batchnorm")
    c_src, d_src, f_src = model(emg, src_imu, alpha=0.5, apply_r=False)
    c_tgt, d_tgt, f_tgt = model(emg, tgt_imu, alpha=0.5, apply_r=True)
    print(f"[LearnableR-CDAN] src class/domain/feat: "
          f"{tuple(c_src.shape)}/{tuple(d_src.shape)}/{tuple(f_src.shape)}  "
          f"tgt: {tuple(c_tgt.shape)}/{tuple(d_tgt.shape)}/{tuple(f_tgt.shape)}  "
          f"(expected (8,10)/(8,2)/(8,512))")

    mmd_model = LearnableRMMD(num_classes=10)
    c_src, f_src = mmd_model(emg, src_imu, apply_r=False)
    c_tgt, f_tgt = mmd_model(emg, tgt_imu, apply_r=True)
    print(f"[LearnableR-MMD]  src class/feat: {tuple(c_src.shape)}/{tuple(f_src.shape)}  "
          f"tgt class/feat: {tuple(c_tgt.shape)}/{tuple(f_tgt.shape)}  "
          f"(expected (8,10)/(8,512))")

    dann_model = LearnableRDANN(num_classes=10)
    c_src, d_src, f_src = dann_model(emg, src_imu, alpha=0.5, apply_r=False)
    c_tgt, d_tgt, f_tgt = dann_model(emg, tgt_imu, alpha=0.5, apply_r=True)
    print(f"[LearnableR-DANN] src class/domain/feat: "
          f"{tuple(c_src.shape)}/{tuple(d_src.shape)}/{tuple(f_src.shape)}  "
          f"tgt: {tuple(c_tgt.shape)}/{tuple(d_tgt.shape)}/{tuple(f_tgt.shape)}  "
          f"(expected (8,10)/(8,2)/(8,512))")
    aug = augment_imu(tgt_imu)
    print(f"  augment_imu shape {tuple(aug.shape)} (expected (8,3,500)); "
          f"|aug| preserved: {torch.allclose(aug.norm(dim=1), tgt_imu.norm(dim=1), atol=1e-4)}")
    L_rot, L_grav_sq = r_unified_losses(dann_model.r.R, src_imu, tgt_imu)
    print(f"  unified losses @ R=I -> rotation={L_rot:.4f} gravity={L_grav_sq:.4f}  (rotation≈0 at identity)")

    L_orth, L_det, L_grav, L_norm = r_geometric_losses(model.r.R, src_imu, tgt_imu)
    print(f"  losses @ R=I  ->  orth={L_orth:.4f} det={L_det:.4f} "
          f"grav={L_grav:.4f} norm={L_norm:.4f}  (orth/det≈0 at identity)")

    total = sum(p.numel() for p in mmd_model.parameters() if p.requires_grad)
    print(f"  LearnableR-MMD trainable params: {total:,}  (R adds {mmd_model.r.R.numel()})")
