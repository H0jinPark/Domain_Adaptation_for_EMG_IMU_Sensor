"""Learnable R 정렬 모델 정의 (IMU 단독 CDAN + 학습 가능한 3x3 정렬 행렬).

기존 IMUOnlyCDAN 백본을 그대로 두고, target IMU 앞단에 학습 가능한 3x3 행렬 R 을
하나 끼운다. 흐름은 다이어그램과 동일하다:

    target IMU --R--> aligned IMU --> (IMU feature extractor) --> label / domain head

source IMU 는 참조 좌표계로 보고 R 을 거치지 않는다(apply_r=False). 즉 R 은 target
센서 좌표계를 source 좌표계로 회전시켜 맞추는 역할만 한다. 전처리 단계의 손수
구한 정렬(pca/kabsch/gravity/permutation)을 학습으로 대체하는 셈이라, 데이터는
정렬 안 한 preprocessed_MM_raw(_isotropic) 를 쓰는 것이 자연스럽다.

최고 조합(pca_grav_cdan_joint, Target 72.93%)이 쓰는 손실만 남긴다:
    L_cls     : source 분류 CE                           (학습 루프)
    L_domain  : CDAN 조건부 적대 정렬                       (학습 루프)
    L_gravity = ||R g_t - g_s||²   target 중력방향 정렬(pitch/roll 2 DOF)
    L_pca     = ||R Ft - Fs||²     on-the-fly PCA 정렬(yaw 관측, physics-informed)

R 은 **SO(3) 재매개화(so3_exp, Rodrigues)** 로 파라미터화한다 → so(3) 3-파라미터 w 가
무엇이든 R 은 **항상 proper rotation**(det=+1, RᵀR=I). 회전임이 구조적으로 보장되므로
회전 정칙화 손실(||RᵀR-I||·|det-1|)이 아예 필요 없다(그래서 제거함). det 붕괴·NaN·
MSE shrinkage 도 구조적으로 불가능. gravity/pca/적대 손실은 R 을 회전만 시킨다.
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
    IMUOnlyBackbone, ReverseLayerF, _build_label_classifier)


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
    """target IMU (B,3,T) 의 센서축을 회전시키는 학습 가능한 **회전** R (SO(3) 전용).

    R 을 so(3) 3-파라미터 w 의 행렬지수(so3_exp)로 둔다 → **R 이 항상 proper rotation**
    (det=+1, RᵀR=I). w 가 무엇이든 회전임이 구조적으로 보장되어 det 붕괴·NaN·MSE
    shrinkage 가 불가능하고, 별도 회전 정칙화 손실이 필요 없다. gravity/pca/적대 손실은
    R 을 회전만 시킨다. init=None → w=0 → R=I(항등)에서 출발. init 에 회전행렬을 주면
    so3_log 로 대응 w 로 warm-start.
    """

    def __init__(self, init=None):
        super().__init__()
        w0 = torch.zeros(3) if init is None else so3_log(init)
        self.w = nn.Parameter(w0.float())

    @property
    def R(self):
        return so3_exp(self.w)

    def forward(self, x):  # x: (B,3,T)
        # aligned[b,i,t] = sum_j R[i,j] x[b,j,t]
        return torch.einsum("ij,bjt->bit", self.R, x)


class LearnableRQuat(nn.Module):
    """단위 쿼터니언 4-파라미터로 R 을 파라미터화 (SO(3) 대안, 아이디어 2a).

    q=(w,x,y,z) nn.Parameter, forward 마다 q/‖q‖ 로 정규화 후 회전행렬로 변환 → R 은
    항상 proper rotation(det=+1, RᵀR=I). init q=(1,0,0,0) → R=I 에서 출발. so3 대비
    파라미터가 3→4 개로 늘고 이중피복(q, -q 가 같은 R)·norm 자유도가 생기지만, exp-map
    의 θ→0 특이점(1/θ)이 없어 항등 근처 gradient landscape 가 더 매끄럽다는 가설을
    검증하기 위한 변형이다. 회전임이 구조적으로 보장되므로 rotation reg 손실은 불필요.
    """

    def __init__(self):
        super().__init__()
        self.q = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0]))

    @property
    def R(self):
        q = self.q / (self.q.norm() + 1e-12)
        w, x, y, z = q[0], q[1], q[2], q[3]
        return torch.stack([
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
            2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y),
        ]).reshape(3, 3)

    def forward(self, x):  # x: (B,3,T)
        return torch.einsum("ij,bjt->bit", self.R, x)


class LearnableRMatrix(nn.Module):
    """제약 없는 3x3 행렬을 그대로 학습 (SO(3) 강제는 손실로, 아이디어 2b).

    M=nn.Parameter(eye(3)) 자유행렬, R=M 그대로. proper rotation 이 구조로 보장되지 않아
    별도 rotation_reg_loss(‖MᵀM−I‖²+(det−1)²)를 손실에 더해(--lambda_rot) SO(3) 로
    끌어당긴다. 모든 물리 제약(회전·중력·PCA)을 손실 항으로 통일해 표현할 수 있는 대신,
    λ_rot 가 약하면 det 붕괴·비회전해로 샐 위험이 있다. init=I 에서 출발.
    """

    def __init__(self):
        super().__init__()
        self.M = nn.Parameter(torch.eye(3))

    @property
    def R(self):
        return self.M

    def forward(self, x):  # x: (B,3,T)
        return torch.einsum("ij,bjt->bit", self.M, x)


def build_learnable_r(r_param="so3", init=None):
    """R 파라미터화 선택 팩토리. so3(기본)/quat/matrix."""
    if r_param == "so3":
        return LearnableR(init)
    if r_param == "quat":
        return LearnableRQuat()
    if r_param == "matrix":
        return LearnableRMatrix()
    raise ValueError(f"unknown r_param: {r_param}")


# ----------------------------------------------------------------------
# 기하 손실
# ----------------------------------------------------------------------
def gravity_dir(x):
    """raw IMU 배치 (B,3,T) 에서 배치 평균 중력 방향 벡터 (3,) 를 추정한다.

    IMU 가 중력 포함 raw accel 이라 시간평균 DC 성분이 ≈ 중력. 샘플별 DC 를 구한 뒤
    배치 평균한다. (정규화는 호출부에서 필요 시.)
    """
    return x.mean(dim=2).mean(dim=0)  # (B,3,T) -> (B,3) -> (3,)


def gravity_loss(R, x_src, x_tgt):
    """중력 정렬 손실 L_gravity = ||R g_t - g_s||² 을 반환한다 (제곱형).

    target 중력방향 g_t 를 R 로 돌려 source 중력방향 g_s 에 맞춘다(pitch/roll 2 DOF,
    yaw 는 못 봄 — yaw 는 PCA prior 담당). g 는 배치 평균 중력 방향(raw accel DC).
    R 이 SO(3) 라 회전 정칙화(proper rotation) 손실은 불필요해 제거했다.
    """
    g_src, g_tgt = gravity_dir(x_src), gravity_dir(x_tgt)
    return (R @ g_tgt - g_src).pow(2).sum()


def pca_frame(x, trim_sigma=0.0):
    """배치 평균 모션 공분산의 주축 프레임 (3,3) 을 부호고정해 반환 (physics-informed, 매 배치).

    x: (B,3,T) raw IMU. DC(중력) 제거 후 모션 공분산 → eigh → 고윳값 내림차순 정렬.
    부호 모호성은 모션 투영 왜도(skewness) 부호로 고정([[research-direction-feature-augment]]
    DirectionFeatures 와 동일 규약). 열 = 주축(col0=최대분산). **계산된 R 주입 아님** — 매
    배치 신호에서 주축을 새로 구하는 물리량이라 캘리브레이션이 아니다.

    trim_sigma>0 이면 **주축 계산 전 이상치 윈도를 제거**한다. per-window 모션 에너지
    (=trace(공분산)=총 분산)가 큰 소수 윈도가 합공분산을 지배하면, 수평 두 주축 고윳값이
    근접한 상황에서 축 순서·부호가 배치 구성마다 뒤집힌다(seed 의존 붕괴의 근본원인). 데이터
    기반 median+MAD 상단 이상치 탐지로 이를 완화: 임계 = median + trim_sigma·1.4826·MAD
    (정규분포 가정 시 1.4826·MAD≈σ 라 trim_sigma 는 σ 배수로 읽힌다). 에너지 산포가 거의
    없으면(진짜 이상치 없음) no-op, 2개 미만만 남을 정도로 과도제거되면 폴백해 전부 유지.
    """
    m = x.mean(dim=2, keepdim=True)                   # (B,3,1) DC≈중력
    xc = x - m                                        # 모션 (중력 제거)
    T = xc.size(2)
    cov_b = torch.einsum("bct,bdt->bcd", xc, xc) / T  # (B,3,3) 윈도별 공분산
    if trim_sigma > 0 and cov_b.size(0) >= 4:
        energy = cov_b.diagonal(dim1=1, dim2=2).sum(dim=1)  # (B,) trace=총 모션 분산
        med = energy.median()
        mad = (energy - med).abs().median()
        if mad > 1e-8 * (med.abs() + 1e-8):           # 에너지에 실제 산포가 있을 때만
            thr = med + trim_sigma * 1.4826 * mad
            keep = energy <= thr                      # 상단(고에너지) 이상치만 제거
            if keep.sum() >= 2:                       # 과도제거 방지 폴백
                cov_b = cov_b[keep]
    cov = cov_b.mean(dim=0)                           # (3,3) 배치평균(이상치 제거 후)
    cov = cov + 1e-6 * torch.eye(3, device=x.device, dtype=x.dtype)  # eigh 안정 jitter
    # MPS 는 linalg.eigh 미구현 → 3x3 분해는 CPU 에서(비용 무시 가능), 결과만 원 device 로.
    _, evecs = torch.linalg.eigh(cov.cpu())           # 고윳값 오름차순
    evecs = evecs.to(x.device)
    F = evecs.flip(1)                                 # 내림차순 (col0=최대분산축)
    proj = torch.einsum("bct,ck->bkt", xc, F)         # (B,3,T) 각 주축 투영
    sign = torch.sign((proj ** 3).sum(dim=(0, 2)))    # (3,) 왜도 부호
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    F = F * sign.unsqueeze(0)                         # (3,3) 열별 부호고정
    # proper rotation 보장(det=+1): 부호고정이 핸디드니스를 뒤집으면 최소분산축 반전.
    # source/target 둘 다 det +1 이라야 정렬 R=Fs Ftᵀ 가 회전이 되어 rotation reg 와 안 싸움.
    if torch.det(F.cpu()) < 0:
        F = torch.cat([F[:, :2], -F[:, 2:3]], dim=1)
    return F


def pca_alignment_loss(R, x_src, x_tgt, trim_sigma=0.0):
    """on-the-fly PCA 정렬 손실: target 주축 프레임을 source 프레임에 R 로 맞춤.

    L_pca = ||R Ft - Fs||²_F   (Fs/Ft = 매 배치 계산한 주축, gravity loss 와 동형 규약).
    gravity 가 못 보는 yaw(수평 모션구조)를 결정적으로 잡는다. 부호고정으로 ± 모호성 완화,
    단 수평 모션이 등방적이거나 고윳값 근접 시 yaw under-determined(한계). trim_sigma>0 이면
    source/target 각각 주축 계산 전 고에너지 이상치 윈도를 제거해 배치 민감도를 낮춘다.
    """
    Fs, Ft = pca_frame(x_src, trim_sigma), pca_frame(x_tgt, trim_sigma)
    return ((R @ Ft) - Fs).pow(2).sum()


def class_pca_frames(x, w):
    """클래스별(soft) 주축 프레임 (C,3,3) 과 클래스 질량 (C,) 을 반환한다.

    x: (B,3,T) raw IMU, w: (B,C) 비음수 가중치. source 는 하드 one-hot, target 은 분류기
    softmax 확률(detach)을 넘긴다. 각 클래스의 가중 평균 모션 공분산을 구해 주축을 뽑되,
    부호(왜도)·det(+1) 고정은 pca_frame 과 동일 규약을 클래스별로 적용한다.

    **운동(클래스)별로 프레임을 나눠 계산**하는 게 핵심 — 배치가 여러 운동을 섞어서 집계
    주축이 배치 구성마다 흔들리던 문제(target 라벨 부재 → 운동 혼합)를 클래스 내부에서
    정렬해 없앤다. w 가 균등(초반 pseudo-label 미성숙)이면 모든 클래스 프레임이 집계로
    뭉개져 기존 집계 정렬로 자연 후퇴(self-annealing)한다.
    """
    T = x.size(2)
    m = x.mean(dim=2, keepdim=True)
    xc = x - m                                              # (B,3,T) 모션(중력 제거)
    cov_b = torch.einsum("bit,bjt->bij", xc, xc) / T        # (B,3,3) 윈도별 공분산
    mass = w.sum(dim=0)                                     # (C,) 클래스 질량
    cov_c = torch.einsum("bk,bij->kij", w, cov_b) / mass.clamp_min(1e-6)[:, None, None]
    cov_c = cov_c + 1e-6 * torch.eye(3, device=x.device, dtype=x.dtype)  # (C,3,3) eigh jitter
    _, evecs = torch.linalg.eigh(cov_c.cpu())              # (C,3,3) 고윳값 오름차순
    frames = evecs.flip(-1).to(x.device)                  # 내림차순(열0=최대분산축)
    proj = torch.einsum("bit,kij->bkjt", xc, frames)       # (B,C,3,T) 각 클래스 주축 투영
    sk = torch.einsum("bk,bkjt->kj", w, proj ** 3)         # (C,3) 가중 왜도
    sign = torch.sign(sk)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    frames = frames * sign[:, None, :]                     # (C,3,3) 열별 부호고정
    dets = torch.linalg.det(frames.cpu()).to(x.device)     # (C,) proper rotation 보장
    neg = dets < 0
    if neg.any():
        frames = frames.clone()
        frames[neg, :, 2] = -frames[neg, :, 2]             # det<0 클래스는 최소분산축 반전
    return frames, mass


def class_pca_alignment_loss(R, x_src, y_src, x_tgt, logits_tgt, num_classes,
                             min_mass=2.0, conf=0.0):
    """클래스 조건부(soft) PCA 정렬 손실: Σ_c ‖R·Ft^c − Fs^c‖² / (유효 클래스 수).

    같은 운동끼리만 프레임을 맞춘다 — source 는 진짜 라벨 one-hot, target 은 분류기 softmax
    확률(detach → self-annealing, R 로만 grad). 양쪽 질량이 min_mass 이상인 클래스만 합산해
    배치에 없는 클래스는 건너뛴다. conf>0 이면 max prob<conf 인 target 윈도를 0 가중(하드
    게이팅). gravity 가 pitch/roll 2 DOF 를 잡으므로 이 항은 사실상 yaw 1 DOF 를 운동별로
    정렬하는 역할 — pseudo-label 이 엉터리여도 망칠 수 있는 건 yaw 뿐이라 위험이 국소적이다.
    """
    w_src = F.one_hot(y_src, num_classes).to(x_src.dtype)   # (B,C) 하드 라벨
    p_tgt = F.softmax(logits_tgt.detach(), dim=1)           # (B,C) soft, detach(라벨배정 grad X)
    if conf > 0:
        maxp = p_tgt.max(dim=1, keepdim=True).values
        p_tgt = p_tgt * (maxp >= conf).to(p_tgt.dtype)
    Fs, mass_s = class_pca_frames(x_src, w_src)
    Ft, mass_t = class_pca_frames(x_tgt, p_tgt)
    Fs, Ft = Fs.detach(), Ft.detach()                      # 프레임은 상수 타깃, R 로만 grad
    valid = (mass_s >= min_mass) & (mass_t >= min_mass)     # (C,) 양쪽 다 충분한 클래스만
    n = valid.sum()
    if n == 0:
        return (R * 0).sum()                               # 유효 클래스 없음(grad 0)
    diff = torch.einsum("il,clj->cij", R, Ft) - Fs         # (C,3,3) 클래스별 R·Ft − Fs
    per_c = diff.pow(2).sum(dim=(1, 2))                     # (C,)
    return (per_c * valid.to(per_c.dtype)).sum() / n


def rotation_reg_loss(R):
    """자유행렬 R 을 SO(3) 로 끌어당기는 정칙화: L_rot = ‖RᵀR−I‖²_F + (det R − 1)².

    LearnableRMatrix(제약없는 3x3)용(아이디어 2b). ‖RᵀR−I‖² 가 orthonormal(축이 직교·
    단위)을, (det−1)² 가 반사 아닌 proper rotation·핸디드니스를 강제한다. so3/quat 는
    구조적으로 회전이라 불필요(λ_rot=0).
    """
    I = torch.eye(3, device=R.device, dtype=R.dtype)
    ortho = (R.transpose(0, 1) @ R - I).pow(2).sum()
    det = (torch.linalg.det(R) - 1.0).pow(2)
    return ortho + det


def post_r_normalize(x, mode, bn=None):
    """R 직후 인코더 입력 정규화 (회전→정규화 순서, 정규화 딜레마 분리책).

    gravity/pca 손실은 학습 루프에서 raw isotropic 배치로 계산하므로 R 은 중력 살아있는
    pre-norm 데이터를 본다(식별성·gravity 유지). 여기 정규화는 R 뒤 공통(source) 프레임에서
    source·target 동일하게 걸어 인코더가 축별 표준화 입력을 받게 한다.
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

    def __init__(self, num_classes=10, feature_dim=512, R_init=None, post_r_norm="none",
                 r_param="so3"):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.post_r_norm = post_r_norm   # "none" | "instance" | "batchnorm"
        self.r_param = r_param            # "so3"(기본) | "quat" | "matrix"

        self.r = build_learnable_r(r_param, R_init)
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

    L_grav = gravity_loss(model.r.R, src_imu, tgt_imu)
    L_pca = pca_alignment_loss(model.r.R, src_imu, tgt_imu)
    print(f"  gravity @ R=I -> {L_grav:.4f}   pca @ R=I -> {L_pca:.4f}")
    # SO(3) 보장 확인: 임의 w 에서도 det=+1·RᵀR=I
    model.r.w.data = torch.tensor([1.3, -0.7, 2.1])
    R = model.r.R.detach()
    print(f"  SO(3) 확인(w≠0): det(R)={torch.linalg.det(R):.6f}  "
          f"||RᵀR-I||={torch.linalg.norm(R.T @ R - torch.eye(3)):.2e}  (항상 회전)")

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  LearnableR-CDAN trainable params: {total:,}  (R adds {model.r.R.numel()})")
