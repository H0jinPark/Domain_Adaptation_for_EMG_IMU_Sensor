"""Physics-informed 보조 손실: IMU 재구성 + jerk + **물리량 정합(orient / axes)**.

전부 **짝꿍 IMU 만 있으면 되고 라벨이 필요 없다.** 그래서 source 뿐 아니라 target
배치에도 그대로 걸 수 있다 — UDA 관점의 이득이 여기서 나온다.

네 손실의 성격
  rec    파형을 점대점으로 맞춘다. 윈도우당 3x500=1500 개 숫자.
  jerk   예측의 매끄러움 prior (정답을 안 본다).
  orient 중력 방향 (윈도우당 3 dof) — 센서가 중력에 대해 어떻게 놓였는가.
  axes   움직임 공분산 (윈도우당 6 dof) — 주축 방향과 축별 세기.

  orient/axes 는 참 IMU 의 함수이므로 **rec 에 없던 정보가 아니다.** 그럼에도 따로 두는
  이유는 차원이 압도적으로 낮기 때문이다. 2026-08-04 스윕 재분석에서 λ 를 고정하면
  r(재구성 충실도, target acc) = -0.50 (n=40, p=0.006) 으로 **파형을 잘 맞출수록 전이가
  나빴다.** 충실한 파형 복원은 source 특유의 EMG 지문까지 끌어와야 가능하기 때문으로 읽힌다.
  방향 3~6 개 숫자에는 기기 지문을 숨길 자리가 없다. 즉 정보를 더 주는 게 아니라
  **물리적으로 의미 있는 저차원으로 제한**하는 것이 이 두 손실의 목적이다.

  덤으로 이 두 양은 라벨 없이 target IMU 에서 계산되면서 클래스와 강하게 상관된다
  (isotropic 실측: 클래스 쌍 사이 중력방향 차이 중앙값 94.2°, 같은 클래스의 도메인간
  차이 32.9°, 주축은 11.5°). target 도메인에서 EMG 표현에 준-지도신호로 작동할 여지가 있다.

단위와 데이터 전제
  값의 단위는 "정규화 가속도"이고 m/s² 가 아니다.
  **orient/axes 는 imu_norm=isotropic 데이터에서만 의미가 있다.** 축별 z-score 는 세션
  평균을 빼 중력 DC 를 지우고 축마다 다른 배율로 늘려 고유방향을 뒤튼다 — 회전이 아니다.
  실측(윈도우 평균가속도 크기 중앙값): preprocessed_MM_pca 0.437 vs _pca_isotropic 0.932.
  학습 스크립트가 첫 배치에서 gravity_magnitude() 로 이걸 확인하고 미달이면 중단한다.

jerk 정의
  물리적 jerk 는 가속도의 시간미분 da/dt 다. 이산 신호에서는
      jerk_t = (a_{t+1} - a_t) * fs          [정규화단위 / s]
  이고, 최적화하는 손실은 fs 를 접어 넣지 않은
      L_jerk = mean( (a_{t+1} - a_t)^2 ) = mean(jerk^2) / fs^2
  를 쓴다. fs=100 이라 fs^2=1e4 배 차이이므로, 이 상수를 손실에 넣으면 λ_jerk 를
  1e-6 같은 값으로 써야 해서 감이 안 온다. 물리적 크기는 jerk_rms() 로 따로 보고한다.

  이 항은 예측에만 건다(정답과의 차이가 아니다). minimum-jerk 궤적 prior
  (Flash & Hogan 1985) 에서 온 정규화이다.

  참고 논문(Iyer & Jeong 2026)과 같지 않다 — 원문 확인 결과 (2026-08-06)
    그쪽 식 (Eq.6):  L_physics = MSE(Δ²â, 0),  Δ²â = â_{t+1} − 2â_t + â_{t−1}
    즉 **2차 차분**이고, 우리는 **1차 차분** 이다. 연산자가 다르다.
    게다가 그쪽 IMU head 출력은 윈도우별 채널 **평균** 하나(R^{B×D})라 시간축이
    없고, 그래서 차분을 "across the batch dimension" 으로 뜬다(§3.3). 배치 순서는
    셔플된 임의 순서다. 우리는 파형 (B,3,T) 을 내고 진짜 시간축으로 차분한다.
    이름만 같을 뿐 물리적 의미가 다른 항이므로 "we follow" 라고 쓰면 안 된다.
    (논문 본문은 jerk 를 "가속도의 시간 미분"=1차 로 정의해 놓고 식은 2차를 쓴다.
     정의에 맞는 쪽은 오히려 우리다.)
"""
import torch
import torch.nn.functional as F

IMU_FS = 100.0   # 전처리 규약 (data_preprocess_MM.IMU_FS)


def imu_reconstruction_loss(pred, true, kind="huber", delta=1.0):
    """복원 IMU 와 실제 IMU 의 재구성 손실. pred/true: (B,3,T)

    기본 huber 인 이유 — z-score 후에도 운동 구간엔 큰 피크가 남는다. 순수 MSE 는
    그 소수 피크에 손실이 지배되어 디코더가 평균만 맞추려 든다. huber 는 큰 오차의
    기울기를 선형으로 눌러 그 쏠림을 줄인다.
    """
    if kind == "huber":
        return F.huber_loss(pred, true, delta=delta)
    if kind == "mse":
        return F.mse_loss(pred, true)
    if kind == "l1":
        return F.l1_loss(pred, true)
    raise ValueError(f"알 수 없는 recon_loss: {kind!r} (huber|mse|l1 중 하나)")


def imu_jerk_loss(pred):
    """예측 IMU 의 jerk 크기 벌점.  pred: (B,3,T) -> scalar

    L = mean( (a_{t+1} - a_t)^2 ).  fs 는 곱하지 않는다(위 docstring 참고).
    """
    d = pred[..., 1:] - pred[..., :-1]
    return (d ** 2).mean()


def _time_mean(x):
    """윈도우별 시간평균 가속도. (B,3,T) -> (B,3).  isotropic 정규화에서 이것이 중력이다."""
    return x.mean(dim=-1)


def _time_cov(x):
    """윈도우별 3x3 공분산(시간축). (B,3,T) -> (B,3,3).

    평균(중력)을 뺀 뒤 계산하므로 순수하게 **움직임** 성분의 2차 모멘트다.
    고유벡터(주축)와 고유값(축별 세기)을 한꺼번에 담고 있으면서, 고유분해와 달리
    부호 모호도 축퇴(λ1≈λ2)도 없다 — 그래서 손실은 공분산으로 걸고 각도는 진단으로만 본다.
    """
    xc = x - x.mean(dim=-1, keepdim=True)
    return torch.einsum("bit,bjt->bij", xc, xc) / xc.size(-1)


def imu_orientation_loss(pred, true, eps=1e-8):
    """복원 IMU 의 **중력 방향**을 참 IMU 와 맞춘다.  pred/true: (B,3,T) -> scalar

    L = mean( 1 - cos(mean_t(pred), mean_t(true)) ),  범위 [0, 2]

    크기가 아니라 방향만 본다. isotropic 정규화에서 정지 성분의 크기는 어차피 ≈1g 로
    고정이고, 우리가 원하는 물리량은 "센서가 중력에 대해 어떻게 놓였는가"이기 때문이다.

    **z-score 데이터에서는 의미가 없다.** 축별 z-score 는 세션 평균을 빼버려서 중력 DC 가
    사라지고(실측: 윈도우 평균가속도 크기 중앙값 0.437 vs isotropic 0.932), 축마다 다른
    배율로 늘리므로 방향 자체가 뒤틀린다. 반드시 --imu_norm isotropic 데이터에서 쓸 것.
    """
    mp, mt = _time_mean(pred), _time_mean(true)
    cos = F.cosine_similarity(mp, mt, dim=1, eps=eps)
    return (1.0 - cos).mean()


def imu_axes_loss(pred, true, eps=1e-8):
    """복원 IMU 의 **주축 구조**(움직임 공분산)를 참 IMU 와 맞춘다. (B,3,T) -> scalar

    L = mean( ||Cov(pred) - Cov(true)||_F^2 / ||Cov(true)||_F^2 )

    참 IMU 의 크기로 나눠 정규화하므로 윈도우마다 움직임 세기가 달라도 손실이 큰 윈도우에
    지배되지 않는다. 고유벡터를 직접 비교하지 않는 이유는 (1) v 와 -v 가 같아 부호가
    모호하고 (2) λ1≈λ2 인 윈도우에서 주축이 불안정하기 때문이다 (isotropic 실측으로도
    λ1/λ2<1.5 인 윈도우가 2.5% 있다). 공분산 매칭은 그 두 문제를 다 피하면서 고유벡터와
    고유값을 모두 포함한다.
    """
    cp, ct = _time_cov(pred), _time_cov(true)
    num = ((cp - ct) ** 2).sum(dim=(1, 2))
    den = (ct ** 2).sum(dim=(1, 2)) + eps
    return (num / den).mean()


@torch.no_grad()
def orientation_angle_deg(pred, true, eps=1e-8):
    """중력 방향 오차(도) — 손실이 아니라 사람이 읽는 진단값."""
    cos = F.cosine_similarity(_time_mean(pred), _time_mean(true), dim=1, eps=eps)
    return float(torch.rad2deg(torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))).mean().item())


@torch.no_grad()
def principal_axis_angle_deg(pred, true, eps=1e-8):
    """제1주축 사이 각도(도).  부호 모호를 |cos| 로 접어 [0, 90] 범위로 본다.

    축퇴 윈도우(λ1≈λ2)에서는 주축 자체가 정의가 흐리므로 이 값도 흔들린다. 그래서
    진단 전용이고 손실로는 쓰지 않는다.
    """
    def top_vec(c):
        # eigh 는 오름차순 고유값 -> 마지막 열이 제1주축
        return torch.linalg.eigh(c.double())[1][..., -1]
    vp, vt = top_vec(_time_cov(pred)), top_vec(_time_cov(true))
    cos = (vp * vt).sum(dim=1).abs().clamp(0, 1 - 1e-7)
    return float(torch.rad2deg(torch.acos(cos)).mean().item())


@torch.no_grad()
def gravity_magnitude(x):
    """윈도우 평균가속도 크기의 중앙값 — 데이터가 isotropic 인지 확인하는 실측값.

    isotropic(단일 스칼라 정규화)이면 ≈0.93, 축별 z-score 면 ≈0.44 로 확연히 갈린다.
    """
    return float(_time_mean(x).norm(dim=1).median().item())


@torch.no_grad()
def jerk_rms(x, fs=IMU_FS):
    """물리 단위(정규화가속도/초)의 jerk RMS — 로깅·진단용. x: (B,3,T) -> float"""
    d = (x[..., 1:] - x[..., :-1]) * fs
    return float(torch.sqrt((d ** 2).mean()).item())


@torch.no_grad()
def recon_corr(pred, true, eps=1e-8):
    """축별 Pearson 상관을 시간축으로 계산해 평균. pred/true: (B,3,T) -> float

    재구성 품질을 스케일 무관하게 보는 지표다. huber/MSE 값만으로는 "그냥 0 근처를
    예측해서 손실이 낮은" 경우와 "파형을 실제로 맞춘" 경우가 구분되지 않는다.
    상관이 0 근처면 디코더가 아무것도 못 배운 것이다.
    """
    p = pred - pred.mean(dim=-1, keepdim=True)
    t = true - true.mean(dim=-1, keepdim=True)
    num = (p * t).mean(dim=-1)
    den = p.std(dim=-1, unbiased=False) * t.std(dim=-1, unbiased=False)
    return float((num / (den + eps)).mean().item())


@torch.no_grad()
def physics_diagnostics(pred, true, fs=IMU_FS):
    """한 배치의 진단 지표 묶음.

    jerk_ratio 가 핵심이다:
      ≈1  예측이 정답과 비슷한 매끄러움
      <<1 과도한 평활화 — λ_jerk 가 너무 커서 디코더가 직선을 뱉고 있다
      >>1 예측이 정답보다 더 튄다 — jerk 항이 사실상 안 걸린 것
    """
    jp, jt = jerk_rms(pred, fs), jerk_rms(true, fs)
    return {
        "recon_corr": recon_corr(pred, true),
        "jerk_rms_pred": jp,
        "jerk_rms_true": jt,
        "jerk_ratio": jp / (jt + 1e-12),
        # 물리량 정합 — 도 단위라 논문 표에 그대로 쓸 수 있다.
        # 무작위 방향 두 개 사이 각도의 기댓값이 90도이므로 그게 "아무것도 못 배움" 기준선이다.
        "orient_deg": orientation_angle_deg(pred, true),
        "axis_deg": principal_axis_angle_deg(pred, true),
    }
