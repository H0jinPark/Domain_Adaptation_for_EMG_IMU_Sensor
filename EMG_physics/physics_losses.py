"""Physics-informed 보조 손실: IMU 재구성 + jerk 최소화.

두 손실 모두 **짝꿍 IMU 만 있으면 되고 라벨이 필요 없다.** 그래서 source 뿐 아니라
target 배치에도 그대로 걸 수 있다 — UDA 관점의 이득이 여기서 나온다.

단위에 대한 주의
  IMU 는 전처리에서 세션별·축별 z-score 를 거쳤다(preprocessed_MM_pca). 따라서 아래
  모든 값의 단위는 "정규화 가속도"이고 m/s² 가 아니다. 축별로 나눈 표준편차가 세션마다
  달라서, 물리량으로서의 jerk 크기를 세션 간에 직접 비교할 수는 없다.
  (회전 구조·축간 크기비까지 보존하려면 imu_norm=isotropic 폴더를 써야 한다.
   MM_DATA_DIR 만 바꾸면 되므로 나중에 갈아끼울 수 있다.)

jerk 정의
  물리적 jerk 는 가속도의 시간미분 da/dt 다. 이산 신호에서는
      jerk_t = (a_{t+1} - a_t) * fs          [정규화단위 / s]
  이고, 최적화하는 손실은 fs 를 접어 넣지 않은
      L_jerk = mean( (a_{t+1} - a_t)^2 ) = mean(jerk^2) / fs^2
  를 쓴다. fs=100 이라 fs^2=1e4 배 차이이므로, 이 상수를 손실에 넣으면 λ_jerk 를
  1e-6 같은 값으로 써야 해서 감이 안 온다. 물리적 크기는 jerk_rms() 로 따로 보고한다.

  이 항은 예측에만 건다(정답과의 차이가 아니다). minimum-jerk 궤적 prior
  (Flash & Hogan 1985) 에서 온 정규화이고, 위 2026 논문도 "예측 가속도의 1차 미분을
  벌점"으로 같은 형태를 쓴다.
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
    }
