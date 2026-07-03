"""클래스 조건부(soft) PCA 정렬 실험 — 같은 운동끼리 프레임을 맞춘다 (SO(3) 유지).

문제 진단(유저 통찰, 2026-07-03): seed 의존 붕괴의 근본원인은 **target 라벨이 없어서
배치가 여러 운동을 섞고, 운동마다 모션 방향이 다르기 때문.** 기존 aggregate `pca_frame`
은 배치 전체를 뭉쳐 주축을 뽑으니, "그 배치에 어떤 운동이 몇 개 들었냐"에 따라 축이
흔들린다. 게다가 source/target 배치는 독립적으로 셔플돼 **운동 혼합비가 서로 다른 두
프레임을 억지로 맞추는** 꼴이라, R 이 맞출 타깃 자체가 매 스텝·seed 마다 요동친다.

처방 = **클래스 조건부 정렬**. 프레임을 운동 무관 집계로 뽑지 말고 운동별로 나눠서
같은 운동끼리 맞춘다:

    L_cpca = Σ_c ‖R · Ft^c − Fs^c‖²  / (유효 클래스 수)

  · source(Fs^c): 진짜 라벨 one-hot per-class 프레임 (깨끗)
  · target(Ft^c): 분류기 softmax 확률로 **가중**한 per-class 프레임 (detach → R 로만 grad)

**초기 pseudo-label 문제를 soft 가중이 스스로 푼다(self-annealing).** 하드로 라벨을 "고르지"
않는다 — 확률로 가중하므로, 초반 분류기가 엉망이면 p 가 균등 → 모든 Ft^c 가 집계로 뭉개져
**기존 집계 정렬로 자연 후퇴**(안전, 파국 없음). 분류기가 샤프해지면 진짜 운동별 정렬로
매끄럽게 전환. 스모크로 diffuse 라벨 시 class-pca 손실이 aggregate 손실과 정확히 일치함을
확인했다. downside 유계: 분류기가 끝까지 target 을 못 잡아도 집계로 수렴 → 지금보다 나빠지지
않는다.

안전장치: (1) gravity(lambda_g)가 pitch/roll 2 DOF 를 라벨·운동 무관하게 잡으므로, 이 항이
사실상 **yaw 1 DOF** 만 운동별로 정렬 → pseudo-label 이 틀려도 망칠 수 있는 게 yaw 뿐(위험
국소화). (2) `--cpca_ramp` 로 λ_cpca 를 초반 0→1 선형 ramp(pseudo 성숙 사이 서서히 키움).

이 실험은 aggregate PCA(`--lambda_pca`)를 끄고(권장 0) 클래스 조건부만 켠다. gravity+CDAN
JOINT+SO(3) 등 나머지 최고 레시피는 그대로. dadelay/pcatrim 등 다른 변형과 무관.

caveat: 운동 하나가 단일 축으로만 움직이면(수평 등방) 그 클래스 프레임의 부축이 부실 →
yaw under-determined 는 여전히 남을 수 있음(gravity 2 DOF 는 무관하게 유지). 운동별
λ2/λ1 진단으로 어느 운동이 이런지 볼 수 있음.

실행 예:
    # 클래스 조건부 PCA (aggregate 끄고, yaw 항 10ep ramp)
    MM_DATA_DIR=preprocessed_MM_raw_isotropic python Learnable_R/learnable_r_cdan_classpca_train.py \
        --multi_seed --epochs 60 --lambda_pca 0 --lambda_cpca 1 --cpca_ramp 10 --tag classpca
"""
import argparse

from learnable_r_cdan_common import add_common_args, run_and_summarize

NAME = "learnable_r_cdan_classpca"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--lambda_cpca", type=float, default=1.0,
                        help="클래스 조건부(soft) PCA 정렬 손실 가중치. 같은 운동끼리 프레임 맞춤")
    parser.add_argument("--cpca_ramp", type=int, default=10,
                        help="λ_cpca 를 초반 이 epoch 동안 0→1 선형 ramp. 0=끔. pseudo-label 성숙 사이 완만히 키움")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_and_summarize(
        args, name=NAME, r_param="so3",
        method_label="Learnable-R CDAN (IMU-only) | class-conditional PCA",
        extra_kw=dict(lambda_rot=0.0, lambda_cpca=args.lambda_cpca, cpca_ramp=args.cpca_ramp))
