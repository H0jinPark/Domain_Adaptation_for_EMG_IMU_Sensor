"""PCA 정렬 이상치 제거 실험 — 주축 계산 전 고에너지 이상치 윈도를 걷어낸다 (SO(3) 유지).

문제 진단: 최고 조합이 seed42 에선 Target 77% 로 잘 가는데 다른 seed 는 붕괴한다(66.6±5.4%,
seed2 58%). 근본원인은 `pca_frame` 이 매 배치 raw 모션 공분산을 eigh 하는 데 있다 —
소수의 **고에너지(대진폭) 윈도가 합공분산을 지배**하면, 수평 두 주축의 고윳값이 근접한
상황에서 축 순서·부호가 **배치 구성마다 뒤집힌다.** 그 배치에 이상치 윈도가 들어갔냐/
아니냐로 R 초기 방향이 갈리고, R 이 한 번 반대로 갔다 돌아오는 사이 인코더가 잘못된 정렬에
먼저 적응해 나쁜 국소해에 갇힌다. "처음부터 옳은 방향으로" 가게 하려면 주축 추정 자체를
배치에 덜 민감하게 만들어야 한다.

처방 = **PCA 주축 계산 전 이상치 윈도 제거**(`--pca_trim`). per-window 모션 에너지
(=trace(공분산)=총 분산)에 대해 데이터 기반 median+MAD 상단 이상치 탐지:

    임계 = median(energy) + pca_trim · 1.4826 · MAD(energy)

(정규분포 가정 시 1.4826·MAD≈σ 라 pca_trim 은 σ 배수로 읽힌다). 임계 초과(고에너지)
윈도를 빼고 나머지로 공분산을 평균한다. 에너지 산포가 거의 없으면(진짜 이상치 없음)
no-op, 2개 미만만 남을 정도로 과도제거되면 폴백해 전부 유지한다. source/target 각각
독립적으로 적용. 로직은 learnable_r_model.pca_frame(x, trim_sigma) 안에 있다.

이 실험은 원본 최고 레시피(SO(3) + gravity + pca + CDAN JOINT)를 그대로 두고 **PCA
이상치 제거만** 켠다 — 순수하게 trim 효과만 격리해 base(trim off) 대비 재현성을 본다.

실행 예:
    # base 대비 (trim off)
    MM_DATA_DIR=preprocessed_MM_raw_isotropic python Learnable_R/learnable_r_cdan_pcatrim_train.py \
        --multi_seed --epochs 60 --pca_trim 0 --tag base
    # PCA 이상치 제거 켬
    MM_DATA_DIR=preprocessed_MM_raw_isotropic python Learnable_R/learnable_r_cdan_pcatrim_train.py \
        --multi_seed --epochs 60 --pca_trim 3.0 --tag trim3
"""
import argparse

from learnable_r_cdan_common import add_common_args, run_and_summarize

NAME = "learnable_r_cdan_pcatrim"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--pca_trim", type=float, default=3.0,
                        help="PCA 주축 계산 전 고에너지 이상치 윈도 제거 임계"
                             "(median+σ·MAD 의 σ 배수). 0=끔. 배치 민감도↓ 목적. 권장 3.0~3.5")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_and_summarize(
        args, name=NAME, r_param="so3",
        method_label="Learnable-R CDAN (IMU-only) | PCA outlier trim",
        extra_kw=dict(lambda_rot=0.0, pca_trim=args.pca_trim))
