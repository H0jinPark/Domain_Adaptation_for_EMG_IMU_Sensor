"""아이디어 2a — 단위 쿼터니언(4-파라미터)으로 R 을 파라미터화.

문제 진단은 lrsched 파일과 동일(R 이 정답 방향으로 가는 게 인코더보다 느려 비-42 seed
가 나쁜 국소해에 갇힘). 여기선 R 의 **표현(파라미터화)** 을 바꿔 학습 궤적을 개선한다.

원본은 so(3) 3-파라미터의 행렬지수(so3_exp, Rodrigues)를 쓴다. 이 맵은 회전각 θ→0
근처에서 sin θ/θ·(1−cos θ)/θ² 같은 항 때문에 항등 부근 gradient 가 왜곡·정체되기
쉽다(모든 R 이 항등에서 출발하므로 초반이 정확히 그 영역). 대신 **단위 쿼터니언**
q=(w,x,y,z)/‖q‖ 로 두면:
  · 파라미터가 3→4 개로 늘어 최적화 여유가 생기고(가설: 초반 방향전환이 쉬움),
  · 항등(q=(1,0,0,0)) 부근에 exp-map 같은 특이점이 없어 landscape 가 매끄럽다.
회전임은 정규화로 구조적으로 보장(det=+1, RᵀR=I)되어 rotation reg 손실은 여전히 불필요
(λ_rot=0). 손실·데이터·CDAN 규약은 원본과 완전히 동일하고 R 표현만 다르다.

실행 예:
    MM_DATA_DIR=preprocessed_MM_raw_isotropic python Learnable_R/learnable_r_cdan_quat_train.py \
        --multi_seed --epochs 60 --tag quat
"""
import argparse

from learnable_r_cdan_common import add_common_args, run_and_summarize

NAME = "learnable_r_cdan_quat"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_and_summarize(
        args, name=NAME, r_param="quat",
        method_label="Learnable-R CDAN (IMU-only) | idea2a quaternion",
        extra_kw=dict(lambda_rot=0.0))
