"""아이디어 2b — 제약 없는 3x3 행렬을 그대로 학습, SO(3) 는 손실로 강제.

문제 진단은 동일(비-42 seed 에서 R 이 늦게 수렴). 여기선 R 을 어떤 재매개화도 없이
자유 3x3 행렬 M(init=I)로 두고, proper rotation 제약을 **손실 항**으로 건다:

    L_rot = ‖MᵀM − I‖²_F + (det M − 1)²        (--lambda_rot)

동기: so3_exp/쿼터니언처럼 회전을 구조로 강제하면 파라미터가 곡면(manifold) 위에 갇혀
초반 방향전환이 굳는다는 가설. 자유행렬은 9개 성분이 유클리드 공간에서 자유롭게 움직여
초반 재정렬이 쉽고, det=1·orthogonal 을 손실로 부드럽게 당기면 **모든 물리 prior(회전·
중력·PCA)를 손실로 통일** 표현할 수 있다. 대가로 λ_rot 가 약하면 det 붕괴·비회전해로
샐 수 있어 가중치 튜닝이 필요하다(그래서 --lambda_rot 를 노출).

so3/quat 변형과 손실·데이터·CDAN 규약은 동일하고, R 표현(자유행렬)과 λ_rot 만 다르다.

실행 예:
    MM_DATA_DIR=preprocessed_MM_raw_isotropic python Learnable_R/learnable_r_cdan_matrix_train.py \
        --multi_seed --epochs 60 --lambda_rot 1.0 --tag matrix
"""
import argparse

from learnable_r_cdan_common import add_common_args, run_and_summarize

NAME = "learnable_r_cdan_matrix"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--lambda_rot", type=float, default=1.0,
                        help="rotation 정칙화 손실 ‖MᵀM−I‖²+(det−1)² 가중치. 자유행렬을 SO(3) 로 당김")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_and_summarize(
        args, name=NAME, r_param="matrix",
        method_label="Learnable-R CDAN (IMU-only) | idea2b free-matrix + rot-reg",
        extra_kw=dict(lambda_rot=args.lambda_rot))
