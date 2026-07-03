"""아이디어 1 — 인코더/판별기 LR 워밍업으로 R 이 먼저 자리잡게 (SO(3) 유지).

문제 진단(멀티시드): R 에는 사실상 정답 방향이 있고 모든 seed 가 결국 거기로 가지만,
비-42 seed 에선 R 이 정해지는 속도가 인코더·판별기보다 느리다. R 이 한 번 반대 방향으로
갔다가 돌아오는 사이, 인코더/판별기가 그 잘못된 정렬에 먼저 적응해버려 나쁜 국소해에
갇힌다(→ seed2 58% 붕괴 등, target 66.6±5.4%).

해결 = R 에게 선두를 준다. 인코더/판별기(optimizer group 0)의 lr 을 초반
`--enc_warmup_epochs` 동안 `--enc_lr_floor`→1.0 으로 선형 워밍업하고, R lr(group 1)은
처음부터 풀강도로 둔다. 초반에는 R 이 물리 prior(PCA/gravity)로 빠르게 정답 방향을
잡고, 인코더/판별기는 그 뒤에 이미 옳게 정렬된 신호에 적응한다. R 파라미터화는 원본과
동일한 SO(3)(so3_exp) 그대로라, 순수하게 "학습 타이밍" 효과만 격리해서 본다.

참고: 인코더/R 의 상대 속도 자체는 원래 --lr/--r_lr 로도 조절되지만, 이 파일이 더하는
것은 "초반에만 인코더를 늦췄다 복귀"시키는 **타이밍(워밍업)** 이다.

실행 예:
    MM_DATA_DIR=preprocessed_MM_raw_isotropic python Learnable_R/learnable_r_cdan_lrsched_train.py \
        --multi_seed --epochs 60 --enc_warmup_epochs 10 --enc_lr_floor 0.1 --tag warmup10
"""
import argparse

from learnable_r_cdan_common import add_common_args, run_and_summarize

NAME = "learnable_r_cdan_lrsched"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--enc_warmup_epochs", type=int, default=10,
                        help="인코더/판별기 lr 을 floor→1.0 선형 워밍업할 초반 epoch 수. 0=끔")
    parser.add_argument("--enc_lr_floor", type=float, default=0.1,
                        help="워밍업 시작 시 인코더/판별기 lr 배수(0~1). R lr 은 항상 풀강도")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_and_summarize(
        args, name=NAME, r_param="so3",
        method_label="Learnable-R CDAN (IMU-only) | idea1 enc-LR-warmup",
        extra_kw=dict(lambda_rot=0.0,
                      enc_warmup_epochs=args.enc_warmup_epochs,
                      enc_lr_floor=args.enc_lr_floor))
