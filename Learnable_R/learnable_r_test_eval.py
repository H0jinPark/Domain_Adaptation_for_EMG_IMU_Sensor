"""learnable-R 계열 학습 스크립트의 공통 test 평가 (7:1.5:1.5 규약).

learnable_r_*_train.py 는 실험별로 서로 독립된 복제본이지만, **test 평가만은 규약**이라
네 곳에 복사하지 않고 여기 한 곳에 둔다. 그래서 이 모듈이 test 로더를 만드는 유일한
지점이고, 호출 시점도 하나뿐이다 — 학습이 끝나 best checkpoint 를 복원한 직후 1회.
학습 루프나 model selection 이 test 를 볼 수 있는 경로가 아예 없다.

  model selection = target val (oracle — target 라벨을 쓴다. 논문에 명시할 것)
  보고 수치        = target TEST

evaluate_r 은 호출하는 쪽이 넘긴다. 네 스크립트 모두 시그니처가
(model, loader, device, apply_r) -> (acc, preds, labels) 로 같아서 그대로 꽂힌다.
source 는 apply_r=False(원 좌표계), target 은 apply_r=True(학습된 R 적용)라는
규약도 학습 때와 동일하게 유지한다.
"""
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "Multimodal"))
from mm_data_loader import get_mm_test_loaders  # noqa: E402

# 결과 JSON 에 함께 적어 두는 규약 메타 — 나중에 표를 만들 때 이 두 필드로
# 옛 2분할 결과와 구분한다.
SELECTION = "target_val (oracle)"
REPORTED_METRIC = "target_test_acc"

# 집계 대상 지표(있는 것만 쓴다). val 3개 + test 3개.
METRIC_KEYS = ("source_acc", "target_acc", "shift",
               "source_test_acc", "target_test_acc", "test_shift")


def evaluate_test(model, evaluate_r, le, device, seed=None, batch_size=64):
    """best checkpoint 를 복원한 모델로 test 를 1회 평가한다.

    반환 (metrics, cms):
      metrics : {"source_test_acc", "target_test_acc", "test_shift"} — 결과 dict 에 그대로 병합
      cms     : {"source": (true, preds), "target": (true, preds)} — 혼동 행렬용
    두 개로 나눈 것은 (true, preds) 배열이 결과 JSON 에 섞여 들어가지 않게 하기 위함이다.
    """
    test_loader, tgt_test_loader = get_mm_test_loaders(le, batch_size=batch_size)
    src_acc, s_preds, s_true = evaluate_r(model, test_loader, device, apply_r=False)
    tgt_acc, t_preds, t_true = evaluate_r(model, tgt_test_loader, device, apply_r=True)

    who = f"seed={seed} | " if seed is not None else ""
    print(f"[test] {who}Source Test: {src_acc:.2f}% | Target Test: {tgt_acc:.2f}% | "
          f"Shift: {src_acc - tgt_acc:.2f}%   <-- 보고 수치")

    metrics = {"source_test_acc": src_acc, "target_test_acc": tgt_acc,
               "test_shift": src_acc - tgt_acc}
    cms = {"source": (s_true, s_preds), "target": (t_true, t_preds)}
    return metrics, cms


def summarize_metrics(results):
    """seed 별 결과 리스트 -> (mean, std) dict. results 에 실제로 있는 지표만 집계한다.

    seed 가 1개면 표본표준편차가 정의되지 않으므로 ddof=0 (=0.0) 으로 떨어뜨린다.
    """
    ddof = 1 if len(results) > 1 else 0
    cols = {k: [r[k] for r in results] for k in METRIC_KEYS if k in results[0]}
    return ({k: float(np.mean(v)) for k, v in cols.items()},
            {k: float(np.std(v, ddof=ddof)) for k, v in cols.items()})
