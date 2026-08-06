"""align-first 학습의 LR 스케줄 — join 시점과 cosine 위상을 분리한다.

문제 (2026-07-30 확인)
  기존 코드는 `CosineAnnealingLR(optimizer, T_max=epochs)` 를 phase 와 무관하게
  매 epoch step 했다. 그래서 CDAN 이 켜지는 순간의 LR 이 target_join 마다 달랐다:

      join   CDAN 켜질 때 LR(base 대비)   합류 후 남은 LR 적분(전체 대비)
        0            100.0%                        100.0%
       20             75.0%                         39.7%
       30             50.0%                         18.7%
       40             25.0%                          6.1%
       50              6.7%                          0.9%
       60              0.0%                          0.0%

  즉 join 스윕의 단조 하락은 "적응 epoch 수"와 "적응 구간 LR"이 뒤엉킨 결과라
  전자만의 효과로 해석할 수 없었다.

해결
  mode="phase" 는 align 구간 [0, join) 과 적응 구간 [join, epochs) 에 각각
  독립적인 cosine 을 씌운다. 두 구간 모두 base LR 에서 시작하므로, 어떤 join 이든
  **적응이 항상 같은 LR 프로파일로 시작**한다. 남는 차이는 적응 구간의 길이뿐이다.

  mode="global" 은 기존 동작(단일 cosine)이며 **기본값**이다. 과거 결과의
  재현성을 깨지 않기 위해 유지한다. `__main__` 의 자체 검증이 PyTorch
  CosineAnnealingLR 과 수치적으로 일치함을 확인한다.

경계 조건 — join=0 이나 join>=epochs 면 구간이 하나뿐이라 phase 와 global 이
동일하다. 그래서 이 스윕에서 join0 / join60 은 두 모드에서 같은 결과가 나오고,
달라지는 건 교락이 실제로 있었던 중간 join 들뿐이다.
"""
import math

MODES = ("global", "phase")


def lr_factor(epoch, epochs, mode="global", join=0):
    """해당 epoch 에서 base LR 에 곱할 비율 (0~1).

    epoch 은 0-indexed 이며 **그 epoch 을 시작하기 전에** 부르는 것을 전제로 한다
    (기존 코드가 epoch 종료 후 scheduler.step() 하던 것과 위상이 같다).
    """
    if mode not in MODES:
        raise ValueError(f"알 수 없는 lr_schedule: {mode!r} ({'|'.join(MODES)} 중 하나)")

    if mode == "global" or join <= 0 or join >= epochs:
        t, span = epoch, epochs
    elif epoch < join:
        t, span = epoch, join                      # align 구간
    else:
        t, span = epoch - join, epochs - join      # 적응 구간 (base LR 에서 재시작)

    return 0.5 * (1.0 + math.cos(math.pi * t / max(1, span)))


def apply_lr(optimizer, base_lrs, factor):
    """param_group 별 base LR 에 factor 를 곱해 넣는다. 갱신된 LR 리스트를 반환."""
    for group, base in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base * factor
    return [g["lr"] for g in optimizer.param_groups]


def schedule_note(epochs, mode, join):
    """로그·결과 JSON 에 남길 한 줄 설명."""
    if mode == "global":
        return f"global cosine (T_max={epochs}) — 기존 동작"
    if join <= 0 or join >= epochs:
        return f"phase cosine — 구간이 하나뿐이라 global 과 동일 (join={join}, epochs={epochs})"
    return (f"phase cosine — align [0,{join}) T={join} + 적응 [{join},{epochs}) "
            f"T={epochs - join}, 적응이 base LR 에서 재시작")


# ----------------------------------------------------------------------
# 자체 검증: global 모드가 PyTorch CosineAnnealingLR 과 같은지 + phase 모드 성질
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import torch
    from torch.optim.lr_scheduler import CosineAnnealingLR

    E, BASE = 60, 1e-3
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([{"params": [p], "lr": BASE}])
    sched = CosineAnnealingLR(opt, T_max=E)

    worst = 0.0
    for e in range(E):
        torch_lr = opt.param_groups[0]["lr"]
        ours = BASE * lr_factor(e, E, "global")
        worst = max(worst, abs(torch_lr - ours))
        sched.step()
    print(f"global vs CosineAnnealingLR(T_max={E}) 최대 오차 = {worst:.3e} "
          f"({'일치' if worst < 1e-12 else '불일치 — 확인 필요'})")

    print(f"\nphase 모드에서 적응 시작 시점 LR (base 대비), epochs={E}")
    print(f"  {'join':>5}{'global':>12}{'phase':>10}   남은 적응 epoch")
    for j in [0, 10, 20, 30, 40, 50, 60]:
        g = lr_factor(min(j, E - 1), E, "global")
        ph = lr_factor(min(j, E - 1), E, "phase", j)
        print(f"  {j:>5}{g*100:>11.1f}%{ph*100:>9.1f}%   {E - j:>10}")

    print("\n적응 구간 LR 적분 (base·epoch)")
    print(f"  {'join':>5}{'global':>10}{'phase':>10}")
    for j in [0, 10, 20, 30, 40, 50]:
        g = sum(lr_factor(t, E, "global") for t in range(j, E))
        ph = sum(lr_factor(t, E, "phase", j) for t in range(j, E))
        print(f"  {j:>5}{g:>10.2f}{ph:>10.2f}")
    print("\n  phase 열은 '적응 epoch 수' 에만 비례한다 = 교락 제거됨")
