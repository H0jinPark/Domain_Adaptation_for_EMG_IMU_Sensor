"""멀티모달 도메인 적응 실험 공용 유틸리티.

Multimodal/ 폴더의 MMD/Coral/DANN/CDAN 학습 스크립트가 공통으로 쓰는 함수 모음.
시드 고정, 정확도 평가, 혼동 행렬 저장, multi-seed 요약을 한곳에 모아 4개 실험의
평가·로깅 절차를 동일하게 유지한다. evaluate 가 (EMG, IMU) 두 입력을 받는 점만
5채널용 da_utils 와 다르다.
"""
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix


# ----------------------------------------------------------------------
# 난수 시드 고정
# ----------------------------------------------------------------------
def set_seed(seed):
    """random / numpy / torch 의 난수 시드를 고정해 실험 재현성을 확보한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------
# 정확도 평가 (EMG/IMU 분리 입력)
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, needs_alpha=False):
    """loader 전체에 대한 정확도(%)와 (예측, 정답) 리스트를 반환한다.

    needs_alpha=True 면 DANN/CDAN 처럼 forward(emg, imu, alpha) 시그니처를 갖는
    모델로 보고 alpha=0.0 으로 호출한다. 두 경우 모두 forward 의 첫 번째 반환값을
    클래스 logits 로 사용한다.
    """
    model.eval()
    preds, labels = [], []
    for emg, imu, y in loader:
        emg, imu = emg.to(device), imu.to(device)
        if needs_alpha:
            out = model(emg, imu, alpha=0.0)[0]
        else:
            out = model(emg, imu)[0]
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(y.numpy())
    acc = accuracy_score(labels, preds) * 100
    return acc, preds, labels


# ----------------------------------------------------------------------
# 혼동 행렬 저장
# ----------------------------------------------------------------------
def save_confusion_matrix(labels, preds, class_names, path, title, cmap="Blues"):
    """혼동 행렬을 히트맵으로 그려 PNG 로 저장한다."""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=16)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# ----------------------------------------------------------------------
# Multi-seed 결과 요약
# ----------------------------------------------------------------------
def summarize_results(results, method_name="", save_path=None):
    """seed 별 정확도를 표로 출력하고 평균±표준편차를 보고한다.

    results 에 target_test_acc 키가 있으면 test 컬럼까지 함께 낸다(없으면 기존과
    동일한 4컬럼). test 를 쓰지 않는 스크립트를 깨지 않으려고 옵션으로 뒀다.
    **논문에 싣는 숫자는 Tgt-Test 컬럼이다** — Source/Target 은 val 이라 model
    selection 에 쓰였으므로 낙관적으로 편향돼 있다.

    save_path 가 주어지면 동일한 요약을 텍스트 파일로도 저장한다.
    """
    source = np.array([r["source_acc"] for r in results])
    target = np.array([r["target_acc"] for r in results])
    shift = np.array([r["shift"] for r in results])
    ddof = 1 if len(results) > 1 else 0

    has_test = all("target_test_acc" in r for r in results)
    if has_test:
        src_te = np.array([r["source_test_acc"] for r in results])
        tgt_te = np.array([r["target_test_acc"] for r in results])

    width = 84 if has_test else 60
    def row(label, a, b, c, d=None, e=None):
        cells = " | ".join(f"{v:>8.2f}%" for v in (a, b, c))
        if has_test:
            cells += " | " + " | ".join(f"{v:>8.2f}%" for v in (d, e))
        return f"{label:>6} | {cells}"

    lines = []
    lines.append("=" * width)
    lines.append(f" Multi-seed Summary  |  {method_name}")
    lines.append(f" seeds = {[r['seed'] for r in results]}  (n={len(results)})")
    if has_test:
        lines.append(" val = model selection 용(낙관 편향) | test = 최종 보고용")
    lines.append("=" * width)
    hdr = f"{'Seed':>6} | {'Src-Val':>9} | {'Tgt-Val':>9} | {'Shift':>9}"
    if has_test:
        hdr += f" | {'Src-Test':>9} | {'Tgt-Test':>9}"
    lines.append(hdr)
    lines.append("-" * width)
    for r in results:
        lines.append(row(r["seed"], r["source_acc"], r["target_acc"], r["shift"],
                         r.get("source_test_acc"), r.get("target_test_acc")))
    lines.append("-" * width)
    lines.append(row("Mean", source.mean(), target.mean(), shift.mean(),
                     src_te.mean() if has_test else None,
                     tgt_te.mean() if has_test else None))
    lines.append(row("Std", source.std(ddof=ddof), target.std(ddof=ddof), shift.std(ddof=ddof),
                     src_te.std(ddof=ddof) if has_test else None,
                     tgt_te.std(ddof=ddof) if has_test else None))
    lines.append("=" * width)
    summary = (
        f" Src-Val  : {source.mean():.2f} ± {source.std(ddof=ddof):.2f} %\n"
        f" Tgt-Val  : {target.mean():.2f} ± {target.std(ddof=ddof):.2f} %\n"
        f" Shift    : {shift.mean():.2f} ± {shift.std(ddof=ddof):.2f} %")
    if has_test:
        summary += (
            f"\n Src-Test : {src_te.mean():.2f} ± {src_te.std(ddof=ddof):.2f} %\n"
            f" Tgt-Test : {tgt_te.mean():.2f} ± {tgt_te.std(ddof=ddof):.2f} %   <-- 보고 수치")
    lines.append(summary)
    lines.append("=" * width)

    text = "\n".join(lines)
    print("\n" + text)

    if save_path is not None:
        with open(save_path, "w") as f:
            f.write(text + "\n")
        print(f"\n요약 텍스트 저장: {save_path}")
