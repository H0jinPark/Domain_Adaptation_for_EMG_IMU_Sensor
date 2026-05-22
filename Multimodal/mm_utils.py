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
    """seed 별 source/target/shift 정확도를 표로 출력하고 평균±표준편차를 보고한다.

    save_path 가 주어지면 동일한 요약을 텍스트 파일로도 저장한다.
    """
    source = np.array([r["source_acc"] for r in results])
    target = np.array([r["target_acc"] for r in results])
    shift = np.array([r["shift"] for r in results])
    ddof = 1 if len(results) > 1 else 0

    lines = []
    lines.append("=" * 60)
    lines.append(f" Multi-seed Summary  |  {method_name}")
    lines.append(f" seeds = {[r['seed'] for r in results]}  (n={len(results)})")
    lines.append("=" * 60)
    lines.append(f"{'Seed':>6} | {'Source':>9} | {'Target':>9} | {'Shift':>9}")
    lines.append("-" * 60)
    for r in results:
        lines.append(f"{r['seed']:>6} | {r['source_acc']:>8.2f}% | "
                     f"{r['target_acc']:>8.2f}% | {r['shift']:>8.2f}%")
    lines.append("-" * 60)
    lines.append(f"{'Mean':>6} | {source.mean():>8.2f}% | "
                 f"{target.mean():>8.2f}% | {shift.mean():>8.2f}%")
    lines.append(f"{'Std':>6} | {source.std(ddof=ddof):>8.2f}% | "
                 f"{target.std(ddof=ddof):>8.2f}% | {shift.std(ddof=ddof):>8.2f}%")
    lines.append("=" * 60)
    lines.append(
        f" Source : {source.mean():.2f} ± {source.std(ddof=ddof):.2f} %\n"
        f" Target : {target.mean():.2f} ± {target.std(ddof=ddof):.2f} %\n"
        f" Shift  : {shift.mean():.2f} ± {shift.std(ddof=ddof):.2f} %")
    lines.append("=" * 60)

    text = "\n".join(lines)
    print("\n" + text)

    if save_path is not None:
        with open(save_path, "w") as f:
            f.write(text + "\n")
        print(f"\n요약 텍스트 저장: {save_path}")
