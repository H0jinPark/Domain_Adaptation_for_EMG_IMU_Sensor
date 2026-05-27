"""pretrain_Multimodal 공용 유틸리티.

mm_utils.py와 동일한 인터페이스를 유지한다.
evaluate()는 (EMG, IMU) 두 입력을 받으며, needs_alpha로 DANN/CDAN 시그니처를 처리한다.
"""
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, needs_alpha=False):
    """loader 전체에 대한 정확도(%)와 (예측, 정답) 리스트를 반환한다."""
    model.eval()
    preds, labels = [], []
    for emg, imu, y in loader:
        emg, imu = emg.to(device), imu.to(device)
        out = model(emg, imu, alpha=0.0)[0] if needs_alpha else model(emg, imu)[0]
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(y.numpy())
    acc = accuracy_score(labels, preds) * 100
    return acc, preds, labels


def save_confusion_matrix(labels, preds, class_names, path, title, cmap="Blues"):
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


def summarize_results(results, method_name="", save_path=None):
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
