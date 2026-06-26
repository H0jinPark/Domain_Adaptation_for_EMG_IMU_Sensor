"""EMG 단독 source-only baseline 학습 + target inference (multi-seed 지원).

실험 의도(사용자 요청):
  1. EMG 단일 인코더만 떼서, 지금까지와 동일한 모델 구조(인코더→256ch joint TCN→
     512 feature→label head)로 baseline 을 **source 데이터만** 으로 학습한다.
  2. 사전학습된(=source-only) 모델로 **target 데이터에 inference** 해서 얼마나 shift
     가 일어나는지(Source Val − Target Val) 측정한다.

도메인 적응(DA) 없음 = Multimodal/no_da_train.py 의 EMG 전용판. 모델만 EMGOnlyClassifier
로 바꾸고, IMU 는 무시한다(데이터로더의 (emg,imu,y) 규약은 그대로). EMG 는 축정렬과
무관하므로 어느 preprocessed_MM_* 폴더든 동일 — 기본 raw 를 쓴다.

출력은 멀티모달/IMU 와 분리해 results/EMG/ 아래 emg_baseline* 으로 저장한다.
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# 프로젝트 루트 + Multimodal 을 import 경로에 추가 (EMG/ → ../ , ../Multimodal)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "Multimodal"))
from mm_data_loader import get_mm_dataloaders
from mm_model import EMGOnlyClassifier
from mm_utils import set_seed, evaluate, save_confusion_matrix, summarize_results

WEIGHT_DIR = os.path.join(PROJECT_ROOT, "weights")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results", "EMG")


def train_emg_baseline(seed=42, epochs=30, batch_size=64, lr=1e-3, save_cm=False, tag=""):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(WEIGHT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    suffix = f"_{tag}" if tag else ""
    save_path = os.path.join(WEIGHT_DIR, f"emg_baseline_seed{seed}{suffix}_best_model.pth")

    # tgt_train 은 source-only 학습에 쓰지 않는다(평가용 tgt_val 만 사용).
    train_loader, val_loader, _, tgt_val_loader, num_classes, le = \
        get_mm_dataloaders(batch_size=batch_size)
    class_names = le.classes_

    model = EMGOnlyClassifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    print("\n" + "=" * 60)
    print(f"EMG-only Source-only Baseline Training Start  |  seed={seed}")
    print("Mode: EMG single-encoder, source-only, no domain adaptation")
    print(f"Targeting {num_classes} classes on {device}")
    print("=" * 60)

    best_val_acc, best_target_acc = 0.0, 0.0

    for epoch in range(epochs):
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Seed {seed} | Epoch [{epoch+1:02d}/{epochs:02d}]")
        for src_emg, _src_imu, src_y in pbar:        # IMU 미사용
            src_emg, src_y = src_emg.to(device), src_y.to(device)

            optimizer.zero_grad()
            class_out, _ = model(src_emg)
            loss = criterion(class_out, src_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += (class_out.argmax(1) == src_y).sum().item()
            total_samples += src_y.size(0)
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Acc":  f"{total_correct / total_samples * 100:.1f}%",
            })

        scheduler.step()

        # 2단계: 매 epoch source val / target val 평가 (evaluate 는 (emg,imu) 넘기지만 IMU 무시)
        val_acc, _, _ = evaluate(model, val_loader, device, needs_alpha=False)
        tgt_acc, _, _ = evaluate(model, tgt_val_loader, device, needs_alpha=False)

        # source-only 이므로 target 라벨을 보지 않고 source val 기준으로 best 선택.
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_target_acc = tgt_acc
            torch.save(model.state_dict(), save_path)

        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | "
              f"Loss: {total_loss / len(train_loader):.4f} | "
              f"Source Val: {val_acc:.2f}% | Target Val: {tgt_acc:.2f}%"
              + ("  (best)" if is_best else ""))

    print(f"\n최종 결과 | seed={seed} | "
          f"Best Source Val: {best_val_acc:.2f}% | "
          f"Target at Best: {best_target_acc:.2f}% | "
          f"Shift: {best_val_acc - best_target_acc:.2f}%")

    if save_cm:
        model.load_state_dict(torch.load(save_path, map_location=device))
        _, v_preds, v_true = evaluate(model, val_loader, device, needs_alpha=False)
        _, t_preds, t_true = evaluate(model, tgt_val_loader, device, needs_alpha=False)
        save_confusion_matrix(
            v_true, v_preds, class_names,
            os.path.join(RESULT_DIR, f"emg_baseline_seed{seed}{suffix}_source_cm.png"),
            f"EMG-only Source (seed={seed}, Val: {best_val_acc:.1f}%)", cmap="Oranges")
        save_confusion_matrix(
            t_true, t_preds, class_names,
            os.path.join(RESULT_DIR, f"emg_baseline_seed{seed}{suffix}_target_cm.png"),
            f"EMG-only Target (seed={seed}, Src: {best_val_acc:.1f}% / Tgt: {best_target_acc:.1f}%)",
            cmap="Oranges")
        print("혼동 행렬 저장 완료.")

    return {
        "seed": seed,
        "source_acc": best_val_acc,
        "target_acc": best_target_acc,
        "shift": best_val_acc - best_target_acc,
    }


def write_result_json(results, tag):
    src = [r["source_acc"] for r in results]
    tgt = [r["target_acc"] for r in results]
    sh  = [r["shift"] for r in results]
    ddof = 1 if len(results) > 1 else 0
    payload = {
        "tag": tag or "default",
        "modality": "emg_only",
        "mode": "source_only_no_da",
        "data_dir": os.environ.get("MM_DATA_DIR", "preprocessed_MM_raw"),
        "seeds": [r["seed"] for r in results],
        "results": results,
        "mean": {"source_acc": float(np.mean(src)), "target_acc": float(np.mean(tgt)),
                 "shift": float(np.mean(sh))},
        "std":  {"source_acc": float(np.std(src, ddof=ddof)), "target_acc": float(np.std(tgt, ddof=ddof)),
                 "shift": float(np.std(sh, ddof=ddof))},
    }
    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, f"emg_baseline_result{('_'+tag) if tag else ''}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"결과 JSON 저장: {path}")
    return path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi_seed", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no_cm", action="store_true")
    parser.add_argument("--tag", type=str, default="", help="체크포인트/결과 파일명 suffix")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    suffix = f"_{args.tag}" if args.tag else ""
    if args.multi_seed:
        results = []
        for seed in args.seeds:
            results.append(train_emg_baseline(
                seed=seed, epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, save_cm=not args.no_cm, tag=args.tag))
        os.makedirs(RESULT_DIR, exist_ok=True)
        summarize_results(results, method_name="EMG-only (source-only)",
                          save_path=os.path.join(RESULT_DIR, f"emg_baseline{suffix}_summary.txt"))
    else:
        results = [train_emg_baseline(
            seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, save_cm=not args.no_cm, tag=args.tag)]

    write_result_json(results, args.tag)
