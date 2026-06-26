"""CDAN 학습 (IMU 단독 + 방향정보 입력증강, multi-seed).

CDAN_IMU_train.py 의 입력증강판. **회전 없이** raw IMU 에 윈도우별 중력방향 ĝ, PCA
이동방향 v1·v2, 그리고 둘의 관계 불변 스칼라(고윳값 비율·중력방향 모션분산비율·|cos|)를
채널로 덧붙여 학습한다(`mm_model.IMUDirCDAN`). 가설: 중력방향과 이동방향의 관계는 회전
불변이라 운동 구분에도, 크로스 디바이스 전이에도 강한 신호다.

축정렬을 하지 않는 게 핵심이므로 기본 데이터는 raw 를 쓴다(MM_DATA_DIR=preprocessed_MM_raw).
출력은 results/IMU/ 아래 imu_cdan_dir* 으로 저장.
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from mm_data_loader import get_mm_dataloaders
from mm_model import IMUDirCDAN
from mm_utils import set_seed, evaluate, save_confusion_matrix, summarize_results

WEIGHT_DIR = os.path.join(PROJECT_ROOT, "weights")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results", "IMU")


def train_cdan(seed, args, tag):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(WEIGHT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    save_path = os.path.join(WEIGHT_DIR, f"imu_cdan_dir{suffix}_seed{seed}_best_model.pth")

    train_loader, val_loader, tgt_train_loader, tgt_val_loader, num_classes, le = \
        get_mm_dataloaders(batch_size=args.batch_size)
    class_names = le.classes_

    model = IMUDirCDAN(num_classes=num_classes, gravity=not args.no_gravity,
                       n_pca=args.n_pca, invariant=not args.no_invariant).to(device)
    in_ch = 3 + model.backbone.dir.extra_dim()

    criterion_class = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_domain = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    print("\n" + "=" * 60)
    print(f"CDAN IMU-Dir Training  |  seed={seed}")
    print(f"Mode: IMU-only + direction features (no rotation) | 입력채널={in_ch} "
          f"(gravity={not args.no_gravity}, n_pca={args.n_pca}, invariant={not args.no_invariant})")
    print(f"Targeting {num_classes} classes on {device}")
    print("=" * 60)

    best_val_acc, best_target_acc = 0.0, 0.0
    for epoch in range(args.epochs):
        model.train()
        len_loader = min(len(train_loader), len(tgt_train_loader))
        total_c_loss, total_d_loss, total_d_acc = 0.0, 0.0, 0.0
        p = float(epoch) / args.epochs
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

        pbar = tqdm(enumerate(zip(train_loader, tgt_train_loader)),
                    total=len_loader, desc=f"Seed {seed} | Epoch [{epoch+1:02d}/{args.epochs:02d}]")
        for i, ((src_emg, src_imu, src_y), (tgt_emg, tgt_imu, tgt_y)) in pbar:
            src_emg, src_imu, src_y = src_emg.to(device), src_imu.to(device), src_y.to(device)
            tgt_emg, tgt_imu = tgt_emg.to(device), tgt_imu.to(device)
            optimizer.zero_grad()

            src_domain_label = torch.zeros(src_imu.size(0), dtype=torch.long, device=device)
            tgt_domain_label = torch.ones(tgt_imu.size(0), dtype=torch.long, device=device)

            src_class_out, src_domain_out = model(src_emg, src_imu, alpha=alpha)
            loss_s_label = criterion_class(src_class_out, src_y)
            loss_s_domain = criterion_domain(src_domain_out, src_domain_label)
            _, tgt_domain_out = model(tgt_emg, tgt_imu, alpha=alpha)
            loss_t_domain = criterion_domain(tgt_domain_out, tgt_domain_label)

            class_loss = loss_s_label
            domain_loss = loss_s_domain + loss_t_domain
            loss = class_loss + args.domain_weight * domain_loss
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                domain_pred = torch.cat([src_domain_out.argmax(1), tgt_domain_out.argmax(1)])
                domain_true = torch.cat([src_domain_label, tgt_domain_label])
                domain_acc = (domain_pred == domain_true).float().mean().item()
            total_c_loss += class_loss.item()
            total_d_loss += domain_loss.item()
            total_d_acc += domain_acc
            pbar.set_postfix({"Class": f"{class_loss.item():.4f}",
                              "Domain": f"{domain_loss.item():.4f}",
                              "DomAcc": f"{domain_acc*100:.2f}%"})

        scheduler.step()
        val_acc, _, _ = evaluate(model, val_loader, device, needs_alpha=True)
        tgt_acc, _, _ = evaluate(model, tgt_val_loader, device, needs_alpha=True)
        is_best = tgt_acc > best_target_acc
        if is_best:
            best_val_acc, best_target_acc = val_acc, tgt_acc
            torch.save(model.state_dict(), save_path)
        print(f"Epoch [{epoch+1:02d}/{args.epochs:02d}] | "
              f"Class: {total_c_loss/len_loader:.4f} | Domain: {total_d_loss/len_loader:.4f} | "
              f"DomAcc: {total_d_acc/len_loader*100:.2f}% | "
              f"Source Val: {val_acc:.2f}% | Target Val: {tgt_acc:.2f}% | Alpha: {alpha:.3f}"
              f"{'  (best)' if is_best else ''}")

    print(f"\n최종 | seed={seed} | Best Source Val: {best_val_acc:.2f}% | "
          f"Target at Best: {best_target_acc:.2f}% | Shift: {best_val_acc - best_target_acc:.2f}%")

    if not args.no_cm:
        model.load_state_dict(torch.load(save_path, map_location=device))
        _, v_preds, v_true = evaluate(model, val_loader, device, needs_alpha=True)
        _, t_preds, t_true = evaluate(model, tgt_val_loader, device, needs_alpha=True)
        save_confusion_matrix(
            v_true, v_preds, class_names,
            os.path.join(RESULT_DIR, f"imu_cdan_dir{suffix}_seed{seed}_source_cm.png"),
            f"CDAN IMU-Dir Source (seed={seed}, Val: {best_val_acc:.1f}%)", cmap="Purples")
        save_confusion_matrix(
            t_true, t_preds, class_names,
            os.path.join(RESULT_DIR, f"imu_cdan_dir{suffix}_seed{seed}_target_cm.png"),
            f"CDAN IMU-Dir Target (seed={seed}, Src: {best_val_acc:.1f}% vs Tgt: {best_target_acc:.1f}%)",
            cmap="Purples")
        print("혼동 행렬 저장 완료.")

    return {"seed": seed, "source_acc": best_val_acc, "target_acc": best_target_acc,
            "shift": best_val_acc - best_target_acc}


def write_result_json(results, tag, args):
    src = [r["source_acc"] for r in results]; tgt = [r["target_acc"] for r in results]
    sh = [r["shift"] for r in results]; ddof = 1 if len(results) > 1 else 0
    payload = {
        "tag": tag or "dir", "modality": "imu_only_dir",
        "features": {"gravity": not args.no_gravity, "n_pca": args.n_pca,
                     "invariant": not args.no_invariant},
        "data_dir": os.environ.get("MM_DATA_DIR", "preprocessed_MM_raw"),
        "seeds": [r["seed"] for r in results], "results": results,
        "mean": {"source_acc": float(np.mean(src)), "target_acc": float(np.mean(tgt)),
                 "shift": float(np.mean(sh))},
        "std": {"source_acc": float(np.std(src, ddof=ddof)), "target_acc": float(np.std(tgt, ddof=ddof)),
                "shift": float(np.std(sh, ddof=ddof))},
    }
    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, f"imu_cdan_result_{tag or 'dir'}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"결과 JSON 저장: {path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--multi_seed", action="store_true")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--domain_weight", type=float, default=1.0)
    p.add_argument("--no_cm", action="store_true")
    p.add_argument("--tag", type=str, default="dir", help="출력 파일 태그")
    # 방향정보 ablation 플래그
    p.add_argument("--no_gravity", action="store_true", help="중력방향 채널 제외")
    p.add_argument("--n_pca", type=int, default=2, help="PCA 이동방향 축 개수(0=사용안함)")
    p.add_argument("--no_invariant", action="store_true", help="관계 불변 스칼라 제외")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    suffix = f"_{args.tag}" if args.tag else ""
    if args.multi_seed:
        results = [train_cdan(s, args, args.tag) for s in args.seeds]
        summarize_results(results, method_name=f"CDAN (IMU-Dir){' | '+args.tag if args.tag else ''}",
                          save_path=os.path.join(RESULT_DIR, f"imu_cdan_dir{suffix}_summary.txt"))
    else:
        results = [train_cdan(args.seed, args, args.tag)]
    write_result_json(results, args.tag, args)
