"""DANN-MM-InterFuse 학습 스크립트 (2-phase: 모달별 CL 사전학습 → intermediate fusion DANN).

Phase 1 - DANN_MM_ssl 과 정확히 동일한 모달별 contrastive 사전학습 (코드 재사용).
          기존 CL 인코더 가중치(dann_mm_ssl_*_encoder_seed*.pth)가 있으면 재사용한다.
Phase 2 - 사전학습 인코더를 DualEncoderDANN_InterFuse 에 이식해 DANN 파인튜닝.
          융합을 인코더 pool 이전(시간 feature map 단계)으로 옮겨, 시간축을 살린
          채로 EMG/IMU 를 합치고 joint conv 로 cross-modal 시간 상관을 학습한다.

→ DANN_MM_ssl / DANN_MM_stack 과 Phase 1 이 동일하므로, 비교 시 융합 방식의
  효과만 분리된다. (ssl: pool 후 cat / stack: pool 후 (2,256) / interfuse: pool 전 융합)
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# 프로젝트 루트를 import 경로에 추가 (DANN/DANN_experiment/ → ../../)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from DANN.DANN_experiment.DANN_MM_ssl_train import (
    set_seed, get_dataloaders, pretrain_contrastive, evaluate, save_confusion_matrix
)
from DANN.DANN_experiment.DANN_MM_interfuse_model import DualEncoderDANN_InterFuse


# -----------------------------------------------------------------------
# Phase 2: intermediate fusion DANN 파인튜닝
# -----------------------------------------------------------------------
def finetune(seed, emg_path, imu_path, epochs, lr, device,
             src_loader, val_loader, tgt_loader, tgt_val_loader,
             num_classes, class_names):
    """사전학습 인코더를 로드해 DualEncoderDANN_InterFuse 를 DANN 방식으로 파인튜닝한다."""

    print(f"\n{'='*55}")
    print(f" Phase 2: DANN Fine-tuning (intermediate fusion)  |  finetune_epochs={epochs}")
    print(f"{'='*55}")

    model = DualEncoderDANN_InterFuse.from_pretrained(emg_path, imu_path, num_classes).to(device)
    criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_dom = nn.CrossEntropyLoss()
    optimizer     = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler     = CosineAnnealingLR(optimizer, T_max=epochs)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" Total params: {total_params:,}\n")

    weight_path  = f'weights/dann_mm_interfuse_seed{seed}_best.pth'
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        p     = epoch / epochs
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

        n_iter   = min(len(src_loader), len(tgt_loader))
        src_iter = iter(src_loader)
        tgt_iter = iter(tgt_loader)

        cls_loss_sum = dom_loss_sum = 0.0

        for _ in range(n_iter):
            src_emg, src_imu, src_y = next(src_iter)
            tgt_emg, tgt_imu, _     = next(tgt_iter)

            src_emg, src_imu, src_y = src_emg.to(device), src_imu.to(device), src_y.to(device)
            tgt_emg, tgt_imu        = tgt_emg.to(device), tgt_imu.to(device)

            src_dom = torch.zeros(src_emg.size(0), dtype=torch.long, device=device)
            tgt_dom = torch.ones (tgt_emg.size(0), dtype=torch.long, device=device)

            optimizer.zero_grad()

            src_cls, src_dom_out = model(src_emg, src_imu, alpha)
            _,       tgt_dom_out = model(tgt_emg, tgt_imu, alpha)

            loss_cls = criterion_cls(src_cls, src_y)
            loss_dom = criterion_dom(src_dom_out, src_dom) + criterion_dom(tgt_dom_out, tgt_dom)
            loss     = loss_cls + loss_dom

            loss.backward()
            optimizer.step()

            cls_loss_sum += loss_cls.item()
            dom_loss_sum += loss_dom.item()

        scheduler.step()

        val_acc, _, _ = evaluate(model, val_loader,     device)
        tgt_acc, _, _ = evaluate(model, tgt_val_loader, device)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            torch.save(model.state_dict(), weight_path)

        best_mark = "  (best)" if is_best else ""
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] | "
            f"Class: {cls_loss_sum/n_iter:.4f} | "
            f"Domain: {dom_loss_sum/n_iter:.4f} | "
            f"Source Val: {val_acc*100:.2f}% | "
            f"Target Val: {tgt_acc*100:.2f}% | "
            f"Alpha: {alpha:.3f}"
            f"{best_mark}"
        )

    print(f"\n Best val acc: {best_val_acc*100:.2f}%  (weights: {weight_path})")
    model.load_state_dict(torch.load(weight_path, map_location=device))

    src_acc, src_preds, src_labels = evaluate(model, val_loader,     device)
    tgt_acc, tgt_preds, tgt_labels = evaluate(model, tgt_val_loader, device)

    print(f" Final Source Acc : {src_acc*100:.2f}%")
    print(f" Final Target Acc : {tgt_acc*100:.2f}%")

    save_confusion_matrix(src_labels, src_preds, class_names,
                          f'results/dann_mm_interfuse_seed{seed}_source_cm.png',
                          f'DANN-MM-InterFuse Source (seed={seed}, acc={src_acc:.4f})')
    save_confusion_matrix(tgt_labels, tgt_preds, class_names,
                          f'results/dann_mm_interfuse_seed{seed}_target_cm.png',
                          f'DANN-MM-InterFuse Target (seed={seed}, acc={tgt_acc:.4f})')
    return src_acc, tgt_acc


# -----------------------------------------------------------------------
# 전체 학습: Phase 1 (CL, 재사용 가능) → Phase 2 (intermediate fusion DANN)
# -----------------------------------------------------------------------
def train(seed=42, ssl_epochs=50, finetune_epochs=30,
          ssl_batch_size=128, batch_size=64, ssl_lr=1e-3, lr=1e-3, temperature=0.5):
    """단일 seed 로 2-phase 학습을 실행하고 Source/Target 최종 정확도를 반환한다."""
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*55}")
    print(f" DANN-MM-InterFuse Training  |  seed={seed}  device={device}")
    print(f"{'='*55}")

    os.makedirs('weights', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    (src_loader, val_loader, tgt_loader, tgt_val_loader,
     ssl_loader, le) = get_dataloaders(batch_size, ssl_batch_size)
    class_names = list(le.classes_)
    num_classes = len(class_names)

    # Phase 1: DANN_MM_ssl 과 정확히 동일한 CL. 기존 인코더가 있으면 재사용한다.
    emg_path = f'weights/dann_mm_ssl_emg_encoder_seed{seed}.pth'
    imu_path = f'weights/dann_mm_ssl_imu_encoder_seed{seed}.pth'
    if os.path.exists(emg_path) and os.path.exists(imu_path):
        print(f"\n기존 CL 사전학습 인코더 재사용:\n  {emg_path}\n  {imu_path}")
    else:
        emg_path, imu_path = pretrain_contrastive(
            seed, ssl_epochs, ssl_lr, temperature, device, ssl_loader
        )

    return finetune(
        seed, emg_path, imu_path, finetune_epochs, lr, device,
        src_loader, val_loader, tgt_loader, tgt_val_loader,
        num_classes, class_names
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',            type=int,   default=42)
    parser.add_argument('--multi_seed',      action='store_true')
    parser.add_argument('--seeds',           type=int,   nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--ssl_epochs',      type=int,   default=50)
    parser.add_argument('--finetune_epochs', type=int,   default=30)
    parser.add_argument('--ssl_batch_size',  type=int,   default=128)
    parser.add_argument('--batch_size',      type=int,   default=64)
    parser.add_argument('--ssl_lr',          type=float, default=1e-3)
    parser.add_argument('--lr',              type=float, default=1e-3)
    parser.add_argument('--temperature',     type=float, default=0.5)
    args = parser.parse_args()

    if args.multi_seed:
        results = []
        for s in args.seeds:
            src_acc, tgt_acc = train(
                seed=s,
                ssl_epochs=args.ssl_epochs,
                finetune_epochs=args.finetune_epochs,
                ssl_batch_size=args.ssl_batch_size,
                batch_size=args.batch_size,
                ssl_lr=args.ssl_lr,
                lr=args.lr,
                temperature=args.temperature,
            )
            results.append((s, src_acc, tgt_acc))
        print("\n" + "=" * 55)
        print(" Multi-seed Summary")
        print(f"{'Seed':>6}  {'Src Acc':>8}  {'Tgt Acc':>8}")
        for s, sa, ta in results:
            print(f"{s:>6}  {sa:>8.4f}  {ta:>8.4f}")
        src_mean = np.mean([r[1] for r in results])
        tgt_mean = np.mean([r[2] for r in results])
        print(f"{'Mean':>6}  {src_mean:>8.4f}  {tgt_mean:>8.4f}")
    else:
        train(
            seed=args.seed,
            ssl_epochs=args.ssl_epochs,
            finetune_epochs=args.finetune_epochs,
            ssl_batch_size=args.ssl_batch_size,
            batch_size=args.batch_size,
            ssl_lr=args.ssl_lr,
            lr=args.lr,
            temperature=args.temperature,
        )
