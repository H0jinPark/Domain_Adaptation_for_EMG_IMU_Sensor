"""DANN-MM-Sep 학습 스크립트 (2-phase: 분리 사전학습 -> 융합 DANN).

Phase 1 - EMG/IMU 인코더를 source 데이터로 각각 독립 분류 학습.
          두 인코더를 같은 루프에서 별도 optimizer 로 동시에 학습한다.
Phase 2 - 사전학습 인코더를 DualEncoderDANN_Sep 에 이식하고 GRL 도메인 적대 학습.
          두 페이즈 모두 인코더 파라미터를 업데이트한다 (동결 없음).
"""
import os
import sys
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 프로젝트 루트를 import 경로에 추가 (DANN/DANN_experiment/ → ../../)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from DANN.DANN_experiment.DANN_MM_sep_model import (
    EMGEncoderWithClassifier, IMUEncoderWithClassifier, DualEncoderDANN_Sep
)

DATA_DIR = 'preprocessed_MM'


class MMDataset(Dataset):
    """EMG/IMU 분리 텐서와 라벨을 담는 멀티모달 Dataset."""

    def __init__(self, emg, imu, y):
        self.emg = torch.tensor(emg, dtype=torch.float32)  # (N, 2, 5000)
        self.imu = torch.tensor(imu, dtype=torch.float32)  # (N, 3,  500)
        self.y   = torch.tensor(y,   dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.emg[idx], self.imu[idx], self.y[idx]


def load_split(prefix, le=None):
    """preprocessed_MM/ 에서 한 split 의 EMG/IMU/label 을 로드한다."""
    emg = np.load(f'{DATA_DIR}/X_emg_{prefix}.npy')
    imu = np.load(f'{DATA_DIR}/X_imu_{prefix}.npy')
    y   = np.load(f'{DATA_DIR}/y_{prefix}.npy', allow_pickle=True)
    if le is None:
        le = LabelEncoder()
        le.fit(y)
    return emg, imu, le.transform(y), le


def get_dataloaders(batch_size=64):
    """Source/Target 의 train/val 4분할 DataLoader 와 LabelEncoder 를 반환한다."""
    emg_tr, imu_tr, y_tr, le   = load_split('train')
    emg_val, imu_val, y_val, _ = load_split('val',          le)
    emg_tt, imu_tt, y_tt, _    = load_split('target_train', le)
    emg_tv, imu_tv, y_tv, _    = load_split('target_val',   le)

    print(f"Classes ({len(le.classes_)}): {le.classes_}")
    print(f"  Source  train={len(y_tr)}  val={len(y_val)}")
    print(f"  Target  train={len(y_tt)}  val={len(y_tv)}")

    def loader(emg, imu, y, shuffle, drop_last):
        return DataLoader(MMDataset(emg, imu, y),
                          batch_size=batch_size, shuffle=shuffle,
                          drop_last=drop_last, num_workers=4)

    return (loader(emg_tr,  imu_tr,  y_tr,  True,  True),
            loader(emg_val, imu_val, y_val, False, False),
            loader(emg_tt,  imu_tt,  y_tt,  True,  True),
            loader(emg_tv,  imu_tv,  y_tv,  False, False),
            le)


def set_seed(seed):
    """난수 시드를 고정해 실험 재현성을 확보한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------
# 평가 헬퍼
# -----------------------------------------------------------------------
@torch.no_grad()
def evaluate_single(model, loader, device, modal):
    """Phase 1 용. modal='emg'|'imu' 에 해당하는 modality 만 사용해 정확도를 반환한다."""
    model.eval()
    preds, labels = [], []
    for emg, imu, y in loader:
        x = emg.to(device) if modal == 'emg' else imu.to(device)
        preds.extend(model(x).argmax(1).cpu().numpy())
        labels.extend(y.numpy())
    return accuracy_score(labels, preds)


@torch.no_grad()
def evaluate(model, loader, device):
    """Phase 2 용. loader 전체에 대한 정확도와 (예측, 정답) 리스트를 반환한다."""
    model.eval()
    preds, labels = [], []
    for emg, imu, y in loader:
        emg, imu = emg.to(device), imu.to(device)
        out, _ = model(emg, imu, alpha=0.0)
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(y.numpy())
    return accuracy_score(labels, preds), preds, labels


def save_confusion_matrix(labels, preds, class_names, path, title):
    """혼동 행렬을 히트맵으로 그려 PNG 로 저장한다."""
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# -----------------------------------------------------------------------
# Phase 1: EMG / IMU 인코더 독립 사전학습
# -----------------------------------------------------------------------
def pretrain(seed, epochs, lr, device, src_loader, val_loader, num_classes):
    """EMG/IMU 인코더를 source 분류로 동시에 독립 사전학습하고 인코더 가중치를 저장한다."""

    print(f"\n{'='*55}")
    print(f" Phase 1: Pre-training  |  pretrain_epochs={epochs}")
    print(f"{'='*55}")

    emg_model = EMGEncoderWithClassifier(num_classes).to(device)
    imu_model = IMUEncoderWithClassifier(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    opt_emg = optim.AdamW(emg_model.parameters(), lr=lr, weight_decay=1e-3)
    opt_imu = optim.AdamW(imu_model.parameters(), lr=lr, weight_decay=1e-3)
    sch_emg = CosineAnnealingLR(opt_emg, T_max=epochs)
    sch_imu = CosineAnnealingLR(opt_imu, T_max=epochs)

    best_emg_acc = best_imu_acc = 0.0
    emg_path = f'weights/dann_mm_sep_emg_encoder_seed{seed}.pth'
    imu_path = f'weights/dann_mm_sep_imu_encoder_seed{seed}.pth'

    for epoch in range(1, epochs + 1):
        emg_model.train()
        imu_model.train()
        emg_loss_sum = imu_loss_sum = 0.0
        n = len(src_loader)

        for emg, imu, y in src_loader:
            emg, imu, y = emg.to(device), imu.to(device), y.to(device)

            # EMG 독립 학습
            opt_emg.zero_grad()
            loss_emg = criterion(emg_model(emg), y)
            loss_emg.backward()
            opt_emg.step()
            emg_loss_sum += loss_emg.item()

            # IMU 독립 학습
            opt_imu.zero_grad()
            loss_imu = criterion(imu_model(imu), y)
            loss_imu.backward()
            opt_imu.step()
            imu_loss_sum += loss_imu.item()

        sch_emg.step()
        sch_imu.step()

        emg_val = evaluate_single(emg_model, val_loader, device, 'emg')
        imu_val = evaluate_single(imu_model, val_loader, device, 'imu')

        # encoder 파라미터만 저장 (Phase 2 에서 재사용)
        if emg_val > best_emg_acc:
            best_emg_acc = emg_val
            torch.save(emg_model.encoder.state_dict(), emg_path)
        if imu_val > best_imu_acc:
            best_imu_acc = imu_val
            torch.save(imu_model.encoder.state_dict(), imu_path)

        print(
            f"PreTrain [{epoch:02d}/{epochs:02d}] | "
            f"EMG Loss: {emg_loss_sum/n:.4f}  Val: {emg_val*100:.2f}% | "
            f"IMU Loss: {imu_loss_sum/n:.4f}  Val: {imu_val*100:.2f}%"
        )

    print(f"\n Best EMG encoder val: {best_emg_acc*100:.2f}%  ({emg_path})")
    print(f" Best IMU encoder val: {best_imu_acc*100:.2f}%  ({imu_path})")
    return emg_path, imu_path


# -----------------------------------------------------------------------
# Phase 2: 융합 DANN 파인튜닝
# -----------------------------------------------------------------------
def finetune(seed, emg_path, imu_path, epochs, lr, device,
             src_loader, val_loader, tgt_loader, tgt_val_loader,
             num_classes, class_names):
    """사전학습 인코더를 로드해 DualEncoderDANN_Sep 를 DANN 방식으로 파인튜닝한다."""

    print(f"\n{'='*55}")
    print(f" Phase 2: DANN Fine-tuning  |  finetune_epochs={epochs}")
    print(f"{'='*55}")

    model = DualEncoderDANN_Sep.from_pretrained(emg_path, imu_path, num_classes).to(device)
    criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_dom = nn.CrossEntropyLoss()
    optimizer     = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler     = CosineAnnealingLR(optimizer, T_max=epochs)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" Total params: {total_params:,}\n")

    weight_path  = f'weights/dann_mm_sep_seed{seed}_best.pth'
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        p     = epoch / epochs
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

        n_iter   = min(len(src_loader), len(tgt_loader))
        src_iter = iter(src_loader)
        tgt_iter = iter(tgt_loader)

        total_loss = cls_loss_sum = dom_loss_sum = 0.0

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

            total_loss   += loss.item()
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
                          f'results/dann_mm_sep_seed{seed}_source_cm.png',
                          f'DANN-MM-Sep Source (seed={seed}, acc={src_acc:.4f})')
    save_confusion_matrix(tgt_labels, tgt_preds, class_names,
                          f'results/dann_mm_sep_seed{seed}_target_cm.png',
                          f'DANN-MM-Sep Target (seed={seed}, acc={tgt_acc:.4f})')
    return src_acc, tgt_acc


# -----------------------------------------------------------------------
# 전체 학습: Phase 1 -> Phase 2
# -----------------------------------------------------------------------
def train(seed=42, pretrain_epochs=20, finetune_epochs=30, batch_size=64, lr=1e-3):
    """단일 seed 로 2-phase 학습을 실행하고 Source/Target 최종 정확도를 반환한다."""
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*55}")
    print(f" DANN-MM-Sep Training  |  seed={seed}  device={device}")
    print(f"{'='*55}")

    os.makedirs('weights', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    src_loader, val_loader, tgt_loader, tgt_val_loader, le = get_dataloaders(batch_size)
    class_names = list(le.classes_)
    num_classes = len(class_names)

    emg_path, imu_path = pretrain(
        seed, pretrain_epochs, lr, device,
        src_loader, val_loader, num_classes
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
    parser.add_argument('--pretrain_epochs', type=int,   default=20)
    parser.add_argument('--finetune_epochs', type=int,   default=30)
    parser.add_argument('--batch_size',      type=int,   default=64)
    parser.add_argument('--lr',              type=float, default=1e-3)
    args = parser.parse_args()

    if args.multi_seed:
        results = []
        for s in args.seeds:
            src_acc, tgt_acc = train(
                seed=s,
                pretrain_epochs=args.pretrain_epochs,
                finetune_epochs=args.finetune_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
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
            pretrain_epochs=args.pretrain_epochs,
            finetune_epochs=args.finetune_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
