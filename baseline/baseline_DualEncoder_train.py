"""DualEncoder baseline 학습 스크립트 (멀티모달, No Domain Adaptation).

DualEncoderModel 을 preprocessed_MM/ 데이터(EMG (N,2,5000) / IMU (N,3,500))로
학습한다. EMG/IMU 를 독립 인코더로 처리한 뒤 concat 해 분류하며, domain adaptation
손실은 적용하지 않는 멀티모달 baseline 이다.
"""
import os
import sys
import random
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
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline.baseline_DualEncoder import DualEncoderModel

DATA_DIR = 'preprocessed_MM'


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class MMDataset(Dataset):
    """EMG/IMU 분리 텐서와 라벨을 담는 멀티모달 Dataset."""

    def __init__(self, emg, imu, y):
        self.emg = torch.tensor(emg, dtype=torch.float32)  # (N, 2, 5000)
        self.imu = torch.tensor(imu, dtype=torch.float32)  # (N, 3, 500)
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
    y_enc = le.transform(y)
    return emg, imu, y_enc, le


def get_dataloaders(batch_size=64):
    """Source train/val 과 Target val DataLoader, LabelEncoder 를 반환한다."""
    emg_tr, imu_tr, y_tr, le = load_split('train')
    emg_val, imu_val, y_val, _  = load_split('val', le)
    emg_tgt, imu_tgt, y_tgt, _  = load_split('target_val', le)

    print(f"Classes ({len(le.classes_)}): {le.classes_}")
    print(f"  Source  train: {len(y_tr):>6}  val: {len(y_val):>6}")
    print(f"  Target  val  : {len(y_tgt):>6}")

    train_loader = DataLoader(MMDataset(emg_tr,  imu_tr,  y_tr),
                              batch_size=batch_size, shuffle=True,  drop_last=True,  num_workers=4)
    val_loader   = DataLoader(MMDataset(emg_val, imu_val, y_val),
                              batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)
    tgt_loader   = DataLoader(MMDataset(emg_tgt, imu_tgt, y_tgt),
                              batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)
    return train_loader, val_loader, tgt_loader, le


# ----------------------------------------------------------------------
# 학습 / 평가 함수
# ----------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    """한 epoch 학습을 수행하고 (평균 손실, 정확도)를 반환한다."""
    model.train()
    total_loss, preds_all, labels_all = 0.0, [], []
    for emg, imu, y in loader:
        emg, imu, y = emg.to(device), imu.to(device), y.to(device)
        optimizer.zero_grad()
        out  = model(emg, imu)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        preds_all.extend(out.argmax(1).cpu().numpy())
        labels_all.extend(y.cpu().numpy())
    n = len(loader.dataset)
    return total_loss / n, accuracy_score(labels_all, preds_all)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """loader 전체에 대한 (평균 손실, 정확도, 예측, 정답)을 반환한다."""
    model.eval()
    total_loss, preds_all, labels_all = 0.0, [], []
    for emg, imu, y in loader:
        emg, imu, y = emg.to(device), imu.to(device), y.to(device)
        out  = model(emg, imu)
        loss = criterion(out, y)
        total_loss += loss.item() * len(y)
        preds_all.extend(out.argmax(1).cpu().numpy())
        labels_all.extend(y.cpu().numpy())
    n = len(loader.dataset)
    return total_loss / n, accuracy_score(labels_all, preds_all), preds_all, labels_all


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


# ----------------------------------------------------------------------
# 메인 학습 루프
# ----------------------------------------------------------------------
def set_seed(seed):
    """난수 시드를 고정해 실험 재현성을 확보한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(seed=42, epochs=30, batch_size=64, lr=1e-3):
    """단일 seed 로 DualEncoder baseline 을 학습하고 Source/Target 정확도를 반환한다."""
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*50}")
    print(f" DualEncoder Baseline Training  |  seed={seed}  device={device}")
    print(f"{'='*50}")

    os.makedirs('weights', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    train_loader, val_loader, tgt_loader, le = get_dataloaders(batch_size)
    class_names = list(le.classes_)
    num_classes = len(class_names)

    model     = DualEncoderModel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" Total params: {total_params:,}\n")

    best_val_acc = 0.0
    weight_path  = f'weights/dual_encoder_seed{seed}_best.pth'

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc           = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _   = evaluate(model, val_loader,  criterion, device)
        _,        tgt_acc, _, _   = evaluate(model, tgt_loader,  criterion, device)
        scheduler.step()

        # Source val 기준 best 모델 저장
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            torch.save(model.state_dict(), weight_path)

        best_mark = "  (best)" if is_best else ""
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] | "
            f"Loss: {tr_loss:.4f} | "
            f"Train Acc: {tr_acc*100:.2f}% | "
            f"Val Acc: {val_acc*100:.2f}% | "
            f"Target Val: {tgt_acc*100:.2f}%"
            f"{best_mark}"
        )

    # 최고 모델로 최종 평가
    print(f"\n Best val acc: {best_val_acc*100:.2f}%  (weights: {weight_path})")
    model.load_state_dict(torch.load(weight_path, map_location=device))

    _, src_acc, src_preds, src_labels = evaluate(model, val_loader, criterion, device)
    _, tgt_acc, tgt_preds, tgt_labels = evaluate(model, tgt_loader, criterion, device)

    print(f" Final Source Acc : {src_acc*100:.2f}%")
    print(f" Final Target Acc : {tgt_acc*100:.2f}%")

    save_confusion_matrix(src_labels, src_preds, class_names,
                          f'results/dual_encoder_seed{seed}_source_cm.png',
                          f'DualEncoder Source (seed={seed}, acc={src_acc:.4f})')
    save_confusion_matrix(tgt_labels, tgt_preds, class_names,
                          f'results/dual_encoder_seed{seed}_target_cm.png',
                          f'DualEncoder Target (seed={seed}, acc={tgt_acc:.4f})')
    return src_acc, tgt_acc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--multi_seed', action='store_true')
    parser.add_argument('--seeds',      type=int,   nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--epochs',     type=int,   default=30)
    parser.add_argument('--batch_size', type=int,   default=64)
    parser.add_argument('--lr',         type=float, default=1e-3)
    args = parser.parse_args()

    if args.multi_seed:
        results = []
        for s in args.seeds:
            src_acc, tgt_acc = train(seed=s, epochs=args.epochs,
                                     batch_size=args.batch_size, lr=args.lr)
            results.append((s, src_acc, tgt_acc))
        print("\n" + "=" * 50)
        print(" Multi-seed Summary")
        print(f"{'Seed':>6}  {'Src Acc':>8}  {'Tgt Acc':>8}")
        for s, sa, ta in results:
            print(f"{s:>6}  {sa:>8.4f}  {ta:>8.4f}")
        src_mean = np.mean([r[1] for r in results])
        tgt_mean = np.mean([r[2] for r in results])
        print(f"{'Mean':>6}  {src_mean:>8.4f}  {tgt_mean:>8.4f}")
    else:
        train(seed=args.seed, epochs=args.epochs,
              batch_size=args.batch_size, lr=args.lr)
