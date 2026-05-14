"""
비교 실험: 기본 전처리 데이터 → 모달리티 분리 → DualEncoderModel

목적:
  MM 전처리(preprocessed_MM/)와 기본 전처리(preprocessed/)의 성능 차이가
  '전처리 방식' 때문인지 확인하기 위한 대조 실험.

  기본 전처리 데이터 (N, 5000, 5) 로드
    → EMG: ch 0,1  (N, 2, 5000)  — 그대로 사용
    → IMU: ch 2,3,4 (N, 5000, 3) → 10배 다운샘플 → (N, 3, 500)
    → DualEncoderModel에 투입 (MM baseline과 동일한 모델/학습 루프)

기본 전처리와 MM 전처리의 핵심 차이:
  - 기본: EMG+IMU 공동 Z-score 정규화 (5채널 한 scaler)
  - MM:   EMG/IMU 독립 Z-score 정규화 (scaler 2개)
  → 이 정규화 방식 차이가 성능에 영향을 주는지 이 실험으로 확인
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline.baseline_DualEncoder import DualEncoderModel

# 기본 전처리 데이터 경로
DATA_DIR  = 'preprocessed'
IMU_STEP  = 10   # 1000Hz → 100Hz (10배 다운샘플)


# =====================================================================
# 기본 전처리 (N, 5000, 5) → EMG (N, 2, 5000) / IMU (N, 3, 500) 분리
# ch 순서: [biceps(0), triceps(1), triceps_X(2), triceps_Y(3), triceps_Z(4)]
# =====================================================================
def split_modalities(X):
    """
    X: (N, 5000, 5) — 기본 전처리 출력 (time-first)
    returns:
      emg: (N, 2, 5000)
      imu: (N, 3,  500)
    """
    emg = X[:, :, :2].transpose(0, 2, 1).astype(np.float32)   # (N, 2, 5000)
    imu = X[:, ::IMU_STEP, 2:].transpose(0, 2, 1).astype(np.float32)  # (N, 3, 500)
    return emg, imu


def load_split(prefix, le=None):
    X   = np.load(f'{DATA_DIR}/X_{prefix}.npy')          # (N, 5000, 5)
    y   = np.load(f'{DATA_DIR}/y_{prefix}.npy', allow_pickle=True)
    emg, imu = split_modalities(X)

    if le is None:
        le = LabelEncoder()
        le.fit(y)
    return emg, imu, le.transform(y), le


# =====================================================================
# Dataset
# =====================================================================
class MMDataset(Dataset):
    def __init__(self, emg, imu, y):
        self.emg = torch.tensor(emg, dtype=torch.float32)
        self.imu = torch.tensor(imu, dtype=torch.float32)
        self.y   = torch.tensor(y,   dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.emg[idx], self.imu[idx], self.y[idx]


def get_dataloaders(batch_size=64):
    emg_tr, imu_tr, y_tr, le     = load_split('train')
    emg_val, imu_val, y_val, _   = load_split('val',          le)
    emg_tt, imu_tt, y_tt, _      = load_split('target_train', le)
    emg_tv, imu_tv, y_tv, _      = load_split('target_val',   le)

    print(f"Classes ({len(le.classes_)}): {le.classes_}")
    print(f"  Source  train={len(y_tr)}  val={len(y_val)}")
    print(f"  Target  train={len(y_tt)}  val={len(y_tv)}")
    print(f"  EMG shape: {emg_tr.shape}  IMU shape: {imu_tr.shape}")

    def loader(emg, imu, y, shuffle, drop_last):
        return DataLoader(MMDataset(emg, imu, y),
                          batch_size=batch_size, shuffle=shuffle,
                          drop_last=drop_last, num_workers=4)

    return (loader(emg_tr,  imu_tr,  y_tr,  True,  True),
            loader(emg_val, imu_val, y_val, False, False),
            loader(emg_tt,  imu_tt,  y_tt,  True,  True),
            loader(emg_tv,  imu_tv,  y_tv,  False, False),
            le)


# =====================================================================
# 학습 / 평가
# =====================================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
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
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


# =====================================================================
# 메인
# =====================================================================
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def train(seed=42, epochs=30, batch_size=64, lr=1e-3):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f" DualEncoder [기본전처리→모달분리] | seed={seed}  device={device}")
    print(f" 비교 대상: baseline_DualEncoder_train.py (MM 전처리)")
    print(f"{'='*60}")

    os.makedirs('weights', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    src_loader, val_loader, tgt_loader, tgt_val_loader, le = get_dataloaders(batch_size)
    class_names = list(le.classes_)
    num_classes = len(class_names)
    weight_path = f'weights/dual_encoder_basicpreprocess_seed{seed}_best.pth'

    model     = DualEncoderModel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" Total params: {total_params:,}\n")

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc         = train_one_epoch(model, src_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader,     criterion, device)
        _,        tgt_acc, _, _ = evaluate(model, tgt_val_loader, criterion, device)
        scheduler.step()

        flag = ''
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), weight_path)
            flag = '  <- best'

        print(f"[{epoch:03d}/{epochs}] "
              f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f}  "
              f"val_acc={val_acc:.4f}  tgt_acc={tgt_acc:.4f}{flag}")

    print(f"\n Best val acc: {best_val_acc:.4f}  (weights: {weight_path})")
    model.load_state_dict(torch.load(weight_path, map_location=device))

    _, src_acc, src_preds, src_labels = evaluate(model, val_loader,     criterion, device)
    _, tgt_acc, tgt_preds, tgt_labels = evaluate(model, tgt_val_loader, criterion, device)

    print(f"\n{'='*60}")
    print(f" [결과 비교]")
    print(f"   기본전처리→모달분리  Source={src_acc:.4f}  Target={tgt_acc:.4f}")
    print(f"   MM전처리 baseline   Source=0.9543        Target=0.5475")
    print(f"{'='*60}")

    save_confusion_matrix(src_labels, src_preds, class_names,
                          f'results/dual_encoder_basicpreprocess_seed{seed}_source_cm.png',
                          f'DualEncoder[BasicPreprocess] Source (seed={seed}, acc={src_acc:.4f})')
    save_confusion_matrix(tgt_labels, tgt_preds, class_names,
                          f'results/dual_encoder_basicpreprocess_seed{seed}_target_cm.png',
                          f'DualEncoder[BasicPreprocess] Target (seed={seed}, acc={tgt_acc:.4f})')
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
        print("\n" + "=" * 60)
        print(" Multi-seed Summary")
        print(f"{'Seed':>6}  {'Src Acc':>8}  {'Tgt Acc':>8}")
        for s, sa, ta in results:
            print(f"{s:>6}  {sa:>8.4f}  {ta:>8.4f}")
        print(f"{'Mean':>6}  {np.mean([r[1] for r in results]):>8.4f}  "
              f"{np.mean([r[2] for r in results]):>8.4f}")
    else:
        train(seed=args.seed, epochs=args.epochs,
              batch_size=args.batch_size, lr=args.lr)
