"""
비교 실험: 기본 전처리 데이터 → 모달리티 분리 → DANN

목적:
  DANN_MM_train.py (MM 전처리) 와 동일한 모델·학습 루프를 사용하되
  기본 전처리(preprocessed/) 데이터로 학습해서 전처리 방식의 영향 확인.

  기본 전처리 (N, 5000, 5) 로드
    → EMG: ch 0,1  → (N, 2, 5000)
    → IMU: ch 2,3,4 → 10배 다운샘플 → (N, 3, 500)
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

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_ROOT)
sys.path.append(os.path.join(_ROOT, 'baseline'))
from DANN.DANN_MM_model import DualEncoderDANN

DATA_DIR = 'preprocessed'
IMU_STEP = 10   # 1000Hz → 100Hz


# =====================================================================
# 모달리티 분리
# =====================================================================
def split_modalities(X):
    """
    X: (N, 5000, 5) — [biceps, triceps, triceps_X, triceps_Y, triceps_Z]
    returns:
      emg: (N, 2, 5000)
      imu: (N, 3,  500)
    """
    emg = X[:, :, :2].transpose(0, 2, 1).astype(np.float32)
    imu = X[:, ::IMU_STEP, 2:].transpose(0, 2, 1).astype(np.float32)
    return emg, imu


def load_split(prefix, le=None):
    X = np.load(f'{DATA_DIR}/X_{prefix}.npy')
    y = np.load(f'{DATA_DIR}/y_{prefix}.npy', allow_pickle=True)
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
    print(f"  EMG: {emg_tr.shape}  IMU: {imu_tr.shape}")

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
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for emg, imu, y in loader:
        emg, imu = emg.to(device), imu.to(device)
        out, _ = model(emg, imu, alpha=0.0)
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(y.numpy())
    return accuracy_score(labels, preds), preds, labels


def save_confusion_matrix(labels, preds, class_names, path, title):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def train(seed=42, epochs=30, batch_size=64, lr=1e-3):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f" DANN-MM [기본전처리→모달분리] | seed={seed}  device={device}")
    print(f" 비교 대상: DANN_MM_train.py (MM 전처리)")
    print(f"{'='*60}")

    os.makedirs('weights', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    src_loader, val_loader, tgt_loader, tgt_val_loader, le = get_dataloaders(batch_size)
    class_names = list(le.classes_)
    num_classes = len(class_names)
    weight_path = f'weights/dann_mm_basicpreprocess_seed{seed}_best.pth'

    model         = DualEncoderDANN(num_classes=num_classes).to(device)
    criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_dom = nn.CrossEntropyLoss()
    optimizer     = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler     = CosineAnnealingLR(optimizer, T_max=epochs)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" Total params: {total_params:,}\n")

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

        flag = ''
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), weight_path)
            flag = '  <- best'

        print(f"[{epoch:03d}/{epochs}] alpha={alpha:.3f}  "
              f"cls={cls_loss_sum/n_iter:.4f}  dom={dom_loss_sum/n_iter:.4f}  "
              f"val={val_acc:.4f}  tgt={tgt_acc:.4f}{flag}")

    print(f"\n Best val acc: {best_val_acc:.4f}  (weights: {weight_path})")
    model.load_state_dict(torch.load(weight_path, map_location=device))

    src_acc, src_preds, src_labels = evaluate(model, val_loader,     device)
    tgt_acc, tgt_preds, tgt_labels = evaluate(model, tgt_val_loader, device)

    print(f"\n{'='*60}")
    print(f" [결과 비교]")
    print(f"   기본전처리→모달분리  Source={src_acc:.4f}  Target={tgt_acc:.4f}")
    print(f"   MM전처리 DANN        Source=0.9400        Target=0.6674")
    print(f"{'='*60}")

    save_confusion_matrix(src_labels, src_preds, class_names,
                          f'results/dann_mm_basicpreprocess_seed{seed}_source_cm.png',
                          f'DANN-MM[BasicPreprocess] Source (seed={seed}, acc={src_acc:.4f})')
    save_confusion_matrix(tgt_labels, tgt_preds, class_names,
                          f'results/dann_mm_basicpreprocess_seed{seed}_target_cm.png',
                          f'DANN-MM[BasicPreprocess] Target (seed={seed}, acc={tgt_acc:.4f})')
    return src_acc, tgt_acc


if __name__ == "__main__":
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
