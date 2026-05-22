"""DANN-MM-SSL 학습 스크립트 (2-phase: 모달별 contrastive 사전학습 → 융합 DANN).

Phase 1 - EMG/IMU 인코더를 라벨 없이 SimCLR 식 contrastive 로 각각 사전학습한다.
          source_train + target_train 을 합쳐 쓰므로 target 라벨을 보지 않고도
          target 분포를 사전학습에 반영한다 (source bias 제거가 목적).
Phase 2 - 사전학습 인코더를 DualEncoderDANN_Sep 에 이식하고 GRL 도메인 적대 학습.
          Phase 2 는 DANN_MM_sep 과 동일하다 (Phase 1 만 다른 ablation).
"""
import os
import sys
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 프로젝트 루트를 import 경로에 추가 (DANN/DANN_experiment/ → ../../)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from DANN.DANN_experiment.DANN_MM_ssl_model import ContrastiveModel, nt_xent_loss
from DANN.DANN_experiment.DANN_MM_sep_model import DualEncoderDANN_Sep
from baseline.baseline_DualEncoder import EMGEncoder, IMUEncoder

DATA_DIR = 'preprocessed_MM'


class MMDataset(Dataset):
    """EMG/IMU 분리 텐서와 라벨을 담는 멀티모달 Dataset (Phase 2 용)."""

    def __init__(self, emg, imu, y):
        self.emg = torch.tensor(emg, dtype=torch.float32)  # (N, 2, 5000)
        self.imu = torch.tensor(imu, dtype=torch.float32)  # (N, 3,  500)
        self.y   = torch.tensor(y,   dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.emg[idx], self.imu[idx], self.y[idx]


class SSLDataset(Dataset):
    """라벨 없는 EMG/IMU 윈도우 Dataset (Phase 1 contrastive 용)."""

    def __init__(self, emg, imu):
        self.emg = torch.tensor(emg, dtype=torch.float32)
        self.imu = torch.tensor(imu, dtype=torch.float32)

    def __len__(self):
        return len(self.emg)

    def __getitem__(self, idx):
        return self.emg[idx], self.imu[idx]


def load_split(prefix, le=None):
    """preprocessed_MM/ 에서 한 split 의 EMG/IMU/label 을 로드한다."""
    emg = np.load(f'{DATA_DIR}/X_emg_{prefix}.npy')
    imu = np.load(f'{DATA_DIR}/X_imu_{prefix}.npy')
    y   = np.load(f'{DATA_DIR}/y_{prefix}.npy', allow_pickle=True)
    if le is None:
        le = LabelEncoder()
        le.fit(y)
    return emg, imu, le.transform(y), le


def get_dataloaders(batch_size, ssl_batch_size):
    """Phase 2 용 4분할 라벨 로더 + Phase 1 용 무라벨 SSL 로더 + LabelEncoder 반환."""
    emg_tr, imu_tr, y_tr, le   = load_split('train')
    emg_val, imu_val, y_val, _ = load_split('val',          le)
    emg_tt, imu_tt, y_tt, _    = load_split('target_train', le)
    emg_tv, imu_tv, y_tv, _    = load_split('target_val',   le)

    print(f"Classes ({len(le.classes_)}): {le.classes_}")
    print(f"  Source  train={len(y_tr)}  val={len(y_val)}")
    print(f"  Target  train={len(y_tt)}  val={len(y_tv)}")

    def loader(ds, shuffle, drop_last, bs):
        return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                          drop_last=drop_last, num_workers=4)

    # Phase 1: source_train + target_train 을 라벨 없이 합친 contrastive 풀
    # (target_val 은 Phase 2 평가 누수 방지로 제외)
    ssl_emg = np.concatenate([emg_tr, emg_tt], axis=0)
    ssl_imu = np.concatenate([imu_tr, imu_tt], axis=0)
    ssl_loader = loader(SSLDataset(ssl_emg, ssl_imu), True, True, ssl_batch_size)
    print(f"  SSL pool (src+tgt train, unlabeled) = {len(ssl_emg)}")

    return (loader(MMDataset(emg_tr,  imu_tr,  y_tr),  True,  True,  batch_size),
            loader(MMDataset(emg_val, imu_val, y_val), False, False, batch_size),
            loader(MMDataset(emg_tt,  imu_tt,  y_tt),  True,  True,  batch_size),
            loader(MMDataset(emg_tv,  imu_tv,  y_tv),  False, False, batch_size),
            ssl_loader, le)


def set_seed(seed):
    """난수 시드를 고정해 실험 재현성을 확보한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------
# 시계열 augmentation (contrastive 뷰 생성용, 전부 (B,C,T) 배치 텐서 GPU 연산)
# -----------------------------------------------------------------------
def _smooth_curve(b, c, t, knots, device):
    """knots 개의 난수를 길이 t 로 선형 보간한 매끄러운 곡선 (b, c, t)."""
    lowres = torch.randn(b, c, knots, device=device)
    return F.interpolate(lowres, size=t, mode='linear', align_corners=True)


def jitter(x, sigma=0.05):
    """가우시안 노이즈 추가."""
    return x + torch.randn_like(x) * sigma


def scaling(x, sigma=0.1):
    """샘플·채널별 곱셈 스케일."""
    factor = torch.randn(x.size(0), x.size(1), 1, device=x.device) * sigma + 1.0
    return x * factor


def magnitude_warp(x, sigma=0.2, knots=4):
    """매끄러운 곡선을 곱해 진폭을 시간에 따라 왜곡."""
    curve = _smooth_curve(x.size(0), x.size(1), x.size(2), knots, x.device)
    return x * (curve * sigma + 1.0)


def time_warp(x, sigma=0.2, knots=4):
    """시간축을 매끄럽게 왜곡 (grid_sample 로 재샘플링)."""
    b, c, t = x.shape
    offset = _smooth_curve(b, 1, t, knots, x.device) * sigma             # (b,1,t)
    base = torch.linspace(-1, 1, t, device=x.device).view(1, 1, t)
    gx = (base + offset).clamp(-1, 1).squeeze(1)                         # (b,t)
    grid = torch.stack([gx, torch.zeros_like(gx)], dim=-1).unsqueeze(1)  # (b,1,t,2)
    out = F.grid_sample(x.unsqueeze(2), grid, mode='bilinear',
                        padding_mode='border', align_corners=True)
    return out.squeeze(2)


def crop_resize(x, min_ratio=0.5):
    """랜덤 구간을 잘라 원래 길이로 리샘플."""
    t = x.size(2)
    length = max(2, int(t * random.uniform(min_ratio, 1.0)))
    start = random.randint(0, t - length)
    cropped = x[:, :, start:start + length]
    return F.interpolate(cropped, size=t, mode='linear', align_corners=True)


def channel_mask(x, p=0.3):
    """채널을 확률 p 로 0 마스킹 (최소 1채널은 보존)."""
    if x.size(1) == 1:
        return x
    mask = (torch.rand(x.size(0), x.size(1), 1, device=x.device) > p).float()
    all_off = mask.sum(dim=1, keepdim=True) == 0
    mask = torch.where(all_off, torch.ones_like(mask), mask)
    return x * mask


def augment(x):
    """augmentation 들을 확률적으로 합성해 contrastive 뷰 하나를 만든다.

    확률은 배치 단위로 적용된다 (한 배치 = 같은 augmentation 조합, 단 파라미터는
    샘플별 랜덤). 두 번 호출하면 서로 다른 두 뷰가 된다.
    """
    if random.random() < 0.8: x = jitter(x)
    if random.random() < 0.8: x = scaling(x)
    if random.random() < 0.5: x = magnitude_warp(x)
    if random.random() < 0.5: x = time_warp(x)
    if random.random() < 0.5: x = crop_resize(x)
    if random.random() < 0.3: x = channel_mask(x)
    return x


# -----------------------------------------------------------------------
# 평가 헬퍼 (Phase 2)
# -----------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    """loader 전체에 대한 정확도와 (예측, 정답) 리스트를 반환한다."""
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
# Phase 1: EMG / IMU 인코더 모달별 contrastive 사전학습
# -----------------------------------------------------------------------
def pretrain_contrastive(seed, epochs, lr, temperature, device, ssl_loader):
    """EMG/IMU 인코더를 모달별 SimCLR contrastive 로 사전학습하고 encoder 를 저장한다."""

    print(f"\n{'='*55}")
    print(f" Phase 1: Contrastive Pre-training  |  epochs={epochs}  temp={temperature}")
    print(f"{'='*55}")

    emg_ssl = ContrastiveModel(EMGEncoder()).to(device)
    imu_ssl = ContrastiveModel(IMUEncoder()).to(device)

    opt_emg = optim.AdamW(emg_ssl.parameters(), lr=lr, weight_decay=1e-4)
    opt_imu = optim.AdamW(imu_ssl.parameters(), lr=lr, weight_decay=1e-4)
    sch_emg = CosineAnnealingLR(opt_emg, T_max=epochs)
    sch_imu = CosineAnnealingLR(opt_imu, T_max=epochs)

    emg_path = f'weights/dann_mm_ssl_emg_encoder_seed{seed}.pth'
    imu_path = f'weights/dann_mm_ssl_imu_encoder_seed{seed}.pth'

    for epoch in range(1, epochs + 1):
        emg_ssl.train()
        imu_ssl.train()
        emg_loss_sum = imu_loss_sum = 0.0
        n = len(ssl_loader)

        for emg, imu in ssl_loader:
            emg, imu = emg.to(device), imu.to(device)
            b = emg.size(0)

            # EMG contrastive: 두 뷰를 한 배치(2B)로 묶어 인코딩 (BN 통계 일관)
            opt_emg.zero_grad()
            z = emg_ssl(torch.cat([augment(emg), augment(emg)], dim=0))
            loss_emg = nt_xent_loss(z[:b], z[b:], temperature)
            loss_emg.backward()
            opt_emg.step()
            emg_loss_sum += loss_emg.item()

            # IMU contrastive
            opt_imu.zero_grad()
            z = imu_ssl(torch.cat([augment(imu), augment(imu)], dim=0))
            loss_imu = nt_xent_loss(z[:b], z[b:], temperature)
            loss_imu.backward()
            opt_imu.step()
            imu_loss_sum += loss_imu.item()

        sch_emg.step()
        sch_imu.step()

        print(
            f"PreTrain [{epoch:03d}/{epochs:03d}] | "
            f"EMG CL Loss: {emg_loss_sum/n:.4f} | "
            f"IMU CL Loss: {imu_loss_sum/n:.4f}"
        )

    # contrastive 사전학습은 검증 라벨이 없어 마지막 epoch encoder 를 보존한다
    torch.save(emg_ssl.encoder.state_dict(), emg_path)
    torch.save(imu_ssl.encoder.state_dict(), imu_path)
    print(f"\n Saved EMG encoder → {emg_path}")
    print(f" Saved IMU encoder → {imu_path}")
    return emg_path, imu_path


# -----------------------------------------------------------------------
# Phase 2: 융합 DANN 파인튜닝 (DANN_MM_sep 와 동일)
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

    weight_path  = f'weights/dann_mm_ssl_seed{seed}_best.pth'
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
                          f'results/dann_mm_ssl_seed{seed}_source_cm.png',
                          f'DANN-MM-SSL Source (seed={seed}, acc={src_acc:.4f})')
    save_confusion_matrix(tgt_labels, tgt_preds, class_names,
                          f'results/dann_mm_ssl_seed{seed}_target_cm.png',
                          f'DANN-MM-SSL Target (seed={seed}, acc={tgt_acc:.4f})')
    return src_acc, tgt_acc


# -----------------------------------------------------------------------
# 전체 학습: Phase 1 → Phase 2
# -----------------------------------------------------------------------
def train(seed=42, ssl_epochs=50, finetune_epochs=30,
          ssl_batch_size=128, batch_size=64, ssl_lr=1e-3, lr=1e-3, temperature=0.5):
    """단일 seed 로 2-phase 학습을 실행하고 Source/Target 최종 정확도를 반환한다."""
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*55}")
    print(f" DANN-MM-SSL Training  |  seed={seed}  device={device}")
    print(f"{'='*55}")

    os.makedirs('weights', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    (src_loader, val_loader, tgt_loader, tgt_val_loader,
     ssl_loader, le) = get_dataloaders(batch_size, ssl_batch_size)
    class_names = list(le.classes_)
    num_classes = len(class_names)

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
