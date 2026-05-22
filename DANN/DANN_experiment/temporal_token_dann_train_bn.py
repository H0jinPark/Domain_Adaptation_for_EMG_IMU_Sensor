"""Temporal Token DANN 학습 스크립트 (멀티모달, EMG/IMU 분리 입력).

EMG/IMU temporal token 을 cross-modal attention 으로 융합한 뒤 DANN 학습을 수행한다.
Source 라벨만 사용하고 Target 라벨은 사용하지 않는 UDA 설정이다.

Source/Target 을 concat-forward 로 한 번에 통과시켜 domain discriminator 를 학습하며,
모델 내부 BatchNorm 은 target train forward 를 통해 target 통계에 적응한다.
최종 평가 전 target train set 으로 BatchNorm running statistics 를 재보정해
AdaBN 효과를 명시적으로 활용한다.
"""
import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
from temporal_token_dann_model_bn import TemporalTokenDANN


DATA_DIR = "preprocessed_MM"


# ----------------------------------------------------------------------
# Dataset
# EMG/IMU 분리 텐서와 라벨을 담는 멀티모달 Dataset.
# ----------------------------------------------------------------------
class MMDataset(Dataset):
    def __init__(self, emg, imu, y):
        self.emg = torch.tensor(emg, dtype=torch.float32)
        self.imu = torch.tensor(imu, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.emg[idx], self.imu[idx], self.y[idx]


# ----------------------------------------------------------------------
# 데이터 로드
# preprocessed_MM/ 에서 source/target train/val 데이터를 불러온다.
# ----------------------------------------------------------------------
def load_split(prefix, le=None):
    emg = np.load(f"{DATA_DIR}/X_emg_{prefix}.npy")
    imu = np.load(f"{DATA_DIR}/X_imu_{prefix}.npy")
    y = np.load(f"{DATA_DIR}/y_{prefix}.npy", allow_pickle=True)

    if le is None:
        le = LabelEncoder()
        le.fit(y)

    return emg, imu, le.transform(y), le


def get_dataloaders(batch_size=64, num_workers=4):
    emg_tr, imu_tr, y_tr, le = load_split("train")
    emg_val, imu_val, y_val, _ = load_split("val", le)
    emg_tt, imu_tt, y_tt, _ = load_split("target_train", le)
    emg_tv, imu_tv, y_tv, _ = load_split("target_val", le)

    print(f"Classes ({len(le.classes_)}): {le.classes_}")
    print(f"Source train={len(y_tr)} val={len(y_val)}")
    print(f"Target train={len(y_tt)} val={len(y_tv)}")

    def loader(emg, imu, y, shuffle, drop_last):
        return DataLoader(
            MMDataset(emg, imu, y),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=True,
        )

    return (
        loader(emg_tr, imu_tr, y_tr, True, True),
        loader(emg_val, imu_val, y_val, False, False),
        loader(emg_tt, imu_tt, y_tt, True, True),
        loader(emg_tv, imu_tv, y_tv, False, False),
        le,
    )


# ----------------------------------------------------------------------
# 실험 유틸리티
# 난수 시드, GRL alpha, AMP context 를 관리한다.
# ----------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def grl_alpha(step, total_steps):
    p = step / max(total_steps, 1)
    return 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0


def amp_autocast(device, enabled):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.amp.autocast("cpu", enabled=False)


# ----------------------------------------------------------------------
# 평가
# validation loader 에 대해 accuracy, macro-F1, 예측값을 반환한다.
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []

    for emg, imu, y in loader:
        emg = emg.to(device, non_blocking=True)
        imu = imu.to(device, non_blocking=True)

        out, _ = model(emg, imu, alpha=0.0)
        preds.extend(out.argmax(dim=1).cpu().numpy())
        labels.extend(y.numpy())

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return acc, f1, preds, labels


# ----------------------------------------------------------------------
# AdaBN
# Target train 데이터를 라벨 없이 forward 하여 BatchNorm running statistics 를 재보정한다.
# ----------------------------------------------------------------------
@torch.no_grad()
def recalibrate_bn(model, loader, device):
    model.train()

    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()

    for emg, imu, _ in loader:
        emg = emg.to(device, non_blocking=True)
        imu = imu.to(device, non_blocking=True)
        model(emg, imu, alpha=0.0)

    model.eval()


# ----------------------------------------------------------------------
# 혼동 행렬 저장
# Source/Target validation 예측 결과를 confusion matrix 로 저장한다.
# ----------------------------------------------------------------------
def save_confusion_matrix(labels, preds, class_names, path, title):
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm)
    ax.figure.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# 학습 루프
# Source/Target concat-forward 로 DANN 을 학습한다.
# ----------------------------------------------------------------------
def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.weight_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    src_loader, val_loader, tgt_loader, tgt_val_loader, le = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    class_names = list(le.classes_)
    num_classes = len(class_names)

    model = TemporalTokenDANN(
        num_classes=num_classes,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_cross_layers=args.num_cross_layers,
        emg_stride=args.emg_stride,
        imu_stride=args.imu_stride,
        dropout=args.dropout,
    ).to(device)

    criterion_cls = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    criterion_dom = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=args.amp and device.type == "cuda",
    )

    n_iter = min(len(src_loader), len(tgt_loader))
    total_steps = args.epochs * n_iter
    global_step = 0

    best_val_acc = 0.0
    best_tgt_acc = 0.0
    weight_path = os.path.join(
        args.weight_dir,
        f"temporal_token_dann_bn_adabn_seed{args.seed}_best.pth",
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 70)
    print(f"Temporal Token DANN Training  |  seed={args.seed}  device={device}")
    print(f"Total params: {total_params:,}")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        model.train()

        src_iter = iter(src_loader)
        tgt_iter = iter(tgt_loader)

        loss_sum = 0.0
        cls_loss_sum = 0.0
        dom_loss_sum = 0.0
        dom_acc_sum = 0.0

        for _ in range(n_iter):
            src_emg, src_imu, src_y = next(src_iter)
            tgt_emg, tgt_imu, _ = next(tgt_iter)

            src_emg = src_emg.to(device, non_blocking=True)
            src_imu = src_imu.to(device, non_blocking=True)
            src_y = src_y.to(device, non_blocking=True)
            tgt_emg = tgt_emg.to(device, non_blocking=True)
            tgt_imu = tgt_imu.to(device, non_blocking=True)

            bs = src_emg.size(0)
            bt = tgt_emg.size(0)

            emg = torch.cat([src_emg, tgt_emg], dim=0)
            imu = torch.cat([src_imu, tgt_imu], dim=0)

            domain_y = torch.cat([
                torch.zeros(bs, dtype=torch.long, device=device),
                torch.ones(bt, dtype=torch.long, device=device),
            ])

            alpha = grl_alpha(global_step, total_steps)
            optimizer.zero_grad(set_to_none=True)

            with amp_autocast(device, args.amp):
                class_out, domain_out = model(emg, imu, alpha=alpha)

                loss_cls = criterion_cls(class_out[:bs], src_y)
                loss_dom = criterion_dom(domain_out, domain_y)
                loss = loss_cls + args.domain_weight * loss_dom

            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                dom_pred = domain_out.argmax(dim=1)
                dom_acc = (dom_pred == domain_y).float().mean().item()

            loss_sum += loss.item()
            cls_loss_sum += loss_cls.item()
            dom_loss_sum += loss_dom.item()
            dom_acc_sum += dom_acc
            global_step += 1

        scheduler.step()

        val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
        tgt_acc, tgt_f1, _, _ = evaluate(model, tgt_val_loader, device)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_tgt_acc = tgt_acc
            torch.save(model.state_dict(), weight_path)

        best_mark = "  (best)" if is_best else ""
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] | "
            f"Loss: {loss_sum / n_iter:.4f} | "
            f"Class: {cls_loss_sum / n_iter:.4f} | "
            f"Domain: {dom_loss_sum / n_iter:.4f} | "
            f"DomAcc: {dom_acc_sum / n_iter * 100:.2f}% | "
            f"Source Val: {val_acc * 100:.2f}% | "
            f"Source F1: {val_f1:.4f} | "
            f"Target Val: {tgt_acc * 100:.2f}% | "
            f"Target F1: {tgt_f1:.4f} | "
            f"Alpha: {alpha:.3f}"
            f"{best_mark}"
        )

    # ----------------------------------------------------------------------
    # 최종 결과 평가 및 시각화 저장
    # ----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Saving Final Results")
    print("=" * 70)

    model.load_state_dict(torch.load(weight_path, map_location=device))

    src_acc, src_f1, src_preds, src_labels = evaluate(model, val_loader, device)

    if args.adabn:
        print("Recalibrating BatchNorm statistics with target train data...")
        recalibrate_bn(model, tgt_loader, device)

    tgt_acc, tgt_f1, tgt_preds, tgt_labels = evaluate(model, tgt_val_loader, device)

    print(
        f"\n최종 결과 | "
        f"Best Source Val Acc: {best_val_acc * 100:.2f}% | "
        f"Target Acc at Best Source: {best_tgt_acc * 100:.2f}% | "
        f"Final Source Acc: {src_acc * 100:.2f}% | "
        f"Final Target Acc: {tgt_acc * 100:.2f}% | "
        f"Final Target F1: {tgt_f1:.4f}"
    )

    save_confusion_matrix(
        src_labels,
        src_preds,
        class_names,
        os.path.join(args.result_dir, f"temporal_token_dann_seed{args.seed}_source_cm.png"),
        f"Temporal Token DANN Source (acc={src_acc:.4f}, f1={src_f1:.4f})",
    )

    save_confusion_matrix(
        tgt_labels,
        tgt_preds,
        class_names,
        os.path.join(args.result_dir, f"temporal_token_dann_seed{args.seed}_target_cm.png"),
        f"Temporal Token DANN Target (acc={tgt_acc:.4f}, f1={tgt_f1:.4f})",
    )

    print("혼동 행렬 시각화가 모두 저장되었습니다.")
    return src_acc, tgt_acc


# ----------------------------------------------------------------------
# 실행 옵션
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--domain_weight", type=float, default=0.5)

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--num_cross_layers", type=int, default=3)
    parser.add_argument("--emg_stride", type=int, default=10)
    parser.add_argument("--imu_stride", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--adabn", action="store_true", default=True)

    parser.add_argument("--weight_dir", type=str, default="weights")
    parser.add_argument("--result_dir", type=str, default="results")

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
