import os
import sys
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import get_dataloaders
from DANN.DANN_MM_split_model import SplitDualEncoderDANN


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        x = x.to(device)
        out, _ = model(x, alpha=0.0)
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(y.numpy())
    return accuracy_score(labels, preds), preds, labels


def save_confusion_matrix(labels, preds, class_names, path, title):
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


def train(seed=42, epochs=30, batch_size=64, lr=1e-3):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f" DANN-MM Split Training  |  seed={seed}  device={device}")
    print(f"{'='*60}")

    os.makedirs('weights', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    src_loader, val_loader, tgt_loader, tgt_val_loader, num_classes, le = get_dataloaders(batch_size)
    class_names = list(le.classes_)
    weight_path = f'weights/dann_mm_split_seed{seed}_best.pth'

    model         = SplitDualEncoderDANN(num_classes=num_classes).to(device)
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

        cls_sum = dom_sum = 0.0

        for _ in range(n_iter):
            src_x, src_y = next(src_iter)
            tgt_x, _     = next(tgt_iter)

            src_x, src_y = src_x.to(device), src_y.to(device)
            tgt_x        = tgt_x.to(device)

            src_dom = torch.zeros(src_x.size(0), dtype=torch.long, device=device)
            tgt_dom = torch.ones (tgt_x.size(0), dtype=torch.long, device=device)

            optimizer.zero_grad()

            src_cls, src_dom_out = model(src_x, alpha)
            _,       tgt_dom_out = model(tgt_x, alpha)

            loss_cls = criterion_cls(src_cls, src_y)
            loss_dom = criterion_dom(src_dom_out, src_dom) + criterion_dom(tgt_dom_out, tgt_dom)
            loss     = loss_cls + loss_dom

            loss.backward()
            optimizer.step()

            cls_sum += loss_cls.item()
            dom_sum += loss_dom.item()

        scheduler.step()

        val_acc, _, _ = evaluate(model, val_loader,     device)
        tgt_acc, _, _ = evaluate(model, tgt_val_loader, device)

        flag = ''
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), weight_path)
            flag = '  <- best'

        print(f"[{epoch:03d}/{epochs}] alpha={alpha:.3f}  "
              f"cls={cls_sum/n_iter:.4f}  dom={dom_sum/n_iter:.4f}  "
              f"val={val_acc:.4f}  tgt={tgt_acc:.4f}{flag}")

    print(f"\n Best val acc: {best_val_acc:.4f}  (weights: {weight_path})")
    model.load_state_dict(torch.load(weight_path, map_location=device))

    src_acc, src_preds, src_labels = evaluate(model, val_loader,     device)
    tgt_acc, tgt_preds, tgt_labels = evaluate(model, tgt_val_loader, device)

    print(f" Final Source Acc : {src_acc:.4f}")
    print(f" Final Target Acc : {tgt_acc:.4f}")

    save_confusion_matrix(src_labels, src_preds, class_names,
                          f'results/dann_mm_split_seed{seed}_source_cm.png',
                          f'DANN-MM Split Source (seed={seed}, acc={src_acc:.4f})')
    save_confusion_matrix(tgt_labels, tgt_preds, class_names,
                          f'results/dann_mm_split_seed{seed}_target_cm.png',
                          f'DANN-MM Split Target (seed={seed}, acc={tgt_acc:.4f})')
    return src_acc, tgt_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--multi_seed', action='store_true')
    parser.add_argument('--seeds',      type=int,   nargs='+', default=[0, 1, 2, 3])
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
        print(f"{'Mean':>6}  {np.mean([r[1] for r in results]):>8.4f}  {np.mean([r[2] for r in results]):>8.4f}")
    else:
        train(seed=args.seed, epochs=args.epochs,
              batch_size=args.batch_size, lr=args.lr)
