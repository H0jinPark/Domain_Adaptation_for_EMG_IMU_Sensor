"""Learnable R 학습 스크립트 (IMU 단독 + 클래스 조건부 Contrastive 정렬, multi-seed).

MMD(분포 전체 정렬) 대신, source 와 R-정렬된 target 을 feature 공간에서 **클래스
조건부 supervised contrastive(SupCon)** 로 당긴다. 같은 (pseudo)클래스의 source-target
쌍을 positive 로 끌어당기고 다른 클래스는 밀어내, R 이 "같은 동작이면 source 좌표계로
겹치도록" 회전하게 만든다.

흐름:
    target IMU --R--> aligned IMU --> 인코더 --> tgt_feat, tgt_logits(=pseudo-label)
    source IMU --------------------> 인코더 --> src_feat, src_logits(=분류 CE)
    loss = CE(src) + λ·SupCon(src_feat ∪ tgt_feat) + 기하 prior(L_orth/L_det)

target 은 라벨이 없으므로(UDA) 분류기 예측을 pseudo-label 로 쓰되, confidence ≥ 임계만
contrastive 에 참여시킨다. pseudo-label argmax 는 detach → 라벨 배정엔 grad 가 안 흐르고,
feature 는 R 을 경유하므로 contrastive gradient 가 R 을 정렬한다.

R 은 L_orth=||RᵀR-I||, L_det=|det(R)-1| 로 회전(SO(3))에 묶는다. 데이터는
preprocessed_MM_raw 기본, 출력은 results/Learnable_R/ 의 learnable_r_cl*.
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "Multimodal"))
from mm_data_loader import get_mm_dataloaders                       # noqa: E402
from mm_utils import set_seed, save_confusion_matrix, summarize_results  # noqa: E402

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from learnable_r_model import LearnableRMMD, r_geometric_losses     # noqa: E402

WEIGHT_DIR = os.path.join(PROJECT_ROOT, "weights")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results", "Learnable_R")
R_DIR = os.path.join(PROJECT_ROOT, "results", "R_matrices")


# ----------------------------------------------------------------------
# 클래스 조건부 supervised contrastive 손실 (cross-domain)
# ----------------------------------------------------------------------
def supcon_loss(feats, labels, valid, temperature=0.1):
    """source+target 합친 feature 에 대한 SupCon 손실.

    feats : (N, d)   L2 정규화 전 feature (내부에서 정규화)
    labels: (N,)     source 는 true label, target 은 pseudo-label
    valid : (N,) bool  contrastive 에 참여할 샘플(=source 전부 + 신뢰 target)
    같은 label 의 다른 valid 샘플을 positive, 나머지 valid 샘플을 negative 로 본다.
    positive 는 도메인을 가리지 않으므로 같은 클래스 source-target 이 서로 당겨진다.
    """
    z = F.normalize(feats, dim=1)
    sim = z @ z.t() / temperature                              # (N, N)
    N = z.size(0)
    self_mask = torch.eye(N, dtype=torch.bool, device=z.device)

    valid_pair = valid.unsqueeze(0) & valid.unsqueeze(1)       # 양쪽 다 유효
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    pos_mask = same & valid_pair & ~self_mask                  # positive
    denom_mask = valid_pair & ~self_mask                       # 분모(자기 제외 유효 전부)

    sim = sim - sim.max(dim=1, keepdim=True).values.detach()   # 수치 안정
    exp_sim = torch.exp(sim) * denom_mask
    log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-12)

    pos_count = pos_mask.sum(1)
    anchor_valid = valid & (pos_count > 0)                     # positive 가 있는 유효 anchor
    if anchor_valid.sum() == 0:
        return z.sum() * 0.0                                   # 아직 positive 쌍 없음 → 0
    per_anchor = -(log_prob * pos_mask).sum(1) / pos_count.clamp(min=1)
    return per_anchor[anchor_valid].mean()


@torch.no_grad()
def evaluate_r(model, loader, device, apply_r):
    """source 는 apply_r=False, target 은 apply_r=True 로 정확도(%) 계산."""
    model.eval()
    preds, labels = [], []
    for emg, imu, y in loader:
        emg, imu = emg.to(device), imu.to(device)
        out = model(emg, imu, apply_r=apply_r)[0]
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(y.numpy())
    return accuracy_score(labels, preds) * 100, preds, labels


def _set_trainable(model, encoder_on, r_on):
    """encoder(backbone+분류기) 와 R 의 requires_grad 를 토글한다."""
    for n, p in model.named_parameters():
        p.requires_grad_(r_on if n.startswith("r.") else encoder_on)


def run_source_epoch(model, optimizer, train_loader, criterion, device, desc):
    """Phase 1: source CE 만으로 인코더+분류기 학습 (R 고정, apply_r 불필요)."""
    model.train()
    total = 0.0
    pbar = tqdm(train_loader, total=len(train_loader), desc=desc)
    for src_emg, src_imu, src_y in pbar:
        src_emg, src_imu, src_y = src_emg.to(device), src_imu.to(device), src_y.to(device)
        optimizer.zero_grad()
        out, _ = model(src_emg, src_imu, apply_r=False)
        loss = criterion(out, src_y)
        loss.backward()
        optimizer.step()
        total += loss.item()
        pbar.set_postfix({"CE": f"{loss.item():.3f}"})
    return total / max(1, len(train_loader))


def run_contrastive_epoch(model, optimizer, train_loader, tgt_train_loader, criterion,
                          device, lam, temperature, conf_thresh,
                          w_orth, w_det, w_gravity, w_norm,
                          use_ce, encoder_eval, desc):
    """contrastive 정렬 1 epoch.

    use_ce=True/encoder_eval=False  : JOINT 모드(인코더+R 동시 학습, source CE 포함).
    use_ce=False/encoder_eval=True  : Phase 2(인코더 freeze, R 만 적응). source 는
        no_grad 로 anchor 만 제공, 손실은 λ·SupCon + 기하 prior. encoder_eval 이면
        model.eval() 로 BN/dropout 을 source 통계에 고정한다(grad 는 R 로 계속 흐름).

    기하 prior = w_orth·||RᵀR-I|| + w_det·|det(R)-1| + w_gravity·||R g_t - g_s|| + w_norm·| |Ra|-|a| |.
    중력항은 라벨-free 직접 신호로 pitch/roll(2 DOF)을 잡고, contrastive(λ ramp)가
    남은 yaw 를 미세조정한다. 중력은 ramp 없이 처음부터 풀강도(r_prior 에 포함).
    """
    model.eval() if encoder_eval else model.train()
    len_loader = min(len(train_loader), len(tgt_train_loader))
    total_c, total_l, total_r, total_g = 0.0, 0.0, 0.0, 0.0
    used_tgt, seen_tgt = 0, 0
    pbar = tqdm(enumerate(zip(train_loader, tgt_train_loader)), total=len_loader, desc=desc)
    for i, ((src_emg, src_imu, src_y), (tgt_emg, tgt_imu, tgt_y)) in pbar:
        src_emg, src_imu, src_y = src_emg.to(device), src_imu.to(device), src_y.to(device)
        tgt_emg, tgt_imu = tgt_emg.to(device), tgt_imu.to(device)
        optimizer.zero_grad()

        if use_ce:
            src_out, src_feat = model(src_emg, src_imu, apply_r=False)
            loss_cls = criterion(src_out, src_y)
        else:
            with torch.no_grad():                 # 인코더 freeze: source 경로 grad 불필요
                _, src_feat = model(src_emg, src_imu, apply_r=False)
            loss_cls = torch.zeros((), device=device)

        # target: R 적용, feature + pseudo-label
        tgt_out, tgt_feat = model(tgt_emg, tgt_imu, apply_r=True)
        with torch.no_grad():                     # pseudo-label argmax 는 detach
            conf, pseudo = F.softmax(tgt_out, dim=1).max(1)
            tgt_keep = conf >= conf_thresh
        seen_tgt += tgt_keep.numel()
        used_tgt += int(tgt_keep.sum())

        feats = torch.cat([src_feat, tgt_feat], dim=0)
        labels = torch.cat([src_y, pseudo], dim=0)
        valid = torch.cat([
            torch.ones(src_feat.size(0), dtype=torch.bool, device=device), tgt_keep], dim=0)
        loss_cl = supcon_loss(feats, labels, valid, temperature)

        L_orth, L_det, L_grav, L_norm = r_geometric_losses(model.r.R, src_imu, tgt_imu)
        r_prior = w_orth * L_orth + w_det * L_det + w_gravity * L_grav + w_norm * L_norm

        loss = loss_cls + lam * loss_cl + r_prior
        loss.backward()
        optimizer.step()

        cl_val, ce_val, r_val, g_val = loss_cl.item(), loss_cls.item(), r_prior.item(), L_grav.item()
        total_c += ce_val
        total_l += cl_val
        total_r += r_val
        total_g += g_val
        pbar.set_postfix({"CE": f"{ce_val:.3f}", "CL": f"{cl_val:.4f}",
                          "Rprior": f"{r_val:.3f}", "Lgrav": f"{g_val:.3f}"})

    n = max(1, len_loader)
    return {"ce": total_c / n, "cl": total_l / n, "rprior": total_r / n,
            "grav": total_g / n, "pseudo_rate": 100.0 * used_tgt / max(1, seen_tgt)}


def train(seed=42, epochs=60, batch_size=64, lr=1e-3, r_lr=1e-2, lambda_cl=1.0,
          temperature=0.1, conf_thresh=0.9, cl_warmup=5,
          freeze_encoder=False, phase1_epochs=15,
          w_orth=1.0, w_det=1.0, w_gravity=1.0, w_norm=0.0, save_cm=False, tag=""):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(WEIGHT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(R_DIR, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    save_path = os.path.join(WEIGHT_DIR, f"learnable_r_cl{suffix}_seed{seed}_best_model.pth")

    train_loader, val_loader, tgt_train_loader, tgt_val_loader, num_classes, le = \
        get_mm_dataloaders(batch_size=batch_size)
    class_names = le.classes_

    model = LearnableRMMD(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print("\n" + "=" * 60)
    print(f"Learnable-R Contrastive Training Start  |  seed={seed}")
    if freeze_encoder:
        print(f"Mode: TWO-PHASE | P1 source-only pretrain {phase1_epochs}ep "
              f"-> P2 freeze encoder, adapt R only {epochs - phase1_epochs}ep")
    else:
        print("Mode: JOINT | target IMU -> R -> encoder, align by cross-domain SupCon")
    print(f"lambda_cl={lambda_cl} temp={temperature} conf_thresh={conf_thresh} "
          f"warmup={cl_warmup} | R-prior: orth={w_orth} det={w_det} "
          f"gravity={w_gravity} norm={w_norm}")
    print(f"Targeting {num_classes} classes on {device}")
    print("=" * 60)

    best_val_acc, best_target_acc = 0.0, 0.0

    def eval_and_log(stage, e, E, extra=""):
        nonlocal best_val_acc, best_target_acc
        val_acc, _, _ = evaluate_r(model, val_loader, device, apply_r=False)
        tgt_acc, _, _ = evaluate_r(model, tgt_val_loader, device, apply_r=True)
        is_best = tgt_acc > best_target_acc
        if is_best:
            best_val_acc, best_target_acc = val_acc, tgt_acc
            torch.save(model.state_dict(), save_path)
        with torch.no_grad():
            Rm = model.r.R.detach()
            det = torch.linalg.det(Rm).item()
            orth = torch.linalg.norm(Rm.transpose(0, 1) @ Rm - torch.eye(3, device=device)).item()
        print(f"[{stage} {e:02d}/{E:02d}] {extra} | det(R): {det:.3f} | ||RᵀR-I||: {orth:.3f} | "
              f"Source Val: {val_acc:.2f}% | Target Val: {tgt_acc:.2f}%{'  (best)' if is_best else ''}")

    if not freeze_encoder:
        # ---- JOINT: 인코더 + R 동시 학습 ----
        r_params = list(model.r.parameters())
        other_params = [p for n, p in model.named_parameters() if not n.startswith("r.")]
        optimizer = optim.AdamW([
            {"params": other_params, "lr": lr, "weight_decay": 1e-3},
            {"params": r_params, "lr": r_lr, "weight_decay": 0.0},
        ])
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        for epoch in range(epochs):
            lam = lambda_cl * min(1.0, (epoch + 1) / max(1, cl_warmup))
            s = run_contrastive_epoch(model, optimizer, train_loader, tgt_train_loader, criterion,
                                      device, lam, temperature, conf_thresh,
                                      w_orth, w_det, w_gravity, w_norm,
                                      use_ce=True, encoder_eval=False,
                                      desc=f"Seed {seed} | Epoch [{epoch+1:02d}/{epochs:02d}]")
            scheduler.step()
            eval_and_log("Epoch", epoch + 1, epochs,
                         extra=f"CE: {s['ce']:.4f} | CL: {s['cl']:.4f} (λ={lam:.2f}) | "
                               f"Rprior: {s['rprior']:.4f} | Lgrav: {s['grav']:.4f} | "
                               f"pseudo≥{conf_thresh}: {s['pseudo_rate']:.1f}%")
    else:
        # ---- TWO-PHASE ----
        assert 0 < phase1_epochs < epochs, "phase1_epochs 는 1..epochs-1 사이여야 함"
        # Phase 1: source-only 인코더 pretrain (R 항등 고정)
        _set_trainable(model, encoder_on=True, r_on=False)
        p1_params = [p for n, p in model.named_parameters() if not n.startswith("r.")]
        opt1 = optim.AdamW(p1_params, lr=lr, weight_decay=1e-3)
        sch1 = CosineAnnealingLR(opt1, T_max=phase1_epochs)
        for epoch in range(phase1_epochs):
            ce = run_source_epoch(model, opt1, train_loader, criterion, device,
                                  desc=f"Seed {seed} | P1 [{epoch+1:02d}/{phase1_epochs:02d}]")
            sch1.step()
            eval_and_log("P1", epoch + 1, phase1_epochs, extra=f"CE: {ce:.4f}")

        # Phase 2: 인코더·분류기 freeze, R 만 contrastive 로 적응
        _set_trainable(model, encoder_on=False, r_on=True)
        opt2 = optim.AdamW(model.r.parameters(), lr=r_lr, weight_decay=0.0)
        phase2_epochs = epochs - phase1_epochs
        sch2 = CosineAnnealingLR(opt2, T_max=max(1, phase2_epochs))
        print(f"--- Phase 2 시작: 인코더 freeze, R 만 학습 (r_lr={r_lr}) ---")
        for j in range(phase2_epochs):
            lam = lambda_cl * min(1.0, (j + 1) / max(1, cl_warmup))
            s = run_contrastive_epoch(model, opt2, train_loader, tgt_train_loader, criterion,
                                      device, lam, temperature, conf_thresh,
                                      w_orth, w_det, w_gravity, w_norm,
                                      use_ce=False, encoder_eval=True,
                                      desc=f"Seed {seed} | P2 [{j+1:02d}/{phase2_epochs:02d}]")
            sch2.step()
            eval_and_log("P2", j + 1, phase2_epochs,
                         extra=f"CL: {s['cl']:.4f} (λ={lam:.2f}) | Rprior: {s['rprior']:.4f} | "
                               f"Lgrav: {s['grav']:.4f} | pseudo≥{conf_thresh}: {s['pseudo_rate']:.1f}%")

    print(f"\n최종 결과 | seed={seed} | Best Source Val: {best_val_acc:.2f}% | "
          f"Target at Best: {best_target_acc:.2f}% | Shift: {best_val_acc - best_target_acc:.2f}%")

    model.load_state_dict(torch.load(save_path, map_location=device))
    R_best = model.r.R.detach().cpu().numpy()
    r_path = os.path.join(R_DIR, f"R_learned_cl{suffix}_seed{seed}.npy")
    np.save(r_path, R_best)
    print(f"학습된 R 저장: {r_path}\n{np.array2string(R_best, precision=4)}")

    if save_cm:
        _, v_preds, v_true = evaluate_r(model, val_loader, device, apply_r=False)
        _, t_preds, t_true = evaluate_r(model, tgt_val_loader, device, apply_r=True)
        save_confusion_matrix(
            v_true, v_preds, class_names,
            os.path.join(RESULT_DIR, f"learnable_r_cl{suffix}_seed{seed}_source_cm.png"),
            f"Learnable-R CL Source (seed={seed}, Val: {best_val_acc:.1f}%)", cmap="Greens")
        save_confusion_matrix(
            t_true, t_preds, class_names,
            os.path.join(RESULT_DIR, f"learnable_r_cl{suffix}_seed{seed}_target_cm.png"),
            f"Learnable-R CL Target (seed={seed}, Src: {best_val_acc:.1f}% vs Tgt: {best_target_acc:.1f}%)",
            cmap="Greens")
        print("혼동 행렬 시각화 저장 완료.")

    return {"seed": seed, "source_acc": best_val_acc, "target_acc": best_target_acc,
            "shift": best_val_acc - best_target_acc}


def write_result_json(results, tag):
    src = [r["source_acc"] for r in results]
    tgt = [r["target_acc"] for r in results]
    sh = [r["shift"] for r in results]
    ddof = 1 if len(results) > 1 else 0
    payload = {
        "tag": tag or "default", "modality": "imu_only", "model": "learnable_r_cl",
        "data_dir": os.environ.get("MM_DATA_DIR", "preprocessed_MM_raw"),
        "seeds": [r["seed"] for r in results], "results": results,
        "mean": {"source_acc": float(np.mean(src)), "target_acc": float(np.mean(tgt)),
                 "shift": float(np.mean(sh))},
        "std": {"source_acc": float(np.std(src, ddof=ddof)), "target_acc": float(np.std(tgt, ddof=ddof)),
                "shift": float(np.std(sh, ddof=ddof))},
    }
    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, f"learnable_r_cl_result_{tag or 'default'}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"결과 JSON 저장: {path}")
    return path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi_seed", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--r_lr", type=float, default=1e-2, help="learnable R 전용 학습률")
    parser.add_argument("--lambda_cl", type=float, default=1.0, help="contrastive 가중치")
    parser.add_argument("--temperature", type=float, default=0.1, help="SupCon 온도")
    parser.add_argument("--conf_thresh", type=float, default=0.9,
                        help="target pseudo-label 채택 confidence 임계")
    parser.add_argument("--cl_warmup", type=int, default=5,
                        help="contrastive 가중치 선형 ramp epoch 수(초기 pseudo-label 노이즈 완화)")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="2단계: source 로 인코더 pretrain 후 freeze, R 만 contrastive 로 적응")
    parser.add_argument("--phase1_epochs", type=int, default=15,
                        help="(freeze_encoder 시) source-only 인코더 pretrain epoch 수, 나머지는 Phase 2")
    parser.add_argument("--w_orth", type=float, default=1.0, help="||RᵀR-I|| 가중치")
    parser.add_argument("--w_det", type=float, default=1.0, help="|det(R)-1| 가중치")
    parser.add_argument("--w_gravity", type=float, default=1.0,
                        help="||R g_t - g_s|| 가중치(라벨-free 중력 정렬, isotropic 데이터에서 부활). "
                             "pitch/roll 2 DOF 를 잡는 메인 신호. zscore 데이터면 0 권장")
    parser.add_argument("--w_norm", type=float, default=0.0,
                        help="| |Ra|-|a| | 가중치(크기 보존, 직교성과 중복 → 기본 0)")
    parser.add_argument("--no_cm", action="store_true")
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    suffix = f"_{args.tag}" if args.tag else ""
    kw = dict(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, r_lr=args.r_lr,
              lambda_cl=args.lambda_cl, temperature=args.temperature,
              conf_thresh=args.conf_thresh, cl_warmup=args.cl_warmup,
              freeze_encoder=args.freeze_encoder, phase1_epochs=args.phase1_epochs,
              w_orth=args.w_orth, w_det=args.w_det, w_gravity=args.w_gravity, w_norm=args.w_norm,
              save_cm=not args.no_cm, tag=args.tag)

    if args.multi_seed:
        results = [train(seed=s, **kw) for s in args.seeds]
        os.makedirs(RESULT_DIR, exist_ok=True)
        summarize_results(results, method_name=f"Learnable-R CL (IMU-only){' | '+args.tag if args.tag else ''}",
                          save_path=os.path.join(RESULT_DIR, f"learnable_r_cl{suffix}_summary.txt"))
    else:
        results = [train(seed=args.seed, **kw)]

    write_result_json(results, args.tag)
