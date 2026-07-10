"""멀티모달(EMG+IMU) align-first learnable-R 학습.

IMU 단독 align-first(learnable_r_cdan_alignfirst_train.py, 현재 최고 재현성 세팅)를
**멀티모달로 확장**한 판. 예전에 사전학습 R(PCA/Kabsch/gravity 등)로 target IMU 를
정렬한 뒤 EMG+IMU 를 함께 넣어 멀티모달 추론하던 구조를, 그 R 을 **학습 가능한 3x3
SO(3) 회전**으로 대체한다. 모델은 LearnableRMMCDAN(InterFusion EMG+IMU + IMU 앞단 R).

핵심 구조(IMU 단독판과 동일):
  · R 은 target IMU 3축에만 적용(apply_r), EMG 는 회전 없음. source 는 apply_r=False.
  · gravity/pca 기하 prior 는 raw IMU + R 로 계산(인코더 무관) → R 정렬.
  · CDAN 은 융합 feature(EMG+IMU)에 조건부 적대 정렬 → 인코더 target 적응.

align-first 2단계(유저 아이디어, 재현성 해결):
  · 1단계(epoch < --target_join_epoch): target 은 인코더/판별기 미경유, gravity/pca 로
    R 만 정렬. source 만 CE. 인코더/BN 이 순수 source 로 형성.
  · 2단계: target 인코더+판별기 합류, CDAN 적응. α 는 합류 시점부터 0→1 ramp.

★ 보고 지표(leakage 주의):
  - best_target : target-val 정확도 peak 로 고른 checkpoint(=IMU 단독 실험과 동일 규약,
                  직접 비교용). **target 라벨로 epoch 를 고르는 oracle 선택이라 낙관 편향**.
  - last{N}_mean: 마지막 N epoch target 정확도 평균/표준편차(--report_last, 기본 10).
                  target 라벨로 epoch 를 고르지 않는 **leakage-free 참고값**. 둘 다 저장한다.

실행 예:
    MM_DATA_DIR=preprocessed_MM_raw_isotropic python Learnable_R/learnable_r_mm_alignfirst_train.py \
        --multi_seed --epochs 60 --target_join_epoch 10 --tag mm_join10
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
from sklearn.metrics import accuracy_score
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "Multimodal"))
from mm_data_loader import get_mm_dataloaders                       # noqa: E402
from mm_utils import set_seed, save_confusion_matrix, summarize_results  # noqa: E402

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from learnable_r_mm_model import LearnableRMMCDAN            # noqa: E402
from learnable_r_model import gravity_loss, pca_alignment_loss  # noqa: E402

WEIGHT_DIR = os.path.join(PROJECT_ROOT, "weights")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results", "Learnable_R")
R_DIR = os.path.join(PROJECT_ROOT, "results", "R_matrices")
NAME = "learnable_r_mm_alignfirst"


def pick_device():
    """CUDA → MPS(Apple Silicon) → CPU 순으로 사용 가능한 device 선택."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate_r(model, loader, device, apply_r):
    """source 는 apply_r=False, target 은 apply_r=True 로 정확도(%) 계산 (EMG+IMU 사용)."""
    model.eval()
    preds, labels = [], []
    for emg, imu, y in loader:
        emg, imu = emg.to(device), imu.to(device)
        out = model(emg, imu, alpha=0.0, apply_r=apply_r)[0]
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(y.numpy())
    return accuracy_score(labels, preds) * 100, preds, labels


def run_epoch(model, optimizer, train_loader, tgt_train_loader, criterion, criterion_domain,
              device, alpha, lam_da, lam_g, lam_pca, target_align_only=False,
              log_r_grads=True, desc=""):
    """통합 손실 1 epoch. target_align_only=True 면 1단계(target 인코더 미경유).

    1단계: L = CE(src) + λ_g·L_gravity + λ_pca·L_pca  (target 은 R 정렬에만, domain off).
    2단계: L = CE(src) + λ_da·L_domain + λ_g·L_gravity + λ_pca·L_pca (target 인코더 합류).
    gravity/pca 는 raw IMU + R 로 계산되어 인코더를 거치지 않는다(EMG 와 무관).
    """
    model.train()
    r_frozen = not model.r.w.requires_grad if hasattr(model.r, "w") else False
    len_loader = min(len(train_loader), len(tgt_train_loader))
    tot = {"ce": 0.0, "dom": 0.0, "grav": 0.0, "pca": 0.0, "dom_acc": 0.0}
    grad_diag = None
    pbar = tqdm(enumerate(zip(train_loader, tgt_train_loader)), total=len_loader, desc=desc)
    for i, ((src_emg, src_imu, src_y), (tgt_emg, tgt_imu, tgt_y)) in pbar:
        src_emg, src_imu, src_y = src_emg.to(device), src_imu.to(device), src_y.to(device)
        tgt_emg, tgt_imu = tgt_emg.to(device), tgt_imu.to(device)
        optimizer.zero_grad()
        zero = torch.zeros((), device=device)

        # ---- source: 분류 CE (2단계에선 domain(0) 도). EMG+IMU 융합, IMU 는 R 우회 ----
        src_out, src_dom, _ = model(src_emg, src_imu, alpha=alpha, apply_r=False)
        loss_cls = criterion(src_out, src_y)

        if target_align_only:
            # 1단계: target 은 인코더/판별기를 거치지 않는다(아래 gravity/pca 로 R 만 정렬).
            loss_domain, dom_acc = zero, 0.0
        else:
            # 2단계: target(EMG+정렬IMU)이 인코더+판별기에 합류.
            _, tgt_dom, _ = model(tgt_emg, tgt_imu, alpha=alpha, apply_r=True)
            src_dlabel = torch.zeros(src_dom.size(0), dtype=torch.long, device=device)
            tgt_dlabel = torch.ones(tgt_dom.size(0), dtype=torch.long, device=device)
            loss_domain = criterion_domain(src_dom, src_dlabel) + criterion_domain(tgt_dom, tgt_dlabel)
            with torch.no_grad():
                dom_pred = torch.cat([src_dom, tgt_dom]).argmax(1)
                dom_true = torch.cat([src_dlabel, tgt_dlabel])
                dom_acc = (dom_pred == dom_true).float().mean().item() * 100

        # ---- 기하 prior: raw target IMU + R (인코더 무관, 두 단계 모두 R 정렬) ----
        L_grav = gravity_loss(model.r.R, src_imu, tgt_imu) if (lam_g > 0 and not r_frozen) else zero
        L_pca = pca_alignment_loss(model.r.R, src_imu, tgt_imu) if (lam_pca > 0 and not r_frozen) else zero

        # ---- (진단) R 파라미터에 대한 손실 항별 gradient 분해 ----
        Rparam = next(model.r.parameters())
        if i == 0 and log_r_grads and Rparam.requires_grad:
            def _gR(term):
                if not term.requires_grad:
                    return torch.zeros_like(Rparam)
                g = torch.autograd.grad(term, Rparam, retain_graph=True, allow_unused=True)[0]
                return g if g is not None else torch.zeros_like(Rparam)
            g_dom, g_grav, g_pca = _gR(lam_da * loss_domain), _gR(lam_g * L_grav), _gR(lam_pca * L_pca)

            def _cos(a, b):
                na, nb = a.norm(), b.norm()
                return (a.flatten() @ b.flatten() / (na * nb + 1e-12)).item() if na > 0 and nb > 0 else 0.0
            grad_diag = {
                "g_dom": g_dom.norm().item(), "g_grav": g_grav.norm().item(),
                "g_pca": g_pca.norm().item(),
                "cos_dom_grav": _cos(g_dom, g_grav), "cos_grav_pca": _cos(g_grav, g_pca),
            }

        loss = loss_cls + lam_da * loss_domain + lam_g * L_grav + lam_pca * L_pca
        loss.backward()
        optimizer.step()

        tot["ce"] += loss_cls.item()
        tot["dom"] += loss_domain.item()
        tot["grav"] += L_grav.item()
        tot["pca"] += L_pca.item()
        tot["dom_acc"] += dom_acc
        pbar.set_postfix({"CE": f"{loss_cls.item():.3f}", "Dom": f"{loss_domain.item():.3f}",
                          "Grav": f"{L_grav.item():.3f}", "Pca": f"{L_pca.item():.3f}",
                          "Dacc": f"{dom_acc:.0f}%"})

    n = max(1, len_loader)
    out = {k: v / n for k, v in tot.items()}
    if grad_diag is not None:
        print(f"           ∂R: dom {grad_diag['g_dom']:.3f} grav {grad_diag['g_grav']:.3f} "
              f"pca {grad_diag['g_pca']:.3f} | "
              f"cos(dom,grav) {grad_diag['cos_dom_grav']:+.2f} "
              f"cos(grav,pca) {grad_diag['cos_grav_pca']:+.2f}")
    out["grad_diag"] = grad_diag
    return out


def train(seed=42, epochs=30, batch_size=64, lr=1e-3, r_lr=1e-2,
          lambda_da=1.0, lambda_g=1.0, lambda_pca=1.0, post_r_norm="batchnorm",
          target_join_epoch=10, freeze_r_epoch=None, report_last=10, save_cm=False, tag=""):
    set_seed(seed)
    device = pick_device()

    os.makedirs(WEIGHT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(R_DIR, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    save_path = os.path.join(WEIGHT_DIR, f"{NAME}{suffix}_seed{seed}_best_model.pth")

    train_loader, val_loader, tgt_train_loader, tgt_val_loader, num_classes, le = \
        get_mm_dataloaders(batch_size=batch_size)
    class_names = le.classes_

    model = LearnableRMMCDAN(num_classes=num_classes, post_r_norm=post_r_norm).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_domain = nn.CrossEntropyLoss()

    print("\n" + "=" * 60)
    print(f"Learnable-R MULTIMODAL(EMG+IMU) CDAN align-first Training Start  |  seed={seed}")
    print("Mode: JOINT | target IMU -> R(SO(3)) -> InterFusion(EMG+IMU), align by CDAN")
    print(f"lambda: da={lambda_da} gravity={lambda_g} pca={lambda_pca} | post_r_norm={post_r_norm}")
    print(f"align-first: 1단계 epoch 0-{target_join_epoch-1} target 은 gravity/pca 로 R 만 정렬"
          f"(인코더 미경유) → 2단계 epoch {target_join_epoch}~ target 인코더/판별기 합류(α ramp)")
    print(f"Targeting {num_classes} classes on {device}")
    print("=" * 60)

    best_val_acc, best_target_acc = 0.0, 0.0
    tgt_hist = []  # 에폭별 target 정확도(leakage-free last-N 집계용)
    eye3 = torch.eye(3, device=device)

    def geo_angle(A, B):
        c = ((A @ B.transpose(0, 1)).diagonal().sum() - 1.0) / 2.0
        return torch.rad2deg(torch.arccos(c.clamp(-1.0, 1.0))).item()

    def eval_and_log(e, E, extra=""):
        nonlocal best_val_acc, best_target_acc
        val_acc, _, _ = evaluate_r(model, val_loader, device, apply_r=False)
        tgt_acc, _, _ = evaluate_r(model, tgt_val_loader, device, apply_r=True)
        tgt_hist.append(tgt_acc)
        is_best = tgt_acc > best_target_acc
        if is_best:
            best_val_acc, best_target_acc = val_acc, tgt_acc
            torch.save(model.state_dict(), save_path)
        with torch.no_grad():
            Rm = model.r.R.detach()
            det = torch.linalg.det(Rm).item()
            ang = geo_angle(Rm, eye3)
        Rstr = np.array2string(Rm.cpu().numpy(), precision=2, suppress_small=True,
                               max_line_width=200).replace("\n", "")
        print(f"[Epoch {e:02d}/{E:02d}] {extra} | det: {det:.3f} | ∠fromI: {ang:5.1f}° | "
              f"Src: {val_acc:.2f}% | Tgt: {tgt_acc:.2f}%{'  (best)' if is_best else ''}")
        print(f"           R = {Rstr}")

    def alpha_at(p):  # GRL 스케줄 0 -> 1
        return 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

    # ---- JOINT: 인코더 + R + 판별기 동시 학습 ----
    r_params = list(model.r.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith("r.")]
    optimizer = optim.AdamW([
        {"params": other_params, "lr": lr, "weight_decay": 1e-3},
        {"params": r_params, "lr": r_lr, "weight_decay": 0.0},
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    for epoch in range(epochs):
        if freeze_r_epoch is not None and epoch == freeze_r_epoch:
            for p in model.r.parameters():
                p.requires_grad_(False)
            print(f"--- Epoch {epoch+1}: R freeze (이후 R 고정·인코더/판별기만 학습) ---")

        target_align_only = epoch < target_join_epoch
        if target_align_only:
            alpha = 0.0
        else:
            alpha = alpha_at((epoch - target_join_epoch) / max(1, epochs - target_join_epoch))
        if target_join_epoch > 0 and epoch == target_join_epoch:
            print(f"--- Epoch {epoch+1}: target 데이터 encoder/discriminator 합류 "
                  f"(R 사전정렬 {target_join_epoch}ep 완료, 이후 CDAN 적응) ---")

        s = run_epoch(model, optimizer, train_loader, tgt_train_loader, criterion,
                      criterion_domain, device, alpha, lambda_da, lambda_g, lambda_pca,
                      target_align_only=target_align_only,
                      desc=f"Seed {seed} | Epoch [{epoch+1:02d}/{epochs:02d}]")
        scheduler.step()
        phase = "align" if target_align_only else "joint"
        eval_and_log(epoch + 1, epochs,
                     extra=f"[{phase}] CE: {s['ce']:.3f} | Dom: {s['dom']:.3f} "
                           f"(Dacc {s['dom_acc']:.0f}%, α={alpha:.2f}) | "
                           f"Grav: {s['grav']:.3f} | Pca: {s['pca']:.3f}")

    # leakage-free 참고값: 마지막 N epoch target 정확도 평균/표준편차
    last = tgt_hist[-report_last:] if len(tgt_hist) >= report_last else tgt_hist
    last_mean, last_std = float(np.mean(last)), float(np.std(last))

    print(f"\n최종 결과 | seed={seed} | Best Source Val: {best_val_acc:.2f}% | "
          f"Target at Best(oracle peak): {best_target_acc:.2f}% | Shift: {best_val_acc - best_target_acc:.2f}%")
    print(f"           leakage-free last-{len(last)} Target: {last_mean:.2f} ± {last_std:.2f}% "
          f"(target 라벨로 epoch 안 고른 참고값)")

    model.load_state_dict(torch.load(save_path, map_location=device))
    R_best = model.r.R.detach().cpu().numpy()
    r_path = os.path.join(R_DIR, f"R_learned_mm_alignfirst{suffix}_seed{seed}.npy")
    np.save(r_path, R_best)
    print(f"학습된 R 저장: {r_path}\n{np.array2string(R_best, precision=4)}")

    if save_cm:
        _, v_preds, v_true = evaluate_r(model, val_loader, device, apply_r=False)
        _, t_preds, t_true = evaluate_r(model, tgt_val_loader, device, apply_r=True)
        save_confusion_matrix(
            v_true, v_preds, class_names,
            os.path.join(RESULT_DIR, f"{NAME}{suffix}_seed{seed}_source_cm.png"),
            f"MM Align-first Source (seed={seed}, Val: {best_val_acc:.1f}%)", cmap="Blues")
        save_confusion_matrix(
            t_true, t_preds, class_names,
            os.path.join(RESULT_DIR, f"{NAME}{suffix}_seed{seed}_target_cm.png"),
            f"MM Align-first Target (seed={seed}, Src: {best_val_acc:.1f}% vs Tgt: {best_target_acc:.1f}%)",
            cmap="Blues")
        print("혼동 행렬 시각화 저장 완료.")

    return {"seed": seed, "source_acc": best_val_acc, "target_acc": best_target_acc,
            "shift": best_val_acc - best_target_acc,
            "target_last_mean": last_mean, "target_last_std": last_std}


def write_result_json(results, tag):
    src = [r["source_acc"] for r in results]
    tgt = [r["target_acc"] for r in results]
    sh = [r["shift"] for r in results]
    last = [r["target_last_mean"] for r in results]
    ddof = 1 if len(results) > 1 else 0
    payload = {
        "tag": tag or "default", "modality": "emg_imu_multimodal", "model": NAME,
        "data_dir": os.environ.get("MM_DATA_DIR", "preprocessed_MM_raw"),
        "seeds": [r["seed"] for r in results], "results": results,
        "mean": {"source_acc": float(np.mean(src)), "target_acc": float(np.mean(tgt)),
                 "shift": float(np.mean(sh)), "target_last_mean": float(np.mean(last))},
        "std": {"source_acc": float(np.std(src, ddof=ddof)), "target_acc": float(np.std(tgt, ddof=ddof)),
                "shift": float(np.std(sh, ddof=ddof)), "target_last_mean": float(np.std(last, ddof=ddof))},
    }
    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, f"{NAME}_result_{tag or 'default'}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"결과 JSON 저장: {path}")
    print(f"  [oracle peak] Target {payload['mean']['target_acc']:.2f} ± "
          f"{payload['std']['target_acc']:.2f}%  |  [leakage-free last-N] "
          f"{payload['mean']['target_last_mean']:.2f} ± {payload['std']['target_last_mean']:.2f}%")
    return path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi_seed", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--r_lr", type=float, default=1e-2, help="learnable R 전용 학습률")
    parser.add_argument("--lambda_da", type=float, default=1.0, help="domain(CDAN) 손실 가중치")
    parser.add_argument("--lambda_g", type=float, default=1.0, help="L_gravity 가중치(isotropic 에서 의미)")
    parser.add_argument("--lambda_pca", type=float, default=1.0, help="on-the-fly PCA 정렬 prior 가중치")
    parser.add_argument("--target_join_epoch", type=int, default=10,
                        help="이 epoch 부터 target 이 인코더/판별기에 합류. 그 전엔 target 은 "
                             "gravity/pca 로 R 만 정렬(인코더 미경유). 0=끔(처음부터 합류)")
    parser.add_argument("--post_r_norm", type=str, default="batchnorm",
                        choices=["none", "instance", "batchnorm"])
    parser.add_argument("--freeze_r_epoch", type=int, default=None,
                        help="이 epoch 까지 R 학습 후 고정. 미지정=끝까지 R 학습")
    parser.add_argument("--report_last", type=int, default=10,
                        help="leakage-free 참고값용 마지막 N epoch target 정확도 평균 (기본 10)")
    parser.add_argument("--no_cm", action="store_true")
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    suffix = f"_{args.tag}" if args.tag else ""
    kw = dict(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, r_lr=args.r_lr,
              lambda_da=args.lambda_da, lambda_g=args.lambda_g, lambda_pca=args.lambda_pca,
              post_r_norm=args.post_r_norm, target_join_epoch=args.target_join_epoch,
              freeze_r_epoch=args.freeze_r_epoch, report_last=args.report_last,
              save_cm=not args.no_cm, tag=args.tag)

    if args.multi_seed:
        results = [train(seed=s, **kw) for s in args.seeds]
        os.makedirs(RESULT_DIR, exist_ok=True)
        summarize_results(results,
                          method_name=f"Learnable-R MM(EMG+IMU) align-first{' | '+args.tag if args.tag else ''}",
                          save_path=os.path.join(RESULT_DIR, f"{NAME}{suffix}_summary.txt"))
    else:
        results = [train(seed=args.seed, **kw)]

    write_result_json(results, args.tag)
