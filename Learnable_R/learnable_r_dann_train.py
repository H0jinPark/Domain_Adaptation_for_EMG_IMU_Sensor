"""Learnable R 학습 스크립트 (IMU 단독 + DANN 정렬 + 통합 손실, multi-seed).

다이어그램의 통합 손실을 그대로 구현한다:

    L_total = L_cls_source
            + λ_da   · L_domain(source, R(target))      # DANN 적대 정렬
            + λ_g    · L_gravity = ||R g_t - g_s||²       # 중력 정렬(pitch/roll 2 DOF)
            + λ_rot  · L_rotation = ||RᵀR-I||² + |det(R)-1|²   # proper rotation
            + λ_cons · L_target_consistency               # target 예측 안정화

흐름:
    target IMU --R--> aligned IMU --> 인코더 --> tgt_feat, tgt_logits, domain_logits
    source IMU --------------------> 인코더 --> src_feat, src_logits(=분류 CE), domain_logits
    L_domain   : GRL 통과 feature 를 domain 판별기로 → source(0) vs R-정렬 target(1) 적대 정렬
    L_consistency: 원본 target 예측 vs 작은 무작위 회전 증강 target 예측의 symmetric KL

R 이 단순 회전에서 끝나지 않고 (1) 물리적으로 가능한 회전이며 (2) target 중력을
source 기준으로 맞추고 (3) source-target feature gap 을 줄이고 (4) target 예측을
안정적으로 유지하는 방향으로 학습되도록 한다.

이전 진단([[pipeline-learnable-r]])을 반영해 두 모드를 지원한다:
  · JOINT          : 인코더+R 동시 학습 (인코더가 정렬을 흡수할 위험)
  · --freeze_encoder: 2단계. P1 source-only 인코더 pretrain → P2 인코더·분류기 freeze,
                      R + domain 판별기만 적대/기하/일관성으로 적응. R 이 실제로 회전.
isotropic 정규화 데이터(preprocessed_MM_raw_isotropic)에서 중력 DC 가 살아 L_gravity 가
의미를 가진다. zscore 데이터면 --w_gravity 0 권장.
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
from learnable_r_model import (  # noqa: E402
    LearnableRDANN, LearnableRCDAN, r_unified_losses, augment_imu, pca_alignment_loss)

WEIGHT_DIR = os.path.join(PROJECT_ROOT, "weights")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results", "Learnable_R")
R_DIR = os.path.join(PROJECT_ROOT, "results", "R_matrices")


# ----------------------------------------------------------------------
# target 예측 일관성 손실 (원본 vs 무작위 회전 증강)
# ----------------------------------------------------------------------
def consistency_loss(logits_a, logits_b):
    """두 target 예측 분포의 symmetric KL. 두 가지 뷰 모두 grad 가 흐른다(Π-model 류).

    같은 target 윈도우의 원본/증강 예측이 일치하도록 → R·인코더가 작은 센서 회전에
    불변한 예측을 내도록 유도해 target 예측을 안정화한다.
    """
    log_pa = F.log_softmax(logits_a, dim=1)
    log_pb = F.log_softmax(logits_b, dim=1)
    pa, pb = log_pa.exp(), log_pb.exp()
    kl_ab = F.kl_div(log_pb, pa, reduction="batchmean")
    kl_ba = F.kl_div(log_pa, pb, reduction="batchmean")
    return 0.5 * (kl_ab + kl_ba)


@torch.no_grad()
def evaluate_r(model, loader, device, apply_r):
    """source 는 apply_r=False, target 은 apply_r=True 로 정확도(%) 계산."""
    model.eval()
    preds, labels = [], []
    for emg, imu, y in loader:
        emg, imu = emg.to(device), imu.to(device)
        out = model(emg, imu, alpha=0.0, apply_r=apply_r)[0]
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(y.numpy())
    return accuracy_score(labels, preds) * 100, preds, labels


def _set_phase2_trainable(model):
    """Phase 2: 인코더(backbone)+label head freeze, R + domain 판별기만 학습.

    BN/dropout 통계 고정을 위해 frozen 모듈은 eval, domain 판별기는 train 유지.
    """
    for n, p in model.named_parameters():
        train = n.startswith("r.") or n.startswith("domain_classifier.")
        p.requires_grad_(train)


def set_phase2_mode(model):
    """Phase 2 forward 시 모듈별 train/eval 모드 설정 (매 epoch 시작 시 호출)."""
    model.backbone.eval()
    model.label_classifier.eval()
    if hasattr(model, "input_bn"):
        model.input_bn.eval()        # source 통계 고정 (P1 에서 학습된 running stats 사용)
    model.domain_classifier.train()  # 판별기는 계속 학습


def run_source_epoch(model, optimizer, train_loader, criterion, device, desc):
    """Phase 1: source CE 만으로 인코더+분류기 학습 (R 고정, apply_r 불필요)."""
    model.train()
    total = 0.0
    pbar = tqdm(train_loader, total=len(train_loader), desc=desc)
    for src_emg, src_imu, src_y in pbar:
        src_emg, src_imu, src_y = src_emg.to(device), src_imu.to(device), src_y.to(device)
        optimizer.zero_grad()
        out = model(src_emg, src_imu, alpha=0.0, apply_r=False)[0]
        loss = criterion(out, src_y)
        loss.backward()
        optimizer.step()
        total += loss.item()
        pbar.set_postfix({"CE": f"{loss.item():.3f}"})
    return total / max(1, len(train_loader))


def run_dann_epoch(model, optimizer, train_loader, tgt_train_loader, criterion, criterion_domain,
                   device, alpha, lam_da, lam_g, lam_rot, lam_cons, aug_angle,
                   use_ce, phase2, desc, lam_pca=0.0, grad_clip=5.0, log_r_grads=True):
    """통합 손실 1 epoch.

    use_ce=True/phase2=False : JOINT 모드(인코더+R+판별기 동시 학습, source CE 포함).
    use_ce=False/phase2=True : Phase 2(인코더·분류기 freeze). source CE 생략, source 는
        domain 판별기용 feature 만 제공. R + 판별기만 적대/기하/일관성으로 학습.

    L = [CE(src)] + λ_da·L_domain + λ_g·L_gravity + λ_rot·L_rotation + λ_cons·L_consistency
    """
    if phase2:
        set_phase2_mode(model)
    else:
        model.train()
    len_loader = min(len(train_loader), len(tgt_train_loader))
    tot = {"ce": 0.0, "dom": 0.0, "rot": 0.0, "grav": 0.0, "cons": 0.0, "pca": 0.0, "dom_acc": 0.0}
    grad_diag = None
    pbar = tqdm(enumerate(zip(train_loader, tgt_train_loader)), total=len_loader, desc=desc)
    for i, ((src_emg, src_imu, src_y), (tgt_emg, tgt_imu, tgt_y)) in pbar:
        src_emg, src_imu, src_y = src_emg.to(device), src_imu.to(device), src_y.to(device)
        tgt_emg, tgt_imu = tgt_emg.to(device), tgt_imu.to(device)
        optimizer.zero_grad()

        # ---- source: 분류 CE + domain(0) ----
        src_out, src_dom, _ = model(src_emg, src_imu, alpha=alpha, apply_r=False)
        loss_cls = criterion(src_out, src_y) if use_ce else torch.zeros((), device=device)

        # ---- target: R 정렬 후 domain(1) + 일관성 ----
        tgt_out, tgt_dom, _ = model(tgt_emg, tgt_imu, alpha=alpha, apply_r=True)

        src_dlabel = torch.zeros(src_dom.size(0), dtype=torch.long, device=device)
        tgt_dlabel = torch.ones(tgt_dom.size(0), dtype=torch.long, device=device)
        loss_domain = criterion_domain(src_dom, src_dlabel) + criterion_domain(tgt_dom, tgt_dlabel)

        # ---- 일관성: 원본 target vs 무작위 회전 증강 ----
        if lam_cons > 0:
            tgt_aug = augment_imu(tgt_imu, max_angle_deg=aug_angle)
            tgt_aug_out = model(tgt_emg, tgt_aug, alpha=0.0, apply_r=True)[0]
            loss_cons = consistency_loss(tgt_out, tgt_aug_out)
        else:
            loss_cons = torch.zeros((), device=device)

        # ---- 기하 prior (제곱형) ----
        L_rot, L_grav = r_unified_losses(model.r.R, src_imu, tgt_imu)
        # ---- on-the-fly PCA 정렬 prior (yaw 포함, physics-informed) ----
        L_pca = pca_alignment_loss(model.r.R, src_imu, tgt_imu) if lam_pca > 0 \
            else torch.zeros((), device=device)

        # ---- (진단) R 파라미터에 대한 손실 항별 gradient 분해: 어느 항이 R 을 흔드는가 ----
        # 매 에폭 첫 배치에서만. 정답(perm) 정보 안 씀 — 이미 있는 손실만 R 파라미터로 미분.
        # SO(3) 모드면 leaf 는 w(3-벡터), matrix 모드면 _Rmat. R(property)이 아니라 leaf 로 미분.
        Rparam = model.r.w if getattr(model.r, "mode", "matrix") == "so3" else model.r._Rmat
        if i == 0 and log_r_grads and Rparam.requires_grad:
            def _gR(term):
                if not term.requires_grad:
                    return torch.zeros_like(Rparam)
                g = torch.autograd.grad(term, Rparam, retain_graph=True, allow_unused=True)[0]
                return g if g is not None else torch.zeros_like(Rparam)
            g_dom = _gR(lam_da * loss_domain)
            g_grav = _gR(lam_g * L_grav)
            g_rot = _gR(lam_rot * L_rot)
            g_cons = _gR(lam_cons * loss_cons)
            g_pca = _gR(lam_pca * L_pca)
            def _cos(a, b):
                na, nb = a.norm(), b.norm()
                return (a.flatten() @ b.flatten() / (na * nb + 1e-12)).item() if na > 0 and nb > 0 else 0.0
            grad_diag = {
                "g_dom": g_dom.norm().item(), "g_grav": g_grav.norm().item(),
                "g_rot": g_rot.norm().item(), "g_cons": g_cons.norm().item(),
                "g_pca": g_pca.norm().item(),
                "cos_dom_grav": _cos(g_dom, g_grav), "cos_dom_cons": _cos(g_dom, g_cons),
                "cos_grav_pca": _cos(g_grav, g_pca),
            }

        loss = loss_cls + lam_da * loss_domain + lam_g * L_grav \
            + lam_rot * L_rot + lam_cons * loss_cons + lam_pca * L_pca
        loss.backward()
        if grad_clip > 0:                       # GRL+CDAN 적대 신호 폭발 방지
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        with torch.no_grad():
            dom_pred = torch.cat([src_dom, tgt_dom]).argmax(1)
            dom_true = torch.cat([src_dlabel, tgt_dlabel])
            dom_acc = (dom_pred == dom_true).float().mean().item() * 100
        tot["ce"] += loss_cls.item()
        tot["dom"] += loss_domain.item()
        tot["rot"] += L_rot.item()
        tot["grav"] += L_grav.item()
        tot["cons"] += loss_cons.item()
        tot["pca"] += L_pca.item()
        tot["dom_acc"] += dom_acc
        pbar.set_postfix({"CE": f"{loss_cls.item():.3f}", "Dom": f"{loss_domain.item():.3f}",
                          "Grav": f"{L_grav.item():.3f}", "Pca": f"{L_pca.item():.3f}",
                          "Cons": f"{loss_cons.item():.4f}", "Dacc": f"{dom_acc:.0f}%"})

    n = max(1, len_loader)
    out = {k: v / n for k, v in tot.items()}
    if grad_diag is not None:
        print(f"           ∂R: dom {grad_diag['g_dom']:.3f} grav {grad_diag['g_grav']:.3f} "
              f"pca {grad_diag['g_pca']:.3f} rot {grad_diag['g_rot']:.3f} cons {grad_diag['g_cons']:.3f} | "
              f"cos(dom,grav) {grad_diag['cos_dom_grav']:+.2f} "
              f"cos(grav,pca) {grad_diag['cos_grav_pca']:+.2f}")
    out["grad_diag"] = grad_diag
    return out


def train(seed=42, epochs=60, batch_size=64, lr=1e-3, r_lr=1e-2,
          lambda_da=1.0, lambda_g=1.0, lambda_rot=1.0, lambda_cons=1.0, lambda_pca=0.0,
          aug_angle=15.0, freeze_encoder=False, phase1_epochs=15, post_r_norm="none",
          domain="dann", grad_clip=5.0, save_cm=False, tag=""):
    set_seed(seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
        else "cpu")

    os.makedirs(WEIGHT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(R_DIR, exist_ok=True)
    name = f"learnable_r_{domain}"          # dann | cdan
    suffix = f"_{tag}" if tag else ""
    save_path = os.path.join(WEIGHT_DIR, f"{name}{suffix}_seed{seed}_best_model.pth")

    train_loader, val_loader, tgt_train_loader, tgt_val_loader, num_classes, le = \
        get_mm_dataloaders(batch_size=batch_size)
    class_names = le.classes_

    if domain == "cdan":
        model = LearnableRCDAN(num_classes=num_classes, post_r_norm=post_r_norm).to(device)
    else:
        model = LearnableRDANN(num_classes=num_classes, post_r_norm=post_r_norm).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_domain = nn.CrossEntropyLoss()

    print("\n" + "=" * 60)
    print(f"Learnable-R {domain.upper()} (unified loss) Training Start  |  seed={seed}")
    if freeze_encoder:
        print(f"Mode: TWO-PHASE | P1 source-only pretrain {phase1_epochs}ep "
              f"-> P2 freeze encoder, adapt R + domain disc {epochs - phase1_epochs}ep")
    else:
        print(f"Mode: JOINT | target IMU -> R -> encoder, align by {domain.upper()} adversarial "
              f"(grad_clip={grad_clip})")
    print(f"lambda: da={lambda_da} gravity={lambda_g} pca={lambda_pca} rotation={lambda_rot} "
          f"consistency={lambda_cons} | aug_angle={aug_angle}deg | post_r_norm={post_r_norm}")
    print(f"Targeting {num_classes} classes on {device}")
    print("=" * 60)

    best_val_acc, best_target_acc = 0.0, 0.0

    # 참조 R(수작업 permutation): 학습 R 이 알려진 정답 정렬로 가는지 거리 추적용
    eye3 = torch.eye(3, device=device)
    ref_path = os.path.join(R_DIR, "R_permutation.npy")
    R_ref = (torch.tensor(np.load(ref_path), dtype=torch.float32, device=device)
             if os.path.exists(ref_path) else None)

    def geo_angle(A, B):
        """두 회전행렬 사이 측지각(도). A,B 가 정확한 회전이 아니어도 근사값."""
        c = ((A @ B.transpose(0, 1)).diagonal().sum() - 1.0) / 2.0
        return torch.rad2deg(torch.arccos(c.clamp(-1.0, 1.0))).item()

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
            orth = torch.linalg.norm(Rm.transpose(0, 1) @ Rm - eye3).item()
            ang = geo_angle(Rm, eye3)                              # 항등에서 얼마나 회전했나
            to_ref = geo_angle(Rm, R_ref) if R_ref is not None else float("nan")
        Rstr = np.array2string(Rm.cpu().numpy(), precision=2, suppress_small=True,
                               max_line_width=200).replace("\n", "")
        print(f"[{stage} {e:02d}/{E:02d}] {extra} | det: {det:.3f} | ∠fromI: {ang:5.1f}° | "
              f"∠→perm: {to_ref:5.1f}° | Src: {val_acc:.2f}% | Tgt: {tgt_acc:.2f}%{'  (best)' if is_best else ''}")
        print(f"           R = {Rstr}")

    def alpha_at(p):  # GRL 스케줄 0 -> 1
        return 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

    if not freeze_encoder:
        # ---- JOINT: 인코더 + R + 판별기 동시 학습 ----
        r_params = list(model.r.parameters())
        other_params = [p for n, p in model.named_parameters() if not n.startswith("r.")]
        optimizer = optim.AdamW([
            {"params": other_params, "lr": lr, "weight_decay": 1e-3},
            {"params": r_params, "lr": r_lr, "weight_decay": 0.0},
        ])
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        for epoch in range(epochs):
            alpha = alpha_at(epoch / epochs)
            s = run_dann_epoch(model, optimizer, train_loader, tgt_train_loader, criterion,
                               criterion_domain, device, alpha, lambda_da, lambda_g, lambda_rot,
                               lambda_cons, aug_angle, use_ce=True, phase2=False,
                               lam_pca=lambda_pca, grad_clip=grad_clip,
                               desc=f"Seed {seed} | Epoch [{epoch+1:02d}/{epochs:02d}]")
            scheduler.step()
            eval_and_log("Epoch", epoch + 1, epochs,
                         extra=f"CE: {s['ce']:.3f} | Dom: {s['dom']:.3f} (Dacc {s['dom_acc']:.0f}%, α={alpha:.2f}) | "
                               f"Grav: {s['grav']:.3f} | Pca: {s['pca']:.3f} | Cons: {s['cons']:.4f}")
    else:
        # ---- TWO-PHASE ----
        assert 0 < phase1_epochs < epochs, "phase1_epochs 는 1..epochs-1 사이여야 함"
        # Phase 1: source-only 인코더 pretrain (R 항등 고정, 판별기 미사용)
        for n, p in model.named_parameters():
            p.requires_grad_(not n.startswith("r."))
        p1_params = [p for n, p in model.named_parameters() if not n.startswith("r.")]
        opt1 = optim.AdamW(p1_params, lr=lr, weight_decay=1e-3)
        sch1 = CosineAnnealingLR(opt1, T_max=phase1_epochs)
        for epoch in range(phase1_epochs):
            ce = run_source_epoch(model, opt1, train_loader, criterion, device,
                                  desc=f"Seed {seed} | P1 [{epoch+1:02d}/{phase1_epochs:02d}]")
            sch1.step()
            eval_and_log("P1", epoch + 1, phase1_epochs, extra=f"CE: {ce:.4f}")

        # Phase 2: 인코더·분류기 freeze, R + domain 판별기만 적대/기하/일관성으로 적응
        _set_phase2_trainable(model)
        p2_params = [p for n, p in model.named_parameters() if p.requires_grad]
        opt2 = optim.AdamW([
            {"params": [p for n, p in model.named_parameters()
                        if n.startswith("r.") and p.requires_grad], "lr": r_lr, "weight_decay": 0.0},
            {"params": [p for n, p in model.named_parameters()
                        if n.startswith("domain_classifier.") and p.requires_grad],
             "lr": lr, "weight_decay": 1e-3},
        ])
        phase2_epochs = epochs - phase1_epochs
        sch2 = CosineAnnealingLR(opt2, T_max=max(1, phase2_epochs))
        print(f"--- Phase 2 시작: 인코더 freeze, R + domain 판별기 학습 (r_lr={r_lr}) ---")
        for j in range(phase2_epochs):
            alpha = alpha_at(j / max(1, phase2_epochs))
            s = run_dann_epoch(model, opt2, train_loader, tgt_train_loader, criterion,
                               criterion_domain, device, alpha, lambda_da, lambda_g, lambda_rot,
                               lambda_cons, aug_angle, use_ce=False, phase2=True,
                               lam_pca=lambda_pca, grad_clip=grad_clip,
                               desc=f"Seed {seed} | P2 [{j+1:02d}/{phase2_epochs:02d}]")
            sch2.step()
            eval_and_log("P2", j + 1, phase2_epochs,
                         extra=f"Dom: {s['dom']:.3f} (Dacc {s['dom_acc']:.0f}%, α={alpha:.2f}) | "
                               f"Grav: {s['grav']:.3f} | Pca: {s['pca']:.3f} | Cons: {s['cons']:.4f}")

    print(f"\n최종 결과 | seed={seed} | Best Source Val: {best_val_acc:.2f}% | "
          f"Target at Best: {best_target_acc:.2f}% | Shift: {best_val_acc - best_target_acc:.2f}%")

    model.load_state_dict(torch.load(save_path, map_location=device))
    R_best = model.r.R.detach().cpu().numpy()
    r_path = os.path.join(R_DIR, f"R_learned_{domain}{suffix}_seed{seed}.npy")
    np.save(r_path, R_best)
    print(f"학습된 R 저장: {r_path}\n{np.array2string(R_best, precision=4)}")

    if save_cm:
        _, v_preds, v_true = evaluate_r(model, val_loader, device, apply_r=False)
        _, t_preds, t_true = evaluate_r(model, tgt_val_loader, device, apply_r=True)
        save_confusion_matrix(
            v_true, v_preds, class_names,
            os.path.join(RESULT_DIR, f"{name}{suffix}_seed{seed}_source_cm.png"),
            f"Learnable-R {domain.upper()} Source (seed={seed}, Val: {best_val_acc:.1f}%)", cmap="Blues")
        save_confusion_matrix(
            t_true, t_preds, class_names,
            os.path.join(RESULT_DIR, f"{name}{suffix}_seed{seed}_target_cm.png"),
            f"Learnable-R {domain.upper()} Target (seed={seed}, Src: {best_val_acc:.1f}% vs Tgt: {best_target_acc:.1f}%)",
            cmap="Blues")
        print("혼동 행렬 시각화 저장 완료.")

    return {"seed": seed, "source_acc": best_val_acc, "target_acc": best_target_acc,
            "shift": best_val_acc - best_target_acc}


def write_result_json(results, tag, domain="dann"):
    src = [r["source_acc"] for r in results]
    tgt = [r["target_acc"] for r in results]
    sh = [r["shift"] for r in results]
    ddof = 1 if len(results) > 1 else 0
    payload = {
        "tag": tag or "default", "modality": "imu_only", "model": f"learnable_r_{domain}",
        "data_dir": os.environ.get("MM_DATA_DIR", "preprocessed_MM_raw"),
        "seeds": [r["seed"] for r in results], "results": results,
        "mean": {"source_acc": float(np.mean(src)), "target_acc": float(np.mean(tgt)),
                 "shift": float(np.mean(sh))},
        "std": {"source_acc": float(np.std(src, ddof=ddof)), "target_acc": float(np.std(tgt, ddof=ddof)),
                "shift": float(np.std(sh, ddof=ddof))},
    }
    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, f"learnable_r_{domain}_result_{tag or 'default'}.json")
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
    parser.add_argument("--domain", type=str, default="dann", choices=["dann", "cdan"],
                        help="도메인 정렬 방식. dann=plain(512→2)·cdan=조건부 외적(feature⊗softmax→판별기)")
    parser.add_argument("--lambda_da", type=float, default=1.0, help="domain 손실 가중치")
    parser.add_argument("--lambda_g", type=float, default=1.0,
                        help="L_gravity=||R g_t - g_s||² 가중치. isotropic 데이터에서 의미. zscore 면 0 권장")
    parser.add_argument("--lambda_rot", type=float, default=1.0,
                        help="L_rotation=||RᵀR-I||²+|det(R)-1|² 가중치(proper rotation)")
    parser.add_argument("--lambda_cons", type=float, default=1.0,
                        help="target 예측 일관성(원본 vs 회전증강) 가중치. 0 이면 끔")
    parser.add_argument("--lambda_pca", type=float, default=0.0,
                        help="on-the-fly PCA 정렬 prior 가중치(‖R Ft - Fs‖², 매 배치 주축 계산). "
                             "gravity 가 못 보는 yaw 를 잡음. 계산된 R 주입 아님(physics-informed). 0=끔")
    parser.add_argument("--aug_angle", type=float, default=15.0,
                        help="일관성 증강의 최대 무작위 회전 각도(도)")
    parser.add_argument("--grad_clip", type=float, default=5.0,
                        help="gradient norm clip (GRL+CDAN 적대신호 폭발 방지). 0=끔")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="2단계: source 로 인코더 pretrain 후 freeze, R+판별기만 적응")
    parser.add_argument("--phase1_epochs", type=int, default=15,
                        help="(freeze_encoder 시) source-only 인코더 pretrain epoch 수, 나머지는 Phase 2")
    parser.add_argument("--post_r_norm", type=str, default="batchnorm",
                        choices=["none", "instance", "batchnorm"],
                        help="R 직후 인코더 입력 정규화(회전→정규화 순서). batchnorm=채널별 BN(running 통계가 "
                             "데이터셋 zscore 근사, 75% 레시피 동형, 권장)·instance=윈도우별 표준화·none=기존. "
                             "gravity 손실은 raw 배치로 계산되어 어느 옵션이든 영향 없음")
    parser.add_argument("--no_cm", action="store_true")
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    suffix = f"_{args.tag}" if args.tag else ""
    kw = dict(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, r_lr=args.r_lr,
              lambda_da=args.lambda_da, lambda_g=args.lambda_g, lambda_rot=args.lambda_rot,
              lambda_cons=args.lambda_cons, lambda_pca=args.lambda_pca, aug_angle=args.aug_angle,
              freeze_encoder=args.freeze_encoder, phase1_epochs=args.phase1_epochs,
              post_r_norm=args.post_r_norm, domain=args.domain, grad_clip=args.grad_clip,
              save_cm=not args.no_cm, tag=args.tag)

    if args.multi_seed:
        results = [train(seed=s, **kw) for s in args.seeds]
        os.makedirs(RESULT_DIR, exist_ok=True)
        summarize_results(results,
                          method_name=f"Learnable-R {args.domain.upper()} (IMU-only){' | '+args.tag if args.tag else ''}",
                          save_path=os.path.join(RESULT_DIR, f"learnable_r_{args.domain}{suffix}_summary.txt"))
    else:
        results = [train(seed=args.seed, **kw)]

    write_result_json(results, args.tag, domain=args.domain)
