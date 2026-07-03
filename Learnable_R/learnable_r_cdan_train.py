"""Learnable R 학습 스크립트 (IMU 단독 + CDAN 정렬 + 통합 손실, multi-seed).

최고 조합(pca_grav_cdan_joint, Target 72.93%)의 통합 손실을 그대로 구현한다:

    L_total = L_cls_source
            + λ_da   · L_domain(source, R(target))      # CDAN 조건부 적대 정렬
            + λ_g    · L_gravity  = ||R g_t - g_s||²      # 중력 정렬(pitch/roll 2 DOF)
            + λ_pca  · L_pca      = ||R Ft - Fs||²        # on-the-fly PCA 정렬(yaw 관측)

흐름:
    target IMU --R--> aligned IMU --> 인코더 --> tgt_feat, tgt_logits, domain_logits
    source IMU --------------------> 인코더 --> src_feat, src_logits(=분류 CE), domain_logits
    L_domain : CDAN(feature⊗softmax 조건부 외적) → GRL → 판별기로 source(0) vs
               R-정렬 target(1) 적대 정렬

R 은 항상 항등에서 출발해(SO(3) 재매개화 so3_exp, det=+1·RᵀR=I 구조적 보장 → 회전
정칙화 손실 불필요) JOINT 로 인코더와 동시에 학습된다. PCA prior 가 R 을 X↔Y yaw
스왑으로 자력 정렬하고, gravity 가 pitch/roll 을 보강하며, CDAN 이 인코더를 target 에
적응시킨다.

isotropic 정규화 데이터(preprocessed_MM_raw_isotropic)에서 중력 DC 가 살아 L_gravity·
L_pca 가 의미를 가진다. zscore 데이터면 --lambda_g / --lambda_pca 0 권장.
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
from learnable_r_model import (  # noqa: E402
    LearnableRCDAN, gravity_loss, pca_alignment_loss)

WEIGHT_DIR = os.path.join(PROJECT_ROOT, "weights")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results", "Learnable_R")
R_DIR = os.path.join(PROJECT_ROOT, "results", "R_matrices")


def pick_device():
    """CUDA → MPS(Apple Silicon) → CPU 순으로 사용 가능한 device 선택."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def run_epoch(model, optimizer, train_loader, tgt_train_loader, criterion, criterion_domain,
              device, alpha, lam_da, lam_g, lam_pca, log_r_grads=True, desc=""):
    """통합 손실 1 epoch (JOINT: 인코더+R+판별기 동시 학습, source CE 포함).

    L = CE(src) + λ_da·L_domain + λ_g·L_gravity + λ_pca·L_pca
    (R 은 SO(3) 라 항상 회전 → 회전 정칙화 손실 없음)

    R 이 freeze(w.requires_grad=False)되면 L_gravity·L_pca 는 어떤 파라미터에도 gradient
    를 못 주므로(R 고정·인코더 미경유) 계산을 건너뛴다 — 특히 pca_frame 의 매 배치 eigh
    낭비 제거. 이후 이 손실들은 0 으로 로깅.
    """
    model.train()
    r_frozen = not model.r.w.requires_grad   # freeze 후엔 gravity/pca 손실이 gradient 0 → skip
    len_loader = min(len(train_loader), len(tgt_train_loader))
    tot = {"ce": 0.0, "dom": 0.0, "grav": 0.0, "pca": 0.0, "dom_acc": 0.0}
    grad_diag = None
    pbar = tqdm(enumerate(zip(train_loader, tgt_train_loader)), total=len_loader, desc=desc)
    for i, ((src_emg, src_imu, src_y), (tgt_emg, tgt_imu, tgt_y)) in pbar:
        src_emg, src_imu, src_y = src_emg.to(device), src_imu.to(device), src_y.to(device)
        tgt_emg, tgt_imu = tgt_emg.to(device), tgt_imu.to(device)
        optimizer.zero_grad()

        # ---- source: 분류 CE + domain(0) ----
        src_out, src_dom, _ = model(src_emg, src_imu, alpha=alpha, apply_r=False)
        loss_cls = criterion(src_out, src_y)

        # ---- target: R 정렬 후 domain(1) ----
        _, tgt_dom, _ = model(tgt_emg, tgt_imu, alpha=alpha, apply_r=True)

        src_dlabel = torch.zeros(src_dom.size(0), dtype=torch.long, device=device)
        tgt_dlabel = torch.ones(tgt_dom.size(0), dtype=torch.long, device=device)
        loss_domain = criterion_domain(src_dom, src_dlabel) + criterion_domain(tgt_dom, tgt_dlabel)

        # ---- 기하 prior (R freeze 후엔 gradient 0 이라 skip) ----
        # 중력 정렬(pitch/roll)
        L_grav = gravity_loss(model.r.R, src_imu, tgt_imu) if (lam_g > 0 and not r_frozen) \
            else torch.zeros((), device=device)
        # on-the-fly PCA 정렬 prior (yaw 포함, physics-informed) — 매 배치 eigh
        L_pca = pca_alignment_loss(model.r.R, src_imu, tgt_imu) if (lam_pca > 0 and not r_frozen) \
            else torch.zeros((), device=device)

        # ---- (진단) R 파라미터에 대한 손실 항별 gradient 분해: 어느 항이 R 을 흔드는가 ----
        # 매 에폭 첫 배치에서만. 정답(perm) 정보 안 씀 — 이미 있는 손실만 R 파라미터로 미분.
        # R 은 SO(3) 라 leaf 는 so(3) 3-벡터 w. R(property)이 아니라 leaf 로 미분.
        Rparam = model.r.w
        if i == 0 and log_r_grads and Rparam.requires_grad:
            def _gR(term):
                if not term.requires_grad:
                    return torch.zeros_like(Rparam)
                g = torch.autograd.grad(term, Rparam, retain_graph=True, allow_unused=True)[0]
                return g if g is not None else torch.zeros_like(Rparam)
            g_dom = _gR(lam_da * loss_domain)
            g_grav = _gR(lam_g * L_grav)
            g_pca = _gR(lam_pca * L_pca)
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

        with torch.no_grad():
            dom_pred = torch.cat([src_dom, tgt_dom]).argmax(1)
            dom_true = torch.cat([src_dlabel, tgt_dlabel])
            dom_acc = (dom_pred == dom_true).float().mean().item() * 100
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
          lambda_da=1.0, lambda_g=1.0, lambda_pca=1.0,
          post_r_norm="batchnorm", freeze_r_epoch=None, save_cm=False, tag=""):
    set_seed(seed)
    device = pick_device()

    os.makedirs(WEIGHT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(R_DIR, exist_ok=True)
    name = "learnable_r_cdan"
    suffix = f"_{tag}" if tag else ""
    save_path = os.path.join(WEIGHT_DIR, f"{name}{suffix}_seed{seed}_best_model.pth")

    train_loader, val_loader, tgt_train_loader, tgt_val_loader, num_classes, le = \
        get_mm_dataloaders(batch_size=batch_size)
    class_names = le.classes_

    model = LearnableRCDAN(num_classes=num_classes, post_r_norm=post_r_norm).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_domain = nn.CrossEntropyLoss()

    print("\n" + "=" * 60)
    print(f"Learnable-R CDAN (unified loss, JOINT) Training Start  |  seed={seed}")
    print("Mode: JOINT | target IMU -> R(SO(3)) -> encoder, align by CDAN adversarial")
    print(f"lambda: da={lambda_da} gravity={lambda_g} pca={lambda_pca} "
          f"| post_r_norm={post_r_norm}")
    print(f"R freeze: {'off (끝까지 학습)' if freeze_r_epoch is None else f'epoch {freeze_r_epoch} 부터 R 고정'}")
    print(f"Targeting {num_classes} classes on {device}")
    print("=" * 60)

    best_val_acc, best_target_acc = 0.0, 0.0

    eye3 = torch.eye(3, device=device)   # ∠fromI(항등에서 회전각) 계산용

    def geo_angle(A, B):
        """두 회전행렬 사이 측지각(도). A,B 가 정확한 회전이 아니어도 근사값."""
        c = ((A @ B.transpose(0, 1)).diagonal().sum() - 1.0) / 2.0
        return torch.rad2deg(torch.arccos(c.clamp(-1.0, 1.0))).item()

    def eval_and_log(e, E, extra=""):
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
            ang = geo_angle(Rm, eye3)                              # 항등에서 얼마나 회전했나
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
        # R freeze: freeze_r_epoch 까지 R 학습 후 고정 (이후 인코더/판별기만 적응).
        # R 이 수렴한 뒤엔 갱신이 무의미하므로 requires_grad off → backprop·update 중단.
        if freeze_r_epoch is not None and epoch == freeze_r_epoch:
            for p in model.r.parameters():
                p.requires_grad_(False)
            print(f"--- Epoch {epoch+1}: R freeze (epoch {freeze_r_epoch}까지 학습 완료, "
                  f"이후 R 고정·인코더/판별기만 학습) ---")
        alpha = alpha_at(epoch / epochs)
        s = run_epoch(model, optimizer, train_loader, tgt_train_loader, criterion,
                      criterion_domain, device, alpha, lambda_da, lambda_g, lambda_pca,
                      desc=f"Seed {seed} | Epoch [{epoch+1:02d}/{epochs:02d}]")
        scheduler.step()
        eval_and_log(epoch + 1, epochs,
                     extra=f"CE: {s['ce']:.3f} | Dom: {s['dom']:.3f} (Dacc {s['dom_acc']:.0f}%, α={alpha:.2f}) | "
                           f"Grav: {s['grav']:.3f} | Pca: {s['pca']:.3f}")

    print(f"\n최종 결과 | seed={seed} | Best Source Val: {best_val_acc:.2f}% | "
          f"Target at Best: {best_target_acc:.2f}% | Shift: {best_val_acc - best_target_acc:.2f}%")

    model.load_state_dict(torch.load(save_path, map_location=device))
    R_best = model.r.R.detach().cpu().numpy()
    r_path = os.path.join(R_DIR, f"R_learned_cdan{suffix}_seed{seed}.npy")
    np.save(r_path, R_best)
    print(f"학습된 R 저장: {r_path}\n{np.array2string(R_best, precision=4)}")

    if save_cm:
        _, v_preds, v_true = evaluate_r(model, val_loader, device, apply_r=False)
        _, t_preds, t_true = evaluate_r(model, tgt_val_loader, device, apply_r=True)
        save_confusion_matrix(
            v_true, v_preds, class_names,
            os.path.join(RESULT_DIR, f"{name}{suffix}_seed{seed}_source_cm.png"),
            f"Learnable-R CDAN Source (seed={seed}, Val: {best_val_acc:.1f}%)", cmap="Blues")
        save_confusion_matrix(
            t_true, t_preds, class_names,
            os.path.join(RESULT_DIR, f"{name}{suffix}_seed{seed}_target_cm.png"),
            f"Learnable-R CDAN Target (seed={seed}, Src: {best_val_acc:.1f}% vs Tgt: {best_target_acc:.1f}%)",
            cmap="Blues")
        print("혼동 행렬 시각화 저장 완료.")

    return {"seed": seed, "source_acc": best_val_acc, "target_acc": best_target_acc,
            "shift": best_val_acc - best_target_acc}


def write_result_json(results, tag):
    src = [r["source_acc"] for r in results]
    tgt = [r["target_acc"] for r in results]
    sh = [r["shift"] for r in results]
    ddof = 1 if len(results) > 1 else 0
    payload = {
        "tag": tag or "default", "modality": "imu_only", "model": "learnable_r_cdan",
        "data_dir": os.environ.get("MM_DATA_DIR", "preprocessed_MM_raw"),
        "seeds": [r["seed"] for r in results], "results": results,
        "mean": {"source_acc": float(np.mean(src)), "target_acc": float(np.mean(tgt)),
                 "shift": float(np.mean(sh))},
        "std": {"source_acc": float(np.std(src, ddof=ddof)), "target_acc": float(np.std(tgt, ddof=ddof)),
                "shift": float(np.std(sh, ddof=ddof))},
    }
    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, f"learnable_r_cdan_result_{tag or 'default'}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"결과 JSON 저장: {path}")
    return path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi_seed", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--r_lr", type=float, default=1e-2, help="learnable R 전용 학습률")
    parser.add_argument("--lambda_da", type=float, default=1.0, help="domain(CDAN) 손실 가중치")
    parser.add_argument("--lambda_g", type=float, default=1.0,
                        help="L_gravity=||R g_t - g_s||² 가중치. isotropic 데이터에서 의미. zscore 면 0 권장")
    parser.add_argument("--lambda_pca", type=float, default=1.0,
                        help="on-the-fly PCA 정렬 prior 가중치(‖R Ft - Fs‖², 매 배치 주축 계산). "
                             "gravity 가 못 보는 yaw 를 잡음. 계산된 R 주입 아님(physics-informed). 0=끔")
    parser.add_argument("--post_r_norm", type=str, default="batchnorm",
                        choices=["none", "instance", "batchnorm"],
                        help="R 직후 인코더 입력 정규화(회전→정규화 순서). batchnorm=채널별 BN(running 통계가 "
                             "데이터셋 zscore 근사, 권장)·instance=윈도우별 표준화·none=기존. "
                             "gravity/pca 손실은 raw 배치로 계산되어 어느 옵션이든 영향 없음")
    parser.add_argument("--freeze_r_epoch", type=int, default=None,
                        help="이 epoch 까지 R 학습 후 고정(이후 인코더/판별기만 적응). 예: 10 이면 "
                             "epoch 1~10 R 학습, 11부터 R 고정. 미지정(None)=끝까지 R 학습")
    parser.add_argument("--no_cm", action="store_true")
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    suffix = f"_{args.tag}" if args.tag else ""
    kw = dict(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, r_lr=args.r_lr,
              lambda_da=args.lambda_da, lambda_g=args.lambda_g, lambda_pca=args.lambda_pca,
              post_r_norm=args.post_r_norm, freeze_r_epoch=args.freeze_r_epoch,
              save_cm=not args.no_cm, tag=args.tag)

    if args.multi_seed:
        results = [train(seed=s, **kw) for s in args.seeds]
        os.makedirs(RESULT_DIR, exist_ok=True)
        summarize_results(results,
                          method_name=f"Learnable-R CDAN (IMU-only){' | '+args.tag if args.tag else ''}",
                          save_path=os.path.join(RESULT_DIR, f"learnable_r_cdan{suffix}_summary.txt"))
    else:
        results = [train(seed=args.seed, **kw)]

    write_result_json(results, args.tag)
