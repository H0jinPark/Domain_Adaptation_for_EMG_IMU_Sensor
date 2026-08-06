"""Align-first(2단계) × 쿼터니언 R × both(중력+PCA) prior 결합 학습.

2026-07-10 loss ablation 에서 both(λ_g=1,λ_pca=1) align-first 세팅이 Target 80.5% 로
가장 안정적으로 나왔다([[result-mm-alignfirst-null]] 흐름의 IMU 단독 갈래). 그 "잘 나온"
세팅을 그대로 두고 **R 의 표현만 SO(3)(so3_exp) → 단위 쿼터니언(4-파라미터)** 으로 바꾼
변형이다. 목적은 [[pipeline-learnable-r]] 의 재현성 문제(비-42 seed 에서 R 이 인코더보다
느리게 수렴해 나쁜 국소해에 갇힘)를 쿼터니언 landscape(항등 부근 exp-map 특이점 없음)로
완화할 수 있는지 align-first 조건에서 확인하는 것.

learnable_r_cdan_alignfirst_train.py(so3 원본)를 복제하되 두 가지만 바꿨다:
  1. 모델 생성 시 r_param="quat" (LearnableRQuat, det=+1·RᵀR=I 구조적 보장).
  2. R 파라미터 접근을 so(3) w → 쿼터니언 q 로 교체(freeze 체크·gradient 진단 두 곳).
손실·데이터·align-first 타이밍(target_join_epoch)·CDAN 규약은 원본과 완전히 동일하다.
쿼터니언은 정규화로 회전임이 보장되므로 rotation reg 는 여전히 불필요(λ_rot=0).

align-first 2단계(원본과 동일):
  · 1단계 (epoch < --target_join_epoch, 기본 10): source 만 인코더→CE, target 은 인코더
    미경유·gravity/pca 로 R(쿼터니언)만 정렬, domain off(α=0).
  · 2단계 (epoch ≥ target_join_epoch): target 도 인코더/판별기 합류, CDAN 적대(α 0→1 ramp).

실행 예(both = 기본 λ_g=1 λ_pca=1):
    MM_DATA_DIR=preprocessed_MM_raw_isotropic python \
        Learnable_R/learnable_r_cdan_alignfirst_quat_train.py \
        --multi_seed --epochs 60 --target_join_epoch 10 --no_cm --tag both_quat
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
from learnable_r_test_eval import (  # noqa: E402
    SELECTION, REPORTED_METRIC, evaluate_test, summarize_metrics)

WEIGHT_DIR = os.path.join(PROJECT_ROOT, "weights")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results", "Learnable_R")
R_DIR = os.path.join(PROJECT_ROOT, "results", "R_matrices")
NAME = "learnable_r_cdan_alignfirst_quat"
R_PARAM = "quat"   # 이 스크립트 전용 R 파라미터화 (원본 so3 대비 유일한 모델 차이)


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
              device, alpha, lam_da, lam_g, lam_pca, target_align_only=False,
              log_r_grads=True, desc=""):
    """통합 손실 1 epoch. target_align_only=True 면 1단계(target 인코더 미경유).

    1단계: L = CE(src) + λ_g·L_gravity + λ_pca·L_pca  (target 은 R 정렬에만, domain off).
    2단계: L = CE(src) + λ_da·L_domain + λ_g·L_gravity + λ_pca·L_pca (target 인코더 합류).
    두 단계 모두 gravity/pca 는 raw IMU + R 로 계산되어 인코더를 거치지 않는다.
    """
    model.train()
    # 쿼터니언 파라미터는 q (so3 원본의 w 자리). freeze 체크·gradient 진단 모두 q 로 본다.
    r_frozen = not model.r.q.requires_grad
    len_loader = min(len(train_loader), len(tgt_train_loader))
    tot = {"ce": 0.0, "dom": 0.0, "grav": 0.0, "pca": 0.0, "dom_acc": 0.0}
    grad_diag = None
    pbar = tqdm(enumerate(zip(train_loader, tgt_train_loader)), total=len_loader, desc=desc)
    for i, ((src_emg, src_imu, src_y), (tgt_emg, tgt_imu, tgt_y)) in pbar:
        src_emg, src_imu, src_y = src_emg.to(device), src_imu.to(device), src_y.to(device)
        tgt_emg, tgt_imu = tgt_emg.to(device), tgt_imu.to(device)
        optimizer.zero_grad()
        zero = torch.zeros((), device=device)

        # ---- source: 분류 CE (2단계에선 domain(0) 도) ----
        src_out, src_dom, _ = model(src_emg, src_imu, alpha=alpha, apply_r=False)
        loss_cls = criterion(src_out, src_y)

        if target_align_only:
            # 1단계: target 은 인코더/판별기를 거치지 않는다(아래 gravity/pca 로 R 만 정렬).
            # 인코더·BN 통계가 순수 source 로만 형성 → 흔들리는 target 이 인코더를 안 흔듦.
            loss_domain, dom_acc = zero, 0.0
        else:
            # 2단계: target 이 인코더+판별기에 합류 (R 정렬 후 domain(1)).
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

        # ---- (진단) R 파라미터(쿼터니언 q)에 대한 손실 항별 gradient 분해 ----
        Rparam = model.r.q
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
          target_join_epoch=10, freeze_r_epoch=None, save_cm=False, tag=""):
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

    model = LearnableRCDAN(num_classes=num_classes, post_r_norm=post_r_norm,
                           r_param=R_PARAM).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_domain = nn.CrossEntropyLoss()

    print("\n" + "=" * 60)
    print(f"Learnable-R CDAN align-first (2-phase) | R=quaternion  |  seed={seed}")
    print("Mode: JOINT | target IMU -> R(quat) -> encoder, align by CDAN adversarial")
    print(f"lambda: da={lambda_da} gravity={lambda_g} pca={lambda_pca} | post_r_norm={post_r_norm}")
    print(f"align-first: 1단계 epoch 0-{target_join_epoch-1} target 은 gravity/pca 로 R 만 정렬"
          f"(인코더 미경유) → 2단계 epoch {target_join_epoch}~ target 인코더/판별기 합류(α ramp)")
    print(f"R freeze: {'off (끝까지 학습)' if freeze_r_epoch is None else f'epoch {freeze_r_epoch} 부터 R 고정'}")
    print(f"Targeting {num_classes} classes on {device}")
    print("=" * 60)

    best_val_acc, best_target_acc = 0.0, 0.0
    eye3 = torch.eye(3, device=device)

    def geo_angle(A, B):
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

        # 1단계(align-only): target 은 인코더 미경유, R 만 gravity/pca 로 정렬. domain off(α=0).
        # 2단계: target 인코더 합류, α 는 합류 시점부터 0→1 ramp(급 GRL 충격 방지).
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

    print(f"\n최종 결과 | seed={seed} | Best Source Val: {best_val_acc:.2f}% | "
          f"Target at Best: {best_target_acc:.2f}% | Shift: {best_val_acc - best_target_acc:.2f}%")

    model.load_state_dict(torch.load(save_path, map_location=device))
    R_best = model.r.R.detach().cpu().numpy()
    r_path = os.path.join(R_DIR, f"R_learned_alignfirst_quat{suffix}_seed{seed}.npy")
    np.save(r_path, R_best)
    print(f"학습된 R 저장: {r_path}\n{np.array2string(R_best, precision=4)}")

    # ---- 최종 test 평가 (학습 종료 후 1회) -------------------------------
    # 여기서 처음으로 test 를 로드한다. 위 학습 루프는 test 로더 자체를 갖고 있지
    # 않으므로 model selection 에 test 가 개입할 여지가 없다.
    test_metrics, test_cm = evaluate_test(model, evaluate_r, le, device,
                                          seed=seed, batch_size=batch_size)

    if save_cm:
        # 혼동 행렬도 보고 수치와 같은 test 기준으로 그린다.
        v_true, v_preds = test_cm["source"]
        t_true, t_preds = test_cm["target"]
        src_te, tgt_te = test_metrics["source_test_acc"], test_metrics["target_test_acc"]
        save_confusion_matrix(
            v_true, v_preds, class_names,
            os.path.join(RESULT_DIR, f"{NAME}{suffix}_seed{seed}_source_test_cm.png"),
            f"Align-first quat Source TEST (seed={seed}, Acc: {src_te:.1f}%)", cmap="Blues")
        save_confusion_matrix(
            t_true, t_preds, class_names,
            os.path.join(RESULT_DIR, f"{NAME}{suffix}_seed{seed}_target_test_cm.png"),
            f"Align-first quat Target TEST (seed={seed}, Src: {src_te:.1f}% vs Tgt: {tgt_te:.1f}%)",
            cmap="Blues")
        print("혼동 행렬 시각화 저장 완료 (test 기준).")

    return {"seed": seed, "source_acc": best_val_acc, "target_acc": best_target_acc,
            "shift": best_val_acc - best_target_acc, **test_metrics}


def write_result_json(results, tag):
    mean, std = summarize_metrics(results)
    payload = {
        "tag": tag or "default", "modality": "imu_only", "model": NAME,
        "data_dir": os.environ.get("MM_DATA_DIR", "preprocessed_MM_raw"),
        "selection": SELECTION,             # model selection 기준 — 보고 시 명시할 것
        "reported_metric": REPORTED_METRIC,
        "seeds": [r["seed"] for r in results], "results": results,
        "mean": mean, "std": std,
    }
    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, f"{NAME}_result_{tag or 'default'}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"결과 JSON 저장: {path}")
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
    parser.add_argument("--lambda_g", type=float, default=1.0, help="L_gravity 가중치(both=1)")
    parser.add_argument("--lambda_pca", type=float, default=1.0, help="on-the-fly PCA 정렬 prior 가중치(both=1)")
    parser.add_argument("--target_join_epoch", type=int, default=10,
                        help="이 epoch 부터 target 이 인코더/판별기에 합류. 그 전엔 target 은 "
                             "gravity/pca 로 R 만 정렬(인코더 미경유). 0=끔(처음부터 합류=원본 base)")
    parser.add_argument("--post_r_norm", type=str, default="batchnorm",
                        choices=["none", "instance", "batchnorm"])
    parser.add_argument("--freeze_r_epoch", type=int, default=None,
                        help="이 epoch 까지 R 학습 후 고정. 미지정=끝까지 R 학습")
    parser.add_argument("--no_cm", action="store_true")
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    suffix = f"_{args.tag}" if args.tag else ""
    kw = dict(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, r_lr=args.r_lr,
              lambda_da=args.lambda_da, lambda_g=args.lambda_g, lambda_pca=args.lambda_pca,
              post_r_norm=args.post_r_norm, target_join_epoch=args.target_join_epoch,
              freeze_r_epoch=args.freeze_r_epoch, save_cm=not args.no_cm, tag=args.tag)

    if args.multi_seed:
        results = [train(seed=s, **kw) for s in args.seeds]
        os.makedirs(RESULT_DIR, exist_ok=True)
        summarize_results(results,
                          method_name=f"Learnable-R CDAN align-first quat{' | '+args.tag if args.tag else ''}",
                          save_path=os.path.join(RESULT_DIR, f"{NAME}{suffix}_summary.txt"))
    else:
        results = [train(seed=args.seed, **kw)]

    write_result_json(results, args.tag)
