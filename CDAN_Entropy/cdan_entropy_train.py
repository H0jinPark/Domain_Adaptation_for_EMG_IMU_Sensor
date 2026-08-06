"""CDAN+E (엔트로피 가중) + align-first 2단계 학습, IMU 단독, Learnable-R(SO(3)/Rodrigues).

기저: Learnable_R/learnable_r_cdan_alignfirst_train.py 를 그대로 가져오되(모델·손실은
Learnable_R.learnable_r_model 에서 import 재사용), **도메인 적대 손실에 CDAN+E 엔트로피
가중만 추가**했다. 나머지(SO(3) Rodrigues R, gravity/pca prior, target-join 10ep,
CosineAnnealingLR)는 원본 검증본과 동일.

문제의식(2026-07-13 진단): seed 별 target acc 편차의 주원인은 LR/R 이 아니라 CDAN 조건부
정렬이 **신뢰 못할 target 자기예측**으로 부트스트랩되는 under-determined 고정점이다. 원
CDAN 은 도메인 손실에 엔트로피 가중 w(H)=1+exp(-H(ŷ)) 를 붙여(=CDAN+E) 불확실한(고엔트로피)
샘플이 정렬을 끌고 가는 걸 억제한다. 이 폴더는 그 가중을 원본 base 에 얹어 재현성/편차가
개선되는지 본다.

CDAN+E 구현(Long et al. 공식 구현 방식):
  · 각 샘플의 class softmax 엔트로피 H = -Σ p·log p → 가중 w = 1 + exp(-H)
    (확신할수록 H↓ → w↑ ≈ 2, 불확실할수록 H↑ → w↓ ≈ 1).
  · source·target 각각 배치 내에서 w 를 합=1 로 정규화 → 가중 도메인 CE(reduction='none').
    정규화로 손실 스케일이 원래 mean 과 같아 λ_da 재튜닝 불필요.
  · 가중치는 detach — "불확실 샘플 다운웨이트"라는 재가중만 하고 추가 적대 결합은 안 함.

ablation: --no_entropy 로 가중을 꺼서 **동일 코드로 CDAN(plain) vs CDAN+E** 를 직접 A/B.

실행 예:
    MM_DATA_DIR=preprocessed_MM_raw_isotropic python CDAN_Entropy/cdan_entropy_train.py \
        --multi_seed --epochs 60 --target_join_epoch 10 --tag cdane10
    # ablation(엔트로피 off, 원본 base 재현):
    MM_DATA_DIR=preprocessed_MM_raw_isotropic python CDAN_Entropy/cdan_entropy_train.py \
        --multi_seed --epochs 60 --target_join_epoch 10 --no_entropy --tag plain10
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
sys.path.append(os.path.join(PROJECT_ROOT, "Learnable_R"))
from mm_data_loader import get_mm_dataloaders                       # noqa: E402
from mm_utils import set_seed, save_confusion_matrix, summarize_results  # noqa: E402
# 모델·기하 손실은 Learnable_R 의 검증본을 그대로 재사용(SO(3) Rodrigues R 포함).
from learnable_r_model import (  # noqa: E402
    LearnableRCDAN, gravity_loss, pca_alignment_loss)

WEIGHT_DIR = os.path.join(PROJECT_ROOT, "weights")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results", "CDAN_Entropy")
R_DIR = os.path.join(PROJECT_ROOT, "results", "R_matrices")
NAME = "cdan_entropy"


def pick_device():
    """CUDA → MPS(Apple Silicon) → CPU 순으로 사용 가능한 device 선택."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def entropy_weight(class_logits):
    """CDAN+E 샘플별 엔트로피 가중 w = 1 + exp(-H), H = -Σ p·log p (detach).

    확신하는(H 낮은) 예측일수록 w 가 커져(≈2), 불확실한(H 높은) 예측은 w≈1 로 억제된다.
    반환은 detach 되어 재가중에만 쓰이고 그래디언트는 흘리지 않는다.
    """
    p = F.softmax(class_logits, dim=1)
    H = -(p * torch.log(p + 1e-8)).sum(dim=1)       # (B,)
    return (1.0 + torch.exp(-H)).detach()           # (B,)


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
              use_entropy=True, log_r_grads=True, desc=""):
    """통합 손실 1 epoch. target_align_only=True 면 1단계(target 인코더 미경유).

    1단계: L = CE(src) + λ_g·L_gravity + λ_pca·L_pca  (target 은 R 정렬에만, domain off).
    2단계: L = CE(src) + λ_da·L_domain + λ_g·L_gravity + λ_pca·L_pca (target 인코더 합류).
    2단계 L_domain 은 use_entropy=True 면 CDAN+E 엔트로피 가중, False 면 원본 plain CDAN(mean).
    두 단계 모두 gravity/pca 는 raw IMU + R 로 계산되어 인코더를 거치지 않는다.
    """
    model.train()
    r_frozen = not model.r.w.requires_grad
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
            # tgt_out(class logits)은 엔트로피 가중 계산에 필요.
            tgt_out, tgt_dom, _ = model(tgt_emg, tgt_imu, alpha=alpha, apply_r=True)
            src_dlabel = torch.zeros(src_dom.size(0), dtype=torch.long, device=device)
            tgt_dlabel = torch.ones(tgt_dom.size(0), dtype=torch.long, device=device)
            if use_entropy:
                # CDAN+E: 확신 샘플 위주로 정렬. source/target 각각 합=1 정규화(스케일 유지).
                w_s = entropy_weight(src_out)
                w_t = entropy_weight(tgt_out)
                w_s = w_s / w_s.sum().clamp_min(1e-8)
                w_t = w_t / w_t.sum().clamp_min(1e-8)
                ce_s = F.cross_entropy(src_dom, src_dlabel, reduction="none")
                ce_t = F.cross_entropy(tgt_dom, tgt_dlabel, reduction="none")
                loss_domain = (w_s * ce_s).sum() + (w_t * ce_t).sum()
            else:
                # plain CDAN(원본 base): 무가중 mean.
                loss_domain = criterion_domain(src_dom, src_dlabel) + criterion_domain(tgt_dom, tgt_dlabel)
            with torch.no_grad():
                dom_pred = torch.cat([src_dom, tgt_dom]).argmax(1)
                dom_true = torch.cat([src_dlabel, tgt_dlabel])
                dom_acc = (dom_pred == dom_true).float().mean().item() * 100

        # ---- 기하 prior: raw target IMU + R (인코더 무관, 두 단계 모두 R 정렬) ----
        L_grav = gravity_loss(model.r.R, src_imu, tgt_imu) if (lam_g > 0 and not r_frozen) else zero
        L_pca = pca_alignment_loss(model.r.R, src_imu, tgt_imu) if (lam_pca > 0 and not r_frozen) else zero

        # ---- (진단) R 파라미터(so(3) w)에 대한 손실 항별 gradient 분해 ----
        Rparam = model.r.w
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


def train(seed=42, shuffle_seed=None, epochs=30, batch_size=64, lr=1e-3, r_lr=1e-2,
          lambda_da=1.0, lambda_g=1.0, lambda_pca=1.0, post_r_norm="batchnorm",
          target_join_epoch=10, freeze_r_epoch=None, use_entropy=True, save_cm=False, tag=""):
    # seed = weight init(및 dropout 등 전역 RNG) 시드. shuffle_seed = 학습 배치 셔플 시드.
    # shuffle_seed=None 이면 둘을 결합(=기존 동작). 정수를 주면 분리 → seed 영향 분해 실험.
    ss = seed if shuffle_seed is None else shuffle_seed
    set_seed(seed)
    device = pick_device()

    os.makedirs(WEIGHT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(R_DIR, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    run_id = f"is{seed}_ss{ss}"    # init-seed / shuffle-seed 식별자
    save_path = os.path.join(WEIGHT_DIR, f"{NAME}{suffix}_{run_id}_best_model.pth")

    # 셔플은 ss 의 별도 generator 로(weight init 시드와 분리). ss==seed 면 결합과 동치.
    train_loader, val_loader, tgt_train_loader, tgt_val_loader, num_classes, le = \
        get_mm_dataloaders(batch_size=batch_size, shuffle_seed=ss)
    class_names = le.classes_

    # 모델 생성 직전 init 시드로 재시드 → weight 초기화가 오직 seed 에만 의존(로더 생성이
    # 소비한 RNG 와 무관). 로더는 자체 generator 를 쓰므로 전역 RNG 는 그대로 보존됨.
    torch.manual_seed(seed)
    model = LearnableRCDAN(num_classes=num_classes, post_r_norm=post_r_norm).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_domain = nn.CrossEntropyLoss()

    print("\n" + "=" * 60)
    print(f"CDAN+E (entropy) align-first (2-phase) Training Start  |  "
          f"init_seed={seed}  shuffle_seed={ss}")
    print("Mode: JOINT | target IMU -> R(SO(3)/Rodrigues) -> encoder, align by CDAN adversarial")
    print(f"entropy weight(CDAN+E): {'ON (w=1+exp(-H), 확신 샘플 위주 정렬)' if use_entropy else 'OFF (plain CDAN)'}")
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
                      target_align_only=target_align_only, use_entropy=use_entropy,
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
    r_path = os.path.join(R_DIR, f"R_{NAME}{suffix}_{run_id}.npy")
    np.save(r_path, R_best)
    print(f"학습된 R 저장: {r_path}\n{np.array2string(R_best, precision=4)}")

    if save_cm:
        _, v_preds, v_true = evaluate_r(model, val_loader, device, apply_r=False)
        _, t_preds, t_true = evaluate_r(model, tgt_val_loader, device, apply_r=True)
        save_confusion_matrix(
            v_true, v_preds, class_names,
            os.path.join(RESULT_DIR, f"{NAME}{suffix}_{run_id}_source_cm.png"),
            f"CDAN+E Source ({run_id}, Val: {best_val_acc:.1f}%)", cmap="Blues")
        save_confusion_matrix(
            t_true, t_preds, class_names,
            os.path.join(RESULT_DIR, f"{NAME}{suffix}_{run_id}_target_cm.png"),
            f"CDAN+E Target ({run_id}, Src: {best_val_acc:.1f}% vs Tgt: {best_target_acc:.1f}%)",
            cmap="Blues")
        print("혼동 행렬 시각화 저장 완료.")

    return {"seed": seed, "shuffle_seed": ss, "source_acc": best_val_acc,
            "target_acc": best_target_acc, "shift": best_val_acc - best_target_acc}


def write_result_json(results, tag, use_entropy, decouple="none"):
    src = [r["source_acc"] for r in results]
    tgt = [r["target_acc"] for r in results]
    sh = [r["shift"] for r in results]
    ddof = 1 if len(results) > 1 else 0
    payload = {
        "tag": tag or "default", "modality": "imu_only", "model": NAME,
        "entropy_weight": use_entropy, "decouple": decouple,
        "data_dir": os.environ.get("MM_DATA_DIR", "preprocessed_MM_raw"),
        "init_seeds": [r["seed"] for r in results],
        "shuffle_seeds": [r.get("shuffle_seed") for r in results],
        "seeds": [r["seed"] for r in results], "results": results,
        "mean": {"source_acc": float(np.mean(src)), "target_acc": float(np.mean(tgt)),
                 "shift": float(np.mean(sh))},
        "std": {"source_acc": float(np.std(src, ddof=ddof)), "target_acc": float(np.std(tgt, ddof=ddof)),
                "shift": float(np.std(sh, ddof=ddof))},
    }
    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, f"{NAME}_result_{tag or 'default'}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"결과 JSON 저장: {path}")
    return path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
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
                             "gravity/pca 로 R 만 정렬(인코더 미경유). 0=끔(처음부터 합류=원본 base)")
    parser.add_argument("--post_r_norm", type=str, default="batchnorm",
                        choices=["none", "instance", "batchnorm"])
    parser.add_argument("--freeze_r_epoch", type=int, default=None,
                        help="이 epoch 까지 R 학습 후 고정. 미지정=끝까지 R 학습")
    parser.add_argument("--no_entropy", action="store_true",
                        help="CDAN+E 엔트로피 가중을 끄고 plain CDAN(원본 base)으로 학습(ablation).")
    parser.add_argument("--decouple", type=str, default="none",
                        choices=["none", "vary_init", "vary_shuffle"],
                        help="seed 영향 분해. vary_init=weight init 만 --seeds 로 변화(셔플은 "
                             "--fixed_seed 고정) → init 기여 측정. vary_shuffle=반대(배치 셔플만 "
                             "변화, init 고정) → 배치 구성 기여 측정. none=결합(기존).")
    parser.add_argument("--fixed_seed", type=int, default=0,
                        help="decouple 시 고정하는 쪽의 seed 값.")
    parser.add_argument("--no_cm", action="store_true")
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    use_entropy = not args.no_entropy
    suffix = f"_{args.tag}" if args.tag else ""
    kw = dict(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, r_lr=args.r_lr,
              lambda_da=args.lambda_da, lambda_g=args.lambda_g, lambda_pca=args.lambda_pca,
              post_r_norm=args.post_r_norm, target_join_epoch=args.target_join_epoch,
              freeze_r_epoch=args.freeze_r_epoch, use_entropy=use_entropy,
              save_cm=not args.no_cm, tag=args.tag)

    # decouple 모드: (init_seed, shuffle_seed) 쌍 목록을 만든다.
    #   vary_init    : init 만 --seeds 로 변화, shuffle 은 --fixed_seed 고정 → init 기여.
    #   vary_shuffle : shuffle 만 --seeds 로 변화, init 은 --fixed_seed 고정 → 배치 구성 기여.
    #   none         : 기존 결합(둘이 같은 seed).
    if args.decouple == "vary_init":
        pairs = [(s, args.fixed_seed) for s in args.seeds]
    elif args.decouple == "vary_shuffle":
        pairs = [(args.fixed_seed, s) for s in args.seeds]
    elif args.multi_seed:
        pairs = [(s, None) for s in args.seeds]
    else:
        pairs = [(args.seed, None)]

    results = [train(seed=isd, shuffle_seed=ssd, **kw) for isd, ssd in pairs]

    os.makedirs(RESULT_DIR, exist_ok=True)
    if len(results) > 1:
        label = "CDAN+E" if use_entropy else "CDAN(plain)"
        dec = f" | {args.decouple}(fix={args.fixed_seed})" if args.decouple != "none" else ""
        summarize_results(results,
                          method_name=f"{label} align-first{dec}{' | '+args.tag if args.tag else ''}",
                          save_path=os.path.join(RESULT_DIR, f"{NAME}{suffix}_summary.txt"))
    write_result_json(results, args.tag, use_entropy, decouple=args.decouple)
