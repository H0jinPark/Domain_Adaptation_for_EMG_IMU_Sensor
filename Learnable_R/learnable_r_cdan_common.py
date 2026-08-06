"""Learnable-R CDAN 학습 공통 로직 (세 R-파라미터화/스케줄 실험이 공유).

원본 learnable_r_cdan_train.py 의 통합 손실 학습을 그대로 담되, 세 실험 변형
(learnable_r_cdan_{lrsched,quat,matrix}_train.py)이 재사용하도록 일반화했다:

    L_total = CE(src) + λ_da·L_domain + λ_g·L_gravity + λ_pca·L_pca + λ_rot·L_rot

일반화 포인트 (원본 대비 바뀐 곳만):
  · R 의 leaf 파라미터를 so3(w,3) / quat(q,4) / matrix(M,3x3) 무관하게
    next(model.r.parameters()) 로 잡는다(freeze 판정·gradient 진단 공용).
  · λ_rot·L_rot(rotation_reg_loss) 추가 — 자유행렬(matrix)만 SO(3) 로 끌기 위해.
    so3/quat 는 구조적으로 회전이라 λ_rot=0.
  · 인코더/판별기 LR 워밍업(enc_warmup_epochs) — 아이디어 1. R 이 먼저 자리잡도록
    초반 인코더·판별기 lr 을 낮췄다가 선형 복귀. R lr 은 처음부터 풀강도.

데이터·모델·평가 규약은 원본과 동일(preprocessed_MM_raw_isotropic, source apply_r=False /
target apply_r=True, JOINT).
"""
import os
import sys
import json

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
    LearnableRCDAN, gravity_loss, pca_alignment_loss, rotation_reg_loss,
    class_pca_alignment_loss)
from learnable_r_test_eval import (  # noqa: E402
    SELECTION, REPORTED_METRIC, evaluate_test, summarize_metrics)

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
              device, alpha, lam_da, lam_g, lam_pca, lam_rot=0.0, pca_trim=0.0,
              lam_cpca=0.0, num_classes=10, log_r_grads=True, desc=""):
    """통합 손실 1 epoch (JOINT: 인코더+R+판별기 동시 학습, source CE 포함).

    L = CE(src) + λ_da·L_domain + λ_g·L_gravity + λ_pca·L_pca + λ_rot·L_rot + λ_cpca·L_cpca

    L_cpca = 클래스 조건부(soft) PCA 정렬 — source 라벨 + target pseudo(softmax)로 같은
    운동끼리 프레임을 맞춘다(운동 혼합 문제 해소). lam_cpca=0 이면 skip(다른 실험 무영향).

    R leaf 는 파라미터화와 무관하게 next(model.r.parameters())(so3=w, quat=q, matrix=M).
    R freeze(requires_grad=False) 시엔 gravity/pca/rot 이 gradient 0 → 계산 skip.
    """
    model.train()
    r_leaf = next(model.r.parameters())
    r_frozen = not r_leaf.requires_grad
    len_loader = min(len(train_loader), len(tgt_train_loader))
    tot = {"ce": 0.0, "dom": 0.0, "grav": 0.0, "pca": 0.0, "rot": 0.0, "cpca": 0.0, "dom_acc": 0.0}
    grad_diag = None
    pbar = tqdm(enumerate(zip(train_loader, tgt_train_loader)), total=len_loader, desc=desc)
    for i, ((src_emg, src_imu, src_y), (tgt_emg, tgt_imu, tgt_y)) in pbar:
        src_emg, src_imu, src_y = src_emg.to(device), src_imu.to(device), src_y.to(device)
        tgt_emg, tgt_imu = tgt_emg.to(device), tgt_imu.to(device)
        optimizer.zero_grad()

        # ---- source: 분류 CE + domain(0) ----
        src_out, src_dom, _ = model(src_emg, src_imu, alpha=alpha, apply_r=False)
        loss_cls = criterion(src_out, src_y)

        # ---- target: R 정렬 후 domain(1) (tgt_out=class-pca pseudo-label 용) ----
        tgt_out, tgt_dom, _ = model(tgt_emg, tgt_imu, alpha=alpha, apply_r=True)

        src_dlabel = torch.zeros(src_dom.size(0), dtype=torch.long, device=device)
        tgt_dlabel = torch.ones(tgt_dom.size(0), dtype=torch.long, device=device)
        loss_domain = criterion_domain(src_dom, src_dlabel) + criterion_domain(tgt_dom, tgt_dlabel)

        # ---- 기하 prior (R freeze 후엔 gradient 0 이라 skip) ----
        zero = torch.zeros((), device=device)
        L_grav = gravity_loss(model.r.R, src_imu, tgt_imu) if (lam_g > 0 and not r_frozen) else zero
        L_pca = pca_alignment_loss(model.r.R, src_imu, tgt_imu, pca_trim) if (lam_pca > 0 and not r_frozen) else zero
        # rotation reg — 자유행렬(matrix)만. so3/quat 는 λ_rot=0 으로 넘어옴.
        L_rot = rotation_reg_loss(model.r.R) if (lam_rot > 0 and not r_frozen) else zero
        # 클래스 조건부 PCA — 같은 운동끼리 프레임 정렬(source 라벨+target pseudo).
        L_cpca = (class_pca_alignment_loss(model.r.R, src_imu, src_y, tgt_imu, tgt_out, num_classes)
                  if (lam_cpca > 0 and not r_frozen) else zero)

        # ---- (진단) R leaf 에 대한 손실 항별 gradient 분해 ----
        if i == 0 and log_r_grads and r_leaf.requires_grad:
            def _gR(term):
                if not term.requires_grad:
                    return torch.zeros_like(r_leaf)
                g = torch.autograd.grad(term, r_leaf, retain_graph=True, allow_unused=True)[0]
                return g if g is not None else torch.zeros_like(r_leaf)
            g_dom, g_grav = _gR(lam_da * loss_domain), _gR(lam_g * L_grav)
            g_pca, g_rot = _gR(lam_pca * L_pca), _gR(lam_rot * L_rot)

            def _cos(a, b):
                na, nb = a.norm(), b.norm()
                return (a.flatten() @ b.flatten() / (na * nb + 1e-12)).item() if na > 0 and nb > 0 else 0.0
            grad_diag = {
                "g_dom": g_dom.norm().item(), "g_grav": g_grav.norm().item(),
                "g_pca": g_pca.norm().item(), "g_rot": g_rot.norm().item(),
                "cos_dom_pca": _cos(g_dom, g_pca), "cos_grav_pca": _cos(g_grav, g_pca),
            }

        loss = (loss_cls + lam_da * loss_domain + lam_g * L_grav + lam_pca * L_pca
                + lam_rot * L_rot + lam_cpca * L_cpca)
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
        tot["rot"] += L_rot.item()
        tot["cpca"] += L_cpca.item()
        tot["dom_acc"] += dom_acc
        pbar.set_postfix({"CE": f"{loss_cls.item():.3f}", "Dom": f"{loss_domain.item():.3f}",
                          "Grav": f"{L_grav.item():.3f}", "Pca": f"{L_pca.item():.3f}",
                          "Cpca": f"{L_cpca.item():.3f}", "Dacc": f"{dom_acc:.0f}%"})

    n = max(1, len_loader)
    out = {k: v / n for k, v in tot.items()}
    if grad_diag is not None:
        print(f"           ∂R: dom {grad_diag['g_dom']:.3f} grav {grad_diag['g_grav']:.3f} "
              f"pca {grad_diag['g_pca']:.3f} rot {grad_diag['g_rot']:.3f} | "
              f"cos(dom,pca) {grad_diag['cos_dom_pca']:+.2f} "
              f"cos(grav,pca) {grad_diag['cos_grav_pca']:+.2f}")
    out["grad_diag"] = grad_diag
    return out


def train(seed, name, r_param, epochs=30, batch_size=64, lr=1e-3, r_lr=1e-2,
          lambda_da=1.0, lambda_g=1.0, lambda_pca=1.0, lambda_rot=0.0,
          post_r_norm="batchnorm", freeze_r_epoch=None,
          enc_warmup_epochs=0, enc_lr_floor=0.1, pca_trim=0.0,
          lambda_cpca=0.0, cpca_ramp=0, save_cm=False, tag=""):
    """단일 seed 학습. name=파일별 식별자(저장/결과 naming), r_param=R 파라미터화.

    enc_warmup_epochs>0 이면 인코더/판별기 group lr 을 초반 enc_warmup_epochs 동안
    floor→1.0 선형 워밍업(아이디어 1). R group lr 은 처음부터 풀강도라 R 이 먼저 수렴.

    pca_trim>0 이면 PCA 정렬 손실의 주축 계산 전 고에너지 이상치 윈도를 제거해 배치
    민감도를 낮춘다(PCA outlier trim 실험).

    lambda_cpca>0 이면 클래스 조건부(soft) PCA 정렬을 켠다(같은 운동끼리 프레임 맞춤).
    cpca_ramp>0 이면 λ_cpca 를 초반 cpca_ramp epoch 동안 0→1 선형 ramp(pseudo-label 이
    성숙하는 사이 yaw 항을 서서히 키움). 보통 aggregate PCA(lambda_pca)는 0 으로 끄고 씀.
    """
    set_seed(seed)
    device = pick_device()

    os.makedirs(WEIGHT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(R_DIR, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    save_path = os.path.join(WEIGHT_DIR, f"{name}{suffix}_seed{seed}_best_model.pth")

    train_loader, val_loader, tgt_train_loader, tgt_val_loader, num_classes, le = \
        get_mm_dataloaders(batch_size=batch_size)
    class_names = le.classes_

    model = LearnableRCDAN(num_classes=num_classes, post_r_norm=post_r_norm,
                           r_param=r_param).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_domain = nn.CrossEntropyLoss()

    print("\n" + "=" * 60)
    print(f"{name} (unified loss, JOINT) Training Start  |  seed={seed}")
    print(f"Mode: JOINT | target IMU -> R({r_param}) -> encoder, align by CDAN adversarial")
    print(f"lambda: da={lambda_da} gravity={lambda_g} pca={lambda_pca} rot={lambda_rot} "
          f"| post_r_norm={post_r_norm}")
    if pca_trim > 0:
        print(f"PCA outlier trim: ON (trim_sigma={pca_trim}) — 주축 계산 전 고에너지 이상치 윈도 제거")
    if lambda_cpca > 0:
        print(f"class-conditional PCA: ON (λ_cpca={lambda_cpca}"
              f"{f', ramp {cpca_ramp}ep' if cpca_ramp > 0 else ''}) — 같은 운동끼리 프레임 정렬"
              f"(source 라벨 + target pseudo softmax)")
    if enc_warmup_epochs > 0:
        print(f"encoder/discriminator LR warmup: {enc_warmup_epochs} epochs "
              f"(floor {enc_lr_floor}→1.0), R lr full from start")
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
    other_params = [p for n_, p in model.named_parameters() if not n_.startswith("r.")]
    optimizer = optim.AdamW([
        {"params": other_params, "lr": lr, "weight_decay": 1e-3},   # group 0: encoder/discriminator
        {"params": r_params, "lr": r_lr, "weight_decay": 0.0},      # group 1: R
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    for epoch in range(epochs):
        if freeze_r_epoch is not None and epoch == freeze_r_epoch:
            for p in model.r.parameters():
                p.requires_grad_(False)
            print(f"--- Epoch {epoch+1}: R freeze (이후 R 고정·인코더/판별기만 학습) ---")

        # 인코더/판별기 LR 워밍업(아이디어 1): cosine 값 위에 이 에폭만 배수를 곱한다.
        # 다음 scheduler.step() 이 base_lr 로 재계산하므로 배수는 누적되지 않는다.
        if enc_warmup_epochs > 0 and epoch < enc_warmup_epochs:
            warm = enc_lr_floor + (1.0 - enc_lr_floor) * (epoch + 1) / enc_warmup_epochs
            optimizer.param_groups[0]["lr"] *= warm

        alpha = alpha_at(epoch / epochs)
        # 클래스 조건부 PCA 가중치 ramp: pseudo-label 이 성숙하는 사이 yaw 항을 서서히 키움.
        lam_cpca = lambda_cpca
        if lambda_cpca > 0 and cpca_ramp > 0:
            lam_cpca = lambda_cpca * min(1.0, (epoch + 1) / cpca_ramp)
        s = run_epoch(model, optimizer, train_loader, tgt_train_loader, criterion,
                      criterion_domain, device, alpha, lambda_da, lambda_g, lambda_pca,
                      lambda_rot, pca_trim=pca_trim, lam_cpca=lam_cpca, num_classes=num_classes,
                      desc=f"Seed {seed} | Epoch [{epoch+1:02d}/{epochs:02d}]")
        scheduler.step()
        eval_and_log(epoch + 1, epochs,
                     extra=f"CE: {s['ce']:.3f} | Dom: {s['dom']:.3f} (Dacc {s['dom_acc']:.0f}%, α={alpha:.2f}) | "
                           f"Grav: {s['grav']:.3f} | Pca: {s['pca']:.3f} | Cpca: {s['cpca']:.3f} (λ={lam_cpca:.2f}) | Rot: {s['rot']:.3f}")

    print(f"\n최종 결과 | seed={seed} | Best Source Val: {best_val_acc:.2f}% | "
          f"Target at Best: {best_target_acc:.2f}% | Shift: {best_val_acc - best_target_acc:.2f}%")

    model.load_state_dict(torch.load(save_path, map_location=device))
    R_best = model.r.R.detach().cpu().numpy()
    r_path = os.path.join(R_DIR, f"R_{name}{suffix}_seed{seed}.npy")
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
            os.path.join(RESULT_DIR, f"{name}{suffix}_seed{seed}_source_test_cm.png"),
            f"{name} Source TEST (seed={seed}, Acc: {src_te:.1f}%)", cmap="Blues")
        save_confusion_matrix(
            t_true, t_preds, class_names,
            os.path.join(RESULT_DIR, f"{name}{suffix}_seed{seed}_target_test_cm.png"),
            f"{name} Target TEST (seed={seed}, Src: {src_te:.1f}% vs Tgt: {tgt_te:.1f}%)",
            cmap="Blues")
        print("혼동 행렬 시각화 저장 완료 (test 기준).")

    return {"seed": seed, "source_acc": best_val_acc, "target_acc": best_target_acc,
            "shift": best_val_acc - best_target_acc, **test_metrics}


def write_result_json(results, name, tag):
    mean, std = summarize_metrics(results)
    payload = {
        "tag": tag or "default", "modality": "imu_only", "model": name,
        "data_dir": os.environ.get("MM_DATA_DIR", "preprocessed_MM_raw"),
        "selection": SELECTION,             # model selection 기준 — 보고 시 명시할 것
        "reported_metric": REPORTED_METRIC,
        "seeds": [r["seed"] for r in results], "results": results,
        "mean": mean, "std": std,
    }
    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, f"{name}_result_{tag or 'default'}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"결과 JSON 저장: {path}")
    return path


def run_and_summarize(args, name, r_param, method_label, extra_kw=None):
    """세 entry 스크립트 공통 실행 흐름: seed 루프 → summary → result json.

    extra_kw 로 아이디어별 추가 인자(예: lambda_rot, enc_warmup_epochs)를 넘긴다.
    """
    suffix = f"_{args.tag}" if args.tag else ""
    kw = dict(name=name, r_param=r_param, epochs=args.epochs, batch_size=args.batch_size,
              lr=args.lr, r_lr=args.r_lr, lambda_da=args.lambda_da, lambda_g=args.lambda_g,
              lambda_pca=args.lambda_pca, post_r_norm=args.post_r_norm,
              freeze_r_epoch=args.freeze_r_epoch, save_cm=not args.no_cm, tag=args.tag)
    if extra_kw:
        kw.update(extra_kw)

    if args.multi_seed:
        results = [train(seed=s, **kw) for s in args.seeds]
        os.makedirs(RESULT_DIR, exist_ok=True)
        summarize_results(results, method_name=method_label + (f" | {args.tag}" if args.tag else ""),
                          save_path=os.path.join(RESULT_DIR, f"{name}{suffix}_summary.txt"))
    else:
        results = [train(seed=args.seed, **kw)]

    write_result_json(results, name, args.tag)
    return results


def add_common_args(parser):
    """세 스크립트 공통 인자(원본 learnable_r_cdan_train.py 와 동일 규약)."""
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi_seed", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3, help="인코더/판별기 학습률")
    parser.add_argument("--r_lr", type=float, default=1e-2, help="learnable R 전용 학습률")
    parser.add_argument("--lambda_da", type=float, default=1.0, help="domain(CDAN) 손실 가중치")
    parser.add_argument("--lambda_g", type=float, default=1.0, help="L_gravity 가중치(isotropic 에서 의미)")
    parser.add_argument("--lambda_pca", type=float, default=1.0, help="on-the-fly PCA 정렬 prior 가중치")
    parser.add_argument("--post_r_norm", type=str, default="batchnorm",
                        choices=["none", "instance", "batchnorm"])
    parser.add_argument("--freeze_r_epoch", type=int, default=None,
                        help="이 epoch 까지 R 학습 후 고정. 미지정=끝까지 학습")
    parser.add_argument("--no_cm", action="store_true")
    parser.add_argument("--tag", type=str, default="")
    return parser
