"""Physics-informed CDAN 학습 (멀티모달 + EMG→IMU 디코더, multi-seed).

Multimodal/CDAN_train.py 와 분류·도메인 경로가 **수치적으로 동일**하고, 거기에
EMG 인코더 출력에서 IMU 를 복원하는 디코더와 두 보조 손실만 더한다:

    L = L_cls + w_d·L_domain + λ_rec·L_rec + λ_jerk·L_jerk

  L_rec  : 복원 IMU vs 실제 IMU (huber 기본)
  L_jerk : 복원 IMU 의 jerk 크기 벌점 (예측에만 걸리는 minimum-jerk prior)

λ_rec=λ_jerk=0 이면 기존 CDAN 과 같은 대조군이 된다(디코더는 계산되지만 손실에
기여하지 않아 gradient 가 백본으로 흐르지 않는다). 이 동치성은 physics_model.py 를
직접 실행하면 검증된다.

보조 손실과 라벨
  두 손실 다 짝꿍 IMU 만 쓰고 라벨을 안 본다 → **target 배치에도 그대로 걸린다.**
  기본값이 그것이고, --no_aux_on_target 으로 source 에만 걸어 기여를 분리한다.
  두 도메인에 걸 때는 합이 아니라 **평균**을 쓴다. 그래야 플래그를 켜고 꺼도 λ 의
  의미가 안 변해서 ablation 이 공정하다.

model selection
  기존 멀티모달 CDAN 과 동일하게 **target val (oracle)** 로 고른다. 비교 대상과
  규약을 맞추기 위한 것이며, 결과 json 에 그대로 명시한다. 보고는 test.

데이터
  MM_DATA_DIR 로 지정(기본 preprocessed_MM_pca). 재구성 타깃이 IMU 이므로 두 도메인이
  같은 축 프레임이어야 한다 — R 이 적용된 폴더를 쓸 것. raw(R=I) 로 돌리면 target 의
  축이 어긋난 채로 복원을 요구하게 되어 보조 손실이 오히려 해가 된다.
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
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "Multimodal"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mm_data_loader import get_mm_dataloaders, get_mm_test_loaders   # noqa: E402
from mm_utils import set_seed, evaluate, save_confusion_matrix, \
    summarize_results                                                # noqa: E402
from physics_model import PhysicsInformedCDAN, fusion_info           # noqa: E402
from physics_losses import imu_reconstruction_loss, imu_jerk_loss, \
    imu_orientation_loss, imu_axes_loss, physics_diagnostics, \
    gravity_magnitude                                                # noqa: E402

# isotropic 판정 임계 — 실측 중앙값이 z-score 0.437 / isotropic 0.932 로 갈려서 중간에 둔다.
GRAVITY_MIN = 0.70

WEIGHT_DIR = os.path.join(PROJECT_ROOT, "weights")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results", "EMG_physics")


# ----------------------------------------------------------------------
# 단일 seed 학습
# ----------------------------------------------------------------------
def train_physics_cdan(seed=42, epochs=30, batch_size=64, lr=1e-3,
                       domain_weight=1.0, lambda_rec=0.0, lambda_jerk=0.0,
                       aux_on_target=True, recon_loss="huber",
                       decoder_width=128, save_cm=False, tag="",
                       imu_interp=False, lambda_orient=0.0, lambda_axes=0.0,
                       skip_isotropic_check=False):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(WEIGHT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    save_path = os.path.join(WEIGHT_DIR, f"phys_cdan{suffix}_seed{seed}_best_model.pth")

    train_loader, val_loader, tgt_train_loader, tgt_val_loader, num_classes, le = \
        get_mm_dataloaders(batch_size=batch_size)
    class_names = le.classes_

    model = PhysicsInformedCDAN(num_classes=num_classes,
                                decoder_width=decoder_width,
                                imu_interp=imu_interp).to(device)

    criterion_class = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_domain = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    print("\n" + "=" * 60)
    print(f"Physics-informed CDAN Training Start  |  seed={seed}")
    print(f"lambda_rec={lambda_rec}  lambda_jerk={lambda_jerk}  "
          f"lambda_orient={lambda_orient}  lambda_axes={lambda_axes}  "
          f"aux_on_target={aux_on_target}  recon={recon_loss}")
    print(f"fusion: {model.fusion_info()}")
    print(f"Targeting {num_classes} classes on {device}")

    # orient/axes 는 중력 DC 와 축간 크기비가 살아 있어야 의미가 있다. 데이터 이름이 아니라
    # 첫 배치의 실측값으로 판정한다 — 폴더명 규칙에 의존하면 조용히 틀린 데이터로 돈다.
    if (lambda_orient > 0 or lambda_axes > 0) and not skip_isotropic_check:
        probe_imu = next(iter(train_loader))[1]
        g = gravity_magnitude(probe_imu)
        print(f"[isotropic 확인] 윈도우 평균가속도 크기 중앙값 = {g:.3f} "
              f"(임계 {GRAVITY_MIN}, isotropic ≈0.93 / 축별 z-score ≈0.44)")
        if g < GRAVITY_MIN:
            raise SystemExit(
                f"중력 DC 가 없는 데이터다 (측정 {g:.3f} < {GRAVITY_MIN}). "
                f"MM_DATA_DIR={os.environ.get('MM_DATA_DIR', 'preprocessed_MM_raw')} 는 "
                f"축별 z-score 판으로 보인다 — orient/axes 손실이 무의미하다.\n"
                f"  해결: MM_DATA_DIR=preprocessed_MM_pca_isotropic 로 실행하거나\n"
                f"        python data_preprocess_MM.py --method pca --imu_norm isotropic 로 생성.\n"
                f"  의도한 것이라면 --skip_isotropic_check 로 넘길 수 있다.")
    print("=" * 60)

    best_val_acc, best_target_acc = 0.0, 0.0

    for epoch in range(epochs):
        model.train()

        len_loader = min(len(train_loader), len(tgt_train_loader))
        total_c_loss, total_d_loss, total_d_acc = 0.0, 0.0, 0.0
        total_rec, total_jerk = 0.0, 0.0
        total_orient, total_axes = 0.0, 0.0
        diag_acc, diag_n = {}, 0

        # GRL alpha 스케줄 (기존 CDAN 과 동일)
        p = float(epoch) / epochs
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

        pbar = tqdm(enumerate(zip(train_loader, tgt_train_loader)),
                    total=len_loader, desc=f"Seed {seed} | Epoch [{epoch+1:02d}/{epochs:02d}]")

        for i, ((src_emg, src_imu, src_y), (tgt_emg, tgt_imu, tgt_y)) in pbar:
            src_emg, src_imu, src_y = src_emg.to(device), src_imu.to(device), src_y.to(device)
            tgt_emg, tgt_imu = tgt_emg.to(device), tgt_imu.to(device)

            optimizer.zero_grad()

            src_domain_label = torch.zeros(src_emg.size(0), dtype=torch.long, device=device)
            tgt_domain_label = torch.ones(tgt_emg.size(0), dtype=torch.long, device=device)

            # Step 1. Source 분리 forward
            src_class_out, src_domain_out, src_imu_hat = model(src_emg, src_imu, alpha=alpha)
            loss_s_label = criterion_class(src_class_out, src_y)
            loss_s_domain = criterion_domain(src_domain_out, src_domain_label)

            # Step 2. Target 분리 forward
            _, tgt_domain_out, tgt_imu_hat = model(tgt_emg, tgt_imu, alpha=alpha)
            loss_t_domain = criterion_domain(tgt_domain_out, tgt_domain_label)

            # Step 3. 물리 보조 손실 — 라벨 불필요라 target 에도 걸린다.
            #         두 도메인에 걸 때는 평균(합 아님) → λ 의미가 플래그와 무관해진다.
            #         λ=0 인 항은 아예 계산하지 않는다 — 0 을 곱해 더하면 값은 같아도
            #         쓸데없는 그래프가 생기고, λ=0 대조군의 "기존 CDAN 과 동일" 이 흐려진다.
            def _both(fn, s_hat, s_true, t_hat, t_true):
                v = fn(s_hat, s_true)
                return 0.5 * (v + fn(t_hat, t_true)) if aux_on_target else v

            zero = torch.zeros((), device=device)
            rec = (_both(lambda p, t: imu_reconstruction_loss(p, t, kind=recon_loss),
                         src_imu_hat, src_imu, tgt_imu_hat, tgt_imu)
                   if lambda_rec > 0 else zero)
            jerk = ((0.5 * (imu_jerk_loss(src_imu_hat) + imu_jerk_loss(tgt_imu_hat))
                     if aux_on_target else imu_jerk_loss(src_imu_hat))
                    if lambda_jerk > 0 else zero)
            orient = (_both(imu_orientation_loss, src_imu_hat, src_imu, tgt_imu_hat, tgt_imu)
                      if lambda_orient > 0 else zero)
            axes = (_both(imu_axes_loss, src_imu_hat, src_imu, tgt_imu_hat, tgt_imu)
                    if lambda_axes > 0 else zero)

            # Step 4. 통합 손실
            class_loss = loss_s_label
            domain_loss = loss_s_domain + loss_t_domain
            loss = (class_loss + domain_weight * domain_loss
                    + lambda_rec * rec + lambda_jerk * jerk
                    + lambda_orient * orient + lambda_axes * axes)

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                domain_pred = torch.cat([src_domain_out.argmax(1), tgt_domain_out.argmax(1)])
                domain_true = torch.cat([src_domain_label, tgt_domain_label])
                domain_acc = (domain_pred == domain_true).float().mean().item()

            total_c_loss += class_loss.item()
            total_d_loss += domain_loss.item()
            total_d_acc += domain_acc
            total_rec += float(rec.item())
            total_jerk += float(jerk.item())
            total_orient += float(orient.item())
            total_axes += float(axes.item())

            # 진단은 매 배치 돌릴 필요 없다 (상관계산이 싸지 않다)
            if i % 20 == 0:
                d = physics_diagnostics(tgt_imu_hat if aux_on_target else src_imu_hat,
                                        tgt_imu if aux_on_target else src_imu)
                for k, v in d.items():
                    diag_acc[k] = diag_acc.get(k, 0.0) + v
                diag_n += 1

            pbar.set_postfix({
                "Class": f"{class_loss.item():.4f}",
                "Domain": f"{domain_loss.item():.4f}",
                "Rec": f"{rec.item():.4f}",
                "Jerk": f"{jerk.item():.4f}",
                "Ori": f"{orient.item():.4f}",
                "Axe": f"{axes.item():.4f}",
            })

        scheduler.step()

        val_acc, _, _ = evaluate(model, val_loader, device, needs_alpha=True)
        tgt_acc, _, _ = evaluate(model, tgt_val_loader, device, needs_alpha=True)

        # 기존 멀티모달 CDAN 과 동일한 규약(target val oracle)으로 고른다.
        is_best = tgt_acc > best_target_acc
        if is_best:
            best_val_acc = val_acc
            best_target_acc = tgt_acc
            torch.save(model.state_dict(), save_path)

        dg = {k: v / max(diag_n, 1) for k, v in diag_acc.items()}
        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | "
              f"Class: {total_c_loss/len_loader:.4f} | "
              f"Domain: {total_d_loss/len_loader:.4f} | "
              f"DomAcc: {total_d_acc/len_loader*100:.2f}% | "
              f"Rec: {total_rec/len_loader:.4f} | Jerk: {total_jerk/len_loader:.4f} | "
              f"Ori: {total_orient/len_loader:.4f} | Axe: {total_axes/len_loader:.4f} | "
              f"corr: {dg.get('recon_corr', 0):.3f} | "
              f"jerkR: {dg.get('jerk_ratio', 0):.3f} | "
              f"ori°: {dg.get('orient_deg', 0):.1f} | "
              f"axe°: {dg.get('axis_deg', 0):.1f} | "
              f"Source Val: {val_acc:.2f}% | Target Val: {tgt_acc:.2f}%"
              f"{'  (best)' if is_best else ''}")

    print(f"\n[val] seed={seed} | Best Source Val: {best_val_acc:.2f}% | "
          f"Target Val at best: {best_target_acc:.2f}% | "
          f"Shift: {best_val_acc - best_target_acc:.2f}%")

    # ---- 최종 test 평가 (학습 종료 후 1회) -------------------------------
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_loader, tgt_test_loader = get_mm_test_loaders(le, batch_size=batch_size)
    src_test_acc, v_preds, v_true = evaluate(model, test_loader, device, needs_alpha=True)
    tgt_test_acc, t_preds, t_true = evaluate(model, tgt_test_loader, device, needs_alpha=True)
    print(f"[test] seed={seed} | Source Test: {src_test_acc:.2f}% | "
          f"Target Test: {tgt_test_acc:.2f}% | "
          f"Shift: {src_test_acc - tgt_test_acc:.2f}%   <-- 보고 수치")

    # 최종 재구성 품질 (target test **전체**) — 물리 손실이 실제로 뭘 배웠는지 남긴다.
    # 배치 하나로 재면 표본이 작아 seed 간 비교(r(진단, 정확도))가 잡음에 묻힌다.
    model.eval()
    acc_diag, nb = {}, 0
    with torch.no_grad():
        for emg_b, imu_b, _ in tgt_test_loader:
            imu_b = imu_b.to(device)
            _, _, hat_b = model(emg_b.to(device), imu_b, alpha=0.0)
            for k, v in physics_diagnostics(hat_b, imu_b).items():
                acc_diag[k] = acc_diag.get(k, 0.0) + v
            nb += 1
    final_diag = {k: v / max(nb, 1) for k, v in acc_diag.items()}
    print(f"[recon] target test 전체({nb}배치) | corr={final_diag['recon_corr']:.3f} "
          f"jerk_ratio={final_diag['jerk_ratio']:.3f} | "
          f"중력방향오차={final_diag['orient_deg']:.1f}° "
          f"주축오차={final_diag['axis_deg']:.1f}°  (무작위 기준선 90°)")

    if save_cm:
        save_confusion_matrix(
            v_true, v_preds, class_names,
            os.path.join(RESULT_DIR, f"phys_cdan{suffix}_seed{seed}_source_test_cm.png"),
            f"Physics-CDAN Source TEST (seed={seed}, Acc: {src_test_acc:.1f}%)", cmap="Purples")
        save_confusion_matrix(
            t_true, t_preds, class_names,
            os.path.join(RESULT_DIR, f"phys_cdan{suffix}_seed{seed}_target_test_cm.png"),
            f"Physics-CDAN Target TEST (seed={seed}, Src: {src_test_acc:.1f}% vs Tgt: {tgt_test_acc:.1f}%)",
            cmap="Purples")
        print("혼동 행렬 저장 완료 (test 기준).")

    return {
        "seed": seed,
        "source_acc": best_val_acc,
        "target_acc": best_target_acc,
        "shift": best_val_acc - best_target_acc,
        "source_test_acc": src_test_acc,
        "target_test_acc": tgt_test_acc,
        "test_shift": src_test_acc - tgt_test_acc,
        **{f"final_{k}": v for k, v in final_diag.items()},
    }


# ----------------------------------------------------------------------
# λ 규모 프로브 — 학습 전에 각 항의 크기를 재서 λ 를 감으로 고르지 않게 한다
# ----------------------------------------------------------------------
def probe(batch_size=64, decoder_width=128, recon_loss="huber", imu_interp=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(0)
    train_loader, _, tgt_train_loader, _, num_classes, _ = \
        get_mm_dataloaders(batch_size=batch_size)
    model = PhysicsInformedCDAN(num_classes=num_classes,
                                decoder_width=decoder_width,
                                imu_interp=imu_interp).to(device)
    model.train()
    criterion_class = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_domain = nn.CrossEntropyLoss()

    (se, si, sy) = next(iter(train_loader))
    (te, ti, _) = next(iter(tgt_train_loader))
    se, si, sy = se.to(device), si.to(device), sy.to(device)
    te, ti = te.to(device), ti.to(device)

    with torch.no_grad():
        sc, sd, sh = model(se, si, alpha=1.0)
        _, td, th = model(te, ti, alpha=1.0)
        c = criterion_class(sc, sy).item()
        d = (criterion_domain(sd, torch.zeros(se.size(0), dtype=torch.long, device=device))
             + criterion_domain(td, torch.ones(te.size(0), dtype=torch.long, device=device))).item()
        r = 0.5 * (imu_reconstruction_loss(sh, si, kind=recon_loss)
                   + imu_reconstruction_loss(th, ti, kind=recon_loss)).item()
        j = 0.5 * (imu_jerk_loss(sh) + imu_jerk_loss(th)).item()
        o = 0.5 * (imu_orientation_loss(sh, si) + imu_orientation_loss(th, ti)).item()
        a = 0.5 * (imu_axes_loss(sh, si) + imu_axes_loss(th, ti)).item()
        dg = physics_diagnostics(th, ti)
        # 참고: 정답 IMU 자체의 항 크기 (디코더가 완벽할 때의 하한/기준)
        j_true = imu_jerk_loss(ti).item()
        g_src, g_tgt = gravity_magnitude(si), gravity_magnitude(ti)

    print("\n" + "=" * 66)
    print(" λ 규모 프로브 — 초기화 직후 1배치, 각 손실 항의 크기")
    print("=" * 66)
    print(f"  L_class  (CE)                 = {c:.4f}")
    print(f"  L_domain (CE x2)              = {d:.4f}")
    print(f"  L_rec    ({recon_loss:5s})              = {r:.4f}")
    print(f"  L_jerk   (예측)                = {j:.6f}")
    print(f"  L_jerk   (정답 IMU 기준값)      = {j_true:.6f}")
    print(f"  L_orient (1-cos, 범위 0~2)     = {o:.4f}   (무작위 방향이면 ≈1.0)")
    print(f"  L_axes   (상대 Frobenius²)     = {a:.4f}   (완전 불일치면 ≈1 이상)")
    print(f"  진단: recon_corr={dg['recon_corr']:.3f}  "
          f"jerk_rms pred={dg['jerk_rms_pred']:.2f} true={dg['jerk_rms_true']:.2f}")
    print(f"        중력방향오차 {dg['orient_deg']:.1f}°  주축오차 {dg['axis_deg']:.1f}°  "
          f"(무작위 기준선 90°)")
    print(f"  데이터: 윈도우 평균가속도 크기 중앙값 src={g_src:.3f} tgt={g_tgt:.3f}  "
          f"(isotropic ≈0.93 / 축별 z-score ≈0.44, 임계 {GRAVITY_MIN})")
    print("\n  λ 고르는 법 — 보조항이 분류 손실의 대략 5~30% 기여가 되게 잡는다:")
    for frac in (0.05, 0.1, 0.3):
        print(f"    기여 {frac*100:>4.0f}% :  λ_rec ≈ {frac*c/max(r,1e-12):>8.3f}"
              f"   λ_jerk ≈ {frac*c/max(j,1e-12):>8.3f}"
              f"   λ_orient ≈ {frac*c/max(o,1e-12):>7.3f}"
              f"   λ_axes ≈ {frac*c/max(a,1e-12):>7.4f}")
    print("\n  주의 — 초기화 직후 값이라 학습이 진행되면 L_rec 은 내려가고 L_class 도")
    print("  내려간다. 위 값은 출발점의 자릿수를 잡는 용도이지 최적 λ 가 아니다.")
    print("  L_jerk 가 정답 기준값보다 훨씬 크면 디코더가 아직 잡음을 뱉는 중이고,")
    print("  학습 후에도 ratio 가 1 을 크게 밑돌면 λ_jerk 과평활이다.")
    print("  **λ_axes 는 특히 이 표를 믿지 말 것** — L_axes 는 참 공분산으로 나눈 상대량이라")
    print("  학습 안 된 디코더의 과대 출력(jerk_rms 비 수십 배)에 그대로 부풀어 있다.")
    print("  디코더 스케일이 잡히면 급락하므로, λ_orient 와 비슷한 기여가 되도록")
    print("  1~2 에폭 돌려보고 실측 L_axes 로 다시 잡는 편이 맞다.")
    print("  L_orient 는 1-cos 라 [0,2] 로 유계이고 무작위 방향이 ≈1 이므로 표를 믿어도 된다.")
    print("=" * 66)


# ----------------------------------------------------------------------
# 결과 JSON
# ----------------------------------------------------------------------
def _split_provenance():
    d = os.environ.get("MM_DATA_DIR", "preprocessed_MM_raw")
    out = {}
    for fname, key in (("split_manifest.json", None), ("preproc_config.json", "preproc")):
        path = os.path.join(PROJECT_ROOT, d, fname)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                m = json.load(f)
        except Exception:
            continue
        if key:
            out[key] = m
        else:
            out["split_by"] = m.get("split_by", "session")
            for k in ("val_subjects", "test_subjects", "split_ratio", "split_seed"):
                if m.get(k) is not None:
                    out[k] = m[k]
    return out


def write_result_json(results, tag, cfg):
    ddof = 1 if len(results) > 1 else 0
    keys = ["source_acc", "target_acc", "shift",
            "source_test_acc", "target_test_acc", "test_shift",
            "final_recon_corr", "final_jerk_ratio",
            "final_orient_deg", "final_axis_deg"]
    cols = {k: [r[k] for r in results] for k in keys if k in results[0]}
    payload = {
        "tag": tag or "default",
        "modality": "multimodal_physics",
        "mode": "cdan_physics_informed",
        "data_dir": os.environ.get("MM_DATA_DIR", "preprocessed_MM_raw"),
        **_split_provenance(),
        **cfg,
        "selection": "target_val (oracle)",
        "reported_metric": "target_test_acc",
        "seeds": [r["seed"] for r in results],
        "results": results,
        "mean": {k: float(np.mean(v)) for k, v in cols.items()},
        "std":  {k: float(np.std(v, ddof=ddof)) for k, v in cols.items()},
    }
    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, f"phys_cdan_result_{tag or 'default'}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"결과 JSON 저장: {path}")
    return path


def parse_args():
    p = argparse.ArgumentParser(description="Physics-informed CDAN (EMG→IMU 보조 손실)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--multi_seed", action="store_true")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--domain_weight", type=float, default=1.0)
    p.add_argument("--lambda_rec", type=float, default=0.0,
                   help="IMU 재구성 손실 가중치. 0 이면 기존 CDAN 과 동일한 대조군.")
    p.add_argument("--lambda_jerk", type=float, default=0.0,
                   help="jerk 최소화 손실 가중치. 0 이면 물리 prior 없음.")
    p.add_argument("--lambda_orient", type=float, default=0.0,
                   help="중력방향 정합 손실 가중치 — 복원 IMU 의 시간평균 방향을 참 IMU 와 "
                        "맞춘다(윈도우당 3 dof). isotropic 데이터 전용. "
                        "이름이 Learnable_R 의 --lambda_g 와 다른 것은 의도한 것으로, "
                        "그쪽은 R 을 학습시키는 항이고 이쪽은 디코더 출력에 거는 항이다.")
    p.add_argument("--lambda_axes", type=float, default=0.0,
                   help="주축 구조 정합 손실 가중치 — 복원 IMU 의 움직임 공분산(3x3)을 참 IMU 와 "
                        "맞춘다(윈도우당 6 dof). 고유벡터를 직접 비교하지 않아 부호 모호와 "
                        "축퇴에 안전하다. Learnable_R 의 --lambda_pca 와는 다른 항이다.")
    p.add_argument("--skip_isotropic_check", action="store_true",
                   help="orient/axes 를 켤 때 첫 배치로 중력 DC 존재를 확인하는 검사를 건너뛴다.")
    p.add_argument("--no_aux_on_target", action="store_true",
                   help="보조 손실을 source 에만 건다(기본은 두 도메인 모두). "
                        "두 손실 다 라벨이 필요 없어 target 에도 걸 수 있는데, "
                        "그 기여를 분리하려면 이 플래그로 끈다.")
    p.add_argument("--recon_loss", default="huber", choices=["huber", "mse", "l1"])
    p.add_argument("--decoder_width", type=int, default=128)
    p.add_argument("--imu_interp", action="store_true",
                   help="옛 동작 복원 — IMU 인코더가 길이를 10배 줄이고(stride 5 + MaxPool2) "
                        "F.interpolate 로 500 까지 되돌려 concat 한다. 기본은 길이 보존 "
                        "인코더(stride 1, MaxPool 제거)로 보간 없이 concat. "
                        "2026-08-04 스윕은 전부 이 플래그 켠 판이므로 섞어 비교하지 말 것.")
    p.add_argument("--no_cm", action="store_true")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--probe", action="store_true",
                   help="학습하지 않고 각 손실 항의 크기만 재서 λ 선택을 돕는다.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.probe:
        probe(batch_size=args.batch_size, decoder_width=args.decoder_width,
              recon_loss=args.recon_loss, imu_interp=args.imu_interp)
        sys.exit(0)

    aux_on_target = not args.no_aux_on_target
    cfg = {
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "domain_weight": args.domain_weight,
        "lambda_rec": args.lambda_rec, "lambda_jerk": args.lambda_jerk,
        "lambda_orient": args.lambda_orient, "lambda_axes": args.lambda_axes,
        "aux_on_target": aux_on_target, "recon_loss": args.recon_loss,
        "decoder_width": args.decoder_width,
        # 융합단 구성 — 08-04 판(interp)과 새 판(nointerp)을 구분하는 결정적 필드
        **fusion_info(args.imu_interp),
    }
    kw = dict(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
              domain_weight=args.domain_weight, lambda_rec=args.lambda_rec,
              lambda_jerk=args.lambda_jerk, aux_on_target=aux_on_target,
              recon_loss=args.recon_loss, decoder_width=args.decoder_width,
              save_cm=not args.no_cm, tag=args.tag, imu_interp=args.imu_interp,
              lambda_orient=args.lambda_orient, lambda_axes=args.lambda_axes,
              skip_isotropic_check=args.skip_isotropic_check)

    suffix = f"_{args.tag}" if args.tag else ""
    if args.multi_seed:
        results = [train_physics_cdan(seed=s, **kw) for s in args.seeds]
        os.makedirs(RESULT_DIR, exist_ok=True)
        summarize_results(results, method_name="Physics-informed CDAN (EMG→IMU)",
                          save_path=os.path.join(RESULT_DIR,
                                                 f"phys_cdan{suffix}_summary.txt"))
    else:
        results = [train_physics_cdan(seed=args.seed, **kw)]

    write_result_json(results, args.tag, cfg)
