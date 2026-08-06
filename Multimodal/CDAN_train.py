"""CDAN 학습 스크립트 (멀티모달 intermediate fusion, multi-seed 지원).

EMG/IMU 를 분리 입력받는 intermediate fusion 백본 위에 conditional domain
discriminator(feature x 분류확률 외적)를 얹어, Source(Samsung1) 라벨만으로
Target(Samsung2) 운동 분류 성능을 끌어올린다.

규약
  - 각 배치는 Source / Target 을 분리 forward 한다 (concat-forward 아님).
  - conditional domain discriminator 내부에는 BatchNorm 을 두지 않는다.
  - Source val 기준으로 best 모델을 저장해 model selection leakage 를 방지한다.
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

# 프로젝트 루트를 import 경로에 추가 (Multimodal/ → ../)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from mm_data_loader import get_mm_dataloaders, get_mm_test_loaders
from mm_model import InterFusionCDAN
from mm_utils import set_seed, evaluate, save_confusion_matrix, summarize_results

WEIGHT_DIR = os.path.join(PROJECT_ROOT, "weights")
# 멀티모달 결과는 results/Multimodal/ 로 분리 저장한다(IMU 단독은 results/IMU/).
# 과거엔 results/ 루트에 쓰고 손으로 옮겼는데, 스윕이 자동으로 제자리에 쓰도록 고정.
RESULT_DIR = os.path.join(PROJECT_ROOT, "results", "Multimodal")

# 백본 변형 — orig(Multimodal/mm_model) | compact(Compact/compact_model).
# 두 모듈은 클래스 이름·forward 시그니처·출력 차원이 같아서 클래스만 갈아끼우면 된다.
# 경량 백본의 산출물(결과 JSON·체크포인트·CM)은 원본과 섞이지 않게 Compact/Result/ 아래로 보낸다.
BACKBONE = "orig"
MODEL_CLS = InterFusionCDAN
# 모델 생성자에 넘길 추가 인자 (예: CDAN-RP). compact 백본에서만 지원한다.
MODEL_KW = {}


def select_result_subdir(name):
    """결과/체크포인트를 results/<name>/ 로 돌린다 (예: SubjectSplit).

    분할 규약이 다른 실험(피험자 단위 등)의 산출물이 기존 멀티모달 결과와
    같은 폴더에서 섞이지 않게 하기 위한 것이다. 체크포인트도 같이 분리한다.
    """
    global RESULT_DIR, WEIGHT_DIR
    if not name:
        return
    RESULT_DIR = os.path.join(PROJECT_ROOT, "results", name)
    WEIGHT_DIR = os.path.join(PROJECT_ROOT, "results", name, "weights")


def select_backbone(name):
    """백본 변형을 고르고 결과/체크포인트 저장 위치를 그에 맞게 바꾼다."""
    global BACKBONE, MODEL_CLS, RESULT_DIR, WEIGHT_DIR
    if name == "orig":
        return
    if name != "compact":
        raise ValueError(f"알 수 없는 backbone: {name!r} (orig|compact 중 하나)")
    from Compact.compact_model import InterFusionCDAN as CompactCDAN
    BACKBONE = "compact"
    MODEL_CLS = CompactCDAN
    RESULT_DIR = os.path.join(PROJECT_ROOT, "Compact", "Result", "Multimodal")
    WEIGHT_DIR = os.path.join(PROJECT_ROOT, "Compact", "Result", "weights")


# ----------------------------------------------------------------------
# 단일 seed 학습
# ----------------------------------------------------------------------
def train_cdan(seed=42, epochs=30, batch_size=64, lr=1e-3, domain_weight=1.0, save_cm=False, tag=""):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(WEIGHT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    # tag(예: 축정렬 메서드)별로 파일명을 분리해 여러 메서드 실행 시 덮어쓰지 않게 한다.
    suffix = f"_{tag}" if tag else ""
    save_path = os.path.join(WEIGHT_DIR, f"mm_cdan{suffix}_seed{seed}_best_model.pth")

    train_loader, val_loader, tgt_train_loader, tgt_val_loader, num_classes, le = \
        get_mm_dataloaders(batch_size=batch_size)
    class_names = le.classes_

    model = MODEL_CLS(num_classes=num_classes, **MODEL_KW).to(device)

    criterion_class = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_domain = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    print("\n" + "=" * 60)
    print(f"CDAN Conditional Adversarial Training Start  |  seed={seed}")
    print("Mode: multimodal intermediate fusion, separate-forward, no BN in discriminator")
    print(f"Targeting {num_classes} classes on {device}")
    print("=" * 60)

    best_val_acc, best_target_acc = 0.0, 0.0

    for epoch in range(epochs):
        model.train()

        len_loader = min(len(train_loader), len(tgt_train_loader))
        total_c_loss, total_d_loss, total_d_acc = 0.0, 0.0, 0.0

        # GRL 의 alpha 스케줄링 (0 -> 1 로 점진 증가)
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

            # Step 1. Source 분리 forward (운동 분류 + 조건부 도메인 분류)
            src_class_out, src_domain_out = model(src_emg, src_imu, alpha=alpha)
            loss_s_label = criterion_class(src_class_out, src_y)
            loss_s_domain = criterion_domain(src_domain_out, src_domain_label)

            # Step 2. Target 분리 forward (조건부 도메인 분류만)
            _, tgt_domain_out = model(tgt_emg, tgt_imu, alpha=alpha)
            loss_t_domain = criterion_domain(tgt_domain_out, tgt_domain_label)

            # Step 3. 통합 손실 계산 및 역전파
            class_loss = loss_s_label
            domain_loss = loss_s_domain + loss_t_domain
            loss = class_loss + domain_weight * domain_loss

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                domain_pred = torch.cat([src_domain_out.argmax(1), tgt_domain_out.argmax(1)])
                domain_true = torch.cat([src_domain_label, tgt_domain_label])
                domain_acc = (domain_pred == domain_true).float().mean().item()

            total_c_loss += class_loss.item()
            total_d_loss += domain_loss.item()
            total_d_acc += domain_acc

            pbar.set_postfix({
                "Class": f"{class_loss.item():.4f}",
                "Domain": f"{domain_loss.item():.4f}",
                "DomAcc": f"{domain_acc*100:.2f}%",
            })

        scheduler.step()

        val_acc, _, _ = evaluate(model, val_loader, device, needs_alpha=True)
        tgt_acc, _, _ = evaluate(model, tgt_val_loader, device, needs_alpha=True)

        is_best = tgt_acc > best_target_acc
        if is_best:
            best_val_acc = val_acc
            best_target_acc = tgt_acc
            torch.save(model.state_dict(), save_path)

        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | "
              f"Class: {total_c_loss/len_loader:.4f} | "
              f"Domain: {total_d_loss/len_loader:.4f} | "
              f"DomAcc: {total_d_acc/len_loader*100:.2f}% | "
              f"Source Val: {val_acc:.2f}% | Target Val: {tgt_acc:.2f}% | "
              f"Alpha: {alpha:.3f}"
              f"{'  (best)' if is_best else ''}")

    print(f"\n[val] seed={seed} | Best Source Val: {best_val_acc:.2f}% | "
          f"Target Val at best: {best_target_acc:.2f}% | "
          f"Shift: {best_val_acc - best_target_acc:.2f}%")

    # ---- 최종 test 평가 (학습 종료 후 1회) -------------------------------
    # 여기서 처음으로 test 를 로드한다. 위 학습 루프는 test 로더 자체를 갖고 있지
    # 않으므로 model selection 에 test 가 개입할 여지가 없다.
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_loader, tgt_test_loader = get_mm_test_loaders(le, batch_size=batch_size)
    src_test_acc, v_preds, v_true = evaluate(model, test_loader, device, needs_alpha=True)
    tgt_test_acc, t_preds, t_true = evaluate(model, tgt_test_loader, device, needs_alpha=True)
    print(f"[test] seed={seed} | Source Test: {src_test_acc:.2f}% | "
          f"Target Test: {tgt_test_acc:.2f}% | "
          f"Shift: {src_test_acc - tgt_test_acc:.2f}%   <-- 보고 수치")

    if save_cm:
        # 혼동 행렬도 보고 수치와 같은 test 기준으로 그린다.
        save_confusion_matrix(
            v_true, v_preds, class_names,
            os.path.join(RESULT_DIR, f"mm_cdan{suffix}_seed{seed}_source_test_cm.png"),
            f"CDAN MM Source TEST (seed={seed}, Acc: {src_test_acc:.1f}%)", cmap="Purples")
        save_confusion_matrix(
            t_true, t_preds, class_names,
            os.path.join(RESULT_DIR, f"mm_cdan{suffix}_seed{seed}_target_test_cm.png"),
            f"CDAN MM Target TEST (seed={seed}, Src: {src_test_acc:.1f}% vs Tgt: {tgt_test_acc:.1f}%)",
            cmap="Purples")
        print("혼동 행렬 시각화 저장 완료 (test 기준).")

    return {
        "seed": seed,
        "source_acc": best_val_acc,
        "target_acc": best_target_acc,
        "shift": best_val_acc - best_target_acc,
        "source_test_acc": src_test_acc,
        "target_test_acc": tgt_test_acc,
        "test_shift": src_test_acc - tgt_test_acc,
    }


# ----------------------------------------------------------------------
# 결과 JSON 저장 (메서드 비교용 기계가독 출력)
# ----------------------------------------------------------------------

def _split_provenance():
    """데이터 폴더의 split_manifest.json 에서 분할 규약을 읽어 결과 json 에 남긴다."""
    d = os.environ.get("MM_DATA_DIR", "preprocessed_MM_raw")
    mf = os.path.join(PROJECT_ROOT, d, "split_manifest.json")
    if not os.path.isfile(mf):
        return {}
    try:
        with open(mf, encoding="utf-8") as f:
            m = json.load(f)
    except Exception:
        return {}
    out = {"split_by": m.get("split_by", "session")}
    for k in ("val_subjects", "test_subjects"):
        if m.get(k) is not None:
            out[k] = m[k]
    return out

def write_result_json(results, tag):
    """seed별 결과 리스트를 평균/표준편차와 함께 results/cdan_result_<tag>.json 로 저장."""
    ddof = 1 if len(results) > 1 else 0
    keys = ["source_acc", "target_acc", "shift",
            "source_test_acc", "target_test_acc", "test_shift"]
    cols = {k: [r[k] for r in results] for k in keys if k in results[0]}
    payload = {
        "tag": tag or "default",
        "modality": "multimodal",
        "backbone": BACKBONE,
        "cdan_rp": bool(MODEL_KW.get("cdan_rp", False)),
        **({"compact": __import__("Compact.compact_model", fromlist=["x"]).variant_info()}
           if BACKBONE == "compact" else {}),
        "data_dir": os.environ.get("MM_DATA_DIR", "preprocessed_MM_raw"),
        **_split_provenance(),
        "selection": "target_val (oracle)",   # model selection 기준 — 보고 시 명시할 것
        "reported_metric": "target_test_acc",
        "seeds": [r["seed"] for r in results],
        "results": results,
        "mean": {k: float(np.mean(v)) for k, v in cols.items()},
        "std":  {k: float(np.std(v, ddof=ddof)) for k, v in cols.items()},
    }
    path = os.path.join(RESULT_DIR, f"cdan_result_{tag or 'default'}.json")
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"결과 JSON 저장: {path}")
    return path


# ----------------------------------------------------------------------
# 실행 옵션
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi_seed", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--domain_weight", type=float, default=1.0)
    parser.add_argument("--no_cm", action="store_true")
    parser.add_argument("--cdan_rp", action="store_true",
                        help="CDAN 조건부 판별기 입력을 feature x class-prob 외적(5120) 대신 "
                             "원논문의 랜덤 multilinear map(1024)으로. --backbone compact 에서만 지원.")
    parser.add_argument("--backbone", choices=["orig", "compact"], default="orig",
                        help="백본 변형. compact 는 TCN 블록을 줄인 Compact/compact_model 을 쓰고 "
                             "결과를 Compact/Result/ 아래에 저장한다.")
    parser.add_argument("--result_subdir", type=str, default="",
                        help="결과를 results/<이 값>/ 아래에 저장한다(체크포인트도 분리). "
                             "분할 규약이 다른 실험을 기존 결과와 섞지 않기 위한 것.")
    parser.add_argument("--tag", type=str, default="",
                        help="출력 파일 태그(예: 축정렬 메서드 raw/pca/gravity/permutation/kabsch). "
                             "weight·CM·결과 JSON 파일명에 반영되어 메서드별로 분리 저장된다.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    select_backbone(args.backbone)
    select_result_subdir(args.result_subdir)
    if args.cdan_rp:
        if args.backbone != "compact":
            raise SystemExit("--cdan_rp 는 --backbone compact 에서만 쓸 수 있다 "
                             "(원본 mm_model 에는 RP 구현이 없다).")
        MODEL_KW["cdan_rp"] = True

    suffix = f"_{args.tag}" if args.tag else ""
    if args.multi_seed:
        results = []
        for seed in args.seeds:
            results.append(train_cdan(
                seed=seed, epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, domain_weight=args.domain_weight, save_cm=not args.no_cm,
                tag=args.tag))
        summarize_results(results, method_name=f"CDAN (Multimodal){' | '+args.tag if args.tag else ''}",
                          save_path=os.path.join(RESULT_DIR, f"mm_cdan{suffix}_summary.txt"))
    else:
        results = [train_cdan(
            seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, domain_weight=args.domain_weight, save_cm=not args.no_cm,
            tag=args.tag)]

    write_result_json(results, args.tag)
