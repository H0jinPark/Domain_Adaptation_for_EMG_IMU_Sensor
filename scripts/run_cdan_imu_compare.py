#!/usr/bin/env python
"""축 정렬 메서드별 IMU 단독 CDAN 학습 + 결과 비교 드라이버.

scripts/run_cdan_compare.py 의 IMU 전용판. 각 메서드(raw/pca/gravity/permutation/
kabsch)를 별도 프로세스로 Multimodal/CDAN_IMU_train.py 로 학습한다. 메서드 전환은
환경변수 MM_DATA_DIR 로 하며(import 시점 1회만 읽히므로 subprocess 분리), 결과는
멀티모달과 섞이지 않게 results/IMU/ 아래에 저장·비교한다.

사용 예
  # 5개 메서드 모두, 5 seed 평균
  python scripts/run_cdan_imu_compare.py --multi_seed --epochs 30
  # 빠른 점검: 단일 seed, 일부 메서드만
  python scripts/run_cdan_imu_compare.py --methods raw kabsch --seed 42 --epochs 30 --no_cm
  # 이미 학습 끝났고 비교만 다시
  python scripts/run_cdan_imu_compare.py --compare_only
"""
import argparse
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")          # headless 저장 전용
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CDAN_TRAIN = os.path.join(ROOT, "Multimodal", "CDAN_IMU_train.py")
RESULT_DIR = os.path.join(ROOT, "results", "IMU")
DEFAULT_METHODS = ["raw", "pca", "gravity", "permutation", "kabsch"]


def run_one(method, args):
    """메서드 하나를 subprocess 로 학습. 전처리 폴더 없으면 skip."""
    data_dir = f"preprocessed_MM_{method}"
    if not os.path.isdir(os.path.join(ROOT, data_dir)):
        print(f"[skip] {method}: {data_dir}/ 없음 — 전처리 먼저 실행")
        return False

    env = dict(os.environ, MM_DATA_DIR=data_dir)
    cmd = [sys.executable, CDAN_TRAIN, "--tag", method,
           "--epochs", str(args.epochs), "--batch_size", str(args.batch_size),
           "--lr", str(args.lr), "--domain_weight", str(args.domain_weight)]
    if args.multi_seed:
        cmd += ["--multi_seed", "--seeds", *map(str, args.seeds)]
    else:
        cmd += ["--seed", str(args.seed)]
    if args.no_cm:
        cmd += ["--no_cm"]

    print("\n" + "#" * 72)
    print(f"# CDAN IMU  method={method}   MM_DATA_DIR={data_dir}")
    print("#" * 72, flush=True)
    return subprocess.run(cmd, env=env, cwd=ROOT).returncode == 0


def load_results(methods):
    """메서드별 imu_cdan_result_<method>.json 로드 → 행 리스트(타깃 정확도 내림차순)."""
    rows = []
    for m in methods:
        p = os.path.join(RESULT_DIR, f"imu_cdan_result_{m}.json")
        if not os.path.exists(p):
            print(f"[warn] 결과 없음: {p}")
            continue
        d = json.load(open(p))
        rows.append({
            "method": m,
            "data_dir": d.get("data_dir", ""),
            "n_seeds": len(d["seeds"]),
            "source_acc": d["mean"]["source_acc"],
            "target_acc": d["mean"]["target_acc"],
            "target_std": d["std"]["target_acc"],
            "shift": d["mean"]["shift"],
        })
    rows.sort(key=lambda r: r["target_acc"], reverse=True)
    return rows


def compare(methods):
    rows = load_results(methods)
    if not rows:
        print("\n비교할 결과 JSON 이 없습니다. 먼저 학습을 실행하세요.")
        return

    # 콘솔 표
    print("\n" + "=" * 68)
    print(f"{'method':>12} | {'n':>2} | {'Source':>8} | {'Target':>8} | {'Tgt±std':>8} | {'Shift':>7}")
    print("-" * 68)
    for r in rows:
        print(f"{r['method']:>12} | {r['n_seeds']:>2} | {r['source_acc']:>7.2f}% | "
              f"{r['target_acc']:>7.2f}% | {r['target_std']:>7.2f} | {r['shift']:>6.2f}%")
    print("=" * 68)
    best = rows[0]
    print(f"best (target): {best['method']}  {best['target_acc']:.2f}%")

    # CSV
    os.makedirs(RESULT_DIR, exist_ok=True)
    csv_path = os.path.join(RESULT_DIR, "imu_cdan_method_comparison.csv")
    with open(csv_path, "w") as f:
        f.write("method,n_seeds,source_acc,target_acc,target_std,shift\n")
        for r in rows:
            f.write(f"{r['method']},{r['n_seeds']},{r['source_acc']:.4f},"
                    f"{r['target_acc']:.4f},{r['target_std']:.4f},{r['shift']:.4f}\n")
    print(f"CSV 저장: {csv_path}")

    # 막대그래프 (target accuracy, 에러바 = seed 표준편차)
    names = [r["method"] for r in rows]
    tgt = [r["target_acc"] for r in rows]
    err = [r["target_std"] for r in rows]
    colors = ["tab:gray" if m == "raw" else "tab:green" for m in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, tgt, yerr=err, capsize=4, color=colors, alpha=.85)
    for b, v in zip(bars, tgt):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}", ha="center", fontsize=9)
    if any(m == "raw" for m in names):
        raw_v = next(r["target_acc"] for r in rows if r["method"] == "raw")
        ax.axhline(raw_v, color="gray", ls="--", lw=1, label=f"raw {raw_v:.1f}%")
        ax.legend(fontsize=8)
    ax.set_ylabel("Target accuracy (%)")
    ax.set_title("IMU-only CDAN target accuracy by axis-alignment method")
    plt.tight_layout()
    png_path = os.path.join(RESULT_DIR, "imu_cdan_method_comparison.png")
    plt.savefig(png_path, dpi=200)
    print(f"그래프 저장: {png_path}")


def parse_args():
    p = argparse.ArgumentParser(description="메서드별 IMU 단독 CDAN 학습 + 비교")
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                   help=f"학습할 메서드 (기본: {DEFAULT_METHODS})")
    p.add_argument("--multi_seed", action="store_true", help="여러 seed 평균(권장)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--seed", type=int, default=42, help="단일 seed 모드에서 사용")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--domain_weight", type=float, default=1.0)
    p.add_argument("--no_cm", action="store_true", help="혼동행렬 PNG 저장 생략")
    p.add_argument("--compare_only", action="store_true",
                   help="학습 건너뛰고 기존 결과 JSON 만 모아 비교")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.compare_only:
        for m in args.methods:
            ok = run_one(m, args)
            if not ok:
                print(f"[!] {m} 학습 실패/스킵 — 다음 메서드로 진행")

    compare(args.methods)
