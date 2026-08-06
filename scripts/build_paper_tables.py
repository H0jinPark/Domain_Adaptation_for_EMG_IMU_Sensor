"""원본(비경량) 백본 결과를 논문용 두 표로 정리한다.

  표 1  고정 R 5방법 (raw/permutation/gravity/kabsch/pca) x IMU단독 / 멀티모달
  표 2  Learnable R  (target_join 스윕) x IMU단독 / 멀티모달 + 변형들

둘 다 **원본 백본만** 쓴다(Compact/Result/ 는 읽지 않는다). 전부 7:1.5:1.5 규약
(`protocol-train-val-test-split`)이라 Src/Tgt 를 val·test 두 벌로 낸다.

  · val  = model selection 에 쓰인 값이라 낙관 편향
  · test = 보고 수치. 단 checkpoint 를 target-val 로 골랐으므로(oracle selection)
           test 도 그 편향을 일부 물려받는다. learnable-R 은 last-N 참고값을 함께 낸다.

    /home/user1/miniconda3/envs/DA/bin/python scripts/build_paper_tables.py
"""
from pathlib import Path
import json

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "tables"

METHODS = ["raw", "permutation", "gravity", "kabsch", "pca"]
JOINS = [0, 5, 10, 15, 20, 30, 40, 50, 60]


def load(path):
    p = RES / path
    return json.load(open(p)) if p.is_file() else None


def cell(j, key, prec=2):
    """mean ± std 문자열. 해당 지표가 없으면 '—'."""
    if not j or key not in j.get("mean", {}):
        return "—"
    return f"{j['mean'][key]:.{prec}f} ± {j['std'][key]:.{prec}f}"


def seeds_of(j, key):
    return [r[key] for r in j["results"]] if j else []


# ----------------------------------------------------------------------
# 표 1 — 고정 R
# ----------------------------------------------------------------------
def table_fixed_r():
    rows = []
    for m in METHODS:
        imu = load(f"IMU/imu_cdan_result_{m}.json")
        mm = load(f"Multimodal/cdan_result_{m}.json")
        rows.append((m, imu, mm))

    lines = []
    lines.append("## 표 1. 고정 R 축정렬 5방법 — 원본 백본, CDAN 30epoch, seeds 0-4\n")
    lines.append("| 정렬 방법 | IMU Src-Val | IMU Tgt-Val | **IMU Src-Test** | **IMU Tgt-Test** "
                 "| MM Src-Val | MM Tgt-Val | **MM Src-Test** | **MM Tgt-Test** |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for m, imu, mm in rows:
        lines.append(
            f"| {m} | {cell(imu,'source_acc')} | {cell(imu,'target_acc')} "
            f"| {cell(imu,'source_test_acc')} | **{cell(imu,'target_test_acc')}** "
            f"| {cell(mm,'source_acc')} | {cell(mm,'target_acc')} "
            f"| {cell(mm,'source_test_acc')} | **{cell(mm,'target_test_acc')}** |")

    # domain shift (Src-Test − Tgt-Test)
    lines.append("\n### 표 1b. 도메인 shift (Src-Test − Tgt-Test)\n")
    lines.append("| 정렬 방법 | IMU 단독 shift | 멀티모달 shift |")
    lines.append("|---|---|---|")
    for m, imu, mm in rows:
        lines.append(f"| {m} | {cell(imu,'test_shift')} | {cell(mm,'test_shift')} |")
    return "\n".join(lines), rows


# ----------------------------------------------------------------------
# 표 2 — Learnable R
# ----------------------------------------------------------------------
def table_learnable_r():
    lines = []
    lines.append("## 표 2. Learnable R (SO(3)) — 원본 백본, 60epoch, seeds 0-4\n")
    lines.append("입력은 `preprocessed_MM_raw_isotropic` (고정 R 정렬 없음, 등방 정규화).  "
                 "`target_join` = target 이 인코더/판별기에 합류하는 epoch "
                 "(= 그 전까지는 gravity/PCA 손실로 R 만 정렬).\n")
    lines.append("| target_join | IMU Src-Test | **IMU Tgt-Test** | IMU last-N | "
                 "MM Src-Test | **MM Tgt-Test** | MM last-N |")
    lines.append("|---|---|---|---|---|---|---|")
    for j in JOINS:
        imu = load(f"Learnable_R/learnable_r_cdan_alignfirst_result_join{j}.json")
        mm = load(f"Learnable_R/learnable_r_mm_alignfirst_result_join{j}.json")
        lines.append(
            f"| join{j} | {cell(imu,'source_test_acc')} | **{cell(imu,'target_test_acc')}** "
            f"| {cell(imu,'target_last_mean')} "
            f"| {cell(mm,'source_test_acc')} | **{cell(mm,'target_test_acc')}** "
            f"| {cell(mm,'target_last_mean')} |")

    lines.append("\n`last-N` = 마지막 N epoch target val 정확도 평균. "
                 "target 라벨로 epoch 을 고르지 않은 leakage-free 참고값이라, "
                 "Tgt-Test 와의 차이가 oracle selection 편향의 크기를 보여준다.\n")

    lines.append("\n### 표 2b. Learnable R — val 지표 (model selection 에 쓰인 값, 낙관 편향)\n")
    lines.append("| target_join | IMU Src-Val | IMU Tgt-Val | MM Src-Val | MM Tgt-Val |")
    lines.append("|---|---|---|---|---|")
    for j in JOINS:
        imu = load(f"Learnable_R/learnable_r_cdan_alignfirst_result_join{j}.json")
        mm = load(f"Learnable_R/learnable_r_mm_alignfirst_result_join{j}.json")
        lines.append(f"| join{j} | {cell(imu,'source_acc')} | {cell(imu,'target_acc')} "
                     f"| {cell(mm,'source_acc')} | {cell(mm,'target_acc')} |")

    # 변형들
    lines.append("\n### 표 2c. Learnable R 변형 (전부 원본 백본)\n")
    lines.append("| 구성 | 모달 | 입력 | Src-Test | **Tgt-Test** | last-N |")
    lines.append("|---|---|---|---|---|---|")
    variants = [
        ("gravity 손실만 (λ_pca=0)", "IMU", "Learnable_R/learnable_r_cdan_alignfirst_result_gravonly.json"),
        ("PCA 손실만 (λ_g=0)", "IMU", "Learnable_R/learnable_r_cdan_alignfirst_result_pcaonly.json"),
        ("join10, 30epoch", "IMU", "Learnable_R/learnable_r_cdan_alignfirst_result_join10_ep30.json"),
        ("join0, 30epoch", "MM", "Learnable_R/learnable_r_mm_alignfirst_result_ep30_join0.json"),
        ("join0, post_r_norm 없음", "MM", "Learnable_R/learnable_r_mm_alignfirst_result_nobn_join0.json"),
    ]
    for name, mod, path in variants:
        j = load(path)
        din = (j or {}).get("data_dir", "—")
        lines.append(f"| {name} | {mod} | `{din}` | {cell(j,'source_test_acc')} "
                     f"| **{cell(j,'target_test_acc')}** | {cell(j,'target_last_mean')} |")

    lines.append("\n#### z-score 정규화 + PCA 손실만 (λ_g=0) — 붕괴 사례\n")
    lines.append("| target_join | Src-Test | **Tgt-Test** | last-N |")
    lines.append("|---|---|---|---|")
    for j in [0, 10, 20, 30, 40, 50, 60]:
        z = load(f"Learnable_R/learnable_r_mm_alignfirst_result_zsc_pcaonly_join{j}.json")
        lines.append(f"| join{j} | {cell(z,'source_test_acc')} | **{cell(z,'target_test_acc')}** "
                     f"| {cell(z,'target_last_mean')} |")

    # 손실 가중치 격자 (IMU 단독)
    lines.append("\n### 표 2d. 손실 가중치 격자 — IMU 단독, Tgt-Test (λ_g × λ_pca)\n")
    grid = ["| λ_g \\ λ_pca | " + " | ".join(f"{p:.1f}" for p in
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) + " |",
            "|---|" + "---|" * 6]
    for g in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        cells = []
        for p in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            j = load(f"Learnable_R/learnable_r_cdan_alignfirst_result_g{g:.1f}_p{p:.1f}.json")
            cells.append(f"{j['mean']['target_test_acc']:.2f}" if j else "—")
        grid.append(f"| **{g:.1f}** | " + " | ".join(cells) + " |")
    lines.append("\n".join(grid))
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t1, rows = table_fixed_r()
    t2 = table_learnable_r()

    header = (
        "# 원본 백본 결과 표 (경량화 모델 제외)\n\n"
        "전부 세션단위 층화 7:1.5:1.5 분할, seeds 0-4, 값은 `mean ± std`(ddof=1, %).\n\n"
        "> **읽을 때 주의**\n"
        "> - 모든 실험이 checkpoint 를 **target val 정확도로 선택**했다(oracle selection). "
        "target val 과 target test 는 같은 피험자·인접 세션이라 상관이 높아, "
        "Tgt-Test 도 편향을 일부 물려받는다.\n"
        "> - 윈도우 90% 중첩 → 유효 표본수는 명목의 약 1/10. 세션 하나가 약 1.37%p 다. "
        "1%p 안쪽 차이는 유의하다고 보기 어렵다.\n"
        "> - source/target 이 subject 를 공유하므로 주장 범위는 "
        "\"같은 사람, 새 디바이스\"까지다.\n\n"
    )
    text = header + t1 + "\n\n---\n\n" + t2 + "\n"
    path = OUT / "original_backbone_tables.md"
    path.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n저장: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
