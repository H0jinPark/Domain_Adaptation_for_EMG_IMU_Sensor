#!/usr/bin/env python
"""두 운동(deadlift + latpulldown) 정지중력으로 yaw 관측가능성 + Kabsch R 진단.

아이디어(A): 중력벡터 1개는 yaw(중력축 둘레 회전)를 못 잡지만, 자세가 다른 두 운동의
정지중력 2개가 비공선이면 paired Kabsch 로 3DOF(yaw 포함) 전체가 닫힌형으로 결정된다.

이 스크립트는 모델·EMG 없이 IMU 정지중력만 써서 "이 연구가 되는지"를 한눈에 본다:
  - 디바이스별 두 중력의 사이각 (= yaw 관측가능성; 클수록 좋음)
  - 두 디바이스에서 그 사이각이 일치하는지 (rigid 가정 정합성)
  - paired Kabsch R 의 잔차(°)·det·조건수
  - 단일중력(tilt-only) 대비 비교

각도가 충분히 크면(>~30°) R 을 results/R_matrices/R_kabsch2.npy 로 저장(--save).
정지시각은 gravity_alignment_check 노트북에서 비디오로 확정한 값을 그대로 쓴다.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from eval_utils import IMU_COLS  # ['triceps_X','triceps_Y','triceps_Z']

# ── 비디오로 확정한 정지 시각 (gravity_alignment_check.ipynb cell 5) ──
# source(samsung1) = absolute_time / target(samsung2) = TimeStamp, ±0.1s 평균.
CALIB = [
    dict(ex="latpulldown",
         s1_file="sub2_latpulldown_plot_and_store_rep_2.143.csv",              s1_t="16:24:25",
         s2_file="Data_NINA11_2025.09.16_10.42.27_sub7_latpulldown_1_59_6.csv", s2_t="10:42:41"),
    dict(ex="deadlift",
         s1_file="sub8_deadllift_plot_and_store_rep_1.40.csv",                 s1_t="14:05:14",
         s2_file="Data_NINA11_2025.09.12_09.06.40_sub7_deadlift_1_40_6.csv",    s2_t="09:06:46"),
]


def unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(unit(a), unit(b)), -1, 1))))


def kabsch(src, tgt):
    """src_i ~= R @ tgt_i 인 proper rotation R (det=+1). 행벡터: src ~= tgt @ R.T."""
    H = np.asarray(tgt).T @ np.asarray(src)
    Um, _, Vt = np.linalg.svd(H)
    V = Vt.T
    d = np.sign(np.linalg.det(V @ Um.T))
    return V @ np.diag([1, 1, d]) @ Um.T


def static_vec(df, sess_col, file, time_col, center_time, half=0.1):
    """정지시각 center_time ±half초 구간의 평균 IMU 를 중력 단위벡터로. (rel_std, n 동반)"""
    s = df[df[sess_col] == file].copy()
    if not len(s):
        raise ValueError(f"세션 없음: {file} (col={sess_col})")
    s[time_col] = pd.to_datetime(s[time_col])
    s = s.sort_values(time_col)
    c = str(center_time).strip()
    center = (pd.to_datetime(f"{s[time_col].iloc[0].date()} {c}")
              if len(c) <= 12 else pd.to_datetime(c))
    w = s[(s[time_col] >= center - pd.Timedelta(seconds=half)) &
          (s[time_col] <= center + pd.Timedelta(seconds=half))]
    M = w[IMU_COLS].to_numpy(float)
    if len(M) == 0:
        raise ValueError(f"정지 윈도우 비어있음: {file} @ {center_time}")
    mean = M.mean(0)
    rel_std = float(np.linalg.norm(M.std(0)) / (np.linalg.norm(mean) + 1e-12))
    return dict(unit=unit(mean), rel_std=rel_std, n=len(M))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--half", type=float, default=0.1, help="정지 윈도우 반폭(초)")
    ap.add_argument("--save", action="store_true",
                    help="관측가능성 충분하면 results/R_matrices/R_kabsch2.npy 저장")
    args = ap.parse_args()

    print("parquet 로드 중(필요 컬럼만)...", flush=True)
    s1 = pd.read_parquet(os.path.join(ROOT, "data", "samsung1.parquet"),
                         columns=IMU_COLS + ["filename", "absolute_time"])
    s2 = pd.read_parquet(os.path.join(ROOT, "data", "samsung2.parquet"),
                         columns=IMU_COLS + ["csv_filename_l", "TimeStamp"])

    recs = []
    for c in CALIB:
        g1 = static_vec(s1, "filename",       c["s1_file"], "absolute_time", c["s1_t"], args.half)
        g2 = static_vec(s2, "csv_filename_l", c["s2_file"], "TimeStamp",     c["s2_t"], args.half)
        recs.append(dict(ex=c["ex"], g1=g1["unit"], g2=g2["unit"],
                         rel1=g1["rel_std"], rel2=g2["rel_std"], n1=g1["n"], n2=g2["n"]))

    print("\n=== 정지중력 단위벡터 (±%.2fs 평균) ===" % args.half)
    for r in recs:
        print(f"  {r['ex']:12s} | s1={np.round(r['g1'],3).tolist()} (rel_std {r['rel1']:.3f}, n{r['n1']})"
              f" | s2={np.round(r['g2'],3).tolist()} (rel_std {r['rel2']:.3f}, n{r['n2']})"
              f" | s1↔s2 각 {angle_deg(r['g1'], r['g2']):.1f}°")

    dl = next(r for r in recs if r["ex"] == "deadlift")
    lp = next(r for r in recs if r["ex"] == "latpulldown")

    # ── 관측가능성: 디바이스별 두 운동 중력 사이각 ──
    ang_s1 = angle_deg(dl["g1"], lp["g1"])
    ang_s2 = angle_deg(dl["g2"], lp["g2"])
    print("\n=== yaw 관측가능성 (두 운동 중력 사이각; 클수록 yaw 잘 잡힘) ===")
    print(f"  samsung1: angle(deadlift, latpulldown) = {ang_s1:.1f}°")
    print(f"  samsung2: angle(deadlift, latpulldown) = {ang_s2:.1f}°")
    print(f"  두 디바이스 사이각 불일치 |Δ| = {abs(ang_s1-ang_s2):.1f}°  (작을수록 rigid 가정 정합)")

    # ── paired Kabsch (두 중력쌍) ──
    P1 = np.vstack([dl["g1"], lp["g1"]])   # source
    P2 = np.vstack([dl["g2"], lp["g2"]])   # target
    R = kabsch(P1, P2)                       # target→source, imu @ R.T
    res = [angle_deg(P1[i], P2[i] @ R.T) for i in range(2)]
    sv = np.linalg.svd(P2.T @ P1, compute_uv=False)
    # 2벡터 문제라 sv3≈0 은 당연. yaw 관측은 sv2 가 0 이 아닌지로 본다(2번째 방향 정보량).
    yaw_info = float(sv[1] / (sv[0] + 1e-12))   # 0 이면 두 중력 공선(yaw 못잡음), 클수록 좋음

    # ── 단일중력(tilt-only, latpulldown) 비교: yaw 미결정이라 잔차 0 이지만 yaw 자유 ──
    from eval_utils import rotation_align
    R_tilt = rotation_align(lp["g2"], lp["g1"])     # latpulldown 중력만; yaw 미결정
    tilt_res_lp = angle_deg(lp["g1"], lp["g2"] @ R_tilt.T)
    tilt_res_dl = angle_deg(dl["g1"], dl["g2"] @ R_tilt.T)  # deadlift 는 안 맞을수록 yaw 필요했다는 신호

    print("\n=== paired Kabsch R (target→source, imu @ R.T) ===")
    print(np.round(R, 4))
    print(f"  det(R) = {np.linalg.det(R):+.4f}  (proper rotation = +1)")
    print(f"  정렬 잔차: deadlift {res[0]:.2f}° · latpulldown {res[1]:.2f}°  (작을수록 두 쌍 동시정렬 성공)")
    print(f"  특이값(P2^T P1) = {np.round(sv,3).tolist()}  | yaw 정보량 sv2/sv1 = {yaw_info:.3f} (0 이면 yaw 못잡음)")
    print("\n--- 단일중력 tilt-only(latpulldown) 비교 ---")
    print(f"  latpulldown 잔차 {tilt_res_lp:.2f}° (당연히 ~0, 자기쌍) · "
          f"deadlift 잔차 {tilt_res_dl:.2f}° (← 이만큼이 yaw 미결정으로 남던 오차)")

    # ── 판정 ──
    ok = min(ang_s1, ang_s2) >= 30.0 and abs(ang_s1 - ang_s2) <= 15.0
    verdict = "GO ✅ (yaw 관측 충분)" if ok else "주의 ⚠️ (각도 작거나 불일치 큼 — 운동쌍 재선택/3개+ 권장)"
    print(f"\n=== 판정: {verdict} ===")
    if min(ang_s1, ang_s2) < 30:
        print("  · 두 중력 사이각이 작아 yaw가 ill-conditioned. 자세 차가 더 큰 운동쌍을 쓰거나 N개로 확장.")
    if abs(ang_s1 - ang_s2) > 15:
        print("  · 디바이스 간 사이각 불일치가 큼 → 정지시각/센서 일관성 점검 필요.")

    if args.save:
        if ok:
            out = os.path.join(ROOT, "results", "R_matrices", "R_kabsch2.npy")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            np.save(out, np.asarray(R, np.float64))
            print(f"\n저장: {out}  → data_preprocess_MM.py --method kabsch2 --R {out}")
        else:
            print("\n[--save 무시] 관측가능성 불충분으로 R 저장 보류.")


if __name__ == "__main__":
    main()
