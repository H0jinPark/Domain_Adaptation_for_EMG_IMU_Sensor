#!/usr/bin/env python
"""두 운동 · 두 사람 정지중력으로 yaw 포함 R 직접 결정 (R_kabsch2).

방법 A 의 정식판. 중력 1개는 yaw 를 못 잡지만, 자세가 다른 두 운동(deadlift,
latpulldown)의 정지중력 2개가 비공선이면 paired Kabsch 로 3DOF(yaw 포함)가 닫힌형
으로 결정된다.

이전 관측가능성 스크립트(yaw_two_exercise_observability.py)에서 드러난 문제:
source 쪽 캘리가 운동마다 다른 피험자(sub2/sub8)를 써서 피험자 간 부착차가 yaw 에
40° 가짜로 섞였다. 그래서 여기서는 규칙을 단순·일관되게 못박는다.

  · 운동 2개: deadlift, latpulldown 고정
  · 사람 2명: source 1명(--src_subject) + target 1명(--tgt_subject), 각자 두 운동 모두 담당
  · 정지시각: 비디오 주석 없이 데이터기반 자동검출(세션별 최저분산·중력크기 구간)
  · 한 사람이 한 운동에 여러 세션이 있으면 그 사람 본인 세션들의 정지중력만 평균
    (다른 사람 섞지 않음 → "여러 명 평균" 아님)

결과 R(target→source, imu @ R.T)을 results/R_matrices/R_kabsch2.npy 로 저장(--save)
하면 기존 파이프라인이 그대로 받는다:
  python data_preprocess_MM.py --method kabsch2 --R results/R_matrices/R_kabsch2.npy

이 스크립트는 어떤 기존 파일도 수정하지 않는다(읽기 전용 import 만).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from eval_utils import IMU_COLS, rotation_align  # 읽기 전용 사용

EXERCISES = ["deadlift", "latpulldown"]


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


def stillest_window(A, w):
    """A:(T,3) 에서 길이 w 의 최저분산(가장 정지) 구간의 (평균중력3, rel_std)."""
    T = len(A)
    w = min(w, T)
    if w < 3:
        m = A.mean(0)
        return m, float(np.linalg.norm(A.std(0)) / (np.linalg.norm(m) + 1e-12))
    cs = np.cumsum(np.vstack([np.zeros(3), A]), 0)
    cs2 = np.cumsum(np.vstack([np.zeros(3), A * A]), 0)
    means = (cs[w:] - cs[:-w]) / w
    meansq = (cs2[w:] - cs2[:-w]) / w
    stdnorm = np.sqrt(np.clip(meansq - means ** 2, 0, None).sum(1))
    i = int(np.argmin(stdnorm))
    m = means[i]
    return m, float(stdnorm[i] / (np.linalg.norm(m) + 1e-12))


def subject_exercise_gravity(df, sess_col, time_col, subject_col, subject, exercise,
                             win_sec, rel_thresh, region="all", region_frac=0.3, verbose=True):
    """한 사람(subject)의 한 운동(exercise) 세션들에서 자동검출한 정지중력 단위벡터(평균).

    region: 'all'=세션 전체에서 최저분산 / 'head'=앞 region_frac / 'tail'=뒤 region_frac.
    동적 운동(latpulldown 등)은 정지 자세가 모호하므로 'head'(시작 휴식)로 국면을 고정하면
    세션 간 산포가 줄어든다.
    """
    sub = df[(df[subject_col] == subject) & (df["exercise"] == exercise)]
    if not len(sub):
        raise ValueError(f"데이터 없음: subject={subject} exercise={exercise}")
    units, rels, used = [], [], 0
    for sess, g in sub.groupby(sess_col):
        g = g.sort_values(time_col)
        t = g[time_col].to_numpy(float)
        dt = np.median(np.diff(t)) if len(t) > 1 else 1.0
        fs = 1.0 / dt if dt > 0 else 1.0
        w = max(5, int(round(win_sec * fs)))
        A = g[IMU_COLS].to_numpy(np.float64)
        T = len(A)
        if region == "head":
            A = A[:max(w, int(region_frac * T))]
        elif region == "tail":
            A = A[min(T - w, int((1 - region_frac) * T)):]
        mean, rel = stillest_window(A, w)
        if rel <= rel_thresh:
            units.append(unit(mean))
            rels.append(rel)
            used += 1
    if not units:
        raise ValueError(f"정지구간 없음(rel>{rel_thresh}): subject={subject} exercise={exercise}")
    units = np.vstack(units)
    g_mean = unit(units.mean(0))
    disp = float(np.mean([angle_deg(u, g_mean) for u in units]))  # 세션 간 산포(°)
    if verbose:
        print(f"    {exercise:12s} | 세션 {used}/{sub[sess_col].nunique()} 사용 "
              f"(rel_std<{rel_thresh}) | 평균중력 {np.round(g_mean,3).tolist()} "
              f"| 세션간 산포 {disp:.1f}° | rel_std {np.mean(rels):.3f}")
    return g_mean, disp, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_subject", default="sub2", help="samsung1 source 피험자 1명")
    ap.add_argument("--tgt_subject", type=int, default=7, help="samsung2 target 피험자 1명(id)")
    ap.add_argument("--win_sec", type=float, default=0.3, help="정지검출 윈도우 길이(초)")
    ap.add_argument("--rel_thresh", type=float, default=0.10,
                    help="세션 채택 임계: 최저분산구간 rel_std 가 이보다 작아야 정지로 인정")
    ap.add_argument("--region", choices=["all", "head", "tail"], default="head",
                    help="정지검출 구간: all=전체, head=시작부, tail=끝부 (동적운동은 head 권장)")
    ap.add_argument("--region_frac", type=float, default=0.3, help="head/tail 비율")
    ap.add_argument("--save", action="store_true",
                    help="관측가능성·정합 충분하면 results/R_matrices/R_kabsch2.npy 저장")
    ap.add_argument("--force", action="store_true",
                    help="자동 판정이 '주의'여도 사용자 결정으로 R 저장 강행")
    args = ap.parse_args()

    print("parquet 로드 중(필요 컬럼만)...", flush=True)
    s1 = pd.read_parquet(os.path.join(ROOT, "data", "samsung1.parquet"),
                         columns=IMU_COLS + ["filename", "timestamp", "subject", "exercise"])
    s2 = pd.read_parquet(os.path.join(ROOT, "data", "samsung2.parquet"),
                         columns=IMU_COLS + ["csv_filename_l", "Index_Time", "subject_id", "exercise"])

    print(f"\n=== 정지중력 자동검출 (윈도우 {args.win_sec}s, rel_std<{args.rel_thresh}, region={args.region}) ===")
    print(f"  [source samsung1] subject={args.src_subject}")
    g1 = {}
    for ex in EXERCISES:
        g1[ex], _, _ = subject_exercise_gravity(
            s1, "filename", "timestamp", "subject", args.src_subject, ex,
            args.win_sec, args.rel_thresh, args.region, args.region_frac)
    print(f"  [target samsung2] subject={args.tgt_subject}")
    g2 = {}
    for ex in EXERCISES:
        g2[ex], _, _ = subject_exercise_gravity(
            s2, "csv_filename_l", "Index_Time", "subject_id", args.tgt_subject, ex,
            args.win_sec, args.rel_thresh, args.region, args.region_frac)

    # ── 관측가능성 + rigid 정합 ──
    ang_s1 = angle_deg(g1["deadlift"], g1["latpulldown"])
    ang_s2 = angle_deg(g2["deadlift"], g2["latpulldown"])
    print("\n=== yaw 관측가능성 & rigid 정합 ===")
    print(f"  source angle(deadlift, latpulldown) = {ang_s1:.1f}°")
    print(f"  target angle(deadlift, latpulldown) = {ang_s2:.1f}°")
    print(f"  디바이스 간 사이각 불일치 |Δ| = {abs(ang_s1-ang_s2):.1f}°  (작을수록 순수 디바이스축 회전)")

    P1 = np.vstack([g1["deadlift"], g1["latpulldown"]])  # source
    P2 = np.vstack([g2["deadlift"], g2["latpulldown"]])  # target
    R = kabsch(P1, P2)                                     # target→source, imu @ R.T
    res = [angle_deg(P1[i], P2[i] @ R.T) for i in range(2)]
    sv = np.linalg.svd(P2.T @ P1, compute_uv=False)
    yaw_info = float(sv[1] / (sv[0] + 1e-12))

    # 단일중력 tilt-only(latpulldown) 비교
    R_tilt = rotation_align(g2["latpulldown"], g1["latpulldown"])
    tilt_res_dl = angle_deg(g1["deadlift"], g2["deadlift"] @ R_tilt.T)

    print("\n=== paired Kabsch R (target→source, imu @ R.T) ===")
    print(np.round(R, 4))
    print(f"  det(R) = {np.linalg.det(R):+.4f}")
    print(f"  정렬 잔차: deadlift {res[0]:.2f}° · latpulldown {res[1]:.2f}°")
    print(f"  특이값 = {np.round(sv,3).tolist()} | yaw 정보량 sv2/sv1 = {yaw_info:.3f} (0이면 yaw 못잡음)")
    print(f"  [비교] 단일중력 tilt-only 면 deadlift 가 {tilt_res_dl:.1f}° 어긋남 → 그만큼이 yaw 로 잡혀야 할 양")

    ok = min(ang_s1, ang_s2) >= 30.0 and abs(ang_s1 - ang_s2) <= 15.0 and yaw_info > 0.05
    print(f"\n=== 판정: {'GO ✅' if ok else '주의 ⚠️'} "
          f"(yaw각 충분 & 디바이스 정합 양호 & yaw정보>0) ===")
    if min(ang_s1, ang_s2) < 30:
        print("  · 두 중력 사이각이 작아 yaw ill-conditioned → 다른 두 사람/윈도우 시도.")
    if abs(ang_s1 - ang_s2) > 15:
        print("  · 디바이스 간 사이각 불일치 큼 → 두 사람이 다른 자세였거나 비강체 요인. "
              "--src_subject/--tgt_subject 바꿔 민감도 확인.")

    if args.save:
        if ok or args.force:
            out = os.path.join(ROOT, "results", "R_matrices", "R_kabsch2.npy")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            np.save(out, np.asarray(R, np.float64))
            tag = "" if ok else "  (--force: 자동판정 '주의'였으나 사용자 결정으로 저장)"
            print(f"\n저장: {out}{tag}")
            print(f"  python data_preprocess_MM.py --method kabsch2 --R {out}")
        else:
            print("\n[--save 보류] 정합 불충분. 사용자 결정이면 --force 로 강행.")


if __name__ == "__main__":
    main()
