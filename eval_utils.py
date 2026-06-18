"""Cross-device 평가 유틸 — DataFrame 입력 → (accuracy, DTW vs samsung1) 반환.

설계 의도
─────────────────────────────────────────────────────────────────────────────
parquet **파일 경로**가 아니라 *이미 메모리에 로드된 DataFrame* 을 받는다.
IMU 축 순열·부호(6 order × 8 sign = 48가지) 탐색을 parquet 48개로 만들지 않고,
메모리 상의 ``triceps_X/Y/Z`` 컬럼만 바꿔 같은 함수로 반복 평가하기 위함이다.

입력 DataFrame 스키마 (data/samsung2.parquet 과 동일):
    EMG     : biceps, triceps
    IMU     : triceps_X, triceps_Y, triceps_Z
    라벨    : exercise
    세션/시간: session_col(기본 csv_filename_l) / time_col(기본 Index_Time)

반환 (evaluate):
    accuracy : baseline_TCN(source-only) 추론 윈도우 정확도(%)
              — notebook/eval_baseline_tcn_source_only 와 동일 로직
    dtw      : samsung1 triceps IMU 와 target triceps IMU 의 운동별
              multivariate DTW 평균 (낮을수록 축 정렬이 잘 됨)

같은 DataFrame 이 들어오면 항상 같은 값을 반환한다(세션 선택·리샘플 모두 결정적).
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# ── import 경로: 이 파일이 repo 루트에 있다고 가정하고 baseline/ 도 추가 ──────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (_ROOT, os.path.join(_ROOT, "baseline")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from baseline_model import AdvancedBaselineModel              # noqa: E402
from data_preprocess_original import DomainAdaptationPreprocessor  # noqa: E402

# ── 상수 ──────────────────────────────────────────────────────────────────────
# 학습 때와 동일하게 알파벳 정렬로 고정된 10개 운동 클래스
CLASS_NAMES = [
    "barbellcurl", "barbellrow", "benchpress", "bte", "deadlift",
    "dips", "latpulldown", "ohp", "pullup", "pushup",
]
IMU_COLS = ["triceps_X", "triceps_Y", "triceps_Z"]
FEATURE_COLS = ["biceps", "triceps", "triceps_X", "triceps_Y", "triceps_Z"]


def get_label_encoder():
    """CLASS_NAMES 로 fit 된 LabelEncoder (추론·학습 라벨 순서 일치 보장)."""
    le = LabelEncoder()
    le.fit(CLASS_NAMES)
    return le


def load_baseline_model(weights_path, device=None, num_classes=None, in_channels=5):
    """Source-only baseline_TCN(5채널) 가중치를 로드해 eval 모드 모델 반환."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = num_classes or len(CLASS_NAMES)
    model = AdvancedBaselineModel(in_channels=in_channels, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def load_mm_model(weights_path, device=None, num_classes=None):
    """Source-only 멀티모달(InterFusionClassifier) 가중치를 로드해 eval 모드 모델 반환.

    EMG/IMU 분리 입력 (B,2,5000)/(B,3,500) 을 받는 intermediate fusion 모델.
    forward 는 (class_logits, features) 튜플을 반환한다.
    """
    from Multimodal.mm_model import InterFusionClassifier  # noqa: E402

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = num_classes or len(CLASS_NAMES)
    model = InterFusionClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


# ── IMU 축 순열/부호 (48가지) ────────────────────────────────────────────────
def apply_imu_permutation(df, order="XYZ", signs=(1, 1, 1)):
    """triceps_X/Y/Z 를 ``order`` 로 재매핑하고 ``signs`` 부호를 곱한 새 DataFrame.

    order 'YXZ' → 새 triceps_X ← 기존 triceps_Y, 새 triceps_Y ← 기존 triceps_X ...
    원본 df 는 변경하지 않는다(copy).
    """
    src = {"X": "triceps_X", "Y": "triceps_Y", "Z": "triceps_Z"}
    out = df.copy()
    new_vals = {
        slot: df[src[axis]].to_numpy() * sgn
        for slot, axis, sgn in zip(IMU_COLS, order.upper(), signs)
    }
    for slot, vals in new_vals.items():
        out[slot] = vals
    return out


def iter_imu_permutations():
    """48가지 (name, order, signs) 생성 — 6 축순열 × 8 부호조합."""
    from itertools import permutations, product

    for order in ("".join(p) for p in permutations("XYZ")):
        for signs in product((1, -1), repeat=3):
            sign_str = "".join("+" if s > 0 else "-" for s in signs)
            yield f"{order}_{sign_str}", order, signs


# ── 정확도 (모델 추론) ────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_accuracy(df, model, *, session_col="csv_filename_l", time_col="Index_Time",
                      preprocessor=None, device=None, batch_size=256, return_preds=False):
    """df 전체를 세션 단위로 전처리→윈도우→추론하고 윈도우 정확도(%)를 반환.

    return_preds=True 면 (acc, y_true, y_pred) 반환.
    """
    device = device or next(model.parameters()).device
    preprocessor = preprocessor or DomainAdaptationPreprocessor(
        out_dir=os.path.join(_ROOT, "preprocessed_original"))
    le = get_label_encoder()

    # groupby 1회로 세션 분할 (df[df[col]==sess] 마스킹은 26M행 풀스캔을 세션수만큼 반복 → 병목)
    y_true, y_pred = [], []
    for _, sess_df in df.groupby(session_col, sort=False):
        X_s, y_s = preprocessor.process_single_session(sess_df, time_col)
        if not X_s:
            continue
        X = np.asarray(X_s, dtype=np.float32)        # (n_win, 5000, 5)
        xb = torch.from_numpy(X).permute(0, 2, 1)    # (N, C, T) — HARDataset 과 동일
        for i in range(0, len(xb), batch_size):
            out = model(xb[i:i + batch_size].to(device))
            y_pred.extend(out.argmax(1).cpu().numpy())
        y_true.extend(le.transform(y_s))

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    acc = accuracy_score(y_true, y_pred) * 100 if len(y_true) else float("nan")
    return (acc, y_true, y_pred) if return_preds else acc


# ── 정확도 (멀티모달 모델 추론) ──────────────────────────────────────────────
@torch.no_grad()
def evaluate_accuracy_mm(df, model, *, session_col="csv_filename_l", time_col="Index_Time",
                         preprocessor=None, device=None, batch_size=256, return_preds=False):
    """멀티모달 버전 evaluate_accuracy.

    EMG/IMU 를 각자의 레이트로 분리 전처리(MultiModalPreprocessor)한다:
      EMG (N,2,5000) 1000Hz / IMU (N,3,500) **네이티브 100Hz**.
    즉 IMU 평가가 MM 모델이 실제로 보는 100Hz 기준으로 수행된다
    (5채널 evaluate_accuracy 는 IMU 를 1000Hz 로 업샘플해 본다는 점이 다름).

    model.forward(emg, imu) 는 (class_logits, features) 튜플을 반환하므로 [0] 을 쓴다.
    return_preds=True 면 (acc, y_true, y_pred) 반환.
    """
    from data_preprocess_MM_original import MultiModalPreprocessor  # noqa: E402

    device = device or next(model.parameters()).device
    preprocessor = preprocessor or MultiModalPreprocessor()
    le = get_label_encoder()

    y_true, y_pred = [], []
    for _, sess_df in df.groupby(session_col, sort=False):
        X_emg, X_imu, y_s = preprocessor.process_single_session(sess_df, time_col)
        if not X_emg:
            continue
        emg = np.asarray(X_emg, dtype=np.float32).transpose(0, 2, 1)  # (N, 2, 5000)
        imu = np.asarray(X_imu, dtype=np.float32).transpose(0, 2, 1)  # (N, 3,  500)
        emg_t, imu_t = torch.from_numpy(emg), torch.from_numpy(imu)
        for i in range(0, len(emg_t), batch_size):
            out = model(emg_t[i:i + batch_size].to(device),
                        imu_t[i:i + batch_size].to(device))
            logits = out[0] if isinstance(out, (tuple, list)) else out
            y_pred.extend(logits.argmax(1).cpu().numpy())
        y_true.extend(le.transform(y_s))

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    acc = accuracy_score(y_true, y_pred) * 100 if len(y_true) else float("nan")
    return (acc, y_true, y_pred) if return_preds else acc


# ── DTW (samsung1 triceps IMU 와의 축 정렬도) ────────────────────────────────
def _zscore(x):
    return (x - x.mean(0)) / (x.std(0) + 1e-8)


def _resample(x, n):
    return x[np.linspace(0, len(x) - 1, n, dtype=int)]


def _dtw_mv(a, b):
    """이미 z-score+resample 된 두 (n, 3) 신호의 multivariate DTW 거리."""
    n = len(a)
    cost = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))  # (n, n)
    D = np.full((n + 1, n + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, n])


def _exercise_segments(df, exercise, session_col, time_col, n_pts, max_sess):
    """운동 exercise 의 대표 세션(정렬상 앞 max_sess개) triceps IMU 를
    z-score+resample 한 (n_pts, 3) 리스트로 반환(결정적)."""
    sub = df[df["exercise"] == exercise]
    segs = []
    for sess in sorted(sub[session_col].unique())[:max_sess]:
        s = sub[sub[session_col] == sess].sort_values(time_col)
        mat = s[IMU_COLS].to_numpy(dtype=np.float64)
        if len(mat) < 20:
            continue
        segs.append(_resample(_zscore(mat), n_pts))
    return segs


def build_dtw_reference(df_source, *, session_col="filename", time_col="timestamp",
                        n_pts=200, max_sess=3):
    """samsung1(source) 운동별 triceps IMU 대표 세그먼트를 한 번만 빌드해 캐시.

    48 permutation 동안 source 는 고정이므로 재사용한다.
    반환: {exercise: [ (n_pts,3), ... ]}
    """
    return {
        ex: _exercise_segments(df_source, ex, session_col, time_col, n_pts, max_sess)
        for ex in CLASS_NAMES
        if (df_source["exercise"] == ex).any()
    }


def compute_dtw(df_target, reference, *, session_col="csv_filename_l",
                time_col="Index_Time", n_pts=200, max_sess=1, return_per_exercise=False):
    """target triceps IMU 와 source reference 사이 운동별 multivariate DTW 평균.

    reference : build_dtw_reference 결과. n_pts 는 reference 빌드 때와 동일해야 한다.
    """
    per_ex = {}
    for ex, src_segs in reference.items():
        if not src_segs:
            continue
        tgt_segs = _exercise_segments(df_target, ex, session_col, time_col, n_pts, max_sess)
        if not tgt_segs:
            continue
        dists = [_dtw_mv(s, t) for s in src_segs for t in tgt_segs]
        per_ex[ex] = float(np.mean(dists))

    mean_dtw = float(np.mean(list(per_ex.values()))) if per_ex else float("nan")
    return (mean_dtw, per_ex) if return_per_exercise else mean_dtw


# ── Gravity alignment (중력 기반 축 정렬) ────────────────────────────────────
# IMU(가속도계)에는 항상 중력 1g 가 실리므로, 세션 평균 가속도를 중력 방향 추정치로
# 쓴다. 각 디바이스의 전역 중력 방향을 공통 canonical 축(기본 -Z)으로 회전시키면
# 디바이스 장착 각도(tilt) 차이가 정규화된다 — source/target 에 동일 규칙을 적용.
#   주의: 이 회전은 proper rotation(det=+1) 이라 축 swap 같은 reflection(handedness
#         flip) 은 보정하지 못한다 — tilt 만 정렬한다.
GRAVITY_CANONICAL = np.array([0.0, 0.0, -1.0])  # samsung1 의 자연 중력 방향과 일치
def _skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def rotation_align(a, b):
    """단위벡터 a 를 b 로 보내는 최소 회전 행렬 R (3,3). x_rot = R @ x."""
    a = np.asarray(a, float); a = a / (np.linalg.norm(a) + 1e-12)
    b = np.asarray(b, float); b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-8:                       # 평행: 같으면 I, 반평행이면 임의축 180°
        if c > 0:
            return np.eye(3)
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis = axis / np.linalg.norm(axis)
        K = _skew(axis)
        return np.eye(3) + 2.0 * (K @ K)
    K = _skew(v)
    return np.eye(3) + K + (K @ K) * ((1.0 - c) / (s ** 2))


def estimate_gravity_direction(df, session_col):
    """세션별 평균 가속도(=중력 proxy)를 단위벡터로 평균낸 전역 중력 방향 (3,)."""
    dirs = []
    for _, g in df.groupby(session_col, sort=False):
        m = g[IMU_COLS].to_numpy(dtype=np.float64).mean(0)
        n = np.linalg.norm(m)
        if n > 1e-8:
            dirs.append(m / n)
    G = np.mean(dirs, axis=0)
    return G / (np.linalg.norm(G) + 1e-12)


def apply_gravity_alignment(df_target, g_source, *, g_target=None,
                            session_col="csv_filename_l"):
    """target 의 중력 방향을 g_source 로 회전시킨 새 DataFrame 과 회전행렬 R 반환.

    EMG(biceps, triceps) 는 그대로 두고 triceps_X/Y/Z 만 회전한다.
    g_target 미지정 시 df_target 에서 추정한다.
    """
    if g_target is None:
        g_target = estimate_gravity_direction(df_target, session_col)
    R = rotation_align(g_target, g_source)        # target frame → source frame
    out = df_target.copy()
    imu = df_target[IMU_COLS].to_numpy(dtype=np.float64)
    out[IMU_COLS] = imu @ R.T                       # 각 행벡터 x → R @ x
    return out, R


# ── 통합 진입점 ──────────────────────────────────────────────────────────────
def evaluate(df_target, model, dtw_reference, *, session_col="csv_filename_l",
             time_col="Index_Time", n_pts=200, preprocessor=None, device=None,
             batch_size=256):
    """target DataFrame 하나에 대해 {accuracy, dtw, n_windows} 반환.

    48 permutation 루프의 핵심 호출:
        ref = build_dtw_reference(df_src)
        for name, order, signs in iter_imu_permutations():
            df_p = apply_imu_permutation(df_tgt, order, signs)
            res  = evaluate(df_p, model, ref)
    """
    acc, y_true, _ = evaluate_accuracy(
        df_target, model, session_col=session_col, time_col=time_col,
        preprocessor=preprocessor, device=device, batch_size=batch_size,
        return_preds=True)
    dtw = compute_dtw(df_target, dtw_reference, session_col=session_col,
                      time_col=time_col, n_pts=n_pts)
    return {"accuracy": acc, "dtw": dtw, "n_windows": int(len(y_true))}


def evaluate_mm(df_target, model, dtw_reference, *, session_col="csv_filename_l",
                time_col="Index_Time", n_pts=200, preprocessor=None, device=None,
                batch_size=256):
    """멀티모달 버전 evaluate — accuracy 는 MM 모델(IMU 100Hz)로, DTW 는 동일.

    DTW 는 모델과 무관하게 raw IMU 컬럼으로 계산되므로 5채널과 같은 값이 나온다
    (축 정렬도 측정용). 48 permutation 루프 사용법은 evaluate 와 동일하되
    evaluate_mm + load_mm_model 을 쓴다.
    """
    acc, y_true, _ = evaluate_accuracy_mm(
        df_target, model, session_col=session_col, time_col=time_col,
        preprocessor=preprocessor, device=device, batch_size=batch_size,
        return_preds=True)
    dtw = compute_dtw(df_target, dtw_reference, session_col=session_col,
                      time_col=time_col, n_pts=n_pts)
    return {"accuracy": acc, "dtw": dtw, "n_windows": int(len(y_true))}
