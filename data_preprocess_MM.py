"""멀티모달(EMG/IMU 분리) 통합 전처리 스크립트.

축 정렬 방법(PCA / gravity alignment / permutation / Kabsch / raw)을 하나의
코드로 처리한다. target IMU 3축에 단일 3x3 회전행렬 R 을 곱해 정렬하며,
R 의 기본값은 단위행렬 eye(3) = "raw"(정렬 없음) 이다. 특정 R 을 입력하면
그 방법에 맞게 전처리되고, 메서드별로 다른 출력 폴더에 저장된다.

도메인 규약 (파일명 변경 반영)
  source = data/samsung1.parquet            (유지)
  target = data/samsung2.parquet            (= 과거 samsung2_original, raw/unaligned 원본)
  * 과거의 samsung2.parquet(YXZ 재라벨링본)은 현재 samsung2_YXZ.parquet 이며,
    이번 실험에서는 사용하지 않는다(permutation R 로 대체됨).

R 적용 규약 (rotation_matrix_direct.ipynb 와 동일)
  - R 은 target→source 회전행렬이며, target IMU 3축(triceps_X,Y,Z)에 다음으로 적용:
        imu_rotated = imu_raw @ R.T      (T,3) @ (3,3)
    노트북 평가(`acc_dtw`)가 `imu @ R.T` 로 측정하므로, 전처리도 동일하게 R.T 를 쓴다.
    → 노트북이 저장한 R_<method>.npy 를 그대로 --R 로 넣으면 평가와 정확히 일치한다.
  - source(samsung1)는 reference 이므로 항상 R=I (정렬하지 않음)
  - 회전은 z-score 정규화 *이전* 에 적용한다(기존 PCA 정렬 파이프라인과 동일).

채널 규약
  EMG: [biceps, triceps]                 -> 1000Hz, window=5000 (5초)
  IMU: [triceps_X, triceps_Y, triceps_Z] ->  100Hz, window=500  (5초)

데이터 분할 (세션 단위 7:1.5:1.5, 층화)
  train / val / test 3분할이다. val 은 model selection 전용, **test 는 최종 보고 전용**
  으로 학습·선택 어디에도 쓰지 않는다.
  - 세션 단위로 자르므로 같은 세션의 윈도우가 split 을 넘나드는 leakage 가 없다.
  - 세션당 운동은 정확히 1종이라, 세션 라벨로 **층화(stratify)** 한다. test 가 15%
    (운동당 ~7세션) 로 작아 층화 없이는 클래스 구성이 split 마다 흔들린다.
  - 분할 결과는 out_dir/split_manifest.json 에 세션명까지 기록된다. --split_manifest
    로 그 파일을 넘기면 **동일한 분할을 그대로 재사용**하므로, 축 정렬 방법(method)이
    달라도 모든 전처리 폴더가 같은 세션 분할을 쓴다는 것이 보장·검증 가능하다.

저장 파일 (preprocessed_MM_<method>/)
  X_emg_{prefix}.npy : (N, 2, 5000)
  X_imu_{prefix}.npy : (N, 3,  500)
  y_{prefix}.npy     : (N,)
  R_used.npy         : (3, 3)   -- 이 실행에 사용한 target IMU 회전행렬(provenance)
  split_manifest.json           -- split 별 세션명 + 비율/seed(provenance)

  prefix = train | val | test | target_train | target_val | target_test

사용 예
  python data_preprocess_MM.py --method raw
  python data_preprocess_MM.py --method gravity    --R path/to/gravity_R.npy
  python data_preprocess_MM.py --method permutation --R path/to/perm_R.npy
  python data_preprocess_MM.py --method kabsch     --R path/to/kabsch_R.npy
  python data_preprocess_MM.py --method pca        --R path/to/pca_R.npy
  # --out 으로 출력 폴더를 직접 지정할 수도 있다(기본은 preprocessed_MM_<method>).

IMU 정규화 (--imu_norm)
  - zscore   (기본): 축별 독립 z-score. 회전 구조·중력 DC 파괴 → 학습 R 식별 불가.
  - isotropic      : 3축 공통 단일 scalar(세션 중력크기 ≈1g)로만 나눔(평균 보존).
                     단위(g/mg) 스케일 매칭 + 회전·중력 방향 보존 → learnable R/중력
                     prior 가 의미를 갖는다. 출력: preprocessed_MM_<method>_isotropic
  - none           : 정규화 없음(raw, 진단용).
  예) python data_preprocess_MM.py --method raw --imu_norm isotropic
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

EMG_FS     = 1000   # EMG 목표 샘플링 레이트
IMU_FS     = 100    # IMU 목표 샘플링 레이트 (다운샘플 배수: EMG_FS // IMU_FS = 10)
EMG_WINDOW = 5000   # EMG 윈도우 크기 (5초)
IMU_WINDOW = 500    # IMU 윈도우 크기 (5초, 동일 절대시간)
EMG_STRIDE = 500    # EMG 기준 스트라이드 (0.5초)

# 도메인별 parquet 경로 / 세션·시간 컬럼명
SOURCE_PARQUET = 'data/samsung1.parquet'
TARGET_PARQUET = 'data/samsung2.parquet'   # 과거 samsung2_original (raw 원본)
SOURCE_SESSION_COL, SOURCE_TIME_COL = 'filename',       'timestamp'
TARGET_SESSION_COL, TARGET_TIME_COL = 'csv_filename_l', 'Index_Time'
# 피험자 단위 분할용 컬럼 (두 parquet 이 스키마가 달라 이름·타입이 다르다)
SOURCE_SUBJECT_COL = 'subject'      # 'sub1' ... 'sub14'  (문자열)
TARGET_SUBJECT_COL = 'subject_id'   #  1 ... 14           (정수)


class MultiModalPreprocessor:
    """EMG/IMU 를 분리 저장하는 멀티모달 전처리기.

    R 이 주어지면 해당 도메인의 IMU 3축에 회전행렬을 곱해 축을 정렬한다.
    """

    def __init__(self, out_dir=None, imu_norm='zscore',
                 emg_antialias=False, emg_aa_cutoff=450.0):
        self.emg_cols  = ['biceps', 'triceps']
        self.imu_cols  = ['triceps_X', 'triceps_Y', 'triceps_Z']
        self.label_col = 'exercise'
        self.imu_step  = EMG_FS // IMU_FS  # 1000Hz -> 100Hz: 10샘플마다 1개
        self.out_dir   = out_dir
        self.imu_norm  = imu_norm          # 'zscore' | 'isotropic' | 'none'
        self.emg_antialias = emg_antialias # 데시메이션 전 안티에일리어싱 LPF 적용 여부
        self.emg_aa_cutoff = emg_aa_cutoff # 그 LPF 의 컷오프(Hz)
        # out_dir 은 저장(save_processed_data)할 때만 필요하다. 평가/추론
        # (eval_utils 의 MMEvalCache·evaluate_accuracy_mm 등 process_single_session 만
        # 쓰는 경우)에는 out_dir 없이 생성할 수 있도록 None 을 허용한다.
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)

    def _bandpass_emg(self, data):
        """EMG 신호에 20~450Hz 밴드패스 필터를 적용한다."""
        nyq = 0.5 * EMG_FS
        b, a = butter(4, [20.0 / nyq, 450.0 / nyq], btype='bandpass', analog=False)
        return filtfilt(b, a, data)

    def _antialias_emg(self, session_df, time_col):
        """**네이티브 레이트에서** EMG 에 안티에일리어싱 LPF 를 건다 (in-place).

        왜 필요한가 —
          아래 1000Hz 리샘플(`resample('1ms').mean()`)은 1ms bin 평균이라 사실상
          2-tap 이동평균이고, 안티에일리어싱 필터 구실을 못 한다. 그래서 500Hz 위
          성분이 20~450Hz 대역 안으로 접혀 들어온다(folding).
          이게 도메인 간 비대칭인 게 문제다:
            samsung1 : 아날로그 롤오프가 뚜렷해 >500Hz 전력이 대역전력의 0.05~0.07%
            samsung2 : ~400Hz 위로 거의 백색 → >500Hz 전력이 5.9~12.5%
          즉 현재 파이프라인은 target 에만 없는 고주파를 만들어 넣고 있고, 측정상
          HF 격차(SSC d=+2.51, HF% Δ+4.72)의 15~26%가 이 접힘에서 온다.
          네이티브에서 먼저 잘라내면 그만큼은 사라진다.

        주의 — 이건 "전이 정확도를 올리는 트릭"이 아니라 신호처리상 옳은 순서다.
        나머지 75~85% 는 samsung2 자체의 대역 내 노이즈 바닥이라 이걸로 안 없어진다.

        IMU 는 손대지 않는다. IMU 도 1000→100Hz 를 strided sampling 으로 떨구고 있어
        같은 문제가 있지만, 원 레이트가 148/66Hz 라 성격이 달라 별건으로 다룬다.
        """
        t = session_df[time_col].to_numpy(dtype=np.float64)
        if len(t) < 64:
            return session_df
        dt = np.median(np.diff(t))
        if not np.isfinite(dt) or dt <= 0:
            return session_df
        fs_native = 1.0 / dt
        # 컷오프가 네이티브 Nyquist 이상이면 걸 필요도, 걸 수도 없다
        if fs_native <= 2.0 * self.emg_aa_cutoff:
            return session_df
        b, a = butter(8, self.emg_aa_cutoff / (0.5 * fs_native), btype='low', analog=False)
        for col in self.emg_cols:
            session_df[col] = filtfilt(b, a, session_df[col].to_numpy(dtype=np.float64))
        return session_df

    def _normalize_imu(self, seg):
        """IMU (T,3) 정규화. self.imu_norm 에 따라 분기한다.

          zscore   : 축별 독립 z-score (기존 기본). 축마다 다른 scale·평균 제거로
                     회전 구조와 중력 DC 가 파괴됨 → 학습 R 이 식별 불가.
          isotropic: 3축에 **단일 scalar 하나**(세션 가속도 크기의 중앙값 ≈ 1g)로만
                     나눈다. 평균은 빼지 않는다. R 이 회전이라 ‖Ra‖=‖a‖ 이므로 이
                     scalar 는 회전불변 → R 과 commute. 단위(g/mg)가 달라도 각 세션이
                     자기 중력 크기로 나뉘어 ~1 로 스케일 매칭되고, 중력 방향·축간
                     크기비(=회전 구조)는 그대로 보존된다.
          none     : 정규화 없음(raw). 단위/스케일 차이 그대로 — 진단·비교용.
        """
        seg = np.asarray(seg, dtype=np.float64)
        if self.imu_norm == 'zscore':
            return StandardScaler().fit_transform(seg)
        if self.imu_norm == 'isotropic':
            mag = np.linalg.norm(seg, axis=1)          # (T,) per-sample ‖a‖ (회전불변)
            s = float(np.median(mag))
            if not np.isfinite(s) or s < 1e-8:
                s = 1.0
            return seg / s
        if self.imu_norm == 'none':
            return seg
        raise ValueError(f"알 수 없는 imu_norm: {self.imu_norm!r} "
                         "(zscore|isotropic|none 중 하나)")

    def process_single_session(self, session_df, time_col, R=None):
        """단일 세션을 전처리하고 (X_emg, X_imu, y) 윈도우 리스트를 반환한다.

        R 이 주어지면(=단위행렬이 아니면) IMU 3축을 회전 정렬한다.
        """
        session_df = session_df.sort_values(by=time_col).copy()

        # 0. [안티에일리어싱] 네이티브 레이트에서 EMG LPF — 반드시 리샘플 *이전*
        if self.emg_antialias:
            session_df = self._antialias_emg(session_df, time_col)

        session_df['datetime'] = pd.to_timedelta(session_df[time_col], unit='s')
        session_df = session_df.set_index('datetime')

        # 1. 1000Hz 리샘플링
        all_cols = self.emg_cols + self.imu_cols
        numeric_df = session_df[all_cols].resample('1ms').mean().interpolate('linear')
        label_df   = session_df[[self.label_col]].resample('1ms').ffill()
        df = pd.concat([numeric_df, label_df], axis=1).dropna()

        if len(df) < EMG_WINDOW:
            return None, None, None

        # 2. EMG 밴드패스 필터 (20~450Hz)
        for col in self.emg_cols:
            df[col] = self._bandpass_emg(df[col].values)

        # 3. [축 정렬] IMU 3축에 회전행렬 R 적용 (z-score 이전)
        #    노트북과 동일 규약: imu_rotated = imu_raw @ R.T  (R = target→source)
        if R is not None:
            seg = df[self.imu_cols].to_numpy(dtype=np.float64)   # (T, 3)
            df[self.imu_cols] = seg @ R.T                        # (T,3) @ (3,3)

        # 4. 정규화: EMG 는 축별 z-score(고정), IMU 는 self.imu_norm 옵션에 따라
        #    (R 적용 *이후* 에 정규화 — isotropic 은 R 과 commute)
        df[self.emg_cols] = StandardScaler().fit_transform(df[self.emg_cols])
        df[self.imu_cols] = self._normalize_imu(df[self.imu_cols].to_numpy())

        emg_vals   = df[self.emg_cols].values   # (T, 2)
        imu_vals   = df[self.imu_cols].values   # (T, 3)
        label_vals = df[self.label_col].values

        X_emg, X_imu, y_all = [], [], []

        for start in range(0, len(df) - EMG_WINDOW + 1, EMG_STRIDE):
            # EMG: 5000 샘플 그대로
            emg_win = emg_vals[start : start + EMG_WINDOW]                  # (5000, 2)

            # IMU: 같은 시간 구간을 10배 다운샘플 -> 500 샘플
            imu_win = imu_vals[start : start + EMG_WINDOW : self.imu_step]  # (500, 3)

            # 윈도우 내 최빈 라벨
            win_labels = label_vals[start : start + EMG_WINDOW]
            u, counts  = np.unique(win_labels, return_counts=True)
            label      = u[np.argmax(counts)]

            X_emg.append(emg_win)
            X_imu.append(imu_win)
            y_all.append(label)

        return X_emg, X_imu, y_all

    def save_processed_data(self, df, session_list, session_col, time_col, prefix, R=None):
        """세션 리스트를 전처리해 EMG/IMU/label 을 out_dir 에 저장한다."""
        if self.out_dir is None:
            raise ValueError(
                "save_processed_data 에는 out_dir 이 필요합니다 — "
                "MultiModalPreprocessor(out_dir=...) 로 생성하세요.")
        all_emg, all_imu, all_y = [], [], []
        # 윈도우 -> 세션 역추적용. 세션마다 길이가 달라 나오는 윈도우 수도 다르고,
        # extend 로 이어붙이면 경계가 사라지므로 세션별 기여 행 수를 여기서 세어 둔다.
        sess_names, sess_counts = [], []
        total = len(session_list)

        for i, sess in enumerate(session_list):
            sess_df = df[df[session_col] == sess]
            X_emg, X_imu, y = self.process_single_session(sess_df, time_col, R=R)
            if X_emg:
                all_emg.extend(X_emg)
                all_imu.extend(X_imu)
                all_y.extend(y)
                sess_names.append(sess)
                sess_counts.append(len(X_emg))

            if (i + 1) % max(1, total // 5) == 0:
                print(f"  [{prefix}] {i+1}/{total} sessions done")

        # (N, T, C) -> (N, C, T) 로 축 교환 후 저장
        emg_np = np.array(all_emg).transpose(0, 2, 1).astype(np.float32)  # (N, 2, 5000)
        imu_np = np.array(all_imu).transpose(0, 2, 1).astype(np.float32)  # (N, 3, 500)
        y_np   = np.array(all_y)

        # 윈도우별 출처 세션명 (y 와 길이·순서 동일). 세션명에 피험자·운동·촬영시각이
        # 들어 있어 피험자별/세션별 정확도 분해에 쓴다. 신호·레이블·순서는 안 바뀐다.
        sess_np = (np.repeat(np.array(sess_names), sess_counts) if sess_names
                   else np.array([], dtype='<U1'))
        assert len(sess_np) == len(y_np), f"{prefix}: 세션 라벨 {len(sess_np)} != y {len(y_np)}"

        np.save(f'{self.out_dir}/X_emg_{prefix}.npy', emg_np)
        np.save(f'{self.out_dir}/X_imu_{prefix}.npy', imu_np)
        np.save(f'{self.out_dir}/y_{prefix}.npy',     y_np)
        np.save(f'{self.out_dir}/sessions_{prefix}.npy', sess_np)

        print(f"  {prefix} -> EMG {emg_np.shape}  IMU {imu_np.shape}  y {y_np.shape}  "
              f"sessions {sess_np.shape} ({len(sess_names)}개 세션)")


# ----------------------------------------------------------------------
# R 로딩 유틸
# ----------------------------------------------------------------------
def load_R(path):
    """3x3 회전행렬을 .npy / .npz 에서 로드한다. .npz 면 'R' 키를 우선 사용한다."""
    obj = np.load(path)
    if isinstance(obj, np.lib.npyio.NpzFile):
        R = obj['R'] if 'R' in obj.files else obj[obj.files[0]]
    else:
        R = obj
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"R 의 shape 가 (3,3) 이 아닙니다: {R.shape} (from {path})")
    return R


# ----------------------------------------------------------------------
# 세션 단위 3분할 (train/val/test, 층화)
# ----------------------------------------------------------------------
def session_labels(df, session_col, label_col):
    """세션 -> 라벨 Series 를 만든다. 세션당 운동이 1종인지 검증한다.

    층화 분할의 전제(세션 = 라벨 1개)가 깨지면 조용히 잘못된 층화가 되므로
    여기서 명시적으로 막는다.
    """
    g = df.groupby(session_col)[label_col]
    multi = g.nunique()
    bad = multi[multi > 1]
    if len(bad) > 0:
        raise ValueError(
            f"세션당 운동이 1종이 아닙니다 ({len(bad)}개 세션). 층화 분할 전제가 깨짐: "
            f"{list(bad.index[:5])}")
    return g.first()


def split_sessions(df, session_col, label_col, ratios, seed):
    """세션을 train/val/test 로 층화 분할한다. 반환: (train, val, test) 세션명 리스트.

    2단계로 나눈다. 먼저 test 를 떼고, 남은 (train+val) 안에서 val 을 뗀다.
    2단계의 test_size 는 전체 기준이 아니라 **남은 것 기준**으로 환산해야 하므로
    val/(train+val) 을 쓴다.
    """
    train_r, val_r, test_r = ratios
    lab = session_labels(df, session_col, label_col)
    sessions = df[session_col].unique()          # parquet 등장 순서 유지(재현성)
    y = lab.reindex(sessions).to_numpy()

    trval, test = train_test_split(
        sessions, test_size=test_r, random_state=seed, stratify=y)
    y_trval = lab.reindex(trval).to_numpy()
    train, val = train_test_split(
        trval, test_size=val_r / (train_r + val_r), random_state=seed, stratify=y_trval)

    return list(train), list(val), list(test)


def subject_key(v):
    """'sub7' / 7 / '7' 을 모두 정수 7 로 정규화한다 (두 도메인 스키마 통일)."""
    return int(str(v).strip().lower().replace('sub', ''))


def split_by_subject(df, session_col, subject_col, val_subjects, test_subjects):
    """피험자 단위로 세션을 3분할한다. 반환: (train, val, test) 세션명 리스트.

    세션 단위 층화와 달리 **같은 사람의 세션이 split 을 넘나들지 않는다.**
    (leakage-audit-session-split: 기존 세션 분할은 피험자가 train∩test 였다.)
    지정되지 않은 피험자는 전부 train 이다.

    주의 — 한 피험자가 특정 운동을 아예 안 한 경우가 있어서, val/test 피험자를
    잘못 고르면 그 split 에 클래스가 통째로 빠진다. 아래 verify_class_coverage 로 막는다.
    """
    val_k = {subject_key(v) for v in val_subjects}
    test_k = {subject_key(v) for v in test_subjects}
    overlap = val_k & test_k
    if overlap:
        raise ValueError(f"val 과 test 에 같은 피험자가 있습니다: {sorted(overlap)}")

    sess_subj = df.groupby(session_col)[subject_col].first()
    multi = df.groupby(session_col)[subject_col].nunique()
    bad = multi[multi > 1]
    if len(bad) > 0:
        raise ValueError(f"세션당 피험자가 1명이 아닙니다: {list(bad.index[:5])}")

    train, val, test = [], [], []
    for sess in df[session_col].unique():           # parquet 등장 순서 유지(재현성)
        k = subject_key(sess_subj[sess])
        (test if k in test_k else val if k in val_k else train).append(sess)

    present = {subject_key(v) for v in sess_subj.unique()}
    missing = (val_k | test_k) - present
    if missing:
        raise ValueError(f"지정한 피험자가 이 도메인에 없습니다: {sorted(missing)}")
    return train, val, test


def verify_class_coverage(df, session_col, label_col, splits, domain):
    """val/test 에 10종 운동이 모두 있는지 확인한다. 빠지면 그 클래스는 평가 불가다."""
    lab = session_labels(df, session_col, label_col)
    all_cls = set(lab.unique())
    ok = True
    for name, sess in zip(('train', 'val', 'test'), splits):
        got = set(lab.reindex(sess).dropna().unique())
        miss = sorted(all_cls - got)
        if miss:
            ok = False
            print(f"  [경고] {domain}/{name}: 없는 운동 {miss}  "
                  f"({len(got)}/{len(all_cls)}종만 존재)")
    if ok:
        print(f"  [OK]   {domain}: train/val/test 모두 {len(all_cls)}종 전부 포함")
    return ok


def report_split(df, session_col, label_col, splits, domain):
    """분할 결과를 운동별 세션 수 표로 출력하고, split 간 중복이 없는지 검증한다."""
    train, val, test = splits
    names = ['train', 'val', 'test']

    # 중복·누락 검증 (leakage 방지의 최종 관문)
    sets = [set(train), set(val), set(test)]
    for i in range(3):
        for j in range(i + 1, 3):
            dup = sets[i] & sets[j]
            if dup:
                raise ValueError(f"[{domain}] {names[i]}/{names[j]} 세션 중복: {list(dup)[:5]}")
    total = len(df[session_col].unique())
    if sum(len(s) for s in sets) != total:
        raise ValueError(f"[{domain}] 세션 수 불일치: 분할합={sum(len(s) for s in sets)} vs 전체={total}")

    lab = session_labels(df, session_col, label_col)
    print(f"\n[{domain}] 세션 분할 (총 {total}) — 운동별 세션 수")
    print(f"  {'exercise':<14}" + "".join(f"{n:>8}" for n in names))
    for ex in sorted(lab.unique()):
        row = [sum(1 for s in sp if lab[s] == ex) for sp in (train, val, test)]
        print(f"  {ex:<14}" + "".join(f"{c:>8}" for c in row))
    print(f"  {'합계':<12}" + "".join(f"{len(sp):>8}" for sp in (train, val, test)))


# ----------------------------------------------------------------------
# 실행 영역: 세션 단위 층화 3분할 후 split 별 저장
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="멀티모달 통합 전처리 (target IMU 축 정렬 R 적용)")
    parser.add_argument('--method', default='raw',
                        help="축 정렬 방법 이름. 출력 폴더 이름에 사용됨 (예: raw, pca, gravity, permutation, kabsch)")
    parser.add_argument('--R', default=None,
                        help="target→source 3x3 회전행렬 .npy/.npz 경로(노트북 R_<method>.npy 그대로). "
                             "imu @ R.T 로 적용. 미지정 시 eye(3)=raw")
    parser.add_argument('--out', default=None,
                        help="출력 폴더 직접 지정 (기본: preprocessed_MM_<method>[_<imu_norm>])")
    parser.add_argument('--emg_antialias', action='store_true',
                        help="1000Hz 리샘플 **이전** 에 네이티브 레이트에서 EMG 안티에일리어싱 "
                             "LPF(기본 450Hz, 8차 Butterworth)를 적용한다. 현재 resample('1ms').mean() "
                             "은 안티에일리어싱 구실을 못 해 >500Hz 성분이 20-450 안으로 접히는데, "
                             "samsung2 는 그 성분이 대역전력의 5.9~12.5%%, samsung1 은 0.05~0.07%% 라 "
                             "도메인 간 비대칭 아티팩트가 생긴다. 출력: preprocessed_MM_<method>[...]_aa")
    parser.add_argument('--emg_aa_cutoff', type=float, default=450.0,
                        help="--emg_antialias 의 컷오프(Hz). 기본 450 = 이후 밴드패스 상한과 동일.")
    parser.add_argument('--imu_norm', default='zscore', choices=['zscore', 'isotropic', 'none'],
                        help="IMU 정규화: zscore(축별, 기존 기본) | "
                             "isotropic(3축 공통 scalar=세션 중력크기, 회전·중력 보존+단위 스케일 매칭) | "
                             "none(raw)")
    parser.add_argument('--split', type=float, nargs=3, default=[0.7, 0.15, 0.15],
                        metavar=('TRAIN', 'VAL', 'TEST'),
                        help="세션 단위 train/val/test 비율 (합=1). 기본 0.7 0.15 0.15")
    parser.add_argument('--split_seed', type=int, default=42,
                        help="세션 분할 random_state (기본 42)")
    parser.add_argument('--split_by', default='session', choices=['session', 'subject'],
                        help="session(기본): 세션 단위 층화 분할. "
                             "subject: 피험자 단위 분할 — 같은 사람이 split 을 넘나들지 "
                             "않아 '처음 보는 사람'까지 주장 범위가 넓어진다.")
    parser.add_argument('--val_subjects', nargs='+', default=['3', '8'],
                        help="--split_by subject 일 때 val 로 뗄 피험자 ('sub3' 또는 '3').")
    parser.add_argument('--test_subjects', nargs='+', default=['11', '12'],
                        help="--split_by subject 일 때 test 로 뗄 피험자. "
                             "기본 11,12 는 양쪽 도메인에서 10종 운동을 모두 가진 "
                             "5명(1,3,8,11,12) 중에서 고른 것이다.")
    parser.add_argument('--split_manifest', default=None,
                        help="기존 split_manifest.json 경로. 주면 그 세션 분할을 그대로 재사용한다 "
                             "(--split/--split_seed 는 무시). 여러 method 폴더가 동일 분할을 쓰도록 보장.")
    args = parser.parse_args()

    if args.split_by == 'session' and abs(sum(args.split) - 1.0) > 1e-6:
        parser.error(f"--split 합이 1이 아닙니다: {args.split} (합={sum(args.split)})")
    if any(r <= 0 for r in args.split):
        parser.error(f"--split 은 모두 양수여야 합니다: {args.split}")

    # 기본 출력 폴더: zscore 는 기존 이름 유지(하위호환), 그 외엔 정규화 접미사
    default_out = f'preprocessed_MM_{args.method}'
    if args.imu_norm != 'zscore':
        default_out += f'_{args.imu_norm}'
    if args.emg_antialias:
        default_out += '_aa'
    out_dir = args.out or default_out

    # target IMU 회전행렬 R 결정 (기본 = 단위행렬 = raw)
    if args.R is None:
        R_target = np.eye(3)
        print(f"[R] 미지정 → eye(3) (raw, 정렬 없음). method='{args.method}'")
    else:
        R_target = load_R(args.R)
        print(f"[R] '{args.R}' 로드:\n{R_target}")
        print(f"    |R|={np.linalg.det(R_target):+.4f}  method='{args.method}'")

    prep = MultiModalPreprocessor(out_dir=out_dir, imu_norm=args.imu_norm,
                                  emg_antialias=args.emg_antialias,
                                  emg_aa_cutoff=args.emg_aa_cutoff)
    print(f"[IMU 정규화] imu_norm='{args.imu_norm}'  -> {out_dir}")
    if args.emg_antialias:
        print(f"[EMG 안티에일리어싱] 네이티브 레이트에서 {args.emg_aa_cutoff:.0f}Hz LPF (8차) — "
              f"1000Hz 리샘플 이전에 적용")
    else:
        print("[EMG 안티에일리어싱] 없음 (기존 동작). >500Hz 성분이 20-450 안으로 접힌다.")
    # provenance: 이번 실행에 사용한 R 저장
    np.save(os.path.join(out_dir, 'R_used.npy'), R_target.astype(np.float64))
    # provenance: 폴더만 보고도 어떤 옵션으로 만들었는지 알 수 있게 남긴다.
    # (러너가 이 파일을 읽어 "AA on/off 두 팔이 정말 다른 설정인지" 검증한다)
    with open(os.path.join(out_dir, 'preproc_config.json'), 'w', encoding='utf-8') as f:
        json.dump({'method': args.method, 'R': args.R, 'imu_norm': args.imu_norm,
                   'emg_antialias': bool(args.emg_antialias),
                   'emg_aa_cutoff': float(args.emg_aa_cutoff) if args.emg_antialias else None,
                   'emg_bandpass': [20.0, 450.0], 'emg_fs': EMG_FS, 'imu_fs': IMU_FS},
                  f, indent=2)

    # ---- 세션 분할 결정 (신규 생성 또는 기존 manifest 재사용) ----
    df_src = pd.read_parquet(SOURCE_PARQUET)
    df_tgt = pd.read_parquet(TARGET_PARQUET)

    if args.split_manifest:
        with open(args.split_manifest, encoding='utf-8') as f:
            manifest = json.load(f)
        src_splits = tuple(manifest['source'][k] for k in ('train', 'val', 'test'))
        tgt_splits = tuple(manifest['target'][k] for k in ('train', 'val', 'test'))
        print(f"\n[분할] 기존 manifest 재사용: {args.split_manifest}")
        print(f"       ratio={manifest['split_ratio']} seed={manifest['split_seed']}")
    elif args.split_by == 'subject':
        vs = [subject_key(v) for v in args.val_subjects]
        ts = [subject_key(v) for v in args.test_subjects]
        print(f"\n[분할] **피험자 단위** 3분할  val={sorted(vs)}  test={sorted(ts)}  "
              f"(나머지 전부 train)")
        print("       같은 사람의 세션이 split 을 넘나들지 않는다 → '처음 보는 사람' 평가")
        src_splits = split_by_subject(df_src, SOURCE_SESSION_COL, SOURCE_SUBJECT_COL, vs, ts)
        tgt_splits = split_by_subject(df_tgt, TARGET_SESSION_COL, TARGET_SUBJECT_COL, vs, ts)
        print("\n[클래스 커버리지 검사]")
        cov = verify_class_coverage(df_src, SOURCE_SESSION_COL, 'exercise', src_splits, 'source')
        cov &= verify_class_coverage(df_tgt, TARGET_SESSION_COL, 'exercise', tgt_splits, 'target')
        if not cov:
            raise SystemExit(
                "val/test 에 빠진 운동이 있습니다. --val_subjects/--test_subjects 를 "
                "양쪽 도메인에서 10종을 모두 가진 피험자(1,3,8,11,12)로 고르세요.")
        manifest = {
            'split_ratio': None,
            'split_seed': None,
            'split_by': 'subject',
            'val_subjects': sorted(vs),
            'test_subjects': sorted(ts),
            'source': dict(zip(('train', 'val', 'test'), map(list, src_splits)),
                           parquet=SOURCE_PARQUET, session_col=SOURCE_SESSION_COL,
                           subject_col=SOURCE_SUBJECT_COL),
            'target': dict(zip(('train', 'val', 'test'), map(list, tgt_splits)),
                           parquet=TARGET_PARQUET, session_col=TARGET_SESSION_COL,
                           subject_col=TARGET_SUBJECT_COL),
        }
    else:
        print(f"\n[분할] 세션 단위 층화 3분할  ratio={args.split}  seed={args.split_seed}")
        src_splits = split_sessions(df_src, SOURCE_SESSION_COL, 'exercise',
                                    args.split, args.split_seed)
        tgt_splits = split_sessions(df_tgt, TARGET_SESSION_COL, 'exercise',
                                    args.split, args.split_seed)
        manifest = {
            'split_ratio': list(args.split),
            'split_seed': args.split_seed,
            'split_by': 'session',
            'source': dict(zip(('train', 'val', 'test'), map(list, src_splits)),
                           parquet=SOURCE_PARQUET, session_col=SOURCE_SESSION_COL),
            'target': dict(zip(('train', 'val', 'test'), map(list, tgt_splits)),
                           parquet=TARGET_PARQUET, session_col=TARGET_SESSION_COL),
        }

    # 재사용이든 신규든 항상 검증한다(manifest 가 다른 parquet 로 만들어졌을 수 있음)
    report_split(df_src, SOURCE_SESSION_COL, 'exercise', src_splits, 'source')
    report_split(df_tgt, TARGET_SESSION_COL, 'exercise', tgt_splits, 'target')

    manifest_path = os.path.join(out_dir, 'split_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\n[분할] manifest 저장: {manifest_path}")

    # 1. Source (samsung1) — reference 이므로 R 적용 안 함(항상 eye)
    print(f"\n[Step 1] Source Data ({SOURCE_PARQUET})...  (R=I, 정렬 없음)")
    for prefix, sess in zip(('train', 'val', 'test'), src_splits):
        prep.save_processed_data(df_src, sess, SOURCE_SESSION_COL, SOURCE_TIME_COL,
                                 prefix, R=None)

    # 2. Target (samsung2 = raw 원본) — target IMU 에 R 적용
    print(f"\n[Step 2] Target Data ({TARGET_PARQUET})...  (R 적용: method='{args.method}')")
    for prefix, sess in zip(('target_train', 'target_val', 'target_test'), tgt_splits):
        prep.save_processed_data(df_tgt, sess, TARGET_SESSION_COL, TARGET_TIME_COL,
                                 prefix, R=R_target)

    print("\n" + "=" * 50)
    print(f"완료. {out_dir}/ 에 저장됨  (method='{args.method}')")
    print("  EMG: (N, 2, 5000)  -- 1000Hz, 5초")
    print("  IMU: (N, 3,  500)  --  100Hz, 5초")
    print(f"  split: train/val/test = {manifest['split_ratio']} (세션 단위·층화)")
    print("  ** test 는 최종 보고 전용 — 학습·model selection 에 쓰지 말 것 **")
    print(f"  R_used.npy 저장됨 (target IMU 회전, |R|={np.linalg.det(R_target):+.4f})")
    print("=" * 50)


if __name__ == "__main__":
    main()
