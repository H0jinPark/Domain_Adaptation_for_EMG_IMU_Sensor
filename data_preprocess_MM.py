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

저장 파일 (preprocessed_MM_<method>/)
  X_emg_{prefix}.npy : (N, 2, 5000)
  X_imu_{prefix}.npy : (N, 3,  500)
  y_{prefix}.npy     : (N,)
  R_used.npy         : (3, 3)   -- 이 실행에 사용한 target IMU 회전행렬(provenance)

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


class MultiModalPreprocessor:
    """EMG/IMU 를 분리 저장하는 멀티모달 전처리기.

    R 이 주어지면 해당 도메인의 IMU 3축에 회전행렬을 곱해 축을 정렬한다.
    """

    def __init__(self, out_dir=None, imu_norm='zscore'):
        self.emg_cols  = ['biceps', 'triceps']
        self.imu_cols  = ['triceps_X', 'triceps_Y', 'triceps_Z']
        self.label_col = 'exercise'
        self.imu_step  = EMG_FS // IMU_FS  # 1000Hz -> 100Hz: 10샘플마다 1개
        self.out_dir   = out_dir
        self.imu_norm  = imu_norm          # 'zscore' | 'isotropic' | 'none'
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
        total = len(session_list)

        for i, sess in enumerate(session_list):
            sess_df = df[df[session_col] == sess]
            X_emg, X_imu, y = self.process_single_session(sess_df, time_col, R=R)
            if X_emg:
                all_emg.extend(X_emg)
                all_imu.extend(X_imu)
                all_y.extend(y)

            if (i + 1) % max(1, total // 5) == 0:
                print(f"  [{prefix}] {i+1}/{total} sessions done")

        # (N, T, C) -> (N, C, T) 로 축 교환 후 저장
        emg_np = np.array(all_emg).transpose(0, 2, 1).astype(np.float32)  # (N, 2, 5000)
        imu_np = np.array(all_imu).transpose(0, 2, 1).astype(np.float32)  # (N, 3, 500)
        y_np   = np.array(all_y)

        np.save(f'{self.out_dir}/X_emg_{prefix}.npy', emg_np)
        np.save(f'{self.out_dir}/X_imu_{prefix}.npy', imu_np)
        np.save(f'{self.out_dir}/y_{prefix}.npy',     y_np)

        print(f"  {prefix} -> EMG {emg_np.shape}  IMU {imu_np.shape}  y {y_np.shape}")


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
# 실행 영역: 세션 단위 8:2 분할 후 split 별 저장
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
    parser.add_argument('--imu_norm', default='zscore', choices=['zscore', 'isotropic', 'none'],
                        help="IMU 정규화: zscore(축별, 기존 기본) | "
                             "isotropic(3축 공통 scalar=세션 중력크기, 회전·중력 보존+단위 스케일 매칭) | "
                             "none(raw)")
    args = parser.parse_args()

    # 기본 출력 폴더: zscore 는 기존 이름 유지(하위호환), 그 외엔 정규화 접미사
    default_out = f'preprocessed_MM_{args.method}'
    if args.imu_norm != 'zscore':
        default_out += f'_{args.imu_norm}'
    out_dir = args.out or default_out

    # target IMU 회전행렬 R 결정 (기본 = 단위행렬 = raw)
    if args.R is None:
        R_target = np.eye(3)
        print(f"[R] 미지정 → eye(3) (raw, 정렬 없음). method='{args.method}'")
    else:
        R_target = load_R(args.R)
        print(f"[R] '{args.R}' 로드:\n{R_target}")
        print(f"    |R|={np.linalg.det(R_target):+.4f}  method='{args.method}'")

    prep = MultiModalPreprocessor(out_dir=out_dir, imu_norm=args.imu_norm)
    print(f"[IMU 정규화] imu_norm='{args.imu_norm}'  -> {out_dir}")
    # provenance: 이번 실행에 사용한 R 저장
    np.save(os.path.join(out_dir, 'R_used.npy'), R_target.astype(np.float64))

    # 1. Source (samsung1) — reference 이므로 R 적용 안 함(항상 eye)
    print(f"\n[Step 1] Source Data ({SOURCE_PARQUET})...  (R=I, 정렬 없음)")
    df_src = pd.read_parquet(SOURCE_PARQUET)
    train_sess, val_sess = train_test_split(
        df_src[SOURCE_SESSION_COL].unique(), test_size=0.2, random_state=42)
    prep.save_processed_data(df_src, train_sess, SOURCE_SESSION_COL, SOURCE_TIME_COL, 'train', R=None)
    prep.save_processed_data(df_src, val_sess,   SOURCE_SESSION_COL, SOURCE_TIME_COL, 'val',   R=None)

    # 2. Target (samsung2 = raw 원본) — target IMU 에 R 적용
    print(f"\n[Step 2] Target Data ({TARGET_PARQUET})...  (R 적용: method='{args.method}')")
    df_tgt = pd.read_parquet(TARGET_PARQUET)
    tgt_train_sess, tgt_val_sess = train_test_split(
        df_tgt[TARGET_SESSION_COL].unique(), test_size=0.2, random_state=42)
    prep.save_processed_data(df_tgt, tgt_train_sess, TARGET_SESSION_COL, TARGET_TIME_COL, 'target_train', R=R_target)
    prep.save_processed_data(df_tgt, tgt_val_sess,   TARGET_SESSION_COL, TARGET_TIME_COL, 'target_val',   R=R_target)

    print("\n" + "=" * 50)
    print(f"완료. {out_dir}/ 에 저장됨  (method='{args.method}')")
    print("  EMG: (N, 2, 5000)  -- 1000Hz, 5초")
    print("  IMU: (N, 3,  500)  --  100Hz, 5초")
    print(f"  R_used.npy 저장됨 (target IMU 회전, |R|={np.linalg.det(R_target):+.4f})")
    print("=" * 50)


if __name__ == "__main__":
    main()
