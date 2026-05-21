"""PCA 축 정렬 전처리 스크립트.

data_preprocess.py 와 동일한 (N, 5000, 5) 윈도우 포맷으로 저장하되, Target IMU 의
축 불일치를 운동별 PCA 회전 행렬로 보정한다.

차이점
  - Target(samsung2_original)의 triceps IMU 3축에 운동별 PCA 회전 W 를 곱해 정렬
  - Source(samsung1)는 그대로 사용 (reference)
  - W 행렬은 samsung1 vs samsung2_original 의 운동별 triceps IMU 신호로 한 번 학습

저장 위치: preprocessed_PCA/  (data_preprocess.py 와 동일 파일명)
  X_train.npy, y_train.npy, X_val.npy, y_val.npy
  X_target_train.npy, y_target_train.npy, X_target_val.npy, y_target_val.npy
"""
import os
import warnings
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------
# PCA 정렬 유틸 (notebook/process.ipynb 의 level1_pca 와 동일 로직)
# ----------------------------------------------------------------------
def _resample_mat(X, n):
    """행렬 X 를 n 개 행으로 균등 리샘플링한다."""
    return X[np.linspace(0, len(X) - 1, n, dtype=int)]


def _zscore_mat(X):
    """행렬 X 를 열 단위 Z-score 정규화한다."""
    return (X - X.mean(0)) / (X.std(0) + 1e-8)


def _fix_signs(W, s1_z, s2_z, n_check=300):
    """PCA 주축의 부호 불확정성을 s1 과의 상관계수로 보정한다."""
    n = min(len(s1_z), len(s2_z), n_check)
    c1 = _resample_mat(s1_z, n)
    c2 = _resample_mat(s2_z @ W, n)
    W = W.copy()
    for j in range(3):
        if np.corrcoef(c1[:, j], c2[:, j])[0, 1] < 0:
            W[:, j] *= -1
    return W


def level1_pca(s1_mat, s2_mat):
    """s2 @ W 가 PCA 주축 기준으로 s1 에 가장 잘 맞도록 하는 직교행렬 W 를 구한다."""
    s1_z = _zscore_mat(s1_mat)
    s2_z = _zscore_mat(s2_mat)
    V1 = PCA(n_components=3).fit(s1_z).components_
    V2 = PCA(n_components=3).fit(s2_z).components_
    W = V2.T @ V1
    return _fix_signs(W, s1_z, s2_z)


# ----------------------------------------------------------------------
# 운동별 W 행렬 학습
#   - samsung1 / samsung2_original 의 triceps_{X,Y,Z} 전체 시계열을 그대로 사용
#   - 운동별로 모든 세션 데이터를 stack 해서 한 번씩만 W 학습
# ----------------------------------------------------------------------
def learn_pca_W_per_exercise(df_src, df_tgt):
    """운동(exercise)별 PCA 회전 행렬 W 를 학습해 dict 로 반환한다."""
    cols = ["triceps_X", "triceps_Y", "triceps_Z"]
    exercises = sorted(set(df_src["exercise"].unique()) & set(df_tgt["exercise"].unique()))

    W_dict = {}
    print("\n[PCA W learning] exercise별 회전 행렬 학습")
    print(f"  exercises: {exercises}")
    for ex in exercises:
        s1_mat = df_src.loc[df_src["exercise"] == ex, cols].to_numpy(dtype=np.float64)
        s2_mat = df_tgt.loc[df_tgt["exercise"] == ex, cols].to_numpy(dtype=np.float64)
        if len(s1_mat) < 100 or len(s2_mat) < 100:
            print(f"  [skip] {ex}: too few samples ({len(s1_mat)}/{len(s2_mat)})")
            continue
        W = level1_pca(s1_mat, s2_mat)
        W_dict[ex] = W
        det = np.linalg.det(W)
        print(f"  [{ex:>12}] |W|={det:+.3f}  ({'rot' if det>0 else 'rot+ref'})")
    return W_dict


# ----------------------------------------------------------------------
# 전처리기 (data_preprocess.py 와 같은 인터페이스, target 에만 PCA 추가 적용)
# ----------------------------------------------------------------------
class PCAPreprocessor:
    """세션 단위 전처리기. W_dict 가 주어지면 Target IMU 에 PCA 정렬을 적용한다."""

    def __init__(self, target_fs=1000, window_size=5000, stride=500):
        self.target_fs = target_fs
        self.window_size = window_size
        self.stride = stride
        self.feature_cols = ["biceps", "triceps", "triceps_X", "triceps_Y", "triceps_Z"]
        self.imu_cols = ["triceps_X", "triceps_Y", "triceps_Z"]
        self.emg_cols = ["biceps", "triceps"]
        self.label_col = "exercise"

    def _butter_bandpass(self):
        """Butterworth 밴드패스 필터 계수 (b, a) 를 생성한다."""
        nyq = 0.5 * self.target_fs
        return butter(4, [20.0 / nyq, 450.0 / nyq], btype="bandpass", analog=False)

    def _apply_emg_filter(self, data):
        """EMG 신호에 20~450Hz 밴드패스 필터를 적용한다."""
        b, a = self._butter_bandpass()
        return filtfilt(b, a, data)

    def process_single_session(self, session_df, time_col, W_dict=None):
        """단일 세션을 전처리해 (X, y) 윈도우 리스트를 반환한다.

        W_dict 가 주어지면 운동별 PCA 회전으로 IMU 축을 정렬한다 (Target 전용).
        """
        session_df = session_df.sort_values(by=time_col).copy()
        session_df["datetime"] = pd.to_timedelta(session_df[time_col], unit="s")
        session_df = session_df.set_index("datetime")

        # 1. 1000Hz 리샘플링
        numeric_df = session_df[self.feature_cols].resample("1ms").mean().interpolate("linear")
        label_df = session_df[[self.label_col]].resample("1ms").ffill()
        df = pd.concat([numeric_df, label_df], axis=1).dropna()
        if len(df) < self.window_size:
            return None, None

        # 2. EMG 밴드패스 필터
        for col in self.emg_cols:
            df[col] = self._apply_emg_filter(df[col].values)

        # 3. [Target 전용] 운동별 IMU PCA 회전 정렬
        if W_dict is not None:
            for ex, idx in df.groupby(self.label_col).groups.items():
                if ex not in W_dict:
                    continue
                W = W_dict[ex]
                seg = df.loc[idx, self.imu_cols].to_numpy(dtype=np.float64)
                df.loc[idx, self.imu_cols] = seg @ W

        # 4. 세션 단위 Z-score 정규화 (data_preprocess.py 와 동일)
        scaler = StandardScaler()
        df[self.feature_cols] = scaler.fit_transform(df[self.feature_cols])

        # 5. 슬라이딩 윈도우 추출
        data_values = df[self.feature_cols].values
        label_values = df[self.label_col].values
        X_session, y_session = [], []
        for start in range(0, len(df) - self.window_size + 1, self.stride):
            X_session.append(data_values[start : start + self.window_size, :])
            win_labels = label_values[start : start + self.window_size]
            u, counts = np.unique(win_labels, return_counts=True)
            y_session.append(u[np.argmax(counts)])
        return X_session, y_session

    def save(self, df, session_list, session_col, time_col, prefix, out_dir, W_dict=None):
        """세션 리스트를 전처리해 X_{prefix}.npy / y_{prefix}.npy 로 저장한다."""
        X_all, y_all = [], []
        total = len(session_list)
        for i, sess in enumerate(session_list):
            sess_df = df[df[session_col] == sess]
            X_s, y_s = self.process_single_session(sess_df, time_col, W_dict=W_dict)
            if X_s:
                X_all.extend(X_s)
                y_all.extend(y_s)
            if (i + 1) % max(1, total // 5) == 0:
                print(f"  > {prefix} progress: {i+1}/{total} sessions")
        X_np = np.array(X_all, dtype=np.float32)
        y_np = np.array(y_all)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, f"X_{prefix}.npy"), X_np)
        np.save(os.path.join(out_dir, f"y_{prefix}.npy"), y_np)
        print(f"  {prefix} saved: X {X_np.shape}  y {y_np.shape}")


# ----------------------------------------------------------------------
# 실행 영역
# ----------------------------------------------------------------------
if __name__ == "__main__":
    OUT_DIR = "preprocessed_PCA"
    os.makedirs(OUT_DIR, exist_ok=True)
    prep = PCAPreprocessor()

    print("\n[Step 1] Load raw parquets")
    df_src = pd.read_parquet("data/samsung1.parquet")
    df_tgt = pd.read_parquet("data/samsung2_original.parquet")
    print(f"  samsung1          : {df_src.shape}")
    print(f"  samsung2_original : {df_tgt.shape}")

    print("\n[Step 2] Learn per-exercise PCA W (samsung2 -> samsung1)")
    W_dict = learn_pca_W_per_exercise(df_src, df_tgt)
    # W 행렬을 저장해두면 디버깅에 편하다.
    np.savez(os.path.join(OUT_DIR, "pca_W_per_exercise.npz"),
             **{k: v for k, v in W_dict.items()})

    print("\n[Step 3] Source (samsung1) -- PCA 적용 안 함")
    src_sessions = df_src["filename"].unique()
    src_train, src_val = train_test_split(src_sessions, test_size=0.2, random_state=42)
    prep.save(df_src, src_train, "filename", "timestamp", "train", OUT_DIR, W_dict=None)
    prep.save(df_src, src_val,   "filename", "timestamp", "val",   OUT_DIR, W_dict=None)

    print("\n[Step 4] Target (samsung2_original) -- 운동별 PCA W 적용")
    tgt_sessions = df_tgt["csv_filename_l"].unique()
    tgt_train, tgt_val = train_test_split(tgt_sessions, test_size=0.2, random_state=42)
    prep.save(df_tgt, tgt_train, "csv_filename_l", "Index_Time", "target_train", OUT_DIR, W_dict=W_dict)
    prep.save(df_tgt, tgt_val,   "csv_filename_l", "Index_Time", "target_val",   OUT_DIR, W_dict=W_dict)

    print("\n" + "=" * 50)
    print(f"Done -> {OUT_DIR}/")
    print("=" * 50)
