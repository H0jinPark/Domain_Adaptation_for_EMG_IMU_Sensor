"""5채널 동기화 전처리 — samsung2_original (raw, unaligned) 버전.

`data_preprocess.py` 와 모든 로직이 동일하나, target 으로
`samsung2_original.parquet` (YXZ 재라벨링 *안 된* 원본) 을 사용한다.
출력은 `preprocessed_original/` 로 저장.

Source(samsung1) 도 동일 로직으로 재처리해서 같은 디렉토리에 저장 — 다른 노트북에서
한 디렉토리만 로드하면 되도록 일관성 유지.

차이 요약 (vs `data_preprocess.py`):
  - target parquet:  `data/samsung2.parquet` → `data/samsung2_original.parquet`
  - 출력 디렉토리:    `preprocessed/`         → `preprocessed_original/`
나머지 (resample 1000Hz, EMG 20-450Hz bandpass, 세션 z-score, 5000 윈도우, 세션 8:2 split) 모두 동일.
"""
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')


class DomainAdaptationPreprocessor:
    """세션 단위 전처리 + 슬라이딩 윈도우 추출 (data_preprocess.py 와 로직 동일, out_dir 만 인자화)."""

    def __init__(self, target_fs=1000, window_size=5000, stride=500,
                 out_dir='preprocessed_original'):
        self.target_fs = target_fs
        self.window_size = window_size
        self.stride = stride
        self.out_dir = out_dir
        self.feature_cols = ['biceps', 'triceps', 'triceps_X', 'triceps_Y', 'triceps_Z']
        self.emg_cols = ['biceps', 'triceps']
        self.label_col = 'exercise'
        os.makedirs(self.out_dir, exist_ok=True)

    def butter_bandpass(self, lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='bandpass', analog=False)
        return b, a

    def apply_emg_filter(self, data):
        b, a = self.butter_bandpass(lowcut=20.0, highcut=450.0, fs=self.target_fs, order=4)
        return filtfilt(b, a, data)

    def process_single_session(self, session_df, time_col):
        session_df = session_df.sort_values(by=time_col)
        session_df['datetime'] = pd.to_timedelta(session_df[time_col], unit='s')
        session_df = session_df.set_index('datetime')

        # 1. 1000Hz 리샘플링
        numeric_df = session_df[self.feature_cols].resample('1ms').mean().interpolate(method='linear')
        label_df = session_df[[self.label_col]].resample('1ms').ffill()
        resampled_df = pd.concat([numeric_df, label_df], axis=1).dropna()

        if len(resampled_df) < self.window_size:
            return None, None

        # 2. EMG 밴드패스 필터
        for col in self.emg_cols:
            resampled_df[col] = self.apply_emg_filter(resampled_df[col].values)

        # 3. 세션 단위 Z-score 정규화
        scaler = StandardScaler()
        resampled_df[self.feature_cols] = scaler.fit_transform(resampled_df[self.feature_cols])

        # 4. 슬라이딩 윈도우 추출
        data_values = resampled_df[self.feature_cols].values
        label_values = resampled_df[self.label_col].values

        X_session, y_session = [], []
        for start_idx in range(0, len(resampled_df) - self.window_size + 1, self.stride):
            X_session.append(data_values[start_idx : start_idx + self.window_size, :])
            win_labels = label_values[start_idx : start_idx + self.window_size]
            u, counts = np.unique(win_labels, return_counts=True)
            y_session.append(u[np.argmax(counts)])

        return X_session, y_session

    def save_processed_data(self, df, session_list, session_col, time_col, prefix):
        X_all, y_all = [], []
        total = len(session_list)

        for i, sess in enumerate(session_list):
            sess_df = df[df[session_col] == sess]
            X_s, y_s = self.process_single_session(sess_df, time_col)
            if X_s:
                X_all.extend(X_s)
                y_all.extend(y_s)

            if (i + 1) % max(1, total // 5) == 0:
                print(f" > {prefix} Progress: {i+1}/{total} sessions done...")

        X_np, y_np = np.array(X_all), np.array(y_all)
        np.save(f'{self.out_dir}/X_{prefix}.npy', X_np)
        np.save(f'{self.out_dir}/y_{prefix}.npy', y_np)
        print(f" {prefix} Saved to {self.out_dir}/: {X_np.shape}")


# ----------------------------------------------------------------------
# 실행 영역
# ----------------------------------------------------------------------
if __name__ == "__main__":
    OUT_DIR = 'preprocessed_original'
    preprocessor = DomainAdaptationPreprocessor(out_dir=OUT_DIR)

    # 1. Source (samsung1) — data_preprocess.py 와 동일하지만 out_dir 만 다름
    print("\n[Step 1] Processing Source Data (samsung1)...")
    df_src = pd.read_parquet('data/samsung1.parquet')
    src_sessions = df_src['filename'].unique()

    train_sess, val_sess = train_test_split(src_sessions, test_size=0.2, random_state=42)
    preprocessor.save_processed_data(df_src, train_sess, 'filename', 'timestamp', 'train')
    preprocessor.save_processed_data(df_src, val_sess, 'filename', 'timestamp', 'val')

    # 2. Target — samsung2_original.parquet (YXZ 재라벨링 안 된 원본)
    print("\n[Step 2] Processing Target Data (samsung2_original)...")
    df_tgt = pd.read_parquet('data/samsung2_original.parquet')
    tgt_sessions = df_tgt['csv_filename_l'].unique()

    tgt_train_sess, tgt_val_sess = train_test_split(tgt_sessions, test_size=0.2, random_state=42)
    preprocessor.save_processed_data(df_tgt, tgt_train_sess, 'csv_filename_l', 'Index_Time', 'target_train')
    preprocessor.save_processed_data(df_tgt, tgt_val_sess, 'csv_filename_l', 'Index_Time', 'target_val')

    print("\n" + "=" * 50)
    print(f"All preprocessing complete. Files are in '{OUT_DIR}/'")
    print("=" * 50)
