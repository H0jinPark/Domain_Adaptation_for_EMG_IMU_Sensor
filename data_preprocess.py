"""5채널 동기화 전처리 스크립트.

Source(samsung1) / Target(samsung2) parquet 을 읽어 세션 단위로 리샘플링·필터링·
정규화·슬라이딩 윈도우를 적용한 뒤, (N, 5000, 5) 형태의 .npy 로 preprocessed/ 에
저장한다. EMG 2ch + IMU 3ch 를 모두 1000Hz 로 동기화한다.
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
    """세션 단위 전처리 + 슬라이딩 윈도우 추출을 담당하는 전처리기."""

    def __init__(self, target_fs=1000, window_size=5000, stride=500):
        self.target_fs = target_fs
        self.window_size = window_size
        self.stride = stride
        self.feature_cols = ['biceps', 'triceps', 'triceps_X', 'triceps_Y', 'triceps_Z']
        self.emg_cols = ['biceps', 'triceps']
        self.label_col = 'exercise'

    def butter_bandpass(self, lowcut, highcut, fs, order=4):
        """Butterworth 밴드패스 필터 계수 (b, a) 를 생성한다."""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='bandpass', analog=False)
        return b, a

    def apply_emg_filter(self, data):
        """EMG 신호에 20~450Hz 밴드패스 필터를 적용한다."""
        b, a = self.butter_bandpass(lowcut=20.0, highcut=450.0, fs=self.target_fs, order=4)
        return filtfilt(b, a, data)

    def process_single_session(self, session_df, time_col):
        """단일 세션을 전처리하고 슬라이딩 윈도우 (X, y) 리스트를 반환한다."""
        session_df = session_df.sort_values(by=time_col)
        session_df['datetime'] = pd.to_timedelta(session_df[time_col], unit='s')
        session_df = session_df.set_index('datetime')

        # 1. 1000Hz 리샘플링
        numeric_df = session_df[self.feature_cols].resample('1ms').mean().interpolate(method='linear')
        label_df = session_df[[self.label_col]].resample('1ms').ffill()
        resampled_df = pd.concat([numeric_df, label_df], axis=1).dropna()

        if len(resampled_df) < self.window_size:
            return None, None

        # 2. EMG 밴드패스 필터 (EMG 채널만)
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
            # 센서 데이터 (5000, 5)
            X_session.append(data_values[start_idx : start_idx + self.window_size, :])

            # 윈도우 내 최빈 라벨
            win_labels = label_values[start_idx : start_idx + self.window_size]
            u, counts = np.unique(win_labels, return_counts=True)
            y_session.append(u[np.argmax(counts)])

        return X_session, y_session

    def save_processed_data(self, df, session_list, session_col, time_col, prefix):
        """세션 리스트를 전처리해 X_{prefix}.npy / y_{prefix}.npy 로 저장한다."""
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
        np.save(f'preprocessed/X_{prefix}.npy', X_np)
        np.save(f'preprocessed/y_{prefix}.npy', y_np)
        print(f" {prefix} Saved: {X_np.shape}")


# ----------------------------------------------------------------------
# 실행 영역: 세션 단위 8:2 분할 후 split 별 저장
# ----------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs('preprocessed', exist_ok=True)
    preprocessor = DomainAdaptationPreprocessor()

    # 1. Source 데이터(samsung1) 로드 및 세션 분리
    print("\n[Step 1] Processing Source Data (samsung1)...")
    df_src = pd.read_parquet('data/samsung1.parquet')
    src_sessions = df_src['filename'].unique()

    # 세션 단위 8:2 분할 (윈도우 leakage 방지)
    train_sess, val_sess = train_test_split(src_sessions, test_size=0.2, random_state=42)

    preprocessor.save_processed_data(df_src, train_sess, 'filename', 'timestamp', 'train')
    preprocessor.save_processed_data(df_src, val_sess, 'filename', 'timestamp', 'val')

    # 2. Target 데이터(samsung2) 로드 및 세션 분리
    print("\n[Step 2] Processing Target Data (samsung2)...")
    df_tgt = pd.read_parquet('data/samsung2.parquet')
    tgt_sessions = df_tgt['csv_filename_l'].unique()

    # Target 도 세션 단위 8:2 분할
    tgt_train_sess, tgt_val_sess = train_test_split(tgt_sessions, test_size=0.2, random_state=42)

    preprocessor.save_processed_data(df_tgt, tgt_train_sess, 'csv_filename_l', 'Index_Time', 'target_train')
    preprocessor.save_processed_data(df_tgt, tgt_val_sess, 'csv_filename_l', 'Index_Time', 'target_val')

    print("\n" + "=" * 50)
    print("All preprocessing complete. Files are in 'preprocessed/'")
    print("=" * 50)
