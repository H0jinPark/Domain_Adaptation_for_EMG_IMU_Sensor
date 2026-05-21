"""멀티모달(EMG/IMU 분리) 전처리 스크립트.

data_preprocess.py 와 달리 EMG 와 IMU 를 각자의 샘플링 레이트로 분리 저장한다.
멀티모달 모델(DualEncoder, DANN-MM, CDAN-MM 등)이 사용한다.

채널 규약
  EMG: [biceps, triceps]                 -> 1000Hz, window=5000 (5초)
  IMU: [triceps_X, triceps_Y, triceps_Z] ->  100Hz, window=500  (5초)

저장 파일 (preprocessed_MM/)
  X_emg_{prefix}.npy : (N, 2, 5000)
  X_imu_{prefix}.npy : (N, 3,  500)
  y_{prefix}.npy     : (N,)
"""
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')

EMG_FS     = 1000   # EMG 목표 샘플링 레이트
IMU_FS     = 100    # IMU 목표 샘플링 레이트 (다운샘플 배수: EMG_FS // IMU_FS = 10)
EMG_WINDOW = 5000   # EMG 윈도우 크기 (5초)
IMU_WINDOW = 500    # IMU 윈도우 크기 (5초, 동일 절대시간)
EMG_STRIDE = 500    # EMG 기준 스트라이드 (0.5초)


class MultiModalPreprocessor:
    """EMG/IMU 를 분리 저장하는 멀티모달 전처리기."""

    def __init__(self):
        self.emg_cols  = ['biceps', 'triceps']
        self.imu_cols  = ['triceps_X', 'triceps_Y', 'triceps_Z']
        self.label_col = 'exercise'
        self.imu_step  = EMG_FS // IMU_FS  # 1000Hz -> 100Hz: 10샘플마다 1개

    def _bandpass_emg(self, data):
        """EMG 신호에 20~450Hz 밴드패스 필터를 적용한다."""
        nyq = 0.5 * EMG_FS
        b, a = butter(4, [20.0 / nyq, 450.0 / nyq], btype='bandpass', analog=False)
        return filtfilt(b, a, data)

    def process_single_session(self, session_df, time_col):
        """단일 세션을 전처리하고 (X_emg, X_imu, y) 윈도우 리스트를 반환한다."""
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

        # 3. 모달리티별 독립 Z-score 정규화
        emg_scaler = StandardScaler()
        imu_scaler = StandardScaler()
        df[self.emg_cols] = emg_scaler.fit_transform(df[self.emg_cols])
        df[self.imu_cols] = imu_scaler.fit_transform(df[self.imu_cols])

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

    def save_processed_data(self, df, session_list, session_col, time_col, prefix):
        """세션 리스트를 전처리해 EMG/IMU/label 을 preprocessed_MM/ 에 저장한다."""
        all_emg, all_imu, all_y = [], [], []
        total = len(session_list)

        for i, sess in enumerate(session_list):
            sess_df = df[df[session_col] == sess]
            X_emg, X_imu, y = self.process_single_session(sess_df, time_col)
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

        os.makedirs('preprocessed_MM', exist_ok=True)
        np.save(f'preprocessed_MM/X_emg_{prefix}.npy', emg_np)
        np.save(f'preprocessed_MM/X_imu_{prefix}.npy', imu_np)
        np.save(f'preprocessed_MM/y_{prefix}.npy',     y_np)

        print(f"  {prefix} -> EMG {emg_np.shape}  IMU {imu_np.shape}  y {y_np.shape}")


# ----------------------------------------------------------------------
# 실행 영역: 세션 단위 8:2 분할 후 split 별 저장
# ----------------------------------------------------------------------
if __name__ == "__main__":
    prep = MultiModalPreprocessor()

    print("\n[Step 1] Source Data (samsung1)...")
    df_src = pd.read_parquet('data/samsung1.parquet')
    train_sess, val_sess = train_test_split(
        df_src['filename'].unique(), test_size=0.2, random_state=42)
    prep.save_processed_data(df_src, train_sess, 'filename', 'timestamp', 'train')
    prep.save_processed_data(df_src, val_sess,   'filename', 'timestamp', 'val')

    print("\n[Step 2] Target Data (samsung2)...")
    df_tgt = pd.read_parquet('data/samsung2.parquet')
    tgt_train_sess, tgt_val_sess = train_test_split(
        df_tgt['csv_filename_l'].unique(), test_size=0.2, random_state=42)
    prep.save_processed_data(df_tgt, tgt_train_sess, 'csv_filename_l', 'Index_Time', 'target_train')
    prep.save_processed_data(df_tgt, tgt_val_sess,   'csv_filename_l', 'Index_Time', 'target_val')

    print("\n" + "=" * 50)
    print("완료. preprocessed_MM/ 에 저장됨")
    print("  EMG: (N, 2, 5000)  -- 1000Hz, 5초")
    print("  IMU: (N, 3,  500)  --  100Hz, 5초")
    print("=" * 50)
