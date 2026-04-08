import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess_single_file(file_path, random_state=42):
    """
    1. 데이터를 로드하고 동작(Window) 단위로 묶습니다.
    2. 피험자 구분 없이 전체 데이터를 7:1.5:1.5 비율로 무작위 분할합니다.
    3. Train 기준으로 스케일링을 수행하여 Data Leakage를 방지합니다.
    """
    print(f"📂 {file_path} 로드 중...")
    data = pd.read_parquet(file_path)

    # EMG 및 IMU 컬럼 추출
    emg_cols = [col for col in data.columns if col.startswith('EMG_') and 'RMS' not in col]
    imu_cols = [col for col in data.columns if col.startswith('IMU_')]

    # 1. 시계열 데이터를 파일(윈도우) 단위로 묶기
    filenames, emg_arrays, imu_arrays, labels = [], [], [], []    
    grouped = data.groupby('filename')
    
    print(f"🔥 {len(grouped)}개의 동작 데이터 배열 변환 시작...")
    for file_name, group in grouped:
        emg_matrix = group[emg_cols].values.astype(np.float32)
        imu_matrix = group[imu_cols].values.astype(np.float32)
        label = group['exercise'].iloc[0]
        
        filenames.append(file_name)
        emg_arrays.append(emg_matrix)
        imu_arrays.append(imu_matrix)
        labels.append(label)

    df_dl_ready = pd.DataFrame({
        'filename': filenames,
        'emg_data': emg_arrays,
        'imu_data': imu_arrays, 
        'label': labels
    })

    # 2. 🌟 무작위 7 : 1.5 : 1.5 분할 (Random Split)
    # 먼저 Train(70%)과 나머지(30%)로 나눕니다.
    train_df, temp_df = train_test_split(
        df_dl_ready, 
        test_size=0.3, 
        random_state=random_state,
        stratify=df_dl_ready['label'] # 클래스 비율 유지
    )
    
    # 남은 30%를 반으로 쪼개서 Val(15%)과 Test(15%)로 만듭니다.
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=random_state,
        stratify=temp_df['label']
    )
    
    # 3. 스케일링 (Fit on Train, Transform on All)
    print("⚖️ Train 데이터 기준으로 스케일링 중...")
    scaler_emg = StandardScaler()
    scaler_imu = StandardScaler()

    train_emg_stacked = np.vstack(train_df['emg_data'].values)
    train_imu_stacked = np.vstack(train_df['imu_data'].values)

    scaler_emg.fit(train_emg_stacked)
    scaler_imu.fit(train_imu_stacked)

    def apply_scaling(df, s_emg, s_imu):
        df = df.copy() # SettingWithCopyWarning 방지
        df['emg_data'] = df['emg_data'].apply(lambda x: s_emg.transform(x).astype(np.float32))
        df['imu_data'] = df['imu_data'].apply(lambda x: s_imu.transform(x).astype(np.float32))
        return df

    train_df = apply_scaling(train_df, scaler_emg, scaler_imu)
    val_df = apply_scaling(val_df, scaler_emg, scaler_imu)
    test_df = apply_scaling(test_df, scaler_emg, scaler_imu)

    # 스케일러 저장
    joblib.dump(scaler_emg, 'scaler_emg.pkl')
    joblib.dump(scaler_imu, 'scaler_imu.pkl')

    print(f"📊 최종 결과: Train {len(train_df)} / Val {len(val_df)} / Test {len(test_df)}")
    return train_df, val_df, test_df