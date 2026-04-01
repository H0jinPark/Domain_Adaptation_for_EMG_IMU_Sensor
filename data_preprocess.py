import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_single_file(file_path, test_size=0.2, random_state=42):
    """
    단일 Parquet 파일을 불러와 EMG/IMU 데이터를 묶고 8:2로 분할합니다.
    """
    # 1. Parquet 데이터 로드
    print(f"📂 {file_path} 로드 중 (Parquet)...")
    data = pd.read_parquet(file_path)

    # 2. EMG와 IMU 열(Column) 이름 추출
    # (RMS 제외 순수 신호만 선택)
    emg_cols = [col for col in data.columns if col.startswith('EMG_') and 'RMS' not in col]
    imu_cols = [col for col in data.columns if col.startswith('IMU_')]

    print(f"✅ 추출된 EMG 채널: {len(emg_cols)}개")
    print(f"✅ 추출된 IMU 채널: {len(imu_cols)}개")

    # 3. 파일(동작) 단위로 데이터 묶기 (Stacking)
    filenames, emg_arrays, imu_arrays, labels = [], [], [], []
    
    # 'filename' 기준으로 그룹화하여 시계열 윈도우 생성
    grouped = data.groupby('filename')
    print(f"🔥 {len(grouped)}개의 동작 파일 묶기 시작...")
    
    for file_name, group in grouped:
        # 각 동작을 (시간, 채널) 형태의 Numpy 배열로 변환
        emg_matrix = group[emg_cols].values
        imu_matrix = group[imu_cols].values
        
        # 라벨(exercise) 추출
        label = group['exercise'].iloc[0]
        
        filenames.append(file_name)
        emg_arrays.append(emg_matrix)
        imu_arrays.append(imu_matrix)
        labels.append(label)

    # 4. 딥러닝용 데이터프레임 구성
    df_dl_ready = pd.DataFrame({
        'filename': filenames,
        'emg_data': emg_arrays,
        'imu_data': imu_arrays,
        'label': labels
    })

    # 5. Stratified Split (8:2)
    # 클래스(label) 비율을 유지하며 나눕니다.
    print(f"✂️ 데이터를 {1-test_size}:{test_size} 비율로 분할 중...")
    
    train_df, test_df = train_test_split(
        df_dl_ready, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df_dl_ready['label']
    )

    print(f"📊 분할 결과:")
    print(f" - Train 세트: {len(train_df)} samples")
    print(f" - Test 세트: {len(test_df)} samples")
    
    return train_df, test_df

if __name__ == "__main__":
    # 방금 성공적으로 변환한 parquet 파일을 넣으시면 됩니다.
    target_file = 'data/sensor_data.parquet'
    
    try:
        train_set, test_set = preprocess_single_file(target_file)
        
        # 나중에 활용하기 위해 pickle로 저장해두면 편리합니다. (Numpy 배열 보존)
        # train_set.to_pickle("train_data.pkl")
        # test_set.to_pickle("test_data.pkl")
        
        print(train_set.head())
        print(test_set.head())
        
    except FileNotFoundError:
        print(f"❌ '{target_file}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")