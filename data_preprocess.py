import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_single_file(file_path, test_size=0.2, random_state=42):
    """
    Parquet 파일을 로드하고, EMG/IMU 데이터를 구분하여 8:2로 분할합니다.
    """
    # 1. 데이터 로드
    print(f"📂 {file_path} 로드 중...")
    data = pd.read_parquet(file_path)

    # 2. 컬럼 정의 (이 부분은 호진님의 데이터 컬럼명에 맞춰 자동 추출됩니다)
    # RMS를 제외한 순수 EMG 신호만 선택
    emg_cols = [col for col in data.columns if col.startswith('EMG_') and 'RMS' not in col]
    imu_cols = [col for col in data.columns if col.startswith('IMU_')]

    print(f"✅ 추출 완료 - EMG: {len(emg_cols)}채널, IMU: {len(imu_cols)}채널")

    # 3. 그룹화 및 배열 변환
    filenames, emg_arrays, imu_arrays, labels = [], [], [], []
    
    grouped = data.groupby('filename')
    print(f"🔥 {len(grouped)}개의 동작 데이터 변환 시작...")
    
    for file_name, group in grouped:
        # float32로 변환하여 메모리 효율성 증대 (RTX 3070 학습 속도 향상)
        emg_matrix = group[emg_cols].values.astype(np.float32)
        imu_matrix = group[imu_cols].values.astype(np.float32)
        
        label = group['exercise'].iloc[0]
        
        filenames.append(file_name)
        emg_arrays.append(emg_matrix)
        imu_arrays.append(imu_matrix)
        labels.append(label)

    # 4. 학습용 데이터프레임 구성
    df_dl_ready = pd.DataFrame({
        'filename': filenames,
        'emg_data': emg_arrays,
        'imu_data': imu_arrays, # 일단 가지고는 있습니다!
        'label': labels
    })

    # 5. Stratified Split (클래스 비율 유지)
    train_df, test_df = train_test_split(
        df_dl_ready, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df_dl_ready['label']
    )

    print(f"📊 최종 결과: Train {len(train_df)} / Test {len(test_df)}")
    
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