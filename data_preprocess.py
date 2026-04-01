import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_and_split(first_data_path, second_data_path, test_size=0.2, random_state=42):
    # 1. 데이터 불러오기 및 합치기
    print("📂 데이터 로드 중...")
    df1 = pd.read_csv(first_data_path)
    df2 = pd.read_csv(second_data_path)
    data = pd.concat([df1, df2], ignore_index=True)

    # 2. EMG와 IMU 열(Column) 이름만 쏙쏙 뽑아내기
    emg_cols = [col for col in data.columns if col.startswith('EMG_') and 'RMS' not in col]
    imu_cols = [col for col in data.columns if col.startswith('IMU_')]

    print(f"✅ 추출할 EMG 채널 수: {len(emg_cols)}개")
    print(f"✅ 추출할 IMU 채널 수: {len(imu_cols)}개")

    # 3. 파일(동작) 단위로 데이터 묶기
    filenames, emg_arrays, imu_arrays, labels = [], [], [], []
    
    grouped = data.groupby('filename')
    print("🔥 데이터 묶기(Stacking) 시작...")
    
    for file_name, group in grouped:
        # Numpy 2D 배열로 변환: (시간 길이, 채널 수)
        emg_matrix = group[emg_cols].values
        imu_matrix = group[imu_cols].values
        
        # 해당 파일의 라벨(운동 종류) 가져오기
        label = group['exercise'].iloc[0]
        
        filenames.append(file_name)
        emg_arrays.append(emg_matrix)
        imu_arrays.append(imu_matrix)
        labels.append(label)

    # 4. 데이터프레임 구성
    df_dl_ready = pd.DataFrame({
        'filename': filenames,
        'emg_data': emg_arrays,
        'imu_data': imu_arrays,
        'label': labels
    })

    # 5. 클래스 비율을 유지하며 8:2로 나누기 (Stratified Split)
    # Domain Shift 연구에서는 클래스 불균형이 성능 왜곡을 줄 수 있어 필수입니다.
    print(f"\n✂️ 데이터를 {1-test_size}:{test_size} 비율로 분할 중...")
    
    train_df, test_df = train_test_split(
        df_dl_ready, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df_dl_ready['label']  # 각 동작(exercise) 비율을 일정하게 유지
    )

    print(f"📊 최종 결과:")
    print(f" - Train 세트 크기: {len(train_df)}")
    print(f" - Test 세트 크기: {len(test_df)}")
    
    return train_df, test_df

if __name__ == "__main__":
    # 메인 실행부 (파일 경로 설정)
    train_set, test_set = preprocess_and_split('first_data.csv', 'second_data.csv')
    
    # 나중에 data_loader.py에서 불러오기 쉽게 저장하거나 메모리에 유지
    # 예: train_set.to_pickle("train_ready.pkl")
    print("\n🎉 전처리 및 분할 완료!")