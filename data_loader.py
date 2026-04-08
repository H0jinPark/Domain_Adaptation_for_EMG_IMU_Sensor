import numpy as np
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder

def create_sliding_windows(df, window_size=512, step_size=256, mode='emg_only'):
    """
    설정된 mode에 따라 데이터를 선택하고 슬라이딩 윈도우로 쪼개어 PyTorch용 입력을 만듭니다.
    mode: 'emg_only', 'imu_only', 'both'
    """
    X_windows = []
    y_windows = []

    print(f"🔥 [{mode}] 모드로 {len(df)}개의 샘플에 대해 윈도우 생성 중... (Size: {window_size})")
    
    for _, row in df.iterrows():
        # 1. 모드에 따른 데이터 선택
        if mode == 'emg_only':
            selected_data = row['emg_data']  # (time_steps, emg_channels)
        elif mode == 'imu_only':
            selected_data = row['imu_data']  # (time_steps, imu_channels)
        else: # 'both'
            selected_data = np.concatenate([row['emg_data'], row['imu_data']], axis=1)
        
        label = row['label_encoded']
        length = selected_data.shape[0]
        
        # 2. window_size만큼 데이터를 썰기
        for start in range(0, length - window_size + 1, step_size):
            end = start + window_size
            window = selected_data[start:end, :] 
            
            X_windows.append(window)
            y_windows.append(label)

    # 3. Numpy 배열 변환 및 축 변경 (PyTorch Conv1D용)
    # (Samples, Window_size, Channels) -> (Samples, Channels, Window_size)
    X_numpy = np.array(X_windows)
    y_numpy = np.array(y_windows, dtype=np.int64)
    
    X_numpy = np.transpose(X_numpy, (0, 2, 1)) 
    
    return X_numpy, y_numpy

def get_dataloaders(train_df, test_df, window_size=512, step_size=256, batch_size=64, mode='emg_only'):
    """
    최종적으로 모델 학습에 사용할 DataLoader를 반환합니다.
    """
    # 1. 라벨 인코딩 (일관성을 위해 전체 데이터 기준으로 fit)
    le = LabelEncoder()
    all_labels = pd.concat([train_df['label'], test_df['label']])
    le.fit(all_labels)
    
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['label_encoded'] = le.transform(train_df['label'])
    test_df['label_encoded'] = le.transform(test_df['label'])
    
    # 2. 윈도우 생성 (mode 전달)
    X_train_np, y_train_np = create_sliding_windows(train_df, window_size, step_size, mode=mode)
    X_test_np, y_test_np = create_sliding_windows(test_df, window_size, step_size, mode=mode)

    # 3. Tensor 변환 (VRAM 효율을 위해 float32 사용)
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.long)

    # 4. DataLoader 생성
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    print(f"✅ DataLoader 생성 완료! 입력 형태: {X_train.shape}")
    return train_loader, test_loader, le

if __name__ == "__main__":
    # 사용 예시: 전처리에서 저장한 pkl 파일을 불러온다고 가정
    # train_df = pd.read_pickle("train_data.pkl")
    # test_df = pd.read_pickle("test_data.pkl")
    # train_loader, test_loader, le = get_dataloaders(train_df, test_df)
    pass