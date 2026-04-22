import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

# =====================================================================
# 1. 커스텀 Dataset 클래스 정의 (기존 유지)
# =====================================================================
class HARDataset(Dataset):
    def __init__(self, X, y):
        """ (N, 5000, 5) -> (N, 5, 5000)으로 변환하여 저장 """
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# =====================================================================
# 2. DataLoader 생성 함수 (4개의 분리된 파일 로드로 수정)
# =====================================================================
def get_dataloaders(batch_size=64):
    print("Loading pre-split numpy arrays from 'preprocessed/'...")
    
    # 1. 소스 데이터 로드
    X_train = np.load('preprocessed/X_train.npy')
    y_train = np.load('preprocessed/y_train.npy')
    X_val = np.load('preprocessed/X_val.npy')
    y_val = np.load('preprocessed/y_val.npy')
    
    # 2. 타겟 데이터 로드 (Train/Val 분리 파일 로드)
    X_tgt_train = np.load('preprocessed/X_target_train.npy')
    y_tgt_train = np.load('preprocessed/y_target_train.npy')
    X_tgt_val = np.load('preprocessed/X_target_val.npy')
    y_tgt_val = np.load('preprocessed/y_target_val.npy')
    
    # 3. 라벨 인코딩 (10가지 운동 유형 전체 라벨을 합쳐서 피팅)
    le = LabelEncoder()
    all_labels = np.concatenate([y_train, y_val, y_tgt_train, y_tgt_val])
    le.fit(all_labels)
    
    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)
    y_tgt_train_enc = le.transform(y_tgt_train)
    y_tgt_val_enc = le.transform(y_tgt_val)
    
    num_classes = len(le.classes_)
    print(f"Classes ({num_classes}): {le.classes_}")
    print(f"Loaded - Source Train: {len(X_train)}, Source Val: {len(X_val)}")
    print(f"Loaded - Target Train: {len(X_tgt_train)}, Target Val: {len(X_tgt_val)}")
    
    # 4. Dataset 및 DataLoader 생성
    # 배치 정규화(BatchNorm) 안정성을 위해 학습용 로더에는 drop_last=True 적용 권장
    train_loader = DataLoader(HARDataset(X_train, y_train_enc), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(HARDataset(X_val, y_val_enc), batch_size=batch_size, shuffle=False)
    
    tgt_train_loader = DataLoader(HARDataset(X_tgt_train, y_tgt_train_enc), batch_size=batch_size, shuffle=True, drop_last=True)
    tgt_val_loader = DataLoader(HARDataset(X_tgt_val, y_tgt_val_enc), batch_size=batch_size, shuffle=False)
    
    # 5. Baseline 및 DA 모델 학습에 필요한 6개의 값 반환
    return train_loader, val_loader, tgt_train_loader, tgt_val_loader, num_classes, le