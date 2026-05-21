"""5채널 동기화 데이터(.npy) 로더.

preprocessed/ (또는 preprocessed_PCA/) 에 저장된 Source/Target 의 train/val
4분할 numpy 배열을 읽어 PyTorch DataLoader 로 변환한다. baseline 및 단일 백본
DA 모델(Coral/MMD/DANN/CDAN)이 공통으로 사용한다.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder


# ----------------------------------------------------------------------
# 커스텀 Dataset
# ----------------------------------------------------------------------
class HARDataset(Dataset):
    """운동 분류용 Dataset. (N, 5000, 5) 입력을 (N, 5, 5000) 으로 변환해 보관한다."""

    def __init__(self, X, y):
        # Conv1d 입력 형식에 맞춰 (N, T, C) -> (N, C, T) 로 축 교환
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ----------------------------------------------------------------------
# DataLoader 생성 함수
# ----------------------------------------------------------------------
def get_dataloaders(batch_size=64, data_dir='preprocessed'):
    """Source/Target 의 train/val DataLoader, 클래스 수, LabelEncoder 를 반환한다."""
    print(f"Loading pre-split numpy arrays from '{data_dir}/'...")

    # 1. Source 데이터 로드
    X_train = np.load(f'{data_dir}/X_train.npy')
    y_train = np.load(f'{data_dir}/y_train.npy')
    X_val = np.load(f'{data_dir}/X_val.npy')
    y_val = np.load(f'{data_dir}/y_val.npy')

    # 2. Target 데이터 로드 (train/val 분리 파일)
    X_tgt_train = np.load(f'{data_dir}/X_target_train.npy')
    y_tgt_train = np.load(f'{data_dir}/y_target_train.npy')
    X_tgt_val = np.load(f'{data_dir}/X_target_val.npy')
    y_tgt_val = np.load(f'{data_dir}/y_target_val.npy')

    # 3. 라벨 인코딩 (4분할 라벨 전체를 합쳐서 피팅)
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
    #    BatchNorm 안정성을 위해 학습용 로더에는 drop_last=True 적용
    train_loader = DataLoader(HARDataset(X_train, y_train_enc), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(HARDataset(X_val, y_val_enc), batch_size=batch_size, shuffle=False)

    tgt_train_loader = DataLoader(HARDataset(X_tgt_train, y_tgt_train_enc), batch_size=batch_size, shuffle=True, drop_last=True)
    tgt_val_loader = DataLoader(HARDataset(X_tgt_val, y_tgt_val_enc), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, tgt_train_loader, tgt_val_loader, num_classes, le
