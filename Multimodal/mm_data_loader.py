"""멀티모달(EMG/IMU 분리) 데이터 로더.

preprocessed_MM/ 에 저장된 Source/Target 의 train/val 4분할 numpy 배열을 읽어
EMG/IMU 를 분리한 채로 PyTorch DataLoader 로 변환한다. Multimodal/ 폴더의
MMD/Coral/DANN/CDAN 학습 스크립트가 공통으로 사용한다.

입력 규약 (preprocessed_MM/)
  X_emg_{prefix}.npy : (N, 2, 5000)  -- EMG 1000Hz, 5초
  X_imu_{prefix}.npy : (N, 3,  500)  -- IMU  100Hz, 5초
  y_{prefix}.npy     : (N,)
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 기본값 = preprocessed_MM_raw (samsung1 → samsung2 raw 원본, R=eye, 정렬 없음).
# 축 정렬 방법별 데이터로 돌리려면 환경변수로 폴더 지정:
#   MM_DATA_DIR=preprocessed_MM_pca | preprocessed_MM_gravity |
#               preprocessed_MM_permutation | preprocessed_MM_kabsch
DATA_DIR = os.path.join(PROJECT_ROOT,
                        os.environ.get("MM_DATA_DIR", "preprocessed_MM_raw"))


# ----------------------------------------------------------------------
# 커스텀 Dataset
# ----------------------------------------------------------------------
class MMDataset(Dataset):
    """EMG/IMU 분리 텐서와 라벨을 담는 멀티모달 Dataset."""

    def __init__(self, emg, imu, y):
        self.emg = torch.tensor(emg, dtype=torch.float32)  # (N, 2, 5000)
        self.imu = torch.tensor(imu, dtype=torch.float32)  # (N, 3,  500)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.emg[idx], self.imu[idx], self.y[idx]


def _load_split(prefix):
    """preprocessed_MM/ 에서 한 split 의 EMG/IMU/label 원본 배열을 로드한다."""
    emg = np.load(os.path.join(DATA_DIR, f"X_emg_{prefix}.npy"))
    imu = np.load(os.path.join(DATA_DIR, f"X_imu_{prefix}.npy"))
    y = np.load(os.path.join(DATA_DIR, f"y_{prefix}.npy"), allow_pickle=True)
    return emg, imu, y


# ----------------------------------------------------------------------
# DataLoader 생성 함수
# ----------------------------------------------------------------------
def get_mm_dataloaders(batch_size=64):
    """Source/Target 의 train/val DataLoader, 클래스 수, LabelEncoder 를 반환한다.

    반환 순서는 5채널 get_dataloaders 와 동일하다:
      (train, val, tgt_train, tgt_val, num_classes, le)
    """
    print(f"Loading multimodal numpy arrays from '{DATA_DIR}'...")

    emg_tr, imu_tr, y_tr = _load_split("train")
    emg_val, imu_val, y_val = _load_split("val")
    emg_tt, imu_tt, y_tt = _load_split("target_train")
    emg_tv, imu_tv, y_tv = _load_split("target_val")

    # 라벨 인코딩 (4분할 라벨 전체를 합쳐서 피팅)
    le = LabelEncoder()
    le.fit(np.concatenate([y_tr, y_val, y_tt, y_tv]))
    num_classes = len(le.classes_)

    print(f"Classes ({num_classes}): {le.classes_}")
    print(f"Loaded - Source Train: {len(y_tr)}, Source Val: {len(y_val)}")
    print(f"Loaded - Target Train: {len(y_tt)}, Target Val: {len(y_tv)}")

    def make_loader(emg, imu, y, shuffle, drop_last):
        # BatchNorm 안정성을 위해 학습용 로더에는 drop_last=True 적용
        ds = MMDataset(emg, imu, le.transform(y))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    train_loader = make_loader(emg_tr, imu_tr, y_tr, True, True)
    val_loader = make_loader(emg_val, imu_val, y_val, False, False)
    tgt_train_loader = make_loader(emg_tt, imu_tt, y_tt, True, True)
    tgt_val_loader = make_loader(emg_tv, imu_tv, y_tv, False, False)

    return train_loader, val_loader, tgt_train_loader, tgt_val_loader, num_classes, le


if __name__ == "__main__":
    tr, val, tt, tv, nc, le = get_mm_dataloaders(batch_size=64)
    emg, imu, y = next(iter(tr))
    print(f"\nbatch EMG: {tuple(emg.shape)}  IMU: {tuple(imu.shape)}  y: {tuple(y.shape)}")
    print(f"num_classes: {nc}")
