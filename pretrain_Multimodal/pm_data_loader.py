"""pretrain_Multimodal 전용 데이터 로더.

Phase 2 (DA 학습): get_pm_dataloaders()
  mm_data_loader.get_mm_dataloaders()와 동일한 인터페이스.

Phase 1 (SSL 사전학습): get_ssl_dataloaders()
  Source train + Target train 데이터를 합쳐 라벨 없이 로드한다.
  SSL은 비지도 학습이므로 source/target 구분 없이 EMG/IMU만 반환한다.
"""
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "preprocessed_MM")

sys.path.append(PROJECT_ROOT)


# -----------------------------------------------------------------------
# Dataset 정의
# -----------------------------------------------------------------------
class MMDataset(Dataset):
    """EMG/IMU 분리 텐서 + 라벨 (Phase 2용)."""

    def __init__(self, emg, imu, y):
        self.emg = torch.tensor(emg, dtype=torch.float32)
        self.imu = torch.tensor(imu, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.emg[idx], self.imu[idx], self.y[idx]


class SSLDataset(Dataset):
    """EMG/IMU 분리 텐서만 반환 (Phase 1 SSL용, 라벨 없음)."""

    def __init__(self, emg, imu):
        self.emg = torch.tensor(emg, dtype=torch.float32)
        self.imu = torch.tensor(imu, dtype=torch.float32)

    def __len__(self):
        return len(self.emg)

    def __getitem__(self, idx):
        return self.emg[idx], self.imu[idx]


# -----------------------------------------------------------------------
# 내부 유틸
# -----------------------------------------------------------------------
def _load_split(prefix):
    emg = np.load(os.path.join(DATA_DIR, f"X_emg_{prefix}.npy"))
    imu = np.load(os.path.join(DATA_DIR, f"X_imu_{prefix}.npy"))
    y = np.load(os.path.join(DATA_DIR, f"y_{prefix}.npy"), allow_pickle=True)
    return emg, imu, y


# -----------------------------------------------------------------------
# Phase 2: DA 학습용 DataLoader
# -----------------------------------------------------------------------
def get_pm_dataloaders(batch_size=64):
    """mm_data_loader.get_mm_dataloaders()와 동일한 인터페이스.

    반환: (train_loader, val_loader, tgt_train_loader, tgt_val_loader, num_classes, le)
    """
    print(f"[Phase2] Loading multimodal arrays from '{DATA_DIR}'...")

    emg_tr, imu_tr, y_tr = _load_split("train")
    emg_val, imu_val, y_val = _load_split("val")
    emg_tt, imu_tt, y_tt = _load_split("target_train")
    emg_tv, imu_tv, y_tv = _load_split("target_val")

    le = LabelEncoder()
    le.fit(np.concatenate([y_tr, y_val, y_tt, y_tv]))
    num_classes = len(le.classes_)

    print(f"Classes ({num_classes}): {le.classes_}")
    print(f"Source Train: {len(y_tr)}, Source Val: {len(y_val)}")
    print(f"Target Train: {len(y_tt)}, Target Val: {len(y_tv)}")

    def make_loader(emg, imu, y, shuffle, drop_last):
        ds = MMDataset(emg, imu, le.transform(y))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    return (
        make_loader(emg_tr, imu_tr, y_tr, True, True),
        make_loader(emg_val, imu_val, y_val, False, False),
        make_loader(emg_tt, imu_tt, y_tt, True, True),
        make_loader(emg_tv, imu_tv, y_tv, False, False),
        num_classes,
        le,
    )


# -----------------------------------------------------------------------
# Phase 1: SSL 사전학습용 DataLoader
# -----------------------------------------------------------------------
def get_ssl_dataloaders(batch_size=256):
    """Source train + Target train을 합쳐 라벨 없이 반환한다.

    SSL은 비지도 학습이므로 두 도메인을 구분하지 않고 모든 데이터를 활용한다.
    반환: (ssl_loader, n_samples)
    """
    print(f"[Phase1] Loading SSL arrays from '{DATA_DIR}'...")

    emg_tr, imu_tr, _ = _load_split("train")
    emg_tt, imu_tt, _ = _load_split("target_train")

    emg_all = np.concatenate([emg_tr, emg_tt], axis=0)
    imu_all = np.concatenate([imu_tr, imu_tt], axis=0)

    ds = SSLDataset(emg_all, imu_all)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    print(f"SSL dataset size: {len(ds)} (source {len(emg_tr)} + target {len(emg_tt)})")
    return loader, len(ds)


def get_hybrid_ssl_dataloaders(batch_size=64):
    """Phase 1 Hybrid용: Source는 라벨 있음, Target은 라벨 없음.

    반환: (src_loader, tgt_loader, num_classes, le)
    """
    print(f"[Phase1-Hybrid] Loading arrays from '{DATA_DIR}'...")

    emg_tr, imu_tr, y_tr = _load_split("train")
    emg_tt, imu_tt, _ = _load_split("target_train")

    le = LabelEncoder()
    le.fit(y_tr)
    num_classes = len(le.classes_)

    print(f"Classes ({num_classes}): {le.classes_}")
    print(f"Source (labeled): {len(y_tr)}  |  Target (unlabeled): {len(emg_tt)}")

    src_ds = MMDataset(emg_tr, imu_tr, le.transform(y_tr))
    tgt_ds = SSLDataset(emg_tt, imu_tt)

    src_loader = DataLoader(src_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    return src_loader, tgt_loader, num_classes, le


if __name__ == "__main__":
    ssl_loader, n = get_ssl_dataloaders(batch_size=64)
    emg, imu = next(iter(ssl_loader))
    print(f"SSL batch  EMG: {tuple(emg.shape)}  IMU: {tuple(imu.shape)}")

    tr, val, tt, tv, nc, le = get_pm_dataloaders(batch_size=64)
    emg, imu, y = next(iter(tr))
    print(f"Phase2 batch  EMG: {tuple(emg.shape)}  IMU: {tuple(imu.shape)}  y: {tuple(y.shape)}")
    print(f"num_classes: {nc}")
