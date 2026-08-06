"""멀티모달(EMG/IMU 분리) 데이터 로더.

preprocessed_MM/ 에 저장된 Source/Target 의 train/val/test 배열을 읽어
EMG/IMU 를 분리한 채로 PyTorch DataLoader 로 변환한다. Multimodal/ 폴더의
MMD/Coral/DANN/CDAN 학습 스크립트가 공통으로 사용한다.

  get_mm_dataloaders  : train/val 만 (학습·model selection 용)
  get_mm_test_loaders : test 만   (학습 종료 후 최종 보고 1회 평가 전용)

두 함수를 나눠 둔 이유는 test 가 학습 루프로 새어들 수 없게 하기 위함이다.

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
def get_mm_dataloaders(batch_size=64, shuffle_seed=None):
    """Source/Target 의 train/val DataLoader, 클래스 수, LabelEncoder 를 반환한다.

    반환 순서는 5채널 get_dataloaders 와 동일하다:
      (train, val, tgt_train, tgt_val, num_classes, le)

    shuffle_seed=None(기본) 이면 학습 로더 셔플이 전역 RNG 를 쓰던 기존 동작 그대로.
    정수를 주면 학습 로더 셔플을 그 seed 의 별도 generator 로 고정 → weight init 시드와
    **셔플 시드를 분리**할 수 있다(seed 영향 분해 실험용). val 로더는 shuffle=False 라 무관.
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

    def make_loader(emg, imu, y, shuffle, drop_last, gen=None):
        # BatchNorm 안정성을 위해 학습용 로더에는 drop_last=True 적용
        ds = MMDataset(emg, imu, le.transform(y))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                          generator=gen)

    # 학습 로더 셔플용 generator (shuffle_seed 지정 시에만). source/target 는 서로 다른
    # 스트림을 쓰도록 seed 를 어긋나게 준다.
    gen_src = gen_tgt = None
    if shuffle_seed is not None:
        gen_src = torch.Generator().manual_seed(int(shuffle_seed))
        gen_tgt = torch.Generator().manual_seed(int(shuffle_seed) + 10007)

    train_loader = make_loader(emg_tr, imu_tr, y_tr, True, True, gen_src)
    val_loader = make_loader(emg_val, imu_val, y_val, False, False)
    tgt_train_loader = make_loader(emg_tt, imu_tt, y_tt, True, True, gen_tgt)
    tgt_val_loader = make_loader(emg_tv, imu_tv, y_tv, False, False)

    return train_loader, val_loader, tgt_train_loader, tgt_val_loader, num_classes, le


def get_mm_test_loaders(le, batch_size=64):
    """Source/Target 의 **test** DataLoader 를 반환한다: (test_loader, tgt_test_loader).

    test 를 get_mm_dataloaders 에서 분리한 것은 의도적이다. 학습 스크립트는 학습이
    끝난 뒤 best checkpoint 를 복원한 다음에만 이 함수를 부르므로, 학습 루프·model
    selection 이 test 를 볼 수 있는 경로가 아예 없다.

    le 는 get_mm_dataloaders 가 반환한 LabelEncoder 를 그대로 넘긴다. 같은 인코딩을
    써야 train/val 과 클래스 인덱스가 일치한다.
    """
    try:
        emg_te, imu_te, y_te = _load_split("test")
        emg_tte, imu_tte, y_tte = _load_split("target_test")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"test split 이 '{DATA_DIR}' 에 없습니다 ({e.filename}). "
            "7:1.5:1.5 분할 이전에 만든 폴더입니다 — data_preprocess_MM.py 로 재전처리하세요."
        ) from None

    # le 가 모르는 클래스가 test 에 있으면 조용히 틀린 인덱스가 되지 않도록 먼저 막는다.
    unseen = set(np.concatenate([y_te, y_tte])) - set(le.classes_)
    if unseen:
        raise ValueError(f"test 에 train/val 에 없던 클래스가 있습니다: {sorted(unseen)}")

    print(f"Loaded - Source Test: {len(y_te)}, Target Test: {len(y_tte)}")

    def make_loader(emg, imu, y):
        return DataLoader(MMDataset(emg, imu, le.transform(y)),
                          batch_size=batch_size, shuffle=False, drop_last=False)

    return make_loader(emg_te, imu_te, y_te), make_loader(emg_tte, imu_tte, y_tte)


if __name__ == "__main__":
    tr, val, tt, tv, nc, le = get_mm_dataloaders(batch_size=64)
    emg, imu, y = next(iter(tr))
    print(f"\nbatch EMG: {tuple(emg.shape)}  IMU: {tuple(imu.shape)}  y: {tuple(y.shape)}")
    print(f"num_classes: {nc}")
