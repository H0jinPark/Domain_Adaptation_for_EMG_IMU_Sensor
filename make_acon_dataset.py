import os
import numpy as np
import torch


PREPROCESSED_DIR = "preprocessed"
SAVE_DIR = "NeurIPS24-ACON/data/samsung_da"


FILE_CANDIDATES = {
    "src_train_x": ["X_train.npy", "X_src_train.npy", "src_X_train.npy", "source_X_train.npy", "X_source.npy"],
    "src_train_y": ["y_train.npy", "y_src_train.npy", "src_y_train.npy", "source_y_train.npy", "y_source.npy"],
    "src_val_x": ["X_src_val.npy", "X_val.npy", "src_X_val.npy", "source_X_val.npy"],
    "src_val_y": ["y_src_val.npy", "y_val.npy", "src_y_val.npy", "source_y_val.npy"],

    "tgt_train_x": ["X_tgt_train.npy", "X_target_train.npy", "tgt_X_train.npy", "target_X_train.npy"],
    "tgt_train_y": ["y_tgt_train.npy", "y_target_train.npy", "tgt_y_train.npy", "target_y_train.npy"],
    "tgt_val_x": ["X_tgt_val.npy", "X_target_val.npy", "tgt_X_val.npy", "target_X_val.npy"],
    "tgt_val_y": ["y_tgt_val.npy", "y_target_val.npy", "tgt_y_val.npy", "target_y_val.npy"],
}


LABEL_MAP = {
    "barbellcurl": 0,
    "barbellrow": 1,
    "benchpress": 2,
    "bte": 3,
    "deadlift": 4,
    "dips": 5,
    "latpulldown": 6,
    "ohp": 7,
    "pullup": 8,
    "pushup": 9,
}


def find_file(key):
    candidates = FILE_CANDIDATES[key]

    for filename in candidates:
        path = os.path.join(PREPROCESSED_DIR, filename)
        if os.path.exists(path):
            return path

    available_files = sorted(os.listdir(PREPROCESSED_DIR))
    raise FileNotFoundError(
        f"{key} 파일을 찾지 못했습니다.\n"
        f"찾은 후보 파일명: {candidates}\n"
        f"현재 preprocessed 폴더 파일 목록: {available_files}"
    )


def load_array(key):
    path = find_file(key)
    arr = np.load(path, allow_pickle=True)
    print(f"{key}: {path} | shape={arr.shape} | dtype={arr.dtype}")
    return arr


def to_channel_first(x):
    x = torch.tensor(x, dtype=torch.float32)

    if x.ndim != 3:
        raise ValueError(f"입력 X는 3차원이어야 합니다. 현재 shape: {x.shape}")

    if x.shape[1] == 5000 and x.shape[2] == 5:
        x = x.permute(0, 2, 1)
    elif x.shape[1] == 5 and x.shape[2] == 5000:
        x = x
    else:
        raise ValueError(
            f"지원하지 않는 X shape입니다: {x.shape}\n"
            f"예상 shape은 [N, 5000, 5] 또는 [N, 5, 5000]입니다."
        )

    return x.contiguous()


def encode_labels(y):
    if y.dtype.kind in {"U", "S", "O"}:
        y = np.array([LABEL_MAP[str(label)] for label in y])

    y = torch.tensor(y, dtype=torch.long)

    if y.ndim != 1:
        y = y.view(-1)

    return y


def validate_pair(x, y, name):
    if x.size(0) != y.size(0):
        raise ValueError(
            f"{name}의 X, y 개수가 다릅니다. X={x.size(0)}, y={y.size(0)}"
        )

    if x.size(1) != 5:
        raise ValueError(f"{name}의 channel 수가 5가 아닙니다. 현재 shape: {x.shape}")

    if x.size(2) != 5000:
        raise ValueError(f"{name}의 sequence length가 5000이 아닙니다. 현재 shape: {x.shape}")

    if y.min().item() < 0 or y.max().item() > 9:
        raise ValueError(f"{name}의 label 범위가 이상합니다. min={y.min().item()}, max={y.max().item()}")


def save_pt(x, y, filename):
    path = os.path.join(SAVE_DIR, filename)
    torch.save(
        {
            "samples": x,
            "labels": y,
        },
        path
    )
    print(f"saved: {path} | samples={tuple(x.shape)} | labels={tuple(y.shape)}")


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    src_train_x = to_channel_first(load_array("src_train_x"))
    src_train_y = encode_labels(load_array("src_train_y"))

    src_val_x = to_channel_first(load_array("src_val_x"))
    src_val_y = encode_labels(load_array("src_val_y"))

    tgt_train_x = to_channel_first(load_array("tgt_train_x"))
    tgt_train_y = encode_labels(load_array("tgt_train_y"))

    tgt_val_x = to_channel_first(load_array("tgt_val_x"))
    tgt_val_y = encode_labels(load_array("tgt_val_y"))

    validate_pair(src_train_x, src_train_y, "source train")
    validate_pair(src_val_x, src_val_y, "source val")
    validate_pair(tgt_train_x, tgt_train_y, "target train")
    validate_pair(tgt_val_x, tgt_val_y, "target val")

    save_pt(src_train_x, src_train_y, "train_0.pt")
    save_pt(src_val_x, src_val_y, "test_0.pt")
    save_pt(tgt_train_x, tgt_train_y, "train_1.pt")
    save_pt(tgt_val_x, tgt_val_y, "test_1.pt")

    print("\nACON dataset 생성 완료")
    print(f"저장 위치: {SAVE_DIR}")
    print("source domain: 0")
    print("target domain: 1")


if __name__ == "__main__":
    main()