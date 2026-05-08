import os
import torch
import numpy as np
from collections import Counter

DATA_DIR = "data/samsung_da"

CLASS_NAMES = [
    "barbellcurl",
    "barbellrow",
    "benchpress",
    "bte",
    "deadlift",
    "dips",
    "latpulldown",
    "ohp",
    "pullup",
    "pushup"
]

FILES = [
    "train_0.pt",
    "test_0.pt",
    "train_1.pt",
    "test_1.pt"
]


def summarize_tensor_x(x, name):
    print(f"\n[{name}] samples")
    print("shape:", tuple(x.shape))
    print("dtype:", x.dtype)
    print("min:", float(torch.nan_to_num(x).min()))
    print("max:", float(torch.nan_to_num(x).max()))
    print("mean:", float(torch.nan_to_num(x).mean()))
    print("std:", float(torch.nan_to_num(x).std()))
    print("has_nan:", bool(torch.isnan(x).any()))
    print("has_inf:", bool(torch.isinf(x).any()))

    if x.ndim == 3:
        ch_mean = x.mean(dim=(0, 2))
        ch_std = x.std(dim=(0, 2))
        print("channel mean:", ch_mean.tolist())
        print("channel std :", ch_std.tolist())


def summarize_y(y, name):
    print(f"\n[{name}] labels")
    print("shape:", tuple(y.shape))
    print("dtype:", y.dtype)
    print("min:", int(y.min()))
    print("max:", int(y.max()))
    print("unique:", sorted(y.unique().tolist()))

    counts = Counter(y.cpu().numpy().tolist())
    for k in sorted(counts.keys()):
        cname = CLASS_NAMES[k] if 0 <= k < len(CLASS_NAMES) else "UNKNOWN"
        print(f"{k:2d} {cname:12s}: {counts[k]}")


def main():
    for filename in FILES:
        path = os.path.join(DATA_DIR, filename)
        print("\n" + "=" * 80)
        print(path)

        data = torch.load(path, map_location="cpu")

        print("keys:", data.keys())

        x = data["samples"]
        y = data["labels"]

        summarize_tensor_x(x, filename)
        summarize_y(y, filename)

        if x.ndim != 3:
            print("WARNING: samples ndim is not 3")

        if x.ndim == 3:
            if x.shape[1] == 5:
                print("shape interpretation: [N, C, L] OK")
            elif x.shape[2] == 5:
                print("shape interpretation: [N, L, C], dataloader permute 필요")
            else:
                print("WARNING: channel dimension not obvious")

        if y.min() < 0 or y.max() > 9:
            print("WARNING: label range abnormal")


if __name__ == "__main__":
    main()