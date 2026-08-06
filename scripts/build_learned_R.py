"""학습된 R 들을 SO(3) 평균 한 개로 모아 `results/R_matrices/R_learned.npy` 로 저장한다.

왜 하나로 합칠 수 있나 (2026-07-30 확인)
  learnable-R 이 찾아낸 회전은 **모달(IMU/MM)·target_join·seed 와 거의 무관**하다.
  구성별 SO(3) 평균끼리 최대 1.33° 차이고, 각 구성 안에서 seed 산포도 1.4~2.5° 다.
  반면 R_pca 로부터는 일관되게 ~20° 떨어져 있다. 즉 "학습된 회전"은 사실상 유일하고,
  R_pca 와는 분명히 다른 지점이다.

무엇에 쓰나
  이 R 을 전처리에 구워서(`data_preprocess_MM.py --R`) **고정 R 파이프라인**으로 돌리면,
  learnable-R 의 이득이 (a) 회전이 R_pca 보다 좋아서인지 (b) isotropic 정규화·60epoch·
  R 이 학습 중 움직이는 것 같은 부수 조건 때문인지 분리된다. 지금 표에서는
  IMU 단독 75.00(고정 pca/zscore/30ep) vs 82.62(learnable/isotropic/60ep) 로 교락돼 있다.

SO(3) 평균 = 산술평균을 SVD 로 회전군에 사영한 것(Frobenius 최근접 회전).
반사(det<0)가 나오지 않도록 마지막 특이값 부호를 보정한다.

    /home/user1/miniconda3/envs/DA/bin/python scripts/build_learned_R.py
"""
from pathlib import Path
import glob

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RDIR = ROOT / "results" / "R_matrices"
OUT = RDIR / "R_learned.npy"

# 원본 백본 learnable-R 산출물만 쓴다(Compact/ 는 제외 — 경량 백본 결과는 별개 축).
PATTERNS = [
    ("MM  align-first", "R_learned_mm_alignfirst_join*_seed*.npy"),
    ("IMU align-first", "R_learned_alignfirst_join*_seed*.npy"),
]


def so3_mean(mats):
    """회전행렬 목록의 Frobenius 최근접 회전(SO(3) 사영 평균).

    저장된 R 은 float32 라 그대로 평균내면 직교성 오차가 1e-7 수준으로 남는다.
    float64 로 승격해 누적오차를 줄인다.
    """
    M = np.mean(np.stack(mats).astype(np.float64), axis=0)
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def geo_angle(A, B):
    return np.degrees(np.arccos(np.clip((np.trace(A @ B.T) - 1.0) / 2.0, -1.0, 1.0)))


def main():
    collected, groups = [], {}
    for label, pat in PATTERNS:
        files = sorted(glob.glob(str(RDIR / pat)))
        if not files:
            print(f"  [skip] {label}: 해당 파일 없음 ({pat})")
            continue
        mats = [np.load(f) for f in files]
        collected += mats
        groups[label] = so3_mean(mats)
        print(f"  [{label}] n={len(mats)}")

    if not collected:
        raise SystemExit("학습된 R 파일을 하나도 못 찾았다. results/R_matrices/ 를 확인하라.")

    R = so3_mean(collected)
    det = np.linalg.det(R)
    orth = np.abs(R @ R.T - np.eye(3)).max()
    # float32 로 저장된 R 을 다시 사영한 값이라 1e-6 수준 잔차는 정상이다.
    if not (abs(det - 1.0) < 1e-6 and orth < 1e-6):
        raise SystemExit(f"SO(3) 평균이 회전행렬이 아니다: det={det:.6f}, 직교오차={orth:.2e}")

    devs = [geo_angle(m, R) for m in collected]
    print(f"\n총 {len(collected)}개 평균 → R_learned")
    print(f"  개별 R 까지의 각: 평균 {np.mean(devs):.2f}°  최대 {np.max(devs):.2f}°")
    print(f"  det={det:+.6f}  직교오차={orth:.2e}")

    print("\n기준 R 대비 각도")
    for name in ["R_pca", "R_gravity", "R_permutation", "R_kabsch"]:
        p = RDIR / f"{name}.npy"
        if p.is_file():
            print(f"  vs {name:<15} {geo_angle(R, np.load(p)):6.2f}°")
    print(f"  vs {'I(단위행렬)':<15} {geo_angle(R, np.eye(3)):6.2f}°")

    print("\n구성별 평균끼리 (하나로 합쳐도 되는지 근거)")
    keys = list(groups)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            print(f"  {a} vs {b} = {geo_angle(groups[a], groups[b]):.2f}°")

    np.save(OUT, R.astype(np.float64))
    print(f"\n저장: {OUT.relative_to(ROOT)}")
    print(np.array2string(R, precision=4, suppress_small=True))


if __name__ == "__main__":
    main()
