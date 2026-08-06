# 2026-07-30 밤샘 실험 보고서

**실험 1** IMU 인코더 보간 제거(NoInterp) · **실험 2** 피험자 단위 분할 고정 R 5방법

| 항목 | 값 |
|---|---|
| 러너 | `run_exp_nointerp_subject_2026-07-30.sh` |
| 실행 | 2026-07-30 17:12 → 07-31 00:06 (413분) |
| 잡 | **14완료 / 0스킵 / 0실패** |
| seed | 0, 1, 2, 3, 4 (전 잡 공통) |
| 백본 | 원본 크기 (compact 아님) |
| 로그 | `logs/exp_nointerp_subject_2026-07-30/driver.log` |
| 결과 | `Compact/Result/NoInterp/Learnable_R/` (실험1), `results/SubjectSplit/` (실험2) |

`flock` 락을 걸어 두었고 driver.log 의 START 로그가 잡당 정확히 1회씩만 찍혔다 —
07-27 의 이중 실행 사고는 재발하지 않았다.

---

## 0. 요약

1. **NoInterp 는 null.** 10배 선형보간을 없애도 paired Δ = **−0.38 ± 6.36 %p** (se 1.64, n=15, t=−0.23).
   수용야 보상판(dilation 4배)도 회복시키지 못했다(−1.60). → 보간은 무해했다. **구조 변경 기각.**
2. **피험자 분할에서 방법 순위가 5/5 그대로 보존됐다.** 하락 폭은 −1.1 ~ −6.0 %p 뿐.
   → 기존 세션분할 숫자가 "같은 사람을 train 에서 본 덕"으로 부풀었던 게 아니다.
   주장 범위를 **"처음 보는 사람 + 새 디바이스"** 로 격상할 근거가 생겼다.
3. **pca 가 평균·분산 양쪽에서 1위.** IMU 73.94 ± 3.02 / MM 86.71 ± 1.44.
   permutation·kabsch 는 seed 표준편차가 8%p 대라 seed 하나로 순위가 뒤집힌다.

---

## 1. 데이터와 분할 — 실험 2

### 1.1 원 데이터

| | source | target |
|---|---|---|
| 디바이스 | Samsung 1 | Samsung 2 |
| 세션 식별 컬럼 | `filename` | `csv_filename_l` |
| 피험자 컬럼 | `subject` (`'sub1'`…`'sub14'`, 문자열) | `subject_id` (`1`…`14`, 정수) |
| 세션명 예시 | `sub11_deadlift_plot_and_store_rep_3.2.csv` | `Data_NINA13_2025.09.09_11.59.39_sub11_deadlift_1_30_6.csv` |

두 도메인의 피험자 스키마가 달라서 `subject_key()` 가 `'sub7'` / `7` / `'7'` 을 모두 정수 7 로
정규화한 뒤 대조한다 (`data_preprocess_MM.py:299`). 이게 있어야 "sub11 은 양쪽 도메인에서
같은 사람"이라는 대응이 성립한다.

### 1.2 윈도잉

| | EMG | IMU |
|---|---|---|
| 원본 레이트 | 1000 Hz | 100 Hz (업샘플 후) |
| 윈도우 | 5000 샘플 = **5초** | 500 샘플 = **5초** (동일 절대시간) |
| 스트라이드 | 500 샘플 = **0.5초** | 〃 |
| 채널 | 2 | 3 (accel x/y/z) |

스트라이드 0.5초 / 윈도우 5초 → **인접 윈도우가 90% 겹친다.** 표에 찍히는 n 은 명목값이고
유효 표본수는 대략 그 1/10 이다. 세션 하나가 통째로 흔들리면 1%p 대가 움직인다 —
아래 어느 표든 "n=3612 이라 유의"라고 읽으면 안 된다.

정규화는 EMG 축별 z-score(고정), IMU 는 `--imu_norm zscore` 를 썼다.

### 1.3 어떻게 나눴나

`data_preprocess_MM.py --split_by subject --val_subjects 3 8 --test_subjects 11 12`
(`split_by_subject()`, `data_preprocess_MM.py:304`)

**분할 단위는 세션이지만, 세션을 피험자로 묶어서 통째로 배정한다.** 지정되지 않은 피험자는
전부 train 이다. 즉 **같은 사람의 세션이 split 을 넘나들지 않는다.**

```
val  = sub3, sub8
test = sub11, sub12
train = 나머지 전원
```

| 도메인 | train | val | test |
|---|---|---|---|
| source (Samsung1) 세션 | 349 | 80 | 79 |
| source 피험자 | 1, 2, 4, 5, 6, 7, 9, 10, 13, 14 | 3, 8 | 11, 12 |
| target (Samsung2) 세션 | 310 | 85 | 89 |
| target 피험자 | 1, 4, 5, 6, 7, 9, 10, 13, 14 | 3, 8 | 11, 12 |

(sub2 는 target 도메인에 아예 없어서 source train 에만 등장한다.)

윈도우 수 및 운동별 분포:

| exercise | SRC tr | SRC val | SRC te | TGT tr | TGT val | TGT te |
|---|---|---|---|---|---|---|
| barbellcurl | 678 | 164 | 178 | 1391 | 311 | 384 |
| barbellrow | 716 | 127 | 148 | 1356 | 256 | 333 |
| benchpress | 1142 | 187 | 181 | 1611 | 326 | 356 |
| bte | 741 | 224 | 257 | 1687 | 345 | 327 |
| deadlift | 1234 | 223 | 235 | 1758 | 439 | 559 |
| dips | 651 | 125 | 158 | 1023 | 284 | 211 |
| latpulldown | 878 | 210 | 192 | 2238 | 468 | 419 |
| ohp | 1082 | 202 | 220 | 1709 | 344 | 375 |
| pullup | 415 | 163 | 84 | 900 | 253 | 264 |
| pushup | 580 | 150 | 161 | 1004 | 314 | 384 |
| **합계** | **8117** | **1775** | **1814** | **14677** | **3340** | **3612** |

### 1.4 왜 하필 sub3,8 / sub11,12 인가

**10종 운동을 양쪽 도메인 모두에서 갖춘 피험자는 1, 3, 8, 11, 12 다섯 명뿐이다.**
val/test 는 반드시 이 안에서 골라야 한다. 예컨대 sub10 은 samsung2 에서 dips/pullup/pushup 이
통째로 없어서 val 로 쓰면 7클래스만 평가된다. 그래서:

- test 에 sub11, sub12 → 두 명
- val 에 sub3, sub8 → 두 명
- sub1 은 남겨서 train 으로 (5명을 전부 빼면 train 이 너무 얇아진다)

전처리 시 `verify_class_coverage()` 가 val/test 에 10종이 다 있는지 확인하고, 빠지면
에러로 죽는다. 이번 실행은 두 도메인 모두 통과했다.

### 1.5 5방법이 같은 분할을 쓰도록 강제한 방법

방법(R 행렬)만 다르고 분할이 달라지면 방법 간 비교가 오염된다. 그래서:

1. `preprocessed_MMsubj_raw` 를 먼저 만들어 **기준 분할**을 확정하고 `split_manifest.json` 으로 저장
2. 나머지 4방법은 `--split_manifest preprocessed_MMsubj_raw/split_manifest.json` 로 **그 분할을 재사용**
3. 전처리 후 5폴더의 manifest 를 서로 대조하는 일치 검사를 돌리고, 하나라도 다르면 스윕 중단

driver.log 기록:

```
[OK] preprocessed_MMsubj_raw: 기준 분할  split_by=subject val=[3, 8] test=[11, 12]
                              source=[349, 80, 79] target=[310, 85, 89]
[OK] preprocessed_MMsubj_permutation: 분할 일치
[OK] preprocessed_MMsubj_gravity:     분할 일치
[OK] preprocessed_MMsubj_kabsch:      분할 일치
[OK] preprocessed_MMsubj_pca:         분할 일치
```

추가로 `report_split()` 이 split 간 세션 중복과 세션 수 총합을 검증한다(중복 있으면 예외).

---

## 2. 실험 1 — 보간 없이 스케일을 맞춘 방법

### 2.1 원본의 문제

원본 intermediate fusion 은 두 인코더의 시간 map 을 채널 방향으로 concat 하는데, 길이가 안 맞는다.

```
EMG: (B,2,5000) → stem stride 5 → 1000 → MaxPool(2) → 500   ⇒ map (B,256,500)
IMU: (B,3, 500) → stem stride 5 →  100 → MaxPool(2) →  50   ⇒ map (B,256, 50)
```

그래서 `F.interpolate(imu_map, size=500, mode="linear")` 로 **10배 늘려서** 맞췄다
(`Multimodal/mm_model.py:145`). 선형 업샘플은 없는 정보를 만들지 않으므로, 융합단이 보는
IMU 는 사실상 **10샘플마다 한 번 갱신되는 계단**이다. 이게 성능을 깎고 있는지 확인하는 게 실험 1.

### 2.2 조정한 파라미터

`Compact/nointerp_model.py` — **IMU 인코더만** 바꿨다. EMG 인코더 · fusion · head · 판별기는 원본 그대로.

| 지점 | 원본 | NoInterp | 효과 |
|---|---|---|---|
| stem `Conv1d` stride | **5** | **1** | 500 → 500 (길이 유지) |
| stem kernel / padding | 11 / 5 | 11 / 5 (동일) | `padding=(k−1)/2` 라 stride 1 이면 길이 보존 |
| 블록 사이 `MaxPool1d(2)` | 있음 | **`nn.Identity()`** | 길이 500 유지 |
| 블록 수 / 채널 스케줄 | 4 / 64→64→128→128→256 | 동일 | — |
| fusion 입력 | interpolate 후 concat | **바로 concat** | `forward` 에서 interpolate 제거 |

MaxPool 을 지우지 않고 `Identity` 로 **자리를 남긴** 이유는, `temporal_map()` 이
`layers[:-1]` 로 뒤쪽 pool 만 떼어내는 원본 규약을 그대로 쓰기 위해서다 — 블록 인덱스가 어긋나지 않는다.

### 2.3 스케일이 맞는다는 근거

```
EMG: 5000샘플/5초 = 1000Hz → stem(/5) 200Hz → MaxPool(/2) 100Hz → 길이 500
IMU:  500샘플/5초 =  100Hz → stem(/1) 100Hz → pool 없음        → 길이 500
```

둘 다 **100 Hz 격자(=10 ms)** 위에 있으므로 concat 했을 때 시각이 실제로 대응한다.
길이가 우연히 같은 게 아니라 샘플링 레이트가 같다는 게 핵심이다.
`forward` 에는 길이 불일치 시 예외를 던지는 assert 가 들어 있다.

### 2.4 용량 교락 없음 — 파라미터 수 동일

stride 와 pooling 만 건드렸고 conv 커널 모양은 하나도 안 바꿨으므로 파라미터 수가 완전히 같다.
`python Compact/nointerp_model.py` 자체 검증 출력:

```
모델                      원본      NoInterp        증감
Classifier       5,796,362     5,796,362     0.0%
DANN             5,960,844     5,960,844     0.0%
CDAN            11,566,092    11,566,092     0.0%

EMG map          (4, 256, 500)
IMU map (신규)   (4, 256, 500)   ← 보간 없이 EMG 와 동일 길이
IMU map (원본)   (4, 256, 50)    → F.interpolate 로 500 까지 10배 확대
forward 안에 interpolate 호출: 없음 ✓
```

→ 성능 차이가 나온다면 그것은 용량이 아니라 **순수하게 시간해상도/보간 효과**다.

### 2.5 유일한 대가 — 수용야, 그리고 그 보상판

원본은 뒤쪽 두 블록이 10 Hz 격자에서 돌아 누적 RF 가 약 **10초**였다. 길이를 유지하면
같은 dilation 이 100 Hz 격자에서 도므로 RF 가 `8×(1+2+4+8) = 120샘플 = **1.2초**` 로 줄어든다.
운동 1회 주기보다 짧아질 수 있는 값이다.

그래서 dilation 사다리를 환경변수로 갈아끼울 수 있게 하고, 보상판을 한 잡 더 돌렸다:

```
NOINTERP_IMU_DILATIONS=4,8,16,32   →  8×(4+8+16+32) = 480샘플 = 4.8초
```

실제 쓰인 값은 `variant_info()` 로 결과 json 에 provenance 로 박힌다.

**이 잡이 회복시키면 원인은 보간이 아니라 수용야**, 회복 못 시키면 둘 다 아니다 — 라는 판정 구조다.

### 2.6 사용 데이터 (실험 2 와 다름에 주의)

실험 1 은 **기존 세션 분할** 데이터 `preprocessed_MM_raw_isotropic` 을 쓴다. 기존 MM
learnable-R 스윕과 **seed 짝 비교**를 하려면 분할이 같아야 하기 때문이다.
`--lr_schedule global`, 60 epoch 도 같은 이유로 기존 설정에 맞췄다.

> 부작용: 이 때문에 cosine LR × join 교락(CDAN 이 켜지는 시점이 join 마다 달라 LR 도 달라지는 문제)은
> 이번에도 해소되지 않았다. 이건 `run_lr_and_R_2026-07-30.sh` 의 `--lr_schedule phase` 스윕 몫이다.

---

## 3. 결과

### 3.1 실험 1 — NoInterp (MM learnable R, target-test acc %)

| join | 원본(보간) | NoInterp | paired Δ | per-seed Δ |
|---|---|---|---|---|
| 0 | 86.21 | 86.12 | −0.09 ± 3.89 (se 1.74) | +0.3 +3.3 −6.8 +1.5 +1.3 |
| 10 | 83.84 | 81.31 | −2.53 ± 7.72 (se 3.45) | +7.3 −5.8 −4.8 −12.3 +3.0 |
| 20 | 82.60 | 84.08 | +1.47 ± 7.54 (se 3.37) | +5.3 +10.6 +1.3 −0.1 −9.8 |
| **풀링** | | | **−0.38 ± 6.36 (se 1.64, n=15, t=−0.23)** | |

RF 보상판 (dil 4,8,16,32) join0: **84.52 ± 5.14** — 기본 NoInterp 대비 **−1.60 ± 2.58**.

**판정: null.** 미리 세워둔 기준("Δ 가 0 근처면 보간은 무해했고 융합단이 IMU 를 저해상도로만
쓰고 있었다")에 그대로 해당한다. RF 보상판이 회복시키지 못했으므로 수용야 가설도 같이 죽었다.

→ **보간판 유지, 구조 변경 기각.** 부수적으로, per-seed Δ 가 −12.3 에서 +10.6 까지 흩어진다.
이 축에서는 5 seed 로 ±2%p 미만의 효과를 검출할 수 없다 — 기존 seed 편차 진단과 일관된다.

### 3.2 실험 2 — 피험자 분할, 고정 R 5방법 (target-test acc %, 5 seed)

| 방법 | IMU Tgt | MM Tgt | IMU Src | MM Src |
|---|---|---|---|---|
| **pca** | **73.94 ± 3.02** | **86.71 ± 1.44** | 76.48 ± 1.39 | 80.55 ± 3.88 |
| gravity | 66.88 ± 3.46 | 86.12 ± 3.17 | 76.64 ± 2.92 | 82.62 ± 5.76 |
| permutation | 68.19 ± 8.32 | 82.87 ± 7.07 | 71.86 ± 3.15 | 76.58 ± 4.57 |
| kabsch | 64.94 ± 8.19 | 79.60 ± 4.89 | 73.99 ± 2.60 | 81.04 ± 8.57 |
| raw | 20.24 ± 4.60 | 29.87 ± 4.16 | 69.07 ± 3.95 | 76.58 ± 4.69 |

seed 별 target-test:

| 방법 | IMU per-seed | MM per-seed |
|---|---|---|
| pca | 74.7 72.4 77.8 69.7 75.1 | 88.3 86.6 88.0 85.0 85.7 |
| gravity | 66.3 65.8 72.0 62.5 67.7 | 89.3 81.1 87.6 85.0 87.5 |
| permutation | 66.1 67.2 82.4 60.9 64.2 | 88.7 86.4 88.7 76.8 73.8 |
| kabsch | 57.6 74.5 73.2 60.0 59.4 | 78.3 84.5 75.7 74.5 85.0 |
| raw | 22.0 18.7 24.6 13.0 22.9 | 30.7 33.3 25.0 34.2 26.1 |

### 3.3 세션 분할 대비 — 피험자 간 일반화 비용

| 방법 | IMU 세션 | IMU 피험자 | Δ | MM 세션 | MM 피험자 | Δ |
|---|---|---|---|---|---|---|
| raw | 21.81 | 20.24 | −1.58 | 31.76 | 29.87 | −1.88 |
| permutation | 74.22 | 68.19 | −6.03 | 87.45 | 82.87 | −4.58 |
| gravity | 70.05 | 66.88 | −3.18 | 88.75 | 86.12 | −2.63 |
| kabsch | 63.78 | 64.94 | +1.17 | 85.56 | 79.60 | −5.96 |
| pca | 75.00 | 73.94 | −1.06 | 89.37 | 86.71 | −2.65 |

**세션분할 vs 피험자분할은 test 집합 자체가 다르므로 seed 짝 비교가 아니다.** 표본이 독립이라
차이의 불확실성이 더 크다. 그래서 개별 Δ 값보다 **순위 보존** 여부를 먼저 봐야 한다 —
그리고 **순위는 5/5 보존됐다** (pca ≳ gravity > permutation > kabsch ≫ raw).

---

## 4. 해석

**1. leakage 우려는 대체로 해소됐다.** 07-27 감사에서 피험자 전원이 train∩test 라서
"같은 사람 새 디바이스"까지만 주장 가능하다고 묶어 뒀는데, 같은 사람을 완전히 배제해도
하락이 −1 ~ −6 %p 에 그치고 순위가 그대로다. 축 정렬의 효과가 피험자 기억에서 온 게 아니다.
**"처음 보는 사람 + 새 디바이스"로 주장 격상 가능.**

**2. pca 를 고를 근거가 하나 늘었다.** 평균 1위인 것에 더해 **seed 산포가 가장 작다**
(IMU 3.02 / MM 1.44 vs permutation·kabsch 8%p 대). permutation 은 IMU seed2 에서 82.4,
seed3 에서 60.9 로 22%p 를 왕복한다 — 이 정도면 seed 하나로 순위가 뒤집힌다.
"평균이 비슷하니 아무거나"가 아니라 **pca 가 안정성에서 확실히 앞선다.**

**3. raw 는 여전히 파국이다** (IMU 20% / MM 30%, chance 10%). 반면 source-test 는 69~77% 로
멀쩡하다 — 축 정렬 없이는 target 에서만 붕괴한다는 그림이 분할 방식과 무관하게 재확인됐다.

**4. NoInterp 는 죽었지만 정보는 남았다.** 융합단이 IMU 를 10샘플 계단으로 받아도 성능이
같다는 건, 이 태스크에서 **IMU 의 기여가 고주파 성분이 아니라 저주파 자세/방향 성분**이라는
뜻이다. 실제로 축 정렬(R)이 그렇게 크게 먹히는 것과 정합적이다.

---

## 5. 남은 한계

- **test 가 sub11, sub12 두 명뿐이다.** 피험자 특이성이 그대로 들어간다. 순위가 뒤집히는
  방법이 있으면 피험자를 바꿔 한 번 더 확인해야 한다. 다만 이번엔 5/5 보존이라 급하진 않다.
- **checkpoint 선택은 여전히 target-val oracle selection.** 실전에서는 target 라벨이 없다.
  절대 수치는 낙관 편향이 있고, 방법 간 비교로만 읽어야 한다.
- **윈도 90% 중첩** → 유효 표본수 명목의 약 1/10.
- **실험 1 의 LR 교락 미해소.** seed 짝을 맞추려고 `--lr_schedule global` 로 돌렸기 때문에,
  join 별 LR 이 다른 문제는 그대로 남아 있다.

---

## 6. 다음 단계

**`run_lr_and_R_2026-07-30.sh` (19잡) 가 아직 안 돌았다** — `logs/lr_and_R_2026-07-30/` 자체가 없다.
`--lr_schedule phase` 로 cosine LR × join 교락을 끊는 스윕이라, 이게 끝나야 기존 join 스윕
해석 보류가 풀린다. GPU 는 지금 비어 있다.

---

## 부록 — 재현

```bash
# 실험 2 전처리 (기준 분할 생성)
python data_preprocess_MM.py --method raw --imu_norm zscore \
       --split_by subject --val_subjects 3 8 --test_subjects 11 12 \
       --out preprocessed_MMsubj_raw

# 나머지 4방법 (같은 분할 재사용)
python data_preprocess_MM.py --method pca --R results/R_matrices/R_pca.npy \
       --imu_norm zscore --split_manifest preprocessed_MMsubj_raw/split_manifest.json \
       --out preprocessed_MMsubj_pca

# 실험 2 학습
MM_DATA_DIR=preprocessed_MMsubj_pca python Multimodal/CDAN_train.py \
       --multi_seed --seeds 0 1 2 3 4 --epochs 30 --result_subdir SubjectSplit --no_cm --tag pca

# 실험 1 학습
MM_DATA_DIR=preprocessed_MM_raw_isotropic python Learnable_R/learnable_r_mm_alignfirst_train.py \
       --multi_seed --seeds 0 1 2 3 4 --epochs 60 --target_join_epoch 0 \
       --backbone nointerp --lr_schedule global --no_cm --tag nointerp_join0

# RF 보상판
NOINTERP_IMU_DILATIONS=4,8,16,32 MM_DATA_DIR=preprocessed_MM_raw_isotropic python ...

# NoInterp 모델 자체 검증 (파라미터 동일성 · 길이 · interpolate 부재)
python Compact/nointerp_model.py

# 전체 스윕
nohup bash run_exp_nointerp_subject_2026-07-30.sh > /dev/null 2>&1 &
```

**핵심 파일**

| 역할 | 경로 |
|---|---|
| 피험자 분할 구현 | `data_preprocess_MM.py:304` `split_by_subject()` |
| 피험자 키 정규화 | `data_preprocess_MM.py:299` `subject_key()` |
| 클래스 커버리지 검증 | `data_preprocess_MM.py:338` `verify_class_coverage()` |
| NoInterp 모델 | `Compact/nointerp_model.py` |
| 원본 보간 지점 | `Multimodal/mm_model.py:145` |
| 실험1 결과 | `Compact/Result/NoInterp/Learnable_R/*.json` |
| 실험2 결과 | `results/SubjectSplit/{cdan,imu_cdan}_result_<method>.json` |
