# Cross-Device Domain Adaptation for EMG·IMU Sensor Data

웨어러블 센서(**EMG + IMU**) 기반 운동 분류에서, 서로 다른 측정 디바이스 간
**domain shift** 를 극복하기 위한 domain adaptation 방법을 비교·실험하는 프로젝트.

- **Source** : Samsung1 — 라벨 있음
- **Target** : Samsung2 — 라벨 없음, 동일 동작이지만 IMU 축 순서·부호가 다름
- **목표** : 라벨이 있는 Source 만으로 Target 운동 분류 정확도를 끌어올림 (UDA)
- **동작 클래스 (10종)** : `barbellcurl`, `barbellrow`, `benchpress`, `bte`,
  `deadlift`, `dips`, `latpulldown`, `ohp`, `pullup`, `pushup`

---

## Project Structure

```
Domain_Adaptation/
├── data/                       # 원본 parquet (samsung1 / samsung2 (Y축, X축 변경한 상태)/ samsung2_original)
├── preprocessed/               # 5채널 동기화 전처리 결과 (.npy)
├── preprocessed_MM/            # 멀티모달(EMG/IMU 분리) 전처리 결과 (.npy)
├── preprocessed_PCA/           # PCA 축 정렬 전처리 결과 (.npy)
│
├── data_preprocess.py          # 5채널 동기화 전처리
├── data_preprocess_MM.py       # EMG/IMU 분리 전처리
├── data_preprocess_PCA.py      # PCA 기반 IMU 축 정렬 전처리
├── data_loader.py              # .npy -> PyTorch DataLoader
│
├── baseline/                   # No-DA 백본 + 멀티모달 인코더
├── Coral/                      # CORAL (covariance alignment)
├── MMD/                        # MMD (maximum mean discrepancy)
├── DANN/                       # DANN 및 멀티모달 변종
├── CDAN/                       # CDAN (conditional adversarial)
│
├── notebook/process.ipynb      # IMU 축 정렬 탐색
└── environment.yaml            # conda 환경 정의
```

---

## Problem Setting

| 항목 | 설명 |
|------|------|
| **Task** | 운동 분류 (10-class classification) |
| **Input** | 시계열 센서 데이터 — EMG 2ch (1000Hz) + IMU 3ch (100Hz) |
| **Domain Shift** | 측정 디바이스 차이 (Samsung1 -> Samsung2) |
| **설정** | 라벨 있는 Source + 라벨 없는 Target -> Unsupervised DA |

**중요 조건**
- 디바이스만 다르고 동작/사용자는 동일한 cross-device shift 를 다룸
- Target IMU 는 축 순서·부호가 Source 와 달라, 전처리 단계의 축 정렬도 별도 실험

---

## Data Format

`data/` 의 parquet 파일은 아래 컬럼을 포함:

| 센서 | 컬럼명 |
|------|--------|
| **EMG** | `biceps`, `triceps` |
| **IMU** | `triceps_X`, `triceps_Y`, `triceps_Z` |
| **기타** | `exercise` (label), 시간/세션 식별 컬럼 |

> 세부 컬럼 처리는 `data_preprocess.py` 참고.

---

## Data Preprocessing

용도에 따라 세 가지 전처리 파이프라인을 제공한다.

| 스크립트 | 출력 폴더 | 포맷 | 설명 |
|----------|-----------|------|------|
| `data_preprocess.py` | `preprocessed/` | `(N, 5000, 5)` | EMG+IMU 를 1000Hz 로 동기화한 5채널 텐서 |
| `data_preprocess_MM.py` | `preprocessed_MM/` | EMG `(N, 2, 5000)` / IMU `(N, 3, 500)` | 모달리티별 샘플레이트 유지, 분리 저장 |
| `data_preprocess_PCA.py` | `preprocessed_PCA/` | `(N, 5000, 5)` | 위 5채널 포맷 + Target IMU 에 운동별 PCA 회전 정렬 적용 |

**공통 처리 과정 (`data_preprocess.py` 기준)**

1. **Resampling** -> 1000Hz
2. **EMG Bandpass Filtering** -> 20~450Hz
3. **세션 단위 Z-score 정규화**
4. **Sliding Window** -> window=5000, stride=500
5. **세션 기준 8:2 분할** -> Source/Target 각각 train·validation

```bash
python data_preprocess.py       # -> preprocessed/
python data_preprocess_MM.py    # -> preprocessed_MM/
python data_preprocess_PCA.py   # -> preprocessed_PCA/
```

---

## Models & Training

### A. Baseline 모델 (No DA) — `baseline/baseline_train.py`

5채널 동기화 입력(`preprocessed/`)을 받는 단일 백본 모델 5종. 모두 **하나의 스크립트**
`baseline_train.py` 로 학습하며, `--model` 인자로 종류를 선택한다.

| `--model` | 클래스 | 구조 요약 |
|-----------|--------|-----------|
| `tcn` (기본) | `AdvancedBaselineModel` | Conv1D + Residual TCN(dilation 1·2·4·8·16·32) + SE block + AdaptiveAvgPool + FC |
| `cnn` | `CNNBaselineModel` | 1D CNN + SE block |
| `transformer` | `TransformerBaselineModel` | Conv embedding + Transformer encoder x4 |
| `gru` | `GRUBaselineModel` | 2-layer Bidirectional GRU |
| `mlp` | `MLPBaselineModel` | flatten 후 fully-connected |

#### 실행 옵션

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--model` | `tcn` | 학습 모델 선택 — `tcn` / `cnn` / `transformer` / `gru` / `mlp` |
| `--seed` | `42` | 단일 실행 시 random seed |
| `--multi_seed` | (off) | 여러 seed 로 반복 학습 후 평균±표준편차를 집계 |
| `--seeds` | `0 1 2 3 4` | `--multi_seed` 일 때 사용할 seed 목록 |
| `--batch_size` | `64` | 배치 크기 |
| `--epochs` | `20` | 학습 epoch 수 |
| `--lr` | `1e-3` | 학습률 (AdamW + CosineAnnealingLR) |
| `--data_dir` | `preprocessed` | 입력 데이터 폴더 (`preprocessed/`, `preprocessed_PCA/` 등) |
| `--tag` | `""` | 결과·가중치 파일명에 붙일 식별자 (예: `pca`) |
| `--no_plot` | (off) | confusion matrix 창을 띄우지 않음 (PNG 저장은 유지) |

#### 사용 예시

```bash
# 기본 — TCN, seed 42, 단일 실행
python baseline/baseline_train.py

# 모델 종류 바꾸기
python baseline/baseline_train.py --model cnn
python baseline/baseline_train.py --model transformer

# multi-seed 실험 — seed 0~4 반복, 평균±std 를 CSV 로 저장
python baseline/baseline_train.py --model tcn --multi_seed

# multi-seed + seed 목록 직접 지정
python baseline/baseline_train.py --model gru --multi_seed --seeds 10 20 30

# PCA 정렬 데이터로 학습 + 파일명 태그 부여
python baseline/baseline_train.py --model tcn --data_dir preprocessed_PCA --tag pca

# 하이퍼파라미터 조정 + 플롯 끄기
python baseline/baseline_train.py --model mlp --epochs 30 --lr 5e-4 --batch_size 128 --no_plot
```

각 실행은 **1) Source 학습 -> 2) Source 평가 -> 3) Target 평가(domain gap 출력)** 순으로
진행되며, 다음 파일을 생성한다.

- 가중치 : `weights/{model}_seed{seed}_baseline{tag}_best.pth`
- Confusion matrix : `results/{model}_seed{seed}_baseline{tag}_{source,target}_confusion_matrix.png`
- multi-seed 요약 : `results/{model}_baseline{tag}_multiseed_results.csv` (seed별 정확도 + 평균/표준편차)

> 멀티모달 baseline(`baseline_DualEncoder`)은 입력 포맷이 달라 `baseline_train.py` 로는
> 학습되지 않는다. 전용 스크립트 `baseline/baseline_DualEncoder_train.py` 를 사용한다 (섹션 C).

### B. Single-modal Domain Adaptation

`preprocessed/` 5채널 입력에 TCN 백본을 공유하고 DA 손실만 추가한다.

| 스크립트 | 방법 |
|----------|------|
| `Coral/Coral_train.py` | CORAL — source/target covariance alignment |
| `MMD/MMD_train.py` | MMD — gaussian-kernel maximum mean discrepancy |
| `DANN/DANN_train.py` | DANN — Gradient Reversal Layer 기반 adversarial DA |
| `CDAN/CDAN_train.py` | CDAN — conditional adversarial DA |

```bash
python Coral/Coral_train.py
python MMD/MMD_train.py
python DANN/DANN_train.py
python CDAN/CDAN_train.py
```

### C. 멀티모달 / 융합 모델

EMG(2ch·1000Hz)와 IMU(3ch·100Hz)를 별도 인코더로 처리하는 모델. 입력은 모달리티가
분리된 `preprocessed_MM/` 또는 5채널 동기화 `preprocessed/` 를 사용한다.

| 스크립트 | 설명 | 입력 |
|----------|------|------|
| `baseline/baseline_DualEncoder_train.py` | EMG/IMU 독립 인코더 + concat, DA 없음 | `preprocessed_MM/` |
| `DANN/DANN_MM_train.py` | DualEncoder + DANN (GRL) | `preprocessed_MM/` |
| `DANN/DANN_MM_dualdisc_train.py` | DualEncoder + 모달리티별 GRL·도메인 판별기 2개 | `preprocessed_MM/` |

```bash
python baseline/baseline_DualEncoder_train.py
python DANN/DANN_MM_train.py
python DANN/DANN_MM_dualdisc_train.py
```

> 모든 명령은 **프로젝트 루트**에서 실행하며, 학습 결과는 `weights/` · `results/` ·
> `logs/` 에 저장된다.

---

## Results

### IMU 축 정렬 전처리

Target IMU 의 축 순서·부호 불일치를 전처리 단계에서 보정하는 실험.
`notebook/process.ipynb` 에서 축 순열·PCA·Procrustes·Affine·Soft-DTW 등을 비교했고,
PCA 회전 정렬이 가장 안정적이었다. 채택된 PCA 정렬은 `data_preprocess_PCA.py` 로
구현되어 있으며, 결과는 `preprocessed_PCA/` 에 저장된다.

| 방법 | Before DTW | After DTW | 개선율 |
|------|-----------:|----------:|-------:|
| 정렬 없음 | 395.50 | — | — |
| PCA 정렬 | 395.50 | 329.98 | 16.6% |

학습·평가 로그와 confusion matrix 는 `logs/`, `results/` 에 저장된다.

---

## Environment

```bash
conda env create -f environment.yaml
conda activate mda
```

---

## Summary

- Cross-device domain shift 의 존재를 확인 — DA 없으면 Target 성능 급락
- IMU 축 불일치는 PCA 정렬 전처리로 일부 보정 가능
- CORAL / MMD / DANN / CDAN 및 멀티모달 융합 모델로 domain adaptation 방법을 비교
