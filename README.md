# Cross-Device Domain Adaptation for Wearable Sensor Data

본 프로젝트는 **EMG + IMU 센서 기반 운동 분류 문제**에서 **디바이스 간 domain shift를 극복하기 위한 domain adaptation 방법론**을 비교합니다.

### 비교 대상
- **Baseline** (No Adaptation)
- **CORAL** (Covariance Alignment)
- **DANN** (Adversarial Domain Adaptation)

---

## 📁 Project Structure

```
DOMAIN_ADAPTATION/
├── baseline/
│   ├── baseline_model.py
│   └── baseline_train.py
├── Coral/
│   └── Coral_train.py
├── DANN/
│   ├── DANN_model.py
│   └── DANN_train.py
├── data/
│   ├── samsung1.parquet       # Source domain
│   ├── samsung2.parquet       # Target domain
│   └── ...
├── preprocessed/              # 전처리된 데이터 (.npy)
├── results/                   # 결과 (confusion matrix 등)
├── weights/                   # 모델 저장
├── data_preprocess.py
├── data_loader.py
├── inference.ipynb
└── README.md
```

---

## 📊 Problem Setting

| 항목 | 설명 |
|------|------|
| **Task** | 운동 분류 (Classification) |
| **Input** | 시계열 센서 데이터 (EMG + IMU) |
| **Domain Shift** | 디바이스 차이 (Same Subject) |

### ✔️ 중요 조건

- 같은 사용자 (Subject)
- 디바이스만 변경됨 (Cross-Device Shift)
- 사용자 간 Generalization은 고려하지 않음

---

## 📦 Data Format

데이터는 아래 컬럼을 포함하면 사용 가능합니다:

| 센서 | 컬럼명 |
|------|--------|
| **EMG** | `biceps`, `triceps` |
| **IMU** | `triceps_X`, `triceps_Y`, `triceps_Z` |
| **기타** | `filename` (또는 `csv_filename_l`), `timestamp` (또는 `Index_Time`), `exercise` (label) |

> 세부 사항은 `data_preprocess.py` 참고

---

## ⚙️ Data Preprocessing

```bash
python data_preprocess.py
```

### 주요 과정

1. **Resampling** → 1000Hz
2. **EMG Bandpass Filtering** → 20~450Hz
3. **세션 단위 Z-score 정규화**
4. **Sliding Window 생성** → window=5000, stride=500
5. **데이터 분할** → 세션 기준 8:2 split
   - Source: samsung1 (train/validation)
   - Target: samsung2 (train/validation)

👉 **전처리 결과는 `preprocessed/` 폴더에 저장됩니다.**

---

## 🚀 Training

세 가지 모델을 각각 실행할 수 있습니다.

### 1️⃣ Baseline
```bash
python baseline/baseline_train.py
```

### 2️⃣ CORAL
```bash
python Coral/Coral_train.py
```

### 3️⃣ DANN
```bash
python DANN/DANN_train.py
```

---

## 📈 Results

- 결과는 `results/` 폴더에 저장됩니다.
- 각 모델은 **Confusion Matrix 이미지**를 생성합니다.

---

## 🔍 Inference & Visualization

직관적인 결과 확인은 다음 notebook에서 가능합니다:

📓 **`inference.ipynb`**

---

## 🧠 Model Architectures

### 1️⃣ Baseline Model

**구조**
- Conv1D + BatchNorm + GELU
- Residual TCN Blocks
- Dilation 기반 temporal modeling
- SE Block (channel attention)
- AdaptiveAvgPool → 512-d feature
- FC classifier

**특징**
- ✓ 시계열 데이터에 최적화된 TCN 구조
- ✓ EMG + IMU 멀티채널 처리 가능
- ✓ 비교적 강력한 baseline

---

### 2️⃣ CORAL Model

**핵심 아이디어**
- Source / Target feature의 covariance alignment

**구조**
- Baseline backbone 사용
- Feature extractor에서 512-d feature 추출
- CORAL loss 적용:
  ```
  Cov(Source) ≈ Cov(Target)
  ```

**특징**
- ✓ Domain classifier 없음
- ✓ 안정적인 학습
- ✓ **본 실험에서 가장 높은 성능** 🏆

---

### 3️⃣ DANN Model

**핵심 아이디어**
- Domain-invariant feature 학습

**구조**
```
Feature Extractor
    ↓
┌─────────────┬─────────────┐
Label Classifier   Domain Classifier
    ↑               (GRL)
    └───────────────┘
```  

- Gradient Reversal Layer (GRL) 사용
- Domain confusion을 통해 일반화된 feature 학습

**특징**
- ✓ Adversarial training 기반
- ⚠️ Hyperparameter에 민감
- ⚠️ CORAL 대비 약간 낮은 성능 (본 실험 기준)

---

## 📊 Experimental Results

| Model | Target Accuracy |
|-------|-----------------|
| **Baseline** | ~0.50 |
| **CORAL** | ~0.97 🏆 |
| **DANN** | ~0.96 |

### 해석

- 디바이스 간 domain shift는 **매우 큼**
- Domain adaptation 없으면 **성능 급락** (50%)
- CORAL/DANN 적용 시 **큰 성능 향상** (96~97%)
- **동일 subject 조건에서 adaptation 효과 매우 큼**

---

## ⚠️ Important Notes

> ⚠️ **본 실험은 target train label을 사용하는 Supervised Domain Adaptation setting입니다.**
> 
> 완전한 Unsupervised DA와는 다르므로 결과 해석 시 이 점을 반드시 고려해야 합니다.

---

## 🧾 Dependencies

```bash
conda env create -f environment.yml
```

---

## 📌 Summary

✅ Cross-device domain shift 존재 확인  
✅ Baseline → Target 성능 급락 (50%)  
✅ CORAL / DANN 적용 → 성능 크게 개선  
✅ **CORAL이 가장 안정적이고 높은 성능** 🏆