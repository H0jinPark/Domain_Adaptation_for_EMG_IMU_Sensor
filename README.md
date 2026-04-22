# Cross-Device Domain Adaptation for Wearable Sensor Data

본 프로젝트는 **EMG + IMU 센서 기반 운동 분류 문제**에서  
**디바이스 간 domain shift를 극복하기 위한 domain adaptation 방법론**을 비교합니다.

- Baseline (No Adaptation)
- CORAL (Covariance Alignment)
- DANN (Adversarial Domain Adaptation)

---

## 📁 Project Structure
DOMAIN_ADAPTATION/
│
├── baseline/
│ ├── baseline_model.py
│ └── baseline_train.py
│
├── Coral/
│ └── Coral_train.py
│
├── DANN/
│ ├── DANN_model.py
│ └── DANN_train.py
│
├── data/
│ ├── samsung1.parquet # Source domain
│ ├── samsung2.parquet # Target domain
│ └── ...
│
├── preprocessed/ # 전처리된 데이터 (.npy)
├── results/ # 결과 (confusion matrix 등)
├── weights/ # 모델 저장
│
├── data_preprocess.py # 전처리 코드
├── data_loader.py
├── inference.ipynb # 결과 확인용 notebook
├── README.md

---

## 📊 Problem Setting

- **Task**: 운동 분류 (classification)
- **Input**: 시계열 센서 데이터 (EMG + IMU)
- **Domain Shift**: 디바이스 차이 (same subject)

### ✔️ 중요 조건

- **같은 사용자 (subject)**
- **디바이스만 변경됨 (cross-device shift)**
- 사용자 간 generalization은 고려하지 않음

---

## 📦 Data Format

데이터는 아래 컬럼을 포함하면 사용 가능합니다.

- `biceps`, `triceps` (EMG)
- `triceps_X`, `triceps_Y`, `triceps_Z` (IMU)
- 파일 고유 식별 컬럼 (`filename` 또는 `csv_filename_l`)
- `timestamp` 또는 `Index_Time`
- `exercise` (label)

세부 사항은 `data_preprocess.py` 참고

---

## ⚙️ Data Preprocessing

전처리는 다음 코드로 수행합니다:

```bash
python data_preprocess.py

주요 과정
Resampling → 1000Hz
EMG Bandpass Filtering (20~450Hz)
세션 단위 Z-score 정규화
Sliding Window 생성 (window=5000, stride=500)
세션 기준 8:2 split
Source: samsung1
Target: samsung2

→ 각각 train / validation 분리

👉 전처리 결과는 preprocessed/ 폴더에 저장됩니다.

🚀 Training

세 가지 모델을 각각 실행할 수 있습니다.

1️⃣ Baseline
python baseline/baseline_train.py
2️⃣ CORAL
python Coral/Coral_train.py
3️⃣ DANN
python DANN/DANN_train.py

📈 Results
결과는 results/ 폴더에 저장됩니다.
각 모델은 confusion matrix 이미지를 생성합니다.
🔍 Inference & Visualization

직관적인 결과 확인은 다음 notebook에서 가능합니다:

inference.ipynb
🧠 Model Architectures
1️⃣ Baseline Model
구조
Conv1D + BatchNorm + GELU
Residual TCN Blocks
Dilation 기반 temporal modeling
SE Block (channel attention)
AdaptiveAvgPool → 512-d feature
FC classifier
특징
시계열 데이터에 최적화된 TCN 구조
EMG + IMU 멀티채널 처리 가능
비교적 강력한 baseline
2️⃣ CORAL Model
핵심 아이디어
Source / Target feature의 covariance alignment
구조
Baseline backbone 사용
Feature extractor에서 512-d feature 추출
CORAL loss 적용:
Cov(Source) ≈ Cov(Target)
특징
domain classifier 없음
안정적인 학습
본 실험에서 가장 높은 성능
3️⃣ DANN Model
핵심 아이디어
Domain-invariant feature 학습
구조
Feature Extractor
    ↓
 ┌─────────────┬─────────────┐
Label Classifier   Domain Classifier
Gradient Reversal Layer (GRL) 사용
domain confusion을 통해 일반화된 feature 학습
특징
adversarial training 기반
hyperparameter에 민감
CORAL 대비 약간 낮은 성능 (본 실험 기준)
📊 Experimental Insight
Model	Target Accuracy
Baseline	~0.50
CORAL	~0.97
DANN	~0.96
해석
디바이스 간 domain shift는 매우 큼
Domain adaptation 없으면 성능 급락
CORAL/DANN 적용 시 큰 성능 향상
동일 subject 조건에서 adaptation 효과 매우 큼
⚠️ Notes
본 실험은 target train label을 사용하는 supervised domain adaptation setting입니다.
완전한 unsupervised DA와는 다릅니다.
결과 해석 시 이 점을 고려해야 합니다.

🧾 Dependencies
conda env create -f environment.yml
📌 Summary
Cross-device domain shift 존재 확인
Baseline → Target 성능 급락
CORAL / DANN → 성능 크게 개선
CORAL이 가장 안정적이고 높은 성능
