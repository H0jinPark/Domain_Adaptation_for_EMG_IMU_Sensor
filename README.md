🦾 Domain Adaptation for EMG/IMU Sensor Data
Addressing Domain Shift in Human Activity Recognition (HAR)

This repository focuses on developing and implementing Domain Adaptation (DA) strategies to mitigate the domain shift challenges in multi-modal sensor data, specifically EMG (Electromyography) and IMU (Inertial Measurement Unit).

🔬 Research Overview
Wearable sensor data often suffers from performance degradation due to variations in users, environments, and sensor placements (Domain Shift). This project explores various DA techniques to ensure robust performance across different domains in the wellness and healthcare service field.

Target Data: Multi-modal time-series data (EMG, IMU)

Key Methodology: Deep Domain Adaptation (DANN, CDAN, etc.)

Application: Healthcare, Wellness services, and Human Activity Recognition.

📂 Project Structure
Plaintext
├── data/               # (Excluded from Git) Large-scale sensor datasets
├── models/             # Architecture definitions for DA
├── utils/              # Data preprocessing & sensor fusion scripts
├── train.py            # Main training script
└── README.md
🛠️ Environment Setup
Framework: PyTorch / TensorFlow

Key Libraries: NumPy, Pandas, Scikit-learn, MNE (if applicable)

✍️ Author
Park Ho-jin (박호진)

Graduate Student, Dept. of Industrial Engineering

AIHC Lab, Sungkyunkwan University (SKKU)

💡 꿀팁: README 더 돋보이게 만들기
배지(Badge) 추가: 나중에 빌드 상태나 라이브러리 버전 배지를 상단에 달면 더 "개발자" 느낌 납니다.

그래프/그림: 데이터의 Domain Shift 현상을 보여주는 간단한 시각화 그림을 한 장 넣으면 논문 포트폴리오로 완벽합니다.