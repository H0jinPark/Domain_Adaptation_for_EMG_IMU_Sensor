"""Physics-informed EMG→IMU 보조 손실 실험 패키지.

EMG 인코더의 시간 feature map 에서 IMU 를 복원하는 디코더를 달고, 재구성 손실과
jerk(가속도 시간미분) 최소화 손실을 분류/도메인 손실에 더한다. 기존 Multimodal/
파이프라인은 건드리지 않는다.
"""
