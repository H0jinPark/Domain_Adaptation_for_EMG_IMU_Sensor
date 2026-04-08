import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. 공통 특징 추출기 (Encoder) ---
class SensorEncoder(nn.Module):
    def __init__(self, input_channels):
        super(SensorEncoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool1d(8)

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1) # (Batch, 2048)
        return x

# --- 2. 대조학습 기반 Cross-Modal Distiller ---
class CrossModalCLModel(nn.Module):
    def __init__(self, imu_channels=21, emg_channels=7, num_classes=10, feature_dim=2048, proj_dim=128):
        super(CrossModalCLModel, self).__init__()
        
        self.imu_encoder = SensorEncoder(input_channels=imu_channels)
        self.emg_encoder = SensorEncoder(input_channels=emg_channels)
        
        self.imu_proj = nn.Sequential(
            nn.Linear(feature_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        self.emg_proj = nn.Sequential(
            nn.Linear(feature_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        
        self.imu_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, num_classes)
        )
        self.emg_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, imu_x, emg_x):
        imu_feat = self.imu_encoder(imu_x)
        emg_feat = self.emg_encoder(emg_x)
        
        imu_z = self.imu_proj(imu_feat)
        emg_z = self.emg_proj(emg_feat)
        
        # 🌟 InfoNCE를 위한 필수 과정: 내적(Dot Product)이 코사인 유사도가 되도록 L2 정규화
        imu_z = F.normalize(imu_z, dim=1)
        emg_z = F.normalize(emg_z, dim=1)
        
        imu_pred = self.imu_classifier(imu_feat)
        emg_pred = self.emg_classifier(emg_feat)
        
        return imu_z, emg_z, imu_pred, emg_pred

# --- 3. 통합 Loss 함수 정의 (🔥🔥 InfoNCE 대조학습으로 수정 🔥🔥) ---
def cl_distillation_loss(imu_z, emg_z, imu_pred, emg_pred, labels, alpha=1.0, temperature=0.1):
    """
    alpha: Contrastive Loss의 비중을 조절하는 계수
    temperature: InfoNCE(Contrastive) 유사도 분포를 날카롭게 만들어주는 계수 (보통 0.07 ~ 0.1)
    """
    # 1. Task Loss (동작 분류 손실)
    loss_cls_imu = F.cross_entropy(imu_pred, labels)
    loss_cls_emg = F.cross_entropy(emg_pred, labels)
    
    # 2. Alignment Loss (🌟 InfoNCE Cross-Modal Contrastive Loss 🌟)
    batch_size = emg_z.size(0)
    
    # (Batch, 128) x (128, Batch) -> (Batch, Batch) 형태의 유사도 행렬(Similarity Matrix) 계산
    # imu_z는 여전히 detach() 하여 Teacher를 고정합니다.
    logits = torch.matmul(emg_z, imu_z.detach().T) / temperature
    
    # 정답 라벨 생성: 대각선 원소 (i, i)가 Positive Pair (동일 샘플의 EMG와 IMU)
    # 나머지는 모두 Negative Pair로 취급되어 멀어지게 됩니다.
    labels_cl = torch.arange(batch_size).to(emg_z.device)
    
    # Cross Entropy를 사용해 대각선(Positive)은 1에 가깝게 당기고, 나머지는 0으로 밀어냅니다.
    loss_align = F.cross_entropy(logits, labels_cl)
    
    # 총 Loss
    total_loss = loss_cls_imu + loss_cls_emg + (alpha * loss_align)
    
    return total_loss, loss_cls_imu, loss_cls_emg, loss_align

if __name__ == "__main__":
    print("🛠️ 진정한 InfoNCE CL_model 구조 테스트 중...")
    model = CrossModalCLModel(imu_channels=21, emg_channels=7, num_classes=10)
    
    dummy_imu = torch.randn(16, 21, 2048)
    dummy_emg = torch.randn(16, 7, 2048)
    dummy_labels = torch.randint(0, 10, (16,))
    
    imu_z, emg_z, imu_pred, emg_pred = model(dummy_imu, dummy_emg)
    loss, l_imu, l_emg, l_align = cl_distillation_loss(imu_z, emg_z, imu_pred, emg_pred, dummy_labels)
    
    print(f"✅ Projection 차원: {emg_z.shape} (기대값: [16, 128])")
    print(f"✅ Prediction 차원: {emg_pred.shape} (기대값: [16, 10])")
    print(f"✅ InfoNCE Align Loss: {l_align.item():.4f}")
    print(f"✅ Total Loss 계산 완료: {loss.item():.4f}")
    print("🚀 완벽합니다! 진짜 대조학습(InfoNCE)으로 업그레이드 완료!")