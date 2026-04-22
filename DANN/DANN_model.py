import torch
import torch.nn as nn
from torch.autograd import Function
from baseline.baseline_model import ResidualTCNBlock, SEBlock1D

# =====================================================================
# 1. Gradient Reversal Layer (GRL)
# - 순전파 때는 그대로 통과, 역전파 때는 그래디언트에 -lambda를 곱함
# =====================================================================
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

# =====================================================================
# 2. DANN Model 구조
# =====================================================================
class DANNModel(nn.Module):
    def __init__(self, in_channels=5, num_classes=10):
        super(DANNModel, self).__init__()
        
        # [1] Feature Extractor: 특징을 뽑아내는 '뇌'
        self.feature_extractor = nn.Sequential(
            # Stem (5000 -> 1000)
            nn.Conv1d(in_channels, 64, kernel_size=11, stride=5, padding=5),
            nn.BatchNorm1d(64),
            nn.GELU(),
            
            # ResTCN Blocks
            ResidualTCNBlock(64, 64, dilation=1),
            ResidualTCNBlock(64, 128, dilation=2),
            nn.MaxPool1d(2), # 500
            
            ResidualTCNBlock(128, 128, dilation=4),
            ResidualTCNBlock(128, 256, dilation=8),
            nn.MaxPool1d(2), # 250
            
            ResidualTCNBlock(256, 256, dilation=16),
            ResidualTCNBlock(256, 512, dilation=32),
            nn.AdaptiveAvgPool1d(1) # (B, 512, 1)
        )
        
        # [2] Label Classifier: 운동 종목을 맞추는 부분
        self.label_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # [3] Domain Discriminator: Source인지 Target인지 맞추는 부분
        # 특징 추출기가 이 녀석을 속여야 함!
        self.domain_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2) # Source(0) vs Target(1)
        )

    def forward(self, x, alpha=1.0):
        # 1. 공통 특징 추출
        features = self.feature_extractor(x)
        features = features.squeeze(-1) # (Batch, 512)
        
        # 2. 라벨 분류 (운동 맞추기)
        class_output = self.label_classifier(features)
        
        # 3. 도메인 분류 (GRL 적용하여 적대적 학습)
        # alpha는 학습이 진행됨에 따라 서서히 높여주는 하이퍼파라미터
        reverse_features = ReverseLayerF.apply(features, alpha)
        domain_output = self.domain_classifier(reverse_features)
        
        return class_output, domain_output

# =====================================================================
# 🚀 모델 테스트
# =====================================================================
if __name__ == "__main__":
    model = DANNModel(in_channels=5, num_classes=10)
    dummy_input = torch.randn(32, 5, 5000)
    
    # 순전파 테스트 (alpha 값은 학습 중 동적으로 변함)
    c_out, d_out = model(dummy_input, alpha=0.1)
    
    print(f"Feature Extractor Output: {c_out.shape}") # (32, 10)
    print(f"Domain Discriminator Output: {d_out.shape}") # (32, 2)
    print("DANN Model ready to fight domain shift! 🥊")