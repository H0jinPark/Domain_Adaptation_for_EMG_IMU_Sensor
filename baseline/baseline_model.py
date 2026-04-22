import torch
import torch.nn as nn

class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock1D, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResidualTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, dilation=1, dropout=0.5):
        super(ResidualTCNBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2
        
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.se = SEBlock1D(out_channels)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.net(x)
        out = self.se(out)
        res = x if self.downsample is None else self.downsample(x)
        return out + res

class AdvancedBaselineModel(nn.Module):
    def __init__(self, in_channels=5, num_classes=10):
        super(AdvancedBaselineModel, self).__init__()
        
        # Stem: 초기 압축 (5000 -> 1000)
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=11, stride=5, padding=5),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        
        # TCN Layers: 어텐션 없이 순수하게 깊게 쌓음
        self.layers = nn.Sequential(
            ResidualTCNBlock(64, 64, dilation=1),
            ResidualTCNBlock(64, 128, dilation=2),
            nn.MaxPool1d(2), # 1000 -> 500
            
            ResidualTCNBlock(128, 128, dilation=4),
            ResidualTCNBlock(128, 256, dilation=8),
            nn.MaxPool1d(2), # 500 -> 250
            
            ResidualTCNBlock(256, 256, dilation=16),
            ResidualTCNBlock(256, 512, dilation=32),
            nn.AdaptiveAvgPool1d(1) # 마지막에 전역 평균 풀링으로 요약
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.5), # 강력한 드롭아웃
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = x.squeeze(-1)
        return self.classifier(x)

# =====================================================================
# 🚀 단독 실행 테스트 (모델 파라미터 수 및 메모리 검증용)
# =====================================================================
if __name__ == "__main__":
    model = AdvancedBaselineModel(in_channels=5, num_classes=10)
    
    # 더미 데이터 생성 (배치 32, 채널 5, 길이 5000)
    dummy_input = torch.randn(32, 5, 5000)
    output = model(dummy_input)
    
    # 모델 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model successfully built!")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape} (Expected: 32, 10)")
    print(f"Total Trainable Parameters: {total_params:,}")