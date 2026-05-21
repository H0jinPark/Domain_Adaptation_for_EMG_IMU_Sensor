"""CDAN 모델 정의 (단일 백본, 5채널 동기화 입력).

Conditional Domain Adversarial Network. TCN feature 와 분류 확률의 외적
(multilinear conditioning)을 도메인 판별기 입력으로 사용해, 클래스 정보를
반영한 도메인 적대 학습을 수행한다.
"""
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

from baseline.baseline_model import ResidualTCNBlock


class ReverseLayerF(Function):
    """Gradient Reversal Layer: 역전파 시 그래디언트에 -alpha 를 곱한다."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class CDANModel(nn.Module):
    """CDAN 분류기. forward(x, alpha) -> (class_logits, domain_logits)."""

    def __init__(self, in_channels=5, num_classes=10, feature_dim=512):
        super(CDANModel, self).__init__()

        self.num_classes = num_classes
        self.feature_dim = feature_dim

        # Feature Extractor: TCN 백본 (5000 -> 250, 512차원 feature)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=11, stride=5, padding=5),
            nn.BatchNorm1d(64),
            nn.GELU(),

            ResidualTCNBlock(64, 64, dilation=1),
            ResidualTCNBlock(64, 128, dilation=2),
            nn.MaxPool1d(2),

            ResidualTCNBlock(128, 128, dilation=4),
            ResidualTCNBlock(128, 256, dilation=8),
            nn.MaxPool1d(2),

            ResidualTCNBlock(256, 256, dilation=16),
            ResidualTCNBlock(256, 512, dilation=32),
            nn.AdaptiveAvgPool1d(1)
        )

        self.label_classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        # 도메인 판별기 입력 = feature 와 분류 확률의 외적 (feature_dim x num_classes)
        cdan_input_dim = feature_dim * num_classes

        self.domain_classifier = nn.Sequential(
            nn.Linear(cdan_input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.5),

            nn.Linear(512, 2)
        )

    def conditional_feature(self, features, class_logits):
        """feature 와 분류 확률의 외적(multilinear map)을 평탄화해 반환한다."""
        class_probs = F.softmax(class_logits, dim=1)

        conditional = torch.bmm(
            class_probs.unsqueeze(2),
            features.unsqueeze(1)
        )

        conditional = conditional.view(features.size(0), -1)

        return conditional

    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        features = features.squeeze(-1)

        class_output = self.label_classifier(features)

        # 클래스 정보로 조건화된 feature 를 GRL 거쳐 도메인 판별기에 전달
        conditional_features = self.conditional_feature(features, class_output)
        reverse_features = ReverseLayerF.apply(conditional_features, alpha)
        domain_output = self.domain_classifier(reverse_features)

        return class_output, domain_output


# ----------------------------------------------------------------------
# 단독 실행 테스트 (출력 shape 검증용)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    model = CDANModel(in_channels=5, num_classes=10)

    dummy_input = torch.randn(32, 5, 5000)

    c_out, d_out = model(dummy_input, alpha=0.1)

    print(f"Class Output : {c_out.shape}")
    print(f"Domain Output: {d_out.shape}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")
