"""MLP 기반 baseline 모델 정의 (MLPBaselineModel).

baseline_train.py 의 --model mlp 로 학습한다. 시계열 입력을 통째로 flatten 한 뒤
fully-connected layer 로 분류하는 가장 단순한 baseline.
"""
import torch
import torch.nn as nn


class MLPBaselineModel(nn.Module):
    """MLP baseline 분류기. 입력 (B, in_channels, seq_len) -> (B, num_classes)."""

    def __init__(
        self,
        in_channels=5,
        seq_len=5000,
        num_classes=10,
        hidden_dim=512,
        dropout=0.5
    ):
        super(MLPBaselineModel, self).__init__()

        # flatten 후 입력 차원 = 채널 수 x 시퀀스 길이
        input_dim = in_channels * seq_len

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


# ----------------------------------------------------------------------
# 단독 실행 테스트 (출력 shape 검증용)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    model = MLPBaselineModel(in_channels=5, seq_len=5000, num_classes=10)

    dummy_input = torch.randn(32, 5, 5000)
    output = model(dummy_input)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model successfully built.")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape} (Expected: 32, 10)")
    print(f"Total Trainable Parameters: {total_params:,}")
