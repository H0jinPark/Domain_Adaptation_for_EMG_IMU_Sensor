"""GRU 기반 baseline 모델 정의 (GRUBaselineModel).

baseline_train.py 의 --model gru 로 학습한다. Conv stem 으로 시퀀스를 축약한 뒤
양방향 GRU 로 시간 의존성을 모델링하는 시계열 분류 백본.
"""
import torch
import torch.nn as nn


class GRUBaselineModel(nn.Module):
    """Bi-GRU baseline 분류기. 입력 (B, in_channels, 5000) -> (B, num_classes)."""

    def __init__(
        self,
        in_channels=5,
        num_classes=10,
        hidden_dim=256,
        num_layers=2,
        dropout=0.5,
        bidirectional=True
    ):
        super(GRUBaselineModel, self).__init__()

        # Stem: Conv1d 로 채널 확장 및 시퀀스 길이 축소 (GRU 입력 부담 완화)
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=11, stride=5, padding=5),
            nn.BatchNorm1d(128),
            nn.GELU()
        )

        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        gru_out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C) : GRU 입력 형식

        out, h_n = self.gru(x)

        # 마지막 hidden state 를 feature 로 사용 (양방향이면 두 방향을 concat)
        if self.gru.bidirectional:
            feat = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            feat = h_n[-1]

        return self.classifier(feat)


# ----------------------------------------------------------------------
# 단독 실행 테스트 (출력 shape 검증용)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    model = GRUBaselineModel(in_channels=5, num_classes=10)

    dummy_input = torch.randn(32, 5, 5000)
    output = model(dummy_input)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model successfully built.")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape} (Expected: 32, 10)")
    print(f"Total Trainable Parameters: {total_params:,}")
