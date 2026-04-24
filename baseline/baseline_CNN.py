import torch
import torch.nn as nn

class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock1D, self).__init__()
        hidden = max(channels // reduction, 1)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)


class ResidualCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, dropout=0.5):
        super(ResidualCNNBlock, self).__init__()
        padding = kernel_size // 2

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
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


class CNNBaselineModel(nn.Module):
    def __init__(self, in_channels=5, num_classes=10):
        super(CNNBaselineModel, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=11, stride=5, padding=5),
            nn.BatchNorm1d(64),
            nn.GELU()
        )

        self.layers = nn.Sequential(
            ResidualCNNBlock(64, 64),
            ResidualCNNBlock(64, 128),
            nn.MaxPool1d(2),

            ResidualCNNBlock(128, 128),
            ResidualCNNBlock(128, 256),
            nn.MaxPool1d(2),

            ResidualCNNBlock(256, 256),
            ResidualCNNBlock(256, 512),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = x.squeeze(-1)
        return self.classifier(x)


if __name__ == "__main__":
    model = CNNBaselineModel(in_channels=5, num_classes=10)

    dummy_input = torch.randn(32, 5, 5000)
    output = model(dummy_input)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model successfully built!")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape} (Expected: 32, 10)")
    print(f"Total Trainable Parameters: {total_params:,}")