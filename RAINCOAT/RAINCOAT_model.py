import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline.baseline_model import ResidualTCNBlock


class TCNEncoder(nn.Module):
    def __init__(self, in_channels=5, feature_dim=512):
        super(TCNEncoder, self).__init__()

        self.encoder = nn.Sequential(
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
            ResidualTCNBlock(256, feature_dim, dilation=32),

            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.squeeze(-1)


class TimeEncoder(nn.Module):
    def __init__(self, in_channels=5, feature_dim=512):
        super(TimeEncoder, self).__init__()
        self.encoder = TCNEncoder(
            in_channels=in_channels,
            feature_dim=feature_dim
        )

    def forward(self, x):
        return self.encoder(x)


class FrequencyEncoder(nn.Module):
    def __init__(self, in_channels=5, feature_dim=512, eps=1e-6):
        super(FrequencyEncoder, self).__init__()
        self.eps = eps

        self.encoder = TCNEncoder(
            in_channels=in_channels,
            feature_dim=feature_dim
        )

    def forward(self, x):
        freq = torch.fft.rfft(x, dim=-1)
        mag = torch.abs(freq)
        mag = torch.log1p(mag)

        mean = mag.mean(dim=-1, keepdim=True)
        std = mag.std(dim=-1, keepdim=True)
        mag = (mag - mean) / (std + self.eps)
        mag = torch.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)

        return self.encoder(mag)


class RAINCOATModel(nn.Module):
    def __init__(self, in_channels=5, num_classes=10, feature_dim=512, dropout=0.5):
        super(RAINCOATModel, self).__init__()

        self.time_encoder = TimeEncoder(
            in_channels=in_channels,
            feature_dim=feature_dim
        )

        self.freq_encoder = FrequencyEncoder(
            in_channels=in_channels,
            feature_dim=feature_dim
        )

        fusion_dim = feature_dim * 2

        self.projector = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.GELU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, normalize_feat=False):
        time_feat = self.time_encoder(x)
        freq_feat = self.freq_encoder(x)

        feat = torch.cat([time_feat, freq_feat], dim=1)
        feat = self.projector(feat)

        if normalize_feat:
            feat = F.normalize(feat, dim=1)
            time_feat = F.normalize(time_feat, dim=1)
            freq_feat = F.normalize(freq_feat, dim=1)

        out = self.classifier(feat)

        return out, feat, time_feat, freq_feat


if __name__ == "__main__":
    model = RAINCOATModel(in_channels=5, num_classes=10)

    dummy_input = torch.randn(32, 5, 5000)
    out, feat, time_feat, freq_feat = model(dummy_input, normalize_feat=True)

    print(f"Class Output: {out.shape}")
    print(f"Fused Feature: {feat.shape}")
    print(f"Time Feature: {time_feat.shape}")
    print(f"Freq Feature: {freq_feat.shape}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")