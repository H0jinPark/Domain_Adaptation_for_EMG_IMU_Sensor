import torch
import torch.nn as nn
from torch.autograd import Function


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class CoDATSModel(nn.Module):
    def __init__(self, in_channels=5, num_classes=10, feature_dim=256):
        super(CoDATSModel, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(256, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.label_classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 2)
        )

    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        features = features.squeeze(-1)

        class_output = self.label_classifier(features)

        reverse_features = ReverseLayerF.apply(features, alpha)
        domain_output = self.domain_classifier(reverse_features)

        return class_output, domain_output


if __name__ == "__main__":
    model = CoDATSModel(in_channels=5, num_classes=10)

    dummy_input = torch.randn(32, 5, 5000)
    c_out, d_out = model(dummy_input, alpha=0.1)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Class Output: {c_out.shape}")
    print(f"Domain Output: {d_out.shape}")
    print(f"Total Trainable Parameters: {total_params:,}")
    print("CoDATS Model ready!")