import torch
import torch.nn as nn


class TransformerBaselineModel(nn.Module):
    def __init__(
        self,
        in_channels=5,
        num_classes=10,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        ff_dim=512,
        dropout=0.2,
        max_len=1000
    ):
        super(TransformerBaselineModel, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, embed_dim, kernel_size=11, stride=5, padding=5),
            nn.BatchNorm1d(embed_dim),
            nn.GELU()
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = x.transpose(1, 2)

        b, seq_len, _ = x.shape

        cls_token = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        x = x + self.pos_embedding[:, :seq_len + 1, :]

        x = self.transformer(x)
        x = self.norm(x)

        cls_feat = x[:, 0, :]

        return self.classifier(cls_feat)


if __name__ == "__main__":
    model = TransformerBaselineModel(in_channels=5, num_classes=10)

    dummy_input = torch.randn(32, 5, 5000)
    output = model(dummy_input)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model successfully built!")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape} (Expected: 32, 10)")
    print(f"Total Trainable Parameters: {total_params:,}")