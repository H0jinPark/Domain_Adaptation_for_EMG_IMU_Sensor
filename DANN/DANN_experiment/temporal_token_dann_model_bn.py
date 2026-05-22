"""Temporal Token DANN 모델 정의 (멀티모달, EMG/IMU 분리 입력).

EMG/IMU 를 각각 temporal token 으로 변환한 뒤, cross-modal attention 으로
서로의 시계열 정보를 참조한다. 이후 attention pooling 기반 융합 feature 에
label classifier 와 domain discriminator(GRL)를 얹어 domain-invariant feature 를
학습한다.

성능 향상을 위해 Conv token encoder 와 MLP head 에 BatchNorm 을 사용한다.
Target train 데이터를 forward 하면 BatchNorm running statistics 가 target 분포를
반영하므로 AdaBN 효과를 함께 활용할 수 있다.
"""
import math
import torch
import torch.nn as nn
from torch.autograd import Function


# ----------------------------------------------------------------------
# Gradient Reversal Layer (GRL)
# 순전파에서는 입력을 그대로 통과시키고, 역전파에서는 그래디언트에 -alpha 를 곱한다.
# ----------------------------------------------------------------------
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


# ----------------------------------------------------------------------
# Sinusoidal Positional Encoding
# temporal token 에 시간 위치 정보를 더한다.
# ----------------------------------------------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ----------------------------------------------------------------------
# Conv Token Encoder
# 원시 EMG/IMU 시계열을 temporal token sequence 로 변환한다.
# BatchNorm 을 사용해 source/target 통계 적응 효과를 활용한다.
# ----------------------------------------------------------------------
class ConvTokenEncoder(nn.Module):
    def __init__(self, in_channels, d_model=128, token_stride=10, depth=3, dropout=0.1):
        super().__init__()

        layers = [
            nn.Conv1d(in_channels, d_model, kernel_size=15, stride=token_stride, padding=7),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        ]

        for _ in range(depth):
            layers.extend([
                nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model),
                nn.Conv1d(d_model, d_model, kernel_size=1),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            ])

        self.net = nn.Sequential(*layers)
        self.pos = SinusoidalPositionalEncoding(d_model)

    def forward(self, x):
        x = self.net(x)          # (B, D, T)
        x = x.transpose(1, 2)    # (B, T, D)
        return self.pos(x)


# ----------------------------------------------------------------------
# Cross-Modal Attention Block
# EMG token 은 IMU token 을, IMU token 은 EMG token 을 참조한다.
# Transformer block 내부는 token 안정성을 위해 LayerNorm 을 유지한다.
# ----------------------------------------------------------------------
class CrossModalBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, ff_mult=4, dropout=0.1):
        super().__init__()

        self.emg_to_imu = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.imu_to_emg = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        self.norm_emg_1 = nn.LayerNorm(d_model)
        self.norm_imu_1 = nn.LayerNorm(d_model)
        self.norm_emg_2 = nn.LayerNorm(d_model)
        self.norm_imu_2 = nn.LayerNorm(d_model)

        self.ff_emg = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )
        self.ff_imu = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, emg, imu):
        emg_attn, _ = self.emg_to_imu(query=emg, key=imu, value=imu, need_weights=False)
        imu_attn, _ = self.imu_to_emg(query=imu, key=emg, value=emg, need_weights=False)

        emg = self.norm_emg_1(emg + emg_attn)
        imu = self.norm_imu_1(imu + imu_attn)

        emg = self.norm_emg_2(emg + self.ff_emg(emg))
        imu = self.norm_imu_2(imu + self.ff_imu(imu))

        return emg, imu


# ----------------------------------------------------------------------
# Temporal Token DANN Model
# EMG/IMU temporal token 을 cross-attention 으로 융합한 뒤 DANN 학습을 수행한다.
# ----------------------------------------------------------------------
class TemporalTokenDANN(nn.Module):
    """Temporal token 기반 멀티모달 DANN.

    forward(x_emg, x_imu, alpha) -> (class_logits, domain_logits)
    get_features(x_emg, x_imu)   -> (B, 512)
    """

    def __init__(
        self,
        num_classes=10,
        emg_channels=2,
        imu_channels=3,
        d_model=128,
        n_heads=4,
        num_cross_layers=3,
        emg_stride=10,
        imu_stride=1,
        dropout=0.2,
    ):
        super().__init__()

        self.emg_encoder = ConvTokenEncoder(
            in_channels=emg_channels,
            d_model=d_model,
            token_stride=emg_stride,
            depth=3,
            dropout=dropout,
        )
        self.imu_encoder = ConvTokenEncoder(
            in_channels=imu_channels,
            d_model=d_model,
            token_stride=imu_stride,
            depth=3,
            dropout=dropout,
        )

        self.emg_type = nn.Parameter(torch.zeros(1, 1, d_model))
        self.imu_type = nn.Parameter(torch.zeros(1, 1, d_model))

        self.cross_blocks = nn.ModuleList([
            CrossModalBlock(d_model=d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(num_cross_layers)
        ])

        self.emg_pool = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )
        self.imu_pool = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

        feature_dim = d_model * 4

        self.fusion = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.label_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

        nn.init.normal_(self.emg_type, std=0.02)
        nn.init.normal_(self.imu_type, std=0.02)

    def attentive_pool(self, tokens, pooler):
        score = pooler(tokens).squeeze(-1)
        weight = torch.softmax(score, dim=1).unsqueeze(-1)
        return (tokens * weight).sum(dim=1)

    def get_features(self, x_emg, x_imu):
        """EMG/IMU token 을 cross-attention 으로 융합해 (B,512) feature 를 반환한다."""
        emg = self.emg_encoder(x_emg) + self.emg_type
        imu = self.imu_encoder(x_imu) + self.imu_type

        for block in self.cross_blocks:
            emg, imu = block(emg, imu)

        emg_attn = self.attentive_pool(emg, self.emg_pool)
        imu_attn = self.attentive_pool(imu, self.imu_pool)
        emg_mean = emg.mean(dim=1)
        imu_mean = imu.mean(dim=1)

        features = torch.cat([emg_attn, imu_attn, emg_mean, imu_mean], dim=1)
        return self.fusion(features)

    def forward(self, x_emg, x_imu, alpha=1.0):
        features = self.get_features(x_emg, x_imu)
        class_output = self.label_classifier(features)
        domain_output = self.domain_classifier(ReverseLayerF.apply(features, alpha))
        return class_output, domain_output


# ----------------------------------------------------------------------
# 단독 실행 테스트 (출력 shape 검증용)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    model = TemporalTokenDANN(num_classes=10)
    emg = torch.randn(32, 2, 5000)
    imu = torch.randn(32, 3, 500)
    cls_out, dom_out = model(emg, imu, alpha=0.5)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Label Classifier Output    : {cls_out.shape}  (expected [32, 10])")
    print(f"Domain Discriminator Output: {dom_out.shape}  (expected [32, 2])")
    print(f"Total params               : {total:,}")
