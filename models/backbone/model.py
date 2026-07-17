"""
LightSeek — CNN-Transformer Backbone (W3)
Author: Prajwal K / Team Integral X
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class LightSeekBackbone(nn.Module):
    """
    1D CNN + Transformer encoder for exoplanet transit detection.
    Input:  (batch, 1, 1000) — normalized flux array
    Output: (batch, 1)       — planet probability
    """

    def __init__(self,
                 d_model: int = 64,
                 n_heads: int = 4,
                 n_transformer_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()

        # CNN backbone — extract local transit features
        self.cnn = nn.Sequential(
            # Block 1: 1000 → 500
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 2: 500 → 250
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 3: 250 → 125
            nn.Conv1d(32, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=200)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_transformer_layers
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # CNN: (batch, 1, 1000) → (batch, 64, 125)
        x = self.cnn(x)

        # Reshape for Transformer: (batch, 125, 64)
        x = x.permute(0, 2, 1)

        # Positional encoding
        x = self.pos_encoder(x)

        # Transformer: (batch, 125, 64)
        x = self.transformer(x)

        # Global average pool: (batch, 64)
        x = x.mean(dim=1)

        # Classify: (batch, 1)
        return self.classifier(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing LightSeek backbone...")
    model = LightSeekBackbone()
    print(f"Parameters: {count_parameters(model):,}")

    x = torch.randn(4, 1, 1000)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Sample outputs: {out.detach().squeeze().tolist()}")
    print("Model OK")