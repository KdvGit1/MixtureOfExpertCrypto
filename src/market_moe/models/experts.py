"""The three temporal experts used by both model families."""

from __future__ import annotations

import torch
from torch import nn


class LocalCNNExpert(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs.transpose(1, 2))


class TrendGRUExpert(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, embed_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(inputs)
        return self.dropout(hidden[-1])


class RegimeTransformerExpert(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, dropout: float) -> None:
        super().__init__()
        heads = next(head for head in (8, 4, 2, 1) if embed_dim % head == 0)
        self.projection = nn.Linear(input_dim, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.normalization = nn.LayerNorm(embed_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.projection(inputs))
        return self.normalization(encoded.mean(dim=1))
