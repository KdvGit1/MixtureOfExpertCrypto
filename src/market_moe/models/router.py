"""Context router with transparent normalized expert weights."""

from __future__ import annotations

import torch
from torch import nn


class ExpertRouter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, expert_count: int = 3) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, expert_count),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.network(inputs[:, -1, :]), dim=-1)
