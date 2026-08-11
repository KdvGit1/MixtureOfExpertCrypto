"""Canonical multi-task MoE implementation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from market_moe.models.experts import LocalCNNExpert, RegimeTransformerExpert, TrendGRUExpert
from market_moe.models.router import ExpertRouter


@dataclass(slots=True)
class MultiTaskOutput:
    expected_return: torch.Tensor
    direction_logit: torch.Tensor
    volatility: torch.Tensor
    uncertainty: torch.Tensor
    expert_weights: torch.Tensor
    auxiliary_returns: torch.Tensor


class MultiTaskMoE(nn.Module):
    expert_names = ("local", "trend", "regime")

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 64,
        router_hidden_dim: int = 32,
        dropout: float = 0.2,
        branch_dropout_probability: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.branch_dropout_probability = branch_dropout_probability
        self.experts = nn.ModuleList(
            [
                LocalCNNExpert(input_dim, embed_dim, dropout),
                TrendGRUExpert(input_dim, embed_dim, dropout),
                RegimeTransformerExpert(input_dim, embed_dim, dropout),
            ]
        )
        self.router = ExpertRouter(input_dim, router_hidden_dim, len(self.experts))
        self.return_heads = nn.ModuleList([nn.Linear(embed_dim, 1) for _ in self.experts])
        self.shared = nn.Sequential(nn.LayerNorm(embed_dim), nn.Dropout(dropout))
        self.return_head = nn.Linear(embed_dim, 1)
        self.direction_head = nn.Linear(embed_dim, 1)
        self.volatility_head = nn.Sequential(nn.Linear(embed_dim, 1), nn.Softplus())
        self.uncertainty_head = nn.Sequential(nn.Linear(embed_dim, 1), nn.Softplus())

    def _drop_branches(self, weights: torch.Tensor) -> torch.Tensor:
        if not self.training or self.branch_dropout_probability <= 0:
            return weights
        keep = torch.rand_like(weights) >= self.branch_dropout_probability
        all_dropped = ~keep.any(dim=1)
        if all_dropped.any():
            keep[all_dropped, weights[all_dropped].argmax(dim=1)] = True
        masked = weights * keep
        return masked / masked.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def forward(self, inputs: torch.Tensor) -> MultiTaskOutput:
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_dim:
            raise ValueError(f"expected [batch, time, {self.input_dim}] input")
        embeddings = torch.stack([expert(inputs) for expert in self.experts], dim=1)
        weights = self._drop_branches(self.router(inputs))
        mixed = self.shared((embeddings * weights.unsqueeze(-1)).sum(dim=1))
        auxiliary = torch.cat(
            [head(embeddings[:, index]) for index, head in enumerate(self.return_heads)], dim=1
        )
        return MultiTaskOutput(
            expected_return=self.return_head(mixed).squeeze(-1),
            direction_logit=self.direction_head(mixed).squeeze(-1),
            volatility=self.volatility_head(mixed).squeeze(-1),
            uncertainty=self.uncertainty_head(mixed).squeeze(-1),
            expert_weights=weights,
            auxiliary_returns=auxiliary,
        )

    def config(self) -> dict[str, float | int]:
        first_router_layer = self.router.network[1]
        dropout_layer = self.shared[1]
        if not isinstance(first_router_layer, nn.Linear) or not isinstance(
            dropout_layer, nn.Dropout
        ):
            raise TypeError("unexpected model layer configuration")
        return {
            "input_dim": self.input_dim,
            "embed_dim": self.embed_dim,
            "router_hidden_dim": first_router_layer.out_features,
            "dropout": dropout_layer.p,
            "branch_dropout_probability": self.branch_dropout_probability,
        }
