"""Multi-task and router regularization losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as functional

from market_moe.models.moe import MultiTaskOutput


@dataclass(frozen=True, slots=True)
class LossWeights:
    regression: float = 1.0
    direction: float = 0.5
    volatility: float = 0.25
    auxiliary: float = 0.15
    load_balance: float = 0.02


DEFAULT_LOSS_WEIGHTS = LossWeights()


def multitask_loss(
    output: MultiTaskOutput,
    target_return: torch.Tensor,
    target_direction: torch.Tensor,
    target_volatility: torch.Tensor,
    weights: LossWeights = DEFAULT_LOSS_WEIGHTS,
) -> tuple[torch.Tensor, dict[str, float]]:
    regression = functional.huber_loss(output.expected_return, target_return)
    direction = functional.binary_cross_entropy_with_logits(
        output.direction_logit, target_direction
    )
    volatility = functional.huber_loss(output.volatility, target_volatility)
    auxiliary = functional.huber_loss(
        output.auxiliary_returns,
        target_return.unsqueeze(1).expand_as(output.auxiliary_returns),
    )
    desired = torch.full_like(output.expert_weights.mean(dim=0), 1 / 3)
    load_balance = functional.mse_loss(output.expert_weights.mean(dim=0), desired)
    total = (
        weights.regression * regression
        + weights.direction * direction
        + weights.volatility * volatility
        + weights.auxiliary * auxiliary
        + weights.load_balance * load_balance
    )
    components = {
        "regression": float(regression.detach()),
        "direction": float(direction.detach()),
        "volatility": float(volatility.detach()),
        "auxiliary": float(auxiliary.detach()),
        "load_balance": float(load_balance.detach()),
        "total": float(total.detach()),
    }
    return total, components
