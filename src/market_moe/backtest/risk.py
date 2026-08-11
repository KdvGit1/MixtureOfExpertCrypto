"""Simulation risk limits; defaults are not investment recommendations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RiskLimits:
    maximum_position_fraction: float = 1.0
    maximum_drawdown: float = 0.5
    maximum_gap_fraction: float = 0.25
    allow_short: bool = False

    def __post_init__(self) -> None:
        if not 0 < self.maximum_position_fraction <= 1:
            raise ValueError("maximum position fraction must be in (0, 1]")
        if not 0 < self.maximum_drawdown <= 1:
            raise ValueError("maximum drawdown must be in (0, 1]")
