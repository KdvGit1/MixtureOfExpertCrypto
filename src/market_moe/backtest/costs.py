"""Explicit commission, spread, slippage and carrying-cost models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CostModel:
    commission_bps: float
    spread_bps: float
    slippage_bps: float
    funding_bps_per_bar: float = 0.0
    borrow_bps_per_bar: float = 0.0

    def __post_init__(self) -> None:
        if any(
            value < 0
            for value in (
                self.commission_bps,
                self.spread_bps,
                self.slippage_bps,
                self.funding_bps_per_bar,
                self.borrow_bps_per_bar,
            )
        ):
            raise ValueError("cost components cannot be negative")

    @property
    def idealized(self) -> bool:
        return (
            self.commission_bps
            + self.spread_bps
            + self.slippage_bps
            + self.funding_bps_per_bar
            + self.borrow_bps_per_bar
            == 0
        )

    def transaction_cost(self, notional: float) -> dict[str, float]:
        absolute = abs(notional)
        return {
            "commission": absolute * self.commission_bps / 10_000,
            "spread": absolute * self.spread_bps / 10_000,
            "slippage": absolute * self.slippage_bps / 10_000,
        }

    def carrying_cost(self, notional: float, position: float) -> float:
        rate = self.funding_bps_per_bar
        if position < 0:
            rate += self.borrow_bps_per_bar
        return abs(notional) * rate / 10_000
