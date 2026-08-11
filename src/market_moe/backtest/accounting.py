"""Position, FX and corporate-action accounting helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Account:
    base_currency: str
    cash: float
    quantity: float = 0.0
    realized_pnl: float = 0.0

    def equity(self, price: float, fx_rate: float = 1.0) -> float:
        if fx_rate <= 0:
            raise ValueError("a positive historical FX rate is required")
        return self.cash + self.quantity * price * fx_rate

    def apply_split(self, ratio: float) -> None:
        if ratio <= 0:
            raise ValueError("split ratio must be positive")
        self.quantity *= ratio

    def apply_dividend(self, amount_per_share: float, fx_rate: float = 1.0) -> None:
        self.cash += self.quantity * amount_per_share * fx_rate
