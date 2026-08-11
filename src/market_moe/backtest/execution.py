"""Conservative next-bar simulated execution."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExitDecision:
    reason: str | None
    price: float | None
    intrabar_ambiguous: bool = False


def stop_target_exit(
    *,
    position: float,
    entry_price: float,
    high: float,
    low: float,
    stop_loss_fraction: float | None,
    take_profit_fraction: float | None,
) -> ExitDecision:
    if position == 0 or (stop_loss_fraction is None and take_profit_fraction is None):
        return ExitDecision(None, None)
    if position > 0:
        stop = entry_price * (1 - stop_loss_fraction) if stop_loss_fraction else None
        target = entry_price * (1 + take_profit_fraction) if take_profit_fraction else None
        stop_hit = stop is not None and low <= stop
        target_hit = target is not None and high >= target
    else:
        stop = entry_price * (1 + stop_loss_fraction) if stop_loss_fraction else None
        target = entry_price * (1 - take_profit_fraction) if take_profit_fraction else None
        stop_hit = stop is not None and high >= stop
        target_hit = target is not None and low <= target
    if stop_hit:
        return ExitDecision("stop_loss", stop, target_hit)
    if target_hit:
        return ExitDecision("take_profit", target)
    return ExitDecision(None, None)
