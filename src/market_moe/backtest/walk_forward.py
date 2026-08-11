"""Aggregate independently fitted walk-forward simulation folds."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from market_moe.backtest.engine import BacktestConfig, BacktestResult, run_backtest


def run_walk_forward_backtests(
    folds: list[pd.DatetimeIndex],
    bars: pd.DataFrame,
    signal_factory: Callable[[pd.DataFrame], pd.Series],
    config: BacktestConfig,
) -> list[BacktestResult]:
    results = []
    indexed = bars.set_index(pd.to_datetime(bars["open_time_utc"], utc=True), drop=False)
    for validation_index in folds:
        fold_bars = indexed.loc[indexed.index.intersection(validation_index)].reset_index(drop=True)
        results.append(run_backtest(fold_bars, signal_factory(fold_bars), config))
    return results
