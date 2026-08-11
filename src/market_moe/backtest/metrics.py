"""Gross/net performance and trade statistics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def performance_metrics(
    equity: pd.Series,
    *,
    periods_per_year: int,
    trades: pd.DataFrame,
    benchmark: pd.Series | None = None,
) -> dict[str, float]:
    returns = equity.pct_change().fillna(0.0)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 1 else 0.0
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    maximum_drawdown = float(drawdown.min())
    annualized_volatility = float(returns.std(ddof=0) * math.sqrt(periods_per_year))
    sharpe = (
        float(returns.mean() / returns.std(ddof=0) * math.sqrt(periods_per_year))
        if returns.std(ddof=0) > 0
        else 0.0
    )
    downside = returns.clip(upper=0).std(ddof=0)
    sortino = (
        float(returns.mean() / downside * math.sqrt(periods_per_year)) if downside > 0 else 0.0
    )
    years = max(len(equity) / periods_per_year, 1 / periods_per_year)
    cagr = float((1 + total_return) ** (1 / years) - 1) if 1 + total_return > 0 else -1.0
    calmar = cagr / abs(maximum_drawdown) if maximum_drawdown < 0 else 0.0
    pnl = trades["net_pnl"] if not trades.empty and "net_pnl" in trades else pd.Series(dtype=float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    profit_factor = float(wins.sum() / abs(losses.sum())) if losses.sum() < 0 else float("inf")
    metrics = {
        "net_return": total_return,
        "cagr": cagr,
        "annualized_volatility": annualized_volatility,
        "maximum_drawdown": maximum_drawdown,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": float(calmar),
        "trade_count": float(len(pnl)),
        "win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
        "profit_factor": profit_factor,
        "expectancy": float(pnl.mean()) if len(pnl) else 0.0,
    }
    if benchmark is not None and len(benchmark) > 1:
        benchmark_return = float(benchmark.iloc[-1] / benchmark.iloc[0] - 1)
        metrics["benchmark_return"] = benchmark_return
        metrics["excess_return"] = total_return - benchmark_return
    return metrics


def bootstrap_return_interval(
    returns: pd.Series, *, samples: int = 1_000, seed: int = 20260811
) -> tuple[float, float]:
    values = returns.dropna().to_numpy(dtype=float)
    if values.size == 0:
        return 0.0, 0.0
    generator = np.random.default_rng(seed)
    paths = generator.choice(values, size=(samples, len(values)), replace=True)
    totals = np.prod(1 + paths, axis=1) - 1
    return float(np.quantile(totals, 0.025)), float(np.quantile(totals, 0.975))
