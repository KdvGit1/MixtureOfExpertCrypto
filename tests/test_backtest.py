from __future__ import annotations

import pandas as pd
import pytest

from market_moe.backtest.costs import CostModel
from market_moe.backtest.engine import BacktestConfig, run_backtest
from market_moe.backtest.reports import write_backtest_report
from market_moe.domain.errors import BacktestError


def test_signal_executes_at_next_open_and_costs_are_reported(crypto_bars, tmp_path) -> None:
    bars = crypto_bars.iloc[:12].copy()
    signals = pd.Series(0.0, index=bars["open_time_utc"])
    signals.iloc[0] = 1.0
    signals.iloc[1:4] = 1.0
    result = run_backtest(
        bars,
        signals,
        BacktestConfig(costs=CostModel(5, 5, 2), periods_per_year=365),
    )
    assert result.equity.iloc[0]["position"] == 0
    assert result.equity.iloc[1]["position"] > 0
    assert result.metrics["execution_policy"] == "signal_at_close_execute_next_open"
    assert result.metrics["total_cost"] > 0
    paths = write_backtest_report(result, tmp_path)
    assert all(path.exists() for path in paths.values())


def test_no_cost_run_is_explicitly_idealized(crypto_bars) -> None:
    bars = crypto_bars.iloc[:8].copy()
    signals = pd.Series(1.0, index=bars["open_time_utc"])
    result = run_backtest(bars, signals, BacktestConfig(costs=CostModel(0, 0, 0)))
    assert result.metrics["idealized_no_cost"] is True
    assert "idealized_no_cost_not_for_production" in result.warnings


def test_fx_and_explicit_corporate_action_accounting(equity_bars) -> None:
    bars = equity_bars.iloc[:10].copy()
    bars["currency"] = "EUR"
    bars["is_adjusted"] = False
    signals = pd.Series(1.0, index=bars["open_time_utc"])
    with pytest.raises(BacktestError):
        run_backtest(bars, signals, BacktestConfig(base_currency="USD"))
    fx = pd.Series(1.1, index=bars["open_time_utc"])
    actions = pd.DataFrame(
        {
            "effective_at_utc": [bars["open_time_utc"].iloc[4]],
            "action_type": ["split"],
            "value": [2.0],
        }
    )
    result = run_backtest(
        bars,
        signals,
        BacktestConfig(base_currency="USD", corporate_action_mode="explicit"),
        fx_rates=fx,
        corporate_actions=actions,
    )
    assert (result.equity["fx_rate"] == 1.1).all()
