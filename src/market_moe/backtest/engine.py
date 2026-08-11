"""Single-instrument event engine with strict t+1 execution."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from market_moe.backtest.costs import CostModel
from market_moe.backtest.execution import stop_target_exit
from market_moe.backtest.metrics import bootstrap_return_interval, performance_metrics
from market_moe.backtest.risk import RiskLimits
from market_moe.data.quality import canonicalize_bar_frame, validate_bar_frame
from market_moe.domain.errors import BacktestError


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    initial_cash: float = 100_000.0
    base_currency: str = "USD"
    periods_per_year: int = 252
    costs: CostModel = field(default_factory=lambda: CostModel(5.0, 5.0, 2.0))
    risk: RiskLimits = field(default_factory=RiskLimits)
    stop_loss_fraction: float | None = None
    take_profit_fraction: float | None = None
    corporate_action_mode: str = "provider_adjusted"


@dataclass(slots=True)
class BacktestResult:
    equity: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float | bool | str]
    warnings: list[str]


def run_backtest(
    bars: pd.DataFrame,
    signals: pd.Series,
    config: BacktestConfig | None = None,
    *,
    fx_rates: pd.Series | None = None,
    corporate_actions: pd.DataFrame | None = None,
) -> BacktestResult:
    """Simulate target directions where signal at t executes at open(t+1)."""

    config = config or BacktestConfig()
    frame = canonicalize_bar_frame(bars).set_index("open_time_utc", drop=False)
    validate_bar_frame(frame.reset_index(drop=True), raise_on_error=True)
    signals = signals.reindex(frame.index).fillna(0.0).clip(-1.0, 1.0)
    currencies = set(frame["currency"].astype(str).str.upper())
    if len(currencies) != 1:
        raise BacktestError("one backtest stream must use a single quote currency")
    quote_currency = currencies.pop()
    stable_usd = config.base_currency == "USD" and quote_currency in {"USD", "USDT", "USDC"}
    if quote_currency == config.base_currency or stable_usd:
        rates = pd.Series(1.0, index=frame.index)
    elif fx_rates is None:
        raise BacktestError(
            f"historical FX rates required for {quote_currency}/{config.base_currency}"
        )
    else:
        rates = fx_rates.reindex(frame.index).ffill()
        if rates.isna().any() or (rates <= 0).any():
            raise BacktestError("FX series is missing or invalid for part of the backtest")
    actions = pd.DataFrame()
    if corporate_actions is not None and not corporate_actions.empty:
        if config.corporate_action_mode != "explicit":
            raise BacktestError("corporate actions supplied but mode is not explicit")
        if frame["is_adjusted"].astype(bool).any():
            raise BacktestError("explicit corporate actions cannot be combined with adjusted bars")
        actions = corporate_actions.copy()
        actions["effective_at_utc"] = pd.to_datetime(actions["effective_at_utc"], utc=True)
        actions = actions.sort_values("effective_at_utc")
    cash = config.initial_cash
    quantity = 0.0
    entry_price = 0.0
    entry_time = None
    entry_cost = 0.0
    peak = config.initial_cash
    records: list[dict[str, object]] = []
    trades: list[dict[str, object]] = []
    ambiguous = 0
    previous_close: float | None = None
    entry_fx_rate = 1.0
    previous_timestamp: pd.Timestamp | None = None

    for offset, (raw_timestamp, bar) in enumerate(frame.iterrows()):
        timestamp = pd.Timestamp(str(raw_timestamp))
        open_price = float(bar["open"])
        close_price = float(bar["close"])
        fx_rate = float(rates.loc[timestamp])
        if not actions.empty:
            effective = actions[actions["effective_at_utc"] <= timestamp]
            if previous_timestamp is not None:
                effective = effective[effective["effective_at_utc"] > previous_timestamp]
            for _action_index, action in effective.iterrows():
                if action["action_type"] == "split" and quantity != 0:
                    ratio = float(action["value"])
                    quantity *= ratio
                    entry_price /= ratio
                elif action["action_type"] == "dividend" and quantity > 0:
                    cash += quantity * float(action["value"]) * fx_rate
        if (
            previous_close
            and abs(open_price / previous_close - 1) > config.risk.maximum_gap_fraction
        ):
            requested = 0.0
        else:
            requested = float(signals.iloc[offset - 1]) if offset > 0 else 0.0
        if not config.risk.allow_short:
            requested = max(0.0, requested)
        current_direction = float(np.sign(quantity))
        equity_at_open = cash + quantity * open_price * fx_rate
        drawdown = equity_at_open / peak - 1
        if drawdown <= -config.risk.maximum_drawdown:
            requested = 0.0

        exit_decision = stop_target_exit(
            position=quantity,
            entry_price=entry_price,
            high=float(bar["high"]),
            low=float(bar["low"]),
            stop_loss_fraction=config.stop_loss_fraction,
            take_profit_fraction=config.take_profit_fraction,
        )
        forced_exit = exit_decision.reason is not None
        if forced_exit:
            requested = 0.0
            if exit_decision.price is None:
                raise RuntimeError("exit decision has no price")
            open_price = exit_decision.price
            ambiguous += int(exit_decision.intrabar_ambiguous)

        if requested != current_direction:
            if quantity != 0:
                notional = quantity * open_price * fx_rate
                components = config.costs.transaction_cost(notional)
                exit_cost = sum(components.values())
                cash += notional - exit_cost
                gross_pnl = quantity * (open_price * fx_rate - entry_price * entry_fx_rate)
                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": timestamp,
                        "direction": current_direction,
                        "quantity": abs(quantity),
                        "entry_price": entry_price,
                        "entry_fx_rate": entry_fx_rate,
                        "exit_price": open_price,
                        "exit_fx_rate": fx_rate,
                        "gross_pnl": gross_pnl,
                        "cost": entry_cost + exit_cost,
                        "net_pnl": gross_pnl - entry_cost - exit_cost,
                        "exit_reason": exit_decision.reason or "signal",
                    }
                )
                quantity = 0.0
            if requested != 0:
                allocation = cash * config.risk.maximum_position_fraction
                quantity = requested * allocation / (open_price * fx_rate)
                components = config.costs.transaction_cost(allocation)
                entry_cost = sum(components.values())
                cash -= quantity * open_price * fx_rate + entry_cost
                entry_price = open_price
                entry_fx_rate = fx_rate
                entry_time = timestamp

        carrying = config.costs.carrying_cost(quantity * close_price * fx_rate, quantity)
        cash -= carrying
        equity = cash + quantity * close_price * fx_rate
        peak = max(peak, equity)
        records.append(
            {
                "timestamp": timestamp,
                "cash": cash,
                "position": quantity,
                "close": close_price,
                "fx_rate": fx_rate,
                "equity": equity,
                "gross_exposure": abs(quantity * close_price),
                "carrying_cost": carrying,
            }
        )
        previous_close = close_price
        previous_timestamp = timestamp

    equity_frame = pd.DataFrame(records).set_index("timestamp")
    trades_frame = pd.DataFrame(trades)
    benchmark_value = frame["close"].astype(float) * rates
    benchmark = benchmark_value / benchmark_value.iloc[0] * config.initial_cash
    metrics: dict[str, float | bool | str] = {
        key: value
        for key, value in performance_metrics(
            equity_frame["equity"],
            periods_per_year=config.periods_per_year,
            trades=trades_frame,
            benchmark=benchmark,
        ).items()
    }
    interval = bootstrap_return_interval(equity_frame["equity"].pct_change())
    metrics.update(
        {
            "gross_return": float(
                (equity_frame["equity"] + equity_frame["carrying_cost"].cumsum()).iloc[-1]
                / config.initial_cash
                - 1
            ),
            "total_cost": float(
                (trades_frame["cost"].sum() if not trades_frame.empty else 0.0)
                + equity_frame["carrying_cost"].sum()
            ),
            "turnover": float(
                trades_frame["quantity"]
                .mul(trades_frame["entry_price"])
                .mul(trades_frame["entry_fx_rate"])
                .sum()
                / config.initial_cash
                if not trades_frame.empty
                else 0.0
            ),
            "exposure": float((equity_frame["gross_exposure"] > 0).mean()),
            "bootstrap_return_ci_low": interval[0],
            "bootstrap_return_ci_high": interval[1],
            "intrabar_ambiguous": float(ambiguous),
            "idealized_no_cost": config.costs.idealized,
            "execution_policy": "signal_at_close_execute_next_open",
        }
    )
    warnings = ["idealized_no_cost_not_for_production"] if config.costs.idealized else []
    if stable_usd and quote_currency != "USD":
        warnings.append("stablecoin_usd_parity_assumed")
    return BacktestResult(equity_frame, trades_frame, metrics, warnings)
