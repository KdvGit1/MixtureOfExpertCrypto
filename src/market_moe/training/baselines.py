"""Transparent baselines required for model acceptance."""

from __future__ import annotations

import numpy as np
import pandas as pd


def baseline_predictions(frame: pd.DataFrame, *, seed: int = 20260811) -> pd.DataFrame:
    returns = frame["log_return_1"].astype(float)
    close = (
        frame["close"].astype(float)
        if "close" in frame
        else pd.Series(np.exp(returns.cumsum().to_numpy()), index=frame.index)
    )
    momentum = returns.shift(1).fillna(0.0)
    moving_average = close.rolling(20, min_periods=1).mean()
    trend = np.sign(close - moving_average) * returns.abs().rolling(20, min_periods=1).mean()
    volatility = returns.rolling(20, min_periods=2).std(ddof=0).replace(0.0, np.nan)
    scaled = momentum / volatility
    random = np.random.default_rng(seed).choice([-1.0, 1.0], size=len(frame)) * returns.abs().mean()
    return pd.DataFrame(
        {
            "zero": 0.0,
            "naive_momentum": momentum,
            "moving_average_trend": trend,
            "random_signal": random,
            "volatility_scaled_momentum": scaled.fillna(0.0),
        },
        index=frame.index,
    )
