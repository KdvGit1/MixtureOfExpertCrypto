"""Pure pandas/numpy technical indicators used instead of native TA-Lib."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    alpha = 1.0 / period
    average_gain = gains.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    average_loss = losses.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    relative_strength = average_gain / average_loss.replace(0.0, np.nan)
    value = 100.0 - (100.0 / (1.0 + relative_strength))
    value = value.where(average_loss != 0.0, 100.0)
    value = value.where(average_gain != 0.0, 0.0)
    return value


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    previous_close = close.shift(1)
    components = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    )
    return components.max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return (
        true_range(high, low, close)
        .ewm(alpha=1.0 / period, adjust=False, min_periods=period)
        .mean()
    )


def bollinger_bands(
    series: pd.Series, period: int = 20, deviations: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    middle = sma(series, period)
    standard_deviation = series.rolling(period, min_periods=period).std(ddof=0)
    upper = middle + deviations * standard_deviation
    lower = middle - deviations * standard_deviation
    return upper, middle, lower


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast_line = ema(series, fast)
    slow_line = ema(series, slow)
    macd_line = fast_line - slow_line
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def rolling_slope(series: pd.Series, period: int) -> pd.Series:
    x = np.arange(period, dtype=float)
    x_centered = x - x.mean()
    denominator = float(np.sum(x_centered**2))

    def slope(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        centered = values - values.mean()
        return float(np.sum(x_centered * centered) / denominator)

    return series.rolling(period, min_periods=period).apply(slope, raw=True)
