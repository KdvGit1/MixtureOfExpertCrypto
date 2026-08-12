"""Helpers shared by provider adapters."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from market_moe.data.protocols import CANONICAL_BAR_COLUMNS, timeframe_delta, utc_now
from market_moe.domain.instruments import AssetClass, Instrument


def sanitize_provider_ohlc(
    frame: pd.DataFrame,
    *,
    maximum_envelope_adjustment: float = 0.005,
    maximum_drop_ratio: float = 0.001,
) -> pd.DataFrame:
    """Repair harmless adjusted-price rounding and drop isolated corrupt bars.

    Some providers multiply adjusted OHLC columns independently.  That can leave
    ``high`` a few floating-point ulps below ``open``/``close`` or ``low`` a few
    ulps above them.  Rejecting a multi-decade series for those rows loses far
    more information than it protects.  Small envelope violations are expanded
    conservatively; isolated larger corrupt rows are removed and flagged.
    Widespread corruption remains untouched so the normal quality gate rejects it.
    """

    if frame.empty:
        return frame
    result = frame.copy()
    price_columns = ["open", "high", "low", "close"]
    prices = result[price_columns].apply(pd.to_numeric, errors="coerce")
    upper = prices[["open", "close"]].max(axis=1)
    lower = prices[["open", "close"]].min(axis=1)
    denominator = upper.abs().clip(lower=np.finfo(float).eps)
    high_gap = ((upper - prices["high"]) / denominator).clip(lower=0.0)
    low_gap = ((prices["low"] - lower) / denominator).clip(lower=0.0)
    range_gap = ((prices["low"] - prices["high"]) / denominator).clip(lower=0.0)
    relative_gap = pd.concat([high_gap, low_gap, range_gap], axis=1).max(axis=1)
    finite_positive = pd.Series(
        np.isfinite(prices.to_numpy()).all(axis=1) & (prices > 0).all(axis=1).to_numpy(),
        index=result.index,
    )
    invalid_envelope = (
        (prices["high"] < upper)
        | (prices["low"] > lower)
        | (prices["high"] < prices["low"])
    )
    repairable = finite_positive & invalid_envelope & (
        relative_gap <= maximum_envelope_adjustment
    )
    if repairable.any():
        result.loc[repairable, "high"] = prices.loc[repairable, ["open", "high", "close"]].max(
            axis=1
        )
        result.loc[repairable, "low"] = prices.loc[repairable, ["open", "low", "close"]].min(
            axis=1
        )
        existing = result.loc[repairable, "quality_flags"].fillna("").astype(str)
        result.loc[repairable, "quality_flags"] = existing.map(
            lambda value: ";".join(filter(None, (value, "ohlc_envelope_repaired")))
        )

    remaining_invalid = (~finite_positive) | (invalid_envelope & ~repairable)
    invalid_count = int(remaining_invalid.sum())
    maximum_drops = max(1, int(len(result) * maximum_drop_ratio))
    if 0 < invalid_count <= maximum_drops:
        result = result.loc[~remaining_invalid].copy()
        if not result.empty:
            first = result.index[0]
            raw_flag = result.at[first, "quality_flags"]
            existing_flag = "" if pd.isna(raw_flag) else str(raw_flag)
            result.at[first, "quality_flags"] = ";".join(
                filter(None, (existing_flag, f"provider_invalid_rows_dropped={invalid_count}"))
            )
    return result.reset_index(drop=True)


def provider_frame_to_canonical(
    frame: pd.DataFrame,
    *,
    instrument: Instrument,
    timeframe: str,
    provider: str,
    provider_symbol: str,
    adjusted: bool,
    ingested_at: datetime | None = None,
    fallback: bool = False,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=CANONICAL_BAR_COLUMNS)

    result = frame.copy()
    result.columns = [str(column).strip().lower() for column in result.columns]
    if "open_time_utc" not in result.columns:
        result = result.reset_index()
        timestamp_column = result.columns[0]
        result = result.rename(columns={timestamp_column: "open_time_utc"})

    result["open_time_utc"] = pd.to_datetime(result["open_time_utc"], utc=True)
    result["close_time_utc"] = result["open_time_utc"] + timeframe_delta(timeframe)
    result["instrument_id"] = instrument.instrument_id
    result["timeframe"] = timeframe
    result["currency"] = instrument.currency
    result["session_type"] = (
        "continuous" if instrument.asset_class == AssetClass.CRYPTO else "regular"
    )
    result["is_adjusted"] = adjusted
    result["provider"] = provider
    result["provider_symbol"] = provider_symbol
    result["ingested_at_utc"] = ingested_at or utc_now()
    result["quality_flags"] = "fallback_provider" if fallback else ""
    if "volume" not in result:
        result["volume"] = 0.0
    if "vwap" not in result:
        result["vwap"] = None
    return result[list(CANONICAL_BAR_COLUMNS)].sort_values("open_time_utc").reset_index(drop=True)
