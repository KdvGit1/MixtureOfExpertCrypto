"""Helpers shared by provider adapters."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from market_moe.data.protocols import CANONICAL_BAR_COLUMNS, timeframe_delta, utc_now
from market_moe.domain.instruments import AssetClass, Instrument


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
