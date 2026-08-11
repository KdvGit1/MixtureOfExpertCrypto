"""Market data provider protocol and canonical frame helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol, runtime_checkable

import pandas as pd

from market_moe.domain.instruments import Instrument

CANONICAL_BAR_COLUMNS = (
    "instrument_id",
    "timeframe",
    "open_time_utc",
    "close_time_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "currency",
    "session_type",
    "is_adjusted",
    "provider",
    "provider_symbol",
    "ingested_at_utc",
    "quality_flags",
)

TIMEFRAME_DELTAS: dict[str, timedelta] = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "1d": timedelta(days=1),
    "1w": timedelta(weeks=1),
}


def timeframe_delta(timeframe: str) -> timedelta:
    try:
        return TIMEFRAME_DELTAS[timeframe]
    except KeyError as exc:
        raise ValueError(f"unsupported timeframe: {timeframe}") from exc


def utc_now() -> datetime:
    return datetime.now(UTC)


def empty_bar_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=CANONICAL_BAR_COLUMNS)


@runtime_checkable
class MarketDataProvider(Protocol):
    name: str
    supported_timeframes: frozenset[str]

    def fetch_bars(
        self,
        instrument: Instrument,
        timeframe: str,
        start: datetime,
        end: datetime,
        *,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        """Return a canonical OHLCV DataFrame sorted by open_time_utc."""

    def healthcheck(self) -> dict[str, object]:
        """Return a non-secret provider health summary."""
