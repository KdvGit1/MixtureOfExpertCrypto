"""Canonical OHLCV validation and transparent quality reporting."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from market_moe.data.protocols import CANONICAL_BAR_COLUMNS
from market_moe.domain.errors import DataQualityError


class DataQualityReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    valid: bool
    row_count: int = Field(ge=0)
    duplicate_timestamps: int = Field(ge=0)
    missing_values: int = Field(ge=0)
    invalid_prices: int = Field(ge=0)
    missing_volume_rows: int = Field(ge=0)
    stale: bool
    start_utc: datetime | None = None
    end_utc: datetime | None = None
    warnings: tuple[str, ...] = ()


def canonicalize_bar_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in CANONICAL_BAR_COLUMNS if column not in frame.columns]
    if missing:
        raise DataQualityError(f"canonical bar columns missing: {missing}")

    result = frame[list(CANONICAL_BAR_COLUMNS)].copy()
    for column in ("open_time_utc", "close_time_utc", "ingested_at_utc"):
        result[column] = pd.to_datetime(result[column], utc=True, errors="coerce")
    for column in ("open", "high", "low", "close", "volume", "vwap"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.sort_values("open_time_utc", kind="stable").reset_index(drop=True)
    return result


def validate_bar_frame(
    frame: pd.DataFrame,
    *,
    stale_after_seconds: float | None = None,
    now: datetime | None = None,
    raise_on_error: bool = False,
) -> DataQualityReport:
    result = canonicalize_bar_frame(frame)
    row_count = len(result)
    duplicate_timestamps = int(result["open_time_utc"].duplicated().sum())
    required = ["open_time_utc", "close_time_utc", "open", "high", "low", "close"]
    missing_values = int(result[required].isna().sum().sum())
    missing_volume_rows = int(result["volume"].isna().sum())

    if row_count:
        price_matrix = result[["open", "high", "low", "close"]]
        nonpositive = (price_matrix <= 0).any(axis=1)
        high_invalid = result["high"] < price_matrix[["open", "close"]].max(axis=1)
        low_invalid = result["low"] > price_matrix[["open", "close"]].min(axis=1)
        range_invalid = result["high"] < result["low"]
        invalid_prices = int((nonpositive | high_invalid | low_invalid | range_invalid).sum())
        start_utc = result["open_time_utc"].iloc[0].to_pydatetime()
        end_utc = result["close_time_utc"].iloc[-1].to_pydatetime()
    else:
        invalid_prices = 0
        start_utc = None
        end_utc = None

    stale = False
    if end_utc and stale_after_seconds is not None:
        reference = now or datetime.now(UTC)
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=UTC)
        stale = (reference - end_utc).total_seconds() > stale_after_seconds

    warnings: list[str] = []
    if row_count == 0:
        warnings.append("empty_dataset")
    if duplicate_timestamps:
        warnings.append("duplicate_timestamps")
    if missing_values:
        warnings.append("missing_required_values")
    if missing_volume_rows:
        warnings.append("missing_volume")
    if invalid_prices:
        warnings.append("invalid_ohlc")
    if stale:
        warnings.append("stale_data")

    valid = row_count > 0 and not duplicate_timestamps and not missing_values and not invalid_prices
    report = DataQualityReport(
        valid=valid,
        row_count=row_count,
        duplicate_timestamps=duplicate_timestamps,
        missing_values=missing_values,
        invalid_prices=invalid_prices,
        missing_volume_rows=missing_volume_rows,
        stale=stale,
        start_utc=start_utc,
        end_utc=end_utc,
        warnings=tuple(warnings),
    )
    if raise_on_error and not report.valid:
        raise DataQualityError(
            "invalid canonical bars: " + ", ".join(report.warnings or ("unknown",))
        )
    return report


def deduplicate_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the most recently ingested copy of each provider bar."""

    result = canonicalize_bar_frame(frame)
    result = result.sort_values(["open_time_utc", "ingested_at_utc"])
    result = result.drop_duplicates("open_time_utc", keep="last")
    result["volume"] = result["volume"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return result.sort_values("open_time_utc").reset_index(drop=True)
