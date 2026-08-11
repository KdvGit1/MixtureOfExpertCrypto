"""Canonical OHLCV bar contract."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SessionType(StrEnum):
    REGULAR = "regular"
    PREMARKET = "premarket"
    AFTER_HOURS = "after_hours"
    CONTINUOUS = "continuous"
    UNKNOWN = "unknown"


class DataQualityFlag(StrEnum):
    INCOMPLETE = "incomplete"
    MISSING_VOLUME = "missing_volume"
    DUPLICATE_TIMESTAMP = "duplicate_timestamp"
    CALENDAR_MISMATCH = "calendar_mismatch"
    PRICE_INVARIANT = "price_invariant"
    STALE = "stale"
    FALLBACK_PROVIDER = "fallback_provider"
    SURVIVORSHIP_BIAS = "survivorship_bias"


class Bar(BaseModel):
    """One canonical market bar with explicit provenance."""

    model_config = ConfigDict(frozen=True)

    instrument_id: str
    timeframe: str
    open_time_utc: datetime
    close_time_utc: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)
    vwap: float | None = Field(default=None, gt=0)
    currency: str
    session_type: SessionType
    is_adjusted: bool
    provider: str
    provider_symbol: str
    ingested_at_utc: datetime
    quality_flags: tuple[DataQualityFlag, ...] = ()

    @model_validator(mode="after")
    def validate_bar(self) -> Bar:
        if self.open_time_utc.tzinfo is None or self.close_time_utc.tzinfo is None:
            raise ValueError("bar timestamps must be timezone-aware")
        if self.close_time_utc <= self.open_time_utc:
            raise ValueError("close_time_utc must be after open_time_utc")
        if self.low > min(self.open, self.close) or self.high < max(self.open, self.close):
            raise ValueError("OHLC price invariant failed")
        if self.high < self.low:
            raise ValueError("high must be greater than or equal to low")
        return self
