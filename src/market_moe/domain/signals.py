"""Strategy output kept separate from model predictions."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class SignalAction(StrEnum):
    STRONG_LONG = "strong_long"
    LONG = "long"
    NEUTRAL = "neutral"
    REDUCE = "reduce"
    SHORT = "short"
    STRONG_SHORT = "strong_short"


class Signal(BaseModel):
    model_config = ConfigDict(frozen=True)

    instrument_id: str
    as_of_utc: datetime
    action: SignalAction
    score: float = Field(ge=-1, le=1)
    reason_codes: tuple[str, ...]
    expected_edge_after_cost: float
    risk_level: str
    prediction_id: str
    strategy_version: str
