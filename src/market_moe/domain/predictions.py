"""Versioned, unit-explicit prediction contract."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from market_moe.domain.instruments import AssetClass


class Prediction(BaseModel):
    model_config = ConfigDict(frozen=True)

    prediction_id: str
    instrument_id: str
    asset_class: AssetClass
    timeframe: str
    as_of_utc: datetime
    horizon: str
    target_type: str = "log_return"
    expected_log_return: float
    expected_return_pct: float
    probability_up: float = Field(ge=0, le=1)
    predicted_volatility: float = Field(ge=0)
    uncertainty: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    expert_weights: dict[str, float]
    raw_model_output: dict[str, Any] = Field(default_factory=dict)
    model_id: str
    model_version: str
    feature_schema_hash: str
    normalization_id: str
    data_freshness_seconds: float = Field(ge=0)
    warnings: tuple[str, ...] = ()

    @field_validator("as_of_utc")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("as_of_utc must be timezone-aware")
        return value

    @field_validator("expert_weights")
    @classmethod
    def validate_expert_weights(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("expert_weights cannot be empty")
        if any(weight < 0 or weight > 1 for weight in value.values()):
            raise ValueError("expert weights must be between 0 and 1")
        if abs(sum(value.values()) - 1.0) > 1e-5:
            raise ValueError("expert weights must sum to 1")
        return value
