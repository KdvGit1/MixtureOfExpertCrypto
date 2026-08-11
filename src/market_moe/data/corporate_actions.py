"""Canonical corporate-action representation."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class CorporateActionType(StrEnum):
    SPLIT = "split"
    DIVIDEND = "dividend"
    SYMBOL_CHANGE = "symbol_change"


class CorporateAction(BaseModel):
    model_config = ConfigDict(frozen=True)

    instrument_id: str
    action_type: CorporateActionType
    effective_at_utc: datetime
    value: float = Field(gt=0)
    currency: str | None = None
    provider: str
    provider_reference: str | None = None
