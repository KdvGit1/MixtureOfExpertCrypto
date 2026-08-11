"""Instrument identity and provider-symbol mapping."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AssetClass(StrEnum):
    CRYPTO = "crypto"
    EQUITY = "equity"
    ETF = "etf"
    INDEX = "index"
    FX = "fx"


class Instrument(BaseModel):
    """Stable identity for a listing, not merely a display ticker."""

    model_config = ConfigDict(frozen=True)

    instrument_id: str
    symbol: str
    display_name: str
    asset_class: AssetClass
    exchange_mic: str
    currency: str
    timezone: str
    calendar: str
    country: str | None = None
    sector: str | None = None
    provider_symbols: dict[str, str] = Field(default_factory=dict)
    active: bool = True
    tradable: bool = True

    @field_validator("symbol", "exchange_mic", "currency")
    @classmethod
    def uppercase_codes(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("instrument_id")
    @classmethod
    def validate_instrument_id(cls, value: str) -> str:
        pieces = value.split(":")
        if len(pieces) != 3 or any(not piece for piece in pieces):
            raise ValueError("instrument_id must be '<asset_class>:<venue>:<symbol>'")
        return value

    def provider_symbol(self, provider: str) -> str:
        """Return the provider-specific symbol or the canonical symbol."""

        return self.provider_symbols.get(provider.lower(), self.symbol)

    @classmethod
    def create(
        cls,
        *,
        symbol: str,
        asset_class: AssetClass,
        exchange_mic: str,
        currency: str,
        timezone: str,
        calendar: str,
        display_name: str | None = None,
        provider_symbols: dict[str, str] | None = None,
        country: str | None = None,
        sector: str | None = None,
    ) -> Instrument:
        normalized_symbol = symbol.strip().upper()
        normalized_mic = exchange_mic.strip().upper()
        return cls(
            instrument_id=f"{asset_class.value}:{normalized_mic}:{normalized_symbol}",
            symbol=normalized_symbol,
            display_name=display_name or normalized_symbol,
            asset_class=asset_class,
            exchange_mic=normalized_mic,
            currency=currency,
            timezone=timezone,
            calendar=calendar,
            provider_symbols=provider_symbols or {},
            country=country,
            sector=sector,
        )
