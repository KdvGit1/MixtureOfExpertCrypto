"""Fetch, validate, cache and catalog free public bars."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from market_moe.data.cache import ParquetBarCache
from market_moe.data.catalog import DataCatalog
from market_moe.data.protocols import MarketDataProvider
from market_moe.data.providers import (
    CCXTPublicProvider,
    LocalFileProvider,
    StooqProvider,
    YFinanceProvider,
)
from market_moe.data.quality import DataQualityReport, validate_bar_frame
from market_moe.domain.instruments import AssetClass, Instrument
from market_moe.settings import Settings


class DataService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cache = ParquetBarCache(settings.normalized_data_dir)
        self.catalog = DataCatalog(settings.catalog_path)

    def provider(self, name: str) -> MarketDataProvider:
        normalized = name.lower()
        if normalized == "yfinance":
            return YFinanceProvider()
        if normalized == "stooq":
            return StooqProvider()
        if normalized in {"ccxt", "ccxt_binance", "binance"}:
            return CCXTPublicProvider("binance")
        if normalized.startswith("ccxt_") and len(normalized) > len("ccxt_"):
            return CCXTPublicProvider(normalized.removeprefix("ccxt_"))
        if normalized == "local":
            import_root = self.settings.data_dir / "imports"
            return LocalFileProvider({}, allowed_root=import_root)
        raise ValueError(f"unknown provider: {name}")

    def default_provider_name(self, instrument: Instrument) -> str:
        return (
            self.settings.default_crypto_provider
            if instrument.asset_class == AssetClass.CRYPTO
            else self.settings.default_equity_provider
        )

    def fetch(
        self,
        instrument: Instrument,
        timeframe: str,
        start: datetime,
        end: datetime,
        *,
        provider_name: str | None = None,
        adjusted: bool = True,
    ) -> tuple[pd.DataFrame, DataQualityReport]:
        provider = self.provider(provider_name or self.default_provider_name(instrument))
        frame = provider.fetch_bars(instrument, timeframe, start, end, adjusted=adjusted)
        report = validate_bar_frame(frame, raise_on_error=True)
        path = self.cache.merge(provider.name, instrument, timeframe, frame)
        self.catalog.upsert(
            provider=provider.name,
            instrument_id=instrument.instrument_id,
            timeframe=timeframe,
            path=path,
            report=report,
        )
        return frame, report

    def fetch_equity_with_fallback(
        self, instrument: Instrument, timeframe: str, start: datetime, end: datetime
    ) -> tuple[pd.DataFrame, DataQualityReport, str]:
        try:
            frame, report = self.fetch(instrument, timeframe, start, end, provider_name="yfinance")
            return frame, report, "yfinance"
        except Exception:
            if timeframe != "1d":
                raise
            frame, report = self.fetch(
                instrument, timeframe, start, end, provider_name="stooq", adjusted=False
            )
            return frame, report, "stooq"
