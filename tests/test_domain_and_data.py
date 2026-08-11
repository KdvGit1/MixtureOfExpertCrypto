from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from market_moe.data.cache import ParquetBarCache
from market_moe.data.quality import deduplicate_bars, validate_bar_frame
from market_moe.domain.errors import DataQualityError
from market_moe.domain.instruments import AssetClass, Instrument


def test_instrument_identity_is_listing_specific() -> None:
    instrument = Instrument.create(
        symbol="thyao",
        asset_class=AssetClass.EQUITY,
        exchange_mic="xist",
        currency="try",
        timezone="Europe/Istanbul",
        calendar="XIST",
        provider_symbols={"yfinance": "THYAO.IS"},
    )
    assert instrument.instrument_id == "equity:XIST:THYAO"
    assert instrument.provider_symbol("yfinance") == "THYAO.IS"


def test_quality_detects_invalid_ohlc(crypto_bars: pd.DataFrame) -> None:
    crypto_bars.loc[3, "low"] = crypto_bars.loc[3, "high"] + 1
    report = validate_bar_frame(crypto_bars)
    assert not report.valid
    assert report.invalid_prices == 1
    with pytest.raises(DataQualityError):
        validate_bar_frame(crypto_bars, raise_on_error=True)


def test_deduplicate_keeps_latest_ingestion(crypto_bars: pd.DataFrame) -> None:
    duplicate = crypto_bars.iloc[[0]].copy()
    duplicate["close"] *= 2
    duplicate["high"] = duplicate[["open", "close"]].max(axis=1) * 1.01
    duplicate["ingested_at_utc"] = datetime(2030, 1, 1, tzinfo=UTC)
    result = deduplicate_bars(pd.concat([crypto_bars, duplicate], ignore_index=True))
    assert len(result) == len(crypto_bars)
    assert result.iloc[0]["close"] == duplicate.iloc[0]["close"]


def test_parquet_cache_round_trip(tmp_path, crypto_bars, crypto_instrument) -> None:
    cache = ParquetBarCache(tmp_path)
    path = cache.save("fixture", crypto_instrument, "1d", crypto_bars)
    assert path.exists()
    loaded = cache.load("fixture", crypto_instrument, "1d")
    assert loaded is not None and len(loaded) == len(crypto_bars)
