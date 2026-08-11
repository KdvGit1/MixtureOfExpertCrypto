from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from market_moe.domain.instruments import AssetClass, Instrument


@pytest.fixture
def crypto_instrument() -> Instrument:
    return Instrument.create(
        symbol="BTC/USDT",
        asset_class=AssetClass.CRYPTO,
        exchange_mic="BINANCE",
        currency="USDT",
        timezone="UTC",
        calendar="24/7",
        provider_symbols={"ccxt": "BTC/USDT"},
    )


@pytest.fixture
def equity_instrument() -> Instrument:
    return Instrument.create(
        symbol="AAPL",
        asset_class=AssetClass.EQUITY,
        exchange_mic="XNAS",
        currency="USD",
        timezone="America/New_York",
        calendar="XNAS",
        provider_symbols={"yfinance": "AAPL", "stooq": "AAPL.US"},
    )


def canonical_bars(instrument: Instrument, rows: int = 320, timeframe: str = "1d") -> pd.DataFrame:
    generator = np.random.default_rng(42)
    delta = timedelta(days=1) if timeframe == "1d" else timedelta(hours=1)
    times = pd.date_range("2020-01-01", periods=rows, freq=delta, tz="UTC")
    close = 100 * np.exp(np.cumsum(generator.normal(0.0005, 0.01, rows)))
    open_price = np.r_[close[0], close[:-1]] * (1 + generator.normal(0, 0.002, rows))
    high = np.maximum(open_price, close) * 1.01
    low = np.minimum(open_price, close) * 0.99
    return pd.DataFrame(
        {
            "instrument_id": instrument.instrument_id,
            "timeframe": timeframe,
            "open_time_utc": times,
            "close_time_utc": times + delta,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": generator.uniform(100, 10_000, rows),
            "vwap": (open_price + close) / 2,
            "currency": instrument.currency,
            "session_type": "continuous"
            if instrument.asset_class == AssetClass.CRYPTO
            else "regular",
            "is_adjusted": instrument.asset_class != AssetClass.CRYPTO,
            "provider": "fixture",
            "provider_symbol": instrument.symbol,
            "ingested_at_utc": datetime.now(UTC),
            "quality_flags": "",
        }
    )


@pytest.fixture
def crypto_bars(crypto_instrument: Instrument) -> pd.DataFrame:
    return canonical_bars(crypto_instrument)


@pytest.fixture
def equity_bars(equity_instrument: Instrument) -> pd.DataFrame:
    return canonical_bars(equity_instrument)
