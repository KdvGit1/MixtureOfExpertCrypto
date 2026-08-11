"""Read-only CCXT provider using public exchange endpoints only."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from market_moe.data.protocols import timeframe_delta
from market_moe.data.providers.common import provider_frame_to_canonical
from market_moe.domain.errors import DataProviderError
from market_moe.domain.instruments import Instrument


class CCXTPublicProvider:
    supported_timeframes = frozenset({"1m", "5m", "15m", "30m", "1h", "4h", "1d"})

    def __init__(self, exchange_name: str = "binance", *, max_pages: int = 500) -> None:
        self.exchange_name = exchange_name.lower()
        self.name = f"ccxt_{self.exchange_name}"
        self.max_pages = max_pages
        self._exchange: Any = None

    def _client(self):
        if self._exchange is None:
            try:
                import ccxt

                exchange_type = getattr(ccxt, self.exchange_name)
                self._exchange = exchange_type(
                    {"enableRateLimit": True, "timeout": 30_000, "options": {"defaultType": "spot"}}
                )
                # No apiKey/secret is ever configured: public market data only.
                self._exchange.load_markets()
            except Exception as exc:
                raise DataProviderError(f"cannot initialize public CCXT provider: {exc}") from exc
        return self._exchange

    def fetch_bars(
        self,
        instrument: Instrument,
        timeframe: str,
        start: datetime,
        end: datetime,
        *,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        if timeframe not in self.supported_timeframes:
            raise DataProviderError(f"{self.name} does not support {timeframe}")
        exchange = self._client()
        symbol = instrument.provider_symbol(self.name)
        if symbol == instrument.symbol:
            symbol = instrument.provider_symbol("ccxt")
        since = int(start.astimezone(UTC).timestamp() * 1000)
        end_ms = int(end.astimezone(UTC).timestamp() * 1000)
        rows: list[list[float]] = []
        for _ in range(self.max_pages):
            try:
                batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            except Exception as exc:
                raise DataProviderError(f"{self.name} OHLCV request failed: {exc}") from exc
            if not batch:
                break
            rows.extend(batch)
            last_ms = int(batch[-1][0])
            if last_ms >= end_ms or len(batch) < 1000:
                break
            since = last_ms + 1
            time.sleep(max(0.0, float(exchange.rateLimit) / 1000.0))

        raw = pd.DataFrame(
            rows, columns=["open_time_utc", "open", "high", "low", "close", "volume"]
        )
        if not raw.empty:
            raw["open_time_utc"] = pd.to_datetime(raw["open_time_utc"], unit="ms", utc=True)
            raw = raw[raw["open_time_utc"] < pd.Timestamp(end)]
            current_cutoff = pd.Timestamp(datetime.now(UTC)) - timeframe_delta(timeframe)
            raw = raw[raw["open_time_utc"] <= current_cutoff]
        return provider_frame_to_canonical(
            raw,
            instrument=instrument,
            timeframe=timeframe,
            provider=self.name,
            provider_symbol=symbol,
            adjusted=False,
        )

    def healthcheck(self) -> dict[str, object]:
        try:
            exchange = self._client()
            return {
                "provider": self.name,
                "healthy": True,
                "authenticated": False,
                "markets": len(exchange.markets),
            }
        except DataProviderError as exc:
            return {"provider": self.name, "healthy": False, "error": str(exc)}
