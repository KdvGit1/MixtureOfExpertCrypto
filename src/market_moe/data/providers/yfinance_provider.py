"""Best-effort global equity data through the unofficial yfinance client."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from market_moe.data.providers.common import provider_frame_to_canonical
from market_moe.domain.errors import DataProviderError
from market_moe.domain.instruments import Instrument


class YFinanceProvider:
    name = "yfinance"
    supported_timeframes = frozenset({"15m", "30m", "1h", "1d", "1wk"})
    _interval_map = {"1wk": "1wk"}

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
            raise DataProviderError(f"yfinance does not support {timeframe}")
        try:
            import yfinance as yf

            symbol = instrument.provider_symbol(self.name)
            raw = yf.Ticker(symbol).history(
                start=start,
                end=end,
                interval=self._interval_map.get(timeframe, timeframe),
                auto_adjust=adjusted,
                actions=False,
                repair=True,
                raise_errors=True,
            )
        except Exception as exc:
            raise DataProviderError(
                f"yfinance request failed for {instrument.symbol}: {exc}"
            ) from exc

        if raw.empty:
            raise DataProviderError(f"yfinance returned no rows for {instrument.symbol}")
        raw = raw.rename(columns=str.lower)
        raw.index.name = "open_time_utc"
        return provider_frame_to_canonical(
            raw,
            instrument=instrument,
            timeframe=timeframe,
            provider=self.name,
            provider_symbol=symbol,
            adjusted=adjusted,
        )

    def healthcheck(self) -> dict[str, object]:
        try:
            import yfinance as yf

            return {
                "provider": self.name,
                "healthy": True,
                "authenticated": False,
                "version": getattr(yf, "__version__", "unknown"),
                "usage": "personal-research-only",
            }
        except ImportError as exc:
            return {"provider": self.name, "healthy": False, "error": str(exc)}
