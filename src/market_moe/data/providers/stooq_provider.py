"""Free daily-data fallback using Stooq's public CSV endpoint."""

from __future__ import annotations

import io
from datetime import datetime

import httpx
import pandas as pd

from market_moe.data.providers.common import provider_frame_to_canonical
from market_moe.domain.errors import DataProviderError
from market_moe.domain.instruments import Instrument


class StooqProvider:
    name = "stooq"
    supported_timeframes = frozenset({"1d"})

    def fetch_bars(
        self,
        instrument: Instrument,
        timeframe: str,
        start: datetime,
        end: datetime,
        *,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        if timeframe != "1d":
            raise DataProviderError("Stooq fallback supports daily bars only")
        symbol = instrument.provider_symbol(self.name)
        try:
            response = httpx.get(
                "https://stooq.com/q/d/l/",
                params={
                    "s": symbol.lower(),
                    "d1": start.strftime("%Y%m%d"),
                    "d2": end.strftime("%Y%m%d"),
                    "i": "d",
                },
                timeout=30.0,
                follow_redirects=True,
            )
            response.raise_for_status()
            raw = pd.read_csv(io.StringIO(response.text))
        except Exception as exc:
            raise DataProviderError(f"Stooq request failed for {symbol}: {exc}") from exc
        if raw.empty or "Date" not in raw.columns:
            raise DataProviderError(f"Stooq returned no rows for {symbol}")
        raw = raw.rename(columns=str.lower)
        raw["open_time_utc"] = pd.to_datetime(raw["date"], utc=True)
        raw = raw.sort_values("open_time_utc")
        return provider_frame_to_canonical(
            raw,
            instrument=instrument,
            timeframe=timeframe,
            provider=self.name,
            provider_symbol=symbol,
            adjusted=adjusted,
            fallback=True,
        )

    def healthcheck(self) -> dict[str, object]:
        try:
            return {
                "provider": self.name,
                "healthy": True,
                "authenticated": False,
                "transport": "public-csv",
                "daily_only": True,
            }
        except Exception as exc:
            return {"provider": self.name, "healthy": False, "error": str(exc)}
