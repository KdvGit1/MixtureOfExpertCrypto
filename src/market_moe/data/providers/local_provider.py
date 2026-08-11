"""CSV/Parquet escape hatch when free network providers are unavailable."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from market_moe.data.providers.common import provider_frame_to_canonical
from market_moe.domain.errors import DataProviderError
from market_moe.domain.instruments import Instrument


class LocalFileProvider:
    name = "local_import"
    supported_timeframes = frozenset({"1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"})

    def __init__(self, paths: dict[str, Path], *, allowed_root: Path) -> None:
        self.paths = paths
        self.allowed_root = allowed_root.resolve()

    def _safe_path(self, instrument: Instrument) -> Path:
        try:
            path = self.paths[instrument.instrument_id].resolve(strict=True)
        except (KeyError, OSError) as exc:
            raise DataProviderError(
                f"no local import registered for {instrument.instrument_id}"
            ) from exc
        if self.allowed_root != path and self.allowed_root not in path.parents:
            raise DataProviderError("local import path is outside the configured data directory")
        return path

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
            raise DataProviderError(f"local provider does not support {timeframe}")
        path = self._safe_path(instrument)
        if path.suffix.lower() == ".parquet":
            raw = pd.read_parquet(path)
        elif path.suffix.lower() == ".csv":
            raw = pd.read_csv(path)
        else:
            raise DataProviderError("local import must be CSV or Parquet")
        raw.columns = [str(column).strip().lower() for column in raw.columns]
        if "open_time_utc" not in raw.columns:
            for candidate in ("date", "datetime", "timestamp"):
                if candidate in raw.columns:
                    raw = raw.rename(columns={candidate: "open_time_utc"})
                    break
        if "open_time_utc" not in raw.columns:
            raise DataProviderError("local import has no timestamp column")
        raw["open_time_utc"] = pd.to_datetime(raw["open_time_utc"], utc=True)
        raw = raw[
            (raw["open_time_utc"] >= pd.Timestamp(start))
            & (raw["open_time_utc"] < pd.Timestamp(end))
        ]
        return provider_frame_to_canonical(
            raw,
            instrument=instrument,
            timeframe=timeframe,
            provider=self.name,
            provider_symbol=path.name,
            adjusted=adjusted,
        )

    def healthcheck(self) -> dict[str, object]:
        return {
            "provider": self.name,
            "healthy": True,
            "authenticated": False,
            "registered_files": len(self.paths),
        }
