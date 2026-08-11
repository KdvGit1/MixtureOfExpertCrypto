"""Atomic local Parquet cache for canonical bars."""

from __future__ import annotations

import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from market_moe.data.quality import deduplicate_bars, validate_bar_frame
from market_moe.domain.instruments import Instrument


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


class ParquetBarCache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, provider: str, instrument: Instrument, timeframe: str) -> Path:
        return (
            self.root
            / _safe_component(provider.lower())
            / _safe_component(instrument.asset_class.value)
            / _safe_component(instrument.exchange_mic)
            / f"{_safe_component(instrument.symbol)}__{_safe_component(timeframe)}.parquet"
        )

    def load(self, provider: str, instrument: Instrument, timeframe: str) -> pd.DataFrame | None:
        path = self.path_for(provider, instrument, timeframe)
        if not path.exists():
            return None
        frame = pd.read_parquet(path)
        validate_bar_frame(frame, raise_on_error=True)
        return frame

    def save(
        self, provider: str, instrument: Instrument, timeframe: str, frame: pd.DataFrame
    ) -> Path:
        clean = deduplicate_bars(frame)
        validate_bar_frame(clean, raise_on_error=True)
        path = self.path_for(provider, instrument, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=path.stem + "-", suffix=".parquet", dir=path.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            clean.to_parquet(temporary_path, index=False)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return path

    def merge(
        self, provider: str, instrument: Instrument, timeframe: str, frame: pd.DataFrame
    ) -> Path:
        existing = self.load(provider, instrument, timeframe)
        combined = frame if existing is None else pd.concat([existing, frame], ignore_index=True)
        return self.save(provider, instrument, timeframe, combined)

    def is_fresh(
        self,
        provider: str,
        instrument: Instrument,
        timeframe: str,
        max_age_seconds: float,
        *,
        now: datetime | None = None,
    ) -> bool:
        path = self.path_for(provider, instrument, timeframe)
        if not path.exists():
            return False
        reference = now or datetime.now(UTC)
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        return (reference - modified).total_seconds() <= max_age_seconds
