"""Small DuckDB catalog for local datasets and generated artifacts."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import duckdb

from market_moe.data.quality import DataQualityReport


class DataCatalog:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._initialize()

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.path))

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    provider VARCHAR NOT NULL,
                    instrument_id VARCHAR NOT NULL,
                    timeframe VARCHAR NOT NULL,
                    path VARCHAR NOT NULL,
                    row_count BIGINT NOT NULL,
                    start_utc TIMESTAMPTZ,
                    end_utc TIMESTAMPTZ,
                    valid BOOLEAN NOT NULL,
                    warnings VARCHAR NOT NULL,
                    updated_at_utc TIMESTAMPTZ NOT NULL,
                    PRIMARY KEY (provider, instrument_id, timeframe)
                )
                """
            )

    def upsert(
        self,
        *,
        provider: str,
        instrument_id: str,
        timeframe: str,
        path: Path,
        report: DataQualityReport,
    ) -> None:
        values = (
            provider,
            instrument_id,
            timeframe,
            str(path),
            report.row_count,
            report.start_utc,
            report.end_utc,
            report.valid,
            ",".join(report.warnings),
            datetime.now(UTC),
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (provider, instrument_id, timeframe) DO UPDATE SET
                    path = EXCLUDED.path,
                    row_count = EXCLUDED.row_count,
                    start_utc = EXCLUDED.start_utc,
                    end_utc = EXCLUDED.end_utc,
                    valid = EXCLUDED.valid,
                    warnings = EXCLUDED.warnings,
                    updated_at_utc = EXCLUDED.updated_at_utc
                """,
                values,
            )

    def list_datasets(self) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM datasets ORDER BY instrument_id, timeframe"
            ).fetchdf()
        return cast(list[dict[str, object]], rows.to_dict(orient="records"))
