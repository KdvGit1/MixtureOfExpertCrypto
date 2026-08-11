"""Local DuckDB index for immutable model bundles."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import duckdb

from market_moe.models.bundle import ModelManifest


class ModelRegistry:
    valid_statuses = frozenset({"candidate", "validated", "production", "retired"})

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = root / "registry.duckdb"
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS models (
                    model_id VARCHAR, version VARCHAR, asset_class VARCHAR,
                    timeframe VARCHAR, status VARCHAR, bundle_path VARCHAR,
                    schema_hash VARCHAR, created_at_utc TIMESTAMP,
                    PRIMARY KEY (model_id, version)
                )
                """
            )

    def _connect(self):
        return duckdb.connect(str(self.database))

    def register(self, manifest: ModelManifest, bundle_path: Path) -> None:
        if manifest.status not in self.valid_statuses:
            raise ValueError(f"invalid registry status: {manifest.status}")
        bundle_path = bundle_path.resolve()
        if self.root.resolve() not in bundle_path.parents:
            raise ValueError("bundle must be inside registry root")
        values = asdict(manifest)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO models VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    values["model_id"],
                    values["version"],
                    values["asset_class"],
                    values["timeframe"],
                    values["status"],
                    str(bundle_path),
                    values["feature_schema_hash"],
                    values["created_at_utc"],
                ],
            )

    def promote(self, model_id: str, version: str, status: str) -> None:
        if status not in self.valid_statuses:
            raise ValueError(status)
        with self._connect() as connection:
            if status == "production":
                connection.execute(
                    "UPDATE models SET status='validated' WHERE model_id=? AND status='production'",
                    [model_id],
                )
            result = connection.execute(
                "UPDATE models SET status=? WHERE model_id=? AND version=? RETURNING version",
                [status, model_id, version],
            ).fetchone()
        if result is None:
            raise KeyError(f"model not registered: {model_id}/{version}")

    def list(self, *, status: str | None = None) -> list[dict[str, object]]:
        query = "SELECT * FROM models"
        parameters: list[str] = []
        if status:
            query += " WHERE status=?"
            parameters.append(status)
        query += " ORDER BY created_at_utc DESC"
        with self._connect() as connection:
            cursor = connection.execute(query, parameters)
            columns = [item[0] for item in cursor.description]
            return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]

    def production(self, model_id: str, timeframe: str) -> Path:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT bundle_path FROM models
                WHERE model_id=? AND timeframe=? AND status='production'
                ORDER BY created_at_utc DESC LIMIT 1""",
                [model_id, timeframe],
            ).fetchone()
        if row is None:
            raise KeyError(f"no production model for {model_id}/{timeframe}")
        return Path(row[0])
