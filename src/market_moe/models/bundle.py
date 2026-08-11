"""Atomic self-describing model bundle persistence."""

from __future__ import annotations

import json
import os
import platform
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from market_moe.domain.errors import ModelCompatibilityError
from market_moe.features.normalization import NormalizationStats
from market_moe.features.schema import FeatureSchema
from market_moe.models.moe import MultiTaskMoE


@dataclass(slots=True)
class ModelManifest:
    model_id: str
    version: str
    asset_class: str
    timeframe: str
    horizon_bars: int
    feature_schema_hash: str
    created_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    status: str = "candidate"
    symbols: list[str] = field(default_factory=list)
    date_ranges: dict[str, Any] = field(default_factory=dict)
    provider: str = "local"
    adjustment_mode: str = "provider_adjusted"
    random_seed: int = 20260811
    git_commit: str = "unknown"
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    loss_config: dict[str, float] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    python_version: str = field(default_factory=platform.python_version)
    torch_version: str = field(default_factory=lambda: torch.__version__)


class ModelBundle:
    required_files = frozenset(
        {
            "model.pt",
            "manifest.json",
            "feature_schema.json",
            "normalization.json",
            "metrics.json",
            "model_card.md",
            "calibration.json",
            "training_history.parquet",
        }
    )

    @staticmethod
    def save(
        path: Path,
        model: MultiTaskMoE,
        manifest: ModelManifest,
        schema: FeatureSchema,
        normalization: NormalizationStats,
        *,
        metrics: dict[str, Any] | None = None,
        calibration: dict[str, Any] | None = None,
        model_card: str = "# Model card\n",
        training_history: list[dict[str, float]] | None = None,
    ) -> Path:
        if manifest.feature_schema_hash != schema.schema_hash:
            raise ModelCompatibilityError("manifest and feature schema hash differ")
        path.mkdir(parents=True, exist_ok=True)
        payloads = {
            "manifest.json": asdict(manifest),
            "feature_schema.json": schema.to_dict(),
            "normalization.json": normalization.to_dict(),
            "metrics.json": metrics or {},
            "calibration.json": calibration or {},
        }
        for filename, payload in payloads.items():
            ModelBundle._atomic_json(path / filename, payload)
        (path / "model_card.md").write_text(model_card.rstrip() + "\n", encoding="utf-8")
        pd.DataFrame(
            training_history or [], columns=["epoch", "train_loss", "validation_loss"]
        ).to_parquet(path / "training_history.parquet", index=False)
        temporary = path / "model.pt.tmp"
        torch.save({"model_config": model.config(), "state_dict": model.state_dict()}, temporary)
        os.replace(temporary, path / "model.pt")
        return path

    @staticmethod
    def load(
        path: Path,
        *,
        expected_schema_hash: str | None = None,
        device: str = "cpu",
    ) -> tuple[MultiTaskMoE, ModelManifest, FeatureSchema, NormalizationStats]:
        missing = ModelBundle.required_files - {item.name for item in path.iterdir()}
        if missing:
            raise FileNotFoundError(f"incomplete model bundle: {sorted(missing)}")
        manifest = ModelManifest(**json.loads((path / "manifest.json").read_text("utf-8")))
        schema = FeatureSchema.from_dict(
            json.loads((path / "feature_schema.json").read_text("utf-8"))
        )
        if manifest.feature_schema_hash != schema.schema_hash:
            raise ModelCompatibilityError("bundle schema checksum is invalid")
        if expected_schema_hash and expected_schema_hash != schema.schema_hash:
            raise ModelCompatibilityError("inference feature schema is incompatible with model")
        normalization = NormalizationStats.load(path / "normalization.json")
        if normalization.feature_names != schema.feature_names:
            raise ModelCompatibilityError("normalization fields do not match feature schema")
        checkpoint = torch.load(path / "model.pt", map_location=device, weights_only=True)
        model = MultiTaskMoE(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device).eval()
        return model, manifest, schema, normalization

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            temporary = Path(stream.name)
        os.replace(temporary, path)
