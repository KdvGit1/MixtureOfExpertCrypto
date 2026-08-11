"""Versioned feature schema with a deterministic content hash."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    version: str
    asset_class: str
    timeframe: str
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...] = (
        "target_log_return",
        "target_direction",
        "target_volatility",
    )

    @property
    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def schema_hash(self) -> str:
        """Semantic alias used by manifests and external API contracts."""

        return self.sha256

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["sha256"] = self.sha256
        return payload

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> FeatureSchema:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> FeatureSchema:
        payload = dict(payload)
        expected = payload.pop("sha256", None)
        raw_features = payload["feature_names"]
        raw_targets = payload["target_names"]
        if not isinstance(raw_features, (list, tuple)) or not isinstance(
            raw_targets, (list, tuple)
        ):
            raise ValueError("schema feature and target names must be arrays")
        schema = cls(
            version=str(payload["version"]),
            asset_class=str(payload["asset_class"]),
            timeframe=str(payload["timeframe"]),
            feature_names=tuple(str(item) for item in raw_features),
            target_names=tuple(str(item) for item in raw_targets),
        )
        if expected and expected != schema.sha256:
            raise ValueError("feature schema checksum mismatch")
        return schema
