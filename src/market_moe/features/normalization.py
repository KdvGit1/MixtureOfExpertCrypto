"""Serializable train-only normalization statistics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from market_moe.domain.errors import FeatureSchemaError


@dataclass(frozen=True, slots=True)
class NormalizationStats:
    feature_names: tuple[str, ...]
    mean: dict[str, float]
    std: dict[str, float]
    fitted_rows: int
    fitted_start: str
    fitted_end: str

    @property
    def normalization_id(self) -> str:
        payload = json.dumps(self.to_dict(include_id=False), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def fit(cls, frame: pd.DataFrame, feature_names: tuple[str, ...]) -> NormalizationStats:
        if frame.empty:
            raise ValueError("cannot fit normalization on an empty frame")
        missing = [name for name in feature_names if name not in frame]
        if missing:
            raise FeatureSchemaError(f"normalization features missing: {missing}")
        values = frame[list(feature_names)]
        means = values.mean()
        standard_deviations = values.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
        index = pd.to_datetime(frame.index, utc=True)
        return cls(
            feature_names=feature_names,
            mean={name: float(means[name]) for name in feature_names},
            std={name: float(standard_deviations[name]) for name in feature_names},
            fitted_rows=len(frame),
            fitted_start=index.min().isoformat(),
            fitted_end=index.max().isoformat(),
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [name for name in self.feature_names if name not in frame]
        if missing:
            raise FeatureSchemaError(f"normalization features missing: {missing}")
        result = frame.copy()
        for name in self.feature_names:
            result[name] = (result[name] - self.mean[name]) / self.std[name]
        return result

    def inverse(self, name: str, value: float) -> float:
        if name not in self.mean:
            raise KeyError(name)
        return value * self.std[name] + self.mean[name]

    def to_dict(self, *, include_id: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "feature_names": list(self.feature_names),
            "mean": self.mean,
            "std": self.std,
            "fitted_rows": self.fitted_rows,
            "fitted_start": self.fitted_start,
            "fitted_end": self.fitted_end,
        }
        if include_id:
            payload["normalization_id"] = self.normalization_id
        return payload

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> NormalizationStats:
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = payload.pop("normalization_id", None)
        payload["feature_names"] = tuple(payload["feature_names"])
        stats = cls(**payload)
        if expected and expected != stats.normalization_id:
            raise ValueError("normalization checksum mismatch")
        return stats
