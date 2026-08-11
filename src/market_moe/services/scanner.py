"""Rank predictions using edge, risk, freshness and expert agreement."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from market_moe.data.cache import ParquetBarCache
from market_moe.data.universe import load_universe
from market_moe.domain.predictions import Prediction
from market_moe.domain.signals import Signal
from market_moe.features.pipeline import FeaturePipeline
from market_moe.models.bundle import ModelBundle
from market_moe.models.inference import predict
from market_moe.models.registry import ModelRegistry
from market_moe.services.strategy import prediction_to_signal
from market_moe.settings import Settings


@dataclass(frozen=True, slots=True)
class ScanResult:
    instrument_id: str
    rank_score: float
    prediction: Prediction
    signal: Signal
    explanation_codes: tuple[str, ...]


def rank_predictions(
    predictions: list[Prediction],
    *,
    estimated_round_trip_cost: float,
    allow_short: bool = False,
) -> list[ScanResult]:
    results = []
    for prediction in predictions:
        signal = prediction_to_signal(
            prediction,
            estimated_round_trip_cost=estimated_round_trip_cost,
            allow_short=allow_short,
        )
        weights = np.asarray(list(prediction.expert_weights.values()), dtype=float)
        agreement = 1.0 - float(weights.std())
        freshness = max(0.0, 1.0 - prediction.data_freshness_seconds / 86_400)
        score = (
            0.35 * signal.score
            + 0.25 * np.tanh(signal.expected_edge_after_cost * 100)
            - 0.15 * min(1.0, prediction.predicted_volatility * 10)
            - 0.15 * min(1.0, prediction.uncertainty)
            + 0.05 * agreement
            + 0.05 * freshness
        )
        results.append(
            ScanResult(
                instrument_id=prediction.instrument_id,
                rank_score=float(np.clip(score, -1, 1)),
                prediction=prediction,
                signal=signal,
                explanation_codes=signal.reason_codes,
            )
        )
    return sorted(results, key=lambda item: item.rank_score, reverse=True)


class ScannerService:
    """Run schema-gated inference using only locally cached bars."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cache = ParquetBarCache(settings.normalized_data_dir)
        self.registry = ModelRegistry(settings.model_dir)
        self.pipeline = FeaturePipeline()

    def scan(
        self, universe_path: Path, timeframe: str, *, estimated_round_trip_cost: float = 0.001
    ) -> dict[str, object]:
        universe = load_universe(universe_path)
        predictions: list[Prediction] = []
        skipped: list[dict[str, str]] = []
        for instrument in universe.instruments:
            model_id = f"{instrument.asset_class.value}_moe"
            try:
                bundle_path = self.registry.production(model_id, timeframe)
            except KeyError as exc:
                skipped.append({"instrument_id": instrument.instrument_id, "reason": str(exc)})
                continue
            providers = (
                ["ccxt_binance"]
                if instrument.asset_class.value == "crypto"
                else ["yfinance", "stooq"]
            )
            bars = next(
                (
                    cached
                    for provider in providers
                    if (cached := self.cache.load(provider, instrument, timeframe)) is not None
                ),
                None,
            )
            if bars is None:
                skipped.append(
                    {"instrument_id": instrument.instrument_id, "reason": "cached bars not found"}
                )
                continue
            featured = self.pipeline.transform(bars, instrument, timeframe)
            try:
                model, manifest, schema, normalization = ModelBundle.load(
                    bundle_path, expected_schema_hash=featured.schema.schema_hash
                )
                window = int(manifest.hyperparameters.get("window", 120))
                predictions.append(
                    predict(
                        model,
                        featured.frame,
                        instrument,
                        manifest,
                        schema,
                        normalization,
                        window=window,
                        calibration=json.loads(
                            (bundle_path / "calibration.json").read_text(encoding="utf-8")
                        ),
                    )
                )
            except Exception as exc:
                skipped.append({"instrument_id": instrument.instrument_id, "reason": str(exc)})
        ranked = rank_predictions(predictions, estimated_round_trip_cost=estimated_round_trip_cost)
        return {
            "universe_id": universe.universe_id,
            "results": [
                {
                    "instrument_id": item.instrument_id,
                    "rank_score": item.rank_score,
                    "signal": item.signal.model_dump(mode="json"),
                    "prediction": item.prediction.model_dump(mode="json"),
                    "explanation_codes": item.explanation_codes,
                }
                for item in ranked
            ],
            "skipped": skipped,
            "survivorship_warning": universe.survivorship_warning,
            "automated_trading": False,
        }
