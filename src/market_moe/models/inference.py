"""Schema-checked inference returning an explicit domain contract."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
import pandas as pd
import torch

from market_moe.domain.instruments import Instrument
from market_moe.domain.predictions import Prediction
from market_moe.features.normalization import NormalizationStats
from market_moe.features.schema import FeatureSchema
from market_moe.models.bundle import ModelManifest
from market_moe.models.moe import MultiTaskMoE


def predict(
    model: MultiTaskMoE,
    frame: pd.DataFrame,
    instrument: Instrument,
    manifest: ModelManifest,
    schema: FeatureSchema,
    normalization: NormalizationStats,
    *,
    window: int,
    predicted_at: datetime | None = None,
    calibration: dict[str, object] | None = None,
) -> Prediction:
    if schema.schema_hash != manifest.feature_schema_hash:
        raise ValueError("model and feature schema are incompatible")
    if len(frame) < window:
        raise ValueError(f"at least {window} feature rows are required")
    normalized = normalization.transform(frame.iloc[-window:])
    values = normalized.loc[:, schema.feature_names].to_numpy(dtype=np.float32)
    inputs = torch.from_numpy(values).unsqueeze(0)
    with torch.inference_mode():
        output = model(inputs)
    weights = output.expert_weights[0].cpu().tolist()
    timestamp = predicted_at or datetime.now(UTC)
    last_index = pd.Timestamp(frame.index[-1])
    if last_index.tzinfo is None:
        last_index = last_index.tz_localize("UTC")
    expected_log_return = float(output.expected_return[0])
    raw_probability_up = float(torch.sigmoid(output.direction_logit[0]))
    probability_up = raw_probability_up
    if calibration and calibration.get("method") == "isotonic":
        x_thresholds = np.asarray(calibration.get("x_thresholds", []), dtype=float)
        y_thresholds = np.asarray(calibration.get("y_thresholds", []), dtype=float)
        if x_thresholds.size >= 2 and x_thresholds.size == y_thresholds.size:
            probability_up = float(np.interp(raw_probability_up, x_thresholds, y_thresholds))
    uncertainty = float(output.uncertainty[0])
    freshness = max(0.0, (timestamp - last_index.to_pydatetime()).total_seconds())
    return Prediction(
        prediction_id=str(uuid4()),
        instrument_id=instrument.instrument_id,
        asset_class=instrument.asset_class,
        timeframe=manifest.timeframe,
        as_of_utc=timestamp,
        horizon=f"{manifest.horizon_bars} bars",
        expected_log_return=expected_log_return,
        expected_return_pct=float(np.expm1(expected_log_return) * 100.0),
        probability_up=probability_up,
        predicted_volatility=float(output.volatility[0]),
        uncertainty=uncertainty,
        confidence=float(np.clip(1.0 - uncertainty, 0.0, 1.0)),
        expert_weights=dict(zip(model.expert_names, weights, strict=True)),
        raw_model_output={
            "direction_logit": float(output.direction_logit[0]),
            "uncalibrated_probability_up": raw_probability_up,
        },
        model_id=manifest.model_id,
        model_version=manifest.version,
        feature_schema_hash=schema.schema_hash,
        normalization_id=normalization.normalization_id,
        data_freshness_seconds=freshness,
        warnings=(),
    )
