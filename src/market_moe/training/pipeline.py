"""Pooled multi-instrument training and locked-fold backtesting.

The Windows automation entrypoint downloads data.  This module deliberately
accepts already loaded frames so that training is deterministic, testable and
resumable without coupling the model code to a network provider.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

from market_moe.backtest.costs import CostModel
from market_moe.backtest.engine import BacktestConfig, run_backtest
from market_moe.backtest.reports import write_backtest_report
from market_moe.backtest.risk import RiskLimits
from market_moe.data.quality import canonicalize_bar_frame
from market_moe.domain.instruments import Instrument
from market_moe.features.normalization import NormalizationStats
from market_moe.features.pipeline import FeaturePipeline
from market_moe.features.schema import FeatureSchema
from market_moe.models.bundle import ModelBundle, ModelManifest
from market_moe.models.calibration import ProbabilityCalibrator
from market_moe.models.moe import MultiTaskMoE
from market_moe.models.registry import ModelRegistry
from market_moe.training.dataset import WindowDataset
from market_moe.training.evaluator import regression_direction_metrics
from market_moe.training.model_card import render_model_card
from market_moe.training.trainer import TrainingConfig, train_model

LogCallback = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class PooledTrainingSpec:
    """Validated settings for one model/timeframe job."""

    job_id: str
    model_id: str
    asset_class: str
    timeframe: str
    version: str
    window: int = 120
    horizon_bars: int = 1
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    embed_dim: int = 96
    router_hidden_dim: int = 32
    dropout: float = 0.20
    branch_dropout_probability: float = 0.20
    batch_size: int = 256
    epochs: int = 30
    patience: int = 7
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    seed: int = 20260811
    branch_dropout_start_epoch: int = 4
    probability_threshold: float = 0.58
    allow_short: bool = False
    periods_per_year: int = 252
    initial_cash: float = 100_000.0
    maximum_position_fraction: float = 1.0
    maximum_drawdown: float = 0.50
    maximum_gap_fraction: float = 0.25
    commission_bps: float = 5.0
    spread_bps: float = 5.0
    slippage_bps: float = 2.0
    funding_bps_per_bar: float = 0.0
    borrow_bps_per_bar: float = 0.0
    providers: tuple[str, ...] = ()
    limitations: tuple[str, ...] = (
        "Free public provider data may be delayed, revised, incomplete or unavailable.",
        "Current-universe constituents introduce survivorship bias in historical tests.",
        "Backtests are simulations and do not represent executable or guaranteed returns.",
    )

    def validate(self) -> None:
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be in (0, 1)")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in (0, 1)")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("train + validation fractions must leave a test fold")
        if self.window < 2 or self.horizon_bars < 1:
            raise ValueError("window and horizon_bars are invalid")
        if not 0.5 < self.probability_threshold < 1:
            raise ValueError("probability_threshold must be in (0.5, 1)")


@dataclass(slots=True)
class PreparedInstrument:
    instrument: Instrument
    bars: pd.DataFrame
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(slots=True)
class PooledTrainingResult:
    bundle_path: Path
    backtest_root: Path
    manifest: ModelManifest
    metrics: dict[str, object]
    backtests: dict[str, dict[str, object]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _global_boundaries(
    frames: list[pd.DataFrame], train_fraction: float, validation_fraction: float
) -> tuple[pd.Timestamp, pd.Timestamp]:
    unique_days = pd.DatetimeIndex(
        sorted(
            {
                pd.Timestamp(stamp).normalize()
                for frame in frames
                for stamp in pd.DatetimeIndex(frame.index)
            }
        )
    )
    if len(unique_days) < 10:
        raise ValueError("not enough distinct dates for a locked chronological split")
    train_offset = min(max(1, int(len(unique_days) * train_fraction)), len(unique_days) - 2)
    validation_offset = min(
        max(train_offset + 1, int(len(unique_days) * (train_fraction + validation_fraction))),
        len(unique_days) - 1,
    )
    return unique_days[train_offset], unique_days[validation_offset]


def _purge(frame: pd.DataFrame, *, left: int = 0, right: int = 0) -> pd.DataFrame:
    stop = len(frame) - right if right else len(frame)
    if left >= stop:
        return frame.iloc[0:0]
    return frame.iloc[left:stop]


def _prepare(
    sources: list[tuple[Instrument, pd.DataFrame]], spec: PooledTrainingSpec
) -> tuple[list[PreparedInstrument], FeatureSchema]:
    pipeline = FeaturePipeline()
    transformed: list[tuple[Instrument, pd.DataFrame, pd.DataFrame, FeatureSchema]] = []
    for instrument, raw_bars in sources:
        if instrument.asset_class.value != spec.asset_class:
            raise ValueError(f"{instrument.instrument_id} has the wrong asset class")
        bars = canonicalize_bar_frame(raw_bars)
        featured = pipeline.transform(
            bars,
            instrument,
            spec.timeframe,
            horizon_bars=spec.horizon_bars,
            include_targets=True,
        )
        if len(featured.frame) >= spec.window * 3:
            transformed.append((instrument, bars, featured.frame, featured.schema))
    if not transformed:
        raise ValueError("no instrument has enough clean feature rows for three folds")
    schema = transformed[0][3]
    for _instrument, _bars, _frame, candidate_schema in transformed[1:]:
        if candidate_schema.schema_hash != schema.schema_hash:
            raise ValueError("pooled instruments produced incompatible feature schemas")
    train_end, validation_end = _global_boundaries(
        [item[2] for item in transformed], spec.train_fraction, spec.validation_fraction
    )
    prepared: list[PreparedInstrument] = []
    for instrument, bars, frame, _schema in transformed:
        days = pd.DatetimeIndex(frame.index).normalize()
        train = _purge(
            frame.loc[days < train_end],
            right=spec.horizon_bars,
        )
        validation = _purge(
            frame.loc[(days >= train_end) & (days < validation_end)],
            left=spec.horizon_bars,
            right=spec.horizon_bars,
        )
        test = _purge(frame.loc[days >= validation_end], left=spec.horizon_bars)
        if min(len(train), len(validation), len(test)) >= spec.window:
            prepared.append(PreparedInstrument(instrument, bars, train, validation, test))
    if not prepared:
        raise ValueError(
            "global split left no eligible instruments; request more history or reduce window"
        )
    return prepared, schema


def _loader(
    frames: list[pd.DataFrame],
    feature_names: tuple[str, ...],
    *,
    window: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
    cuda: bool,
) -> DataLoader:
    datasets = [WindowDataset(frame, feature_names, window=window) for frame in frames]
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        ConcatDataset(datasets),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=cuda,
        generator=generator,
    )


def _evaluate(
    model: MultiTaskMoE, loader: DataLoader, device: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    actual: list[np.ndarray] = []
    predicted: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    volatility: list[np.ndarray] = []
    model.to(device).eval()
    with torch.inference_mode():
        for features, target_return, _direction, _target_volatility in loader:
            output = model(features.to(device, non_blocking=True))
            actual.append(target_return.numpy())
            predicted.append(output.expected_return.detach().cpu().numpy())
            probabilities.append(torch.sigmoid(output.direction_logit).detach().cpu().numpy())
            volatility.append(output.volatility.detach().cpu().numpy())
    return (
        np.concatenate(actual),
        np.concatenate(predicted),
        np.concatenate(probabilities),
        np.concatenate(volatility),
    )


def _fit_calibration(
    probabilities: np.ndarray, actual: np.ndarray
) -> tuple[dict[str, object], Callable[[np.ndarray], np.ndarray]]:
    labels = (actual > 0).astype(int)
    if np.unique(labels).size < 2:
        constant = float(labels[0])
        payload: dict[str, object] = {
            "method": "isotonic",
            "x_thresholds": [0.0, 1.0],
            "y_thresholds": [constant, constant],
            "warning": "validation fold contained only one direction class",
        }
        return payload, lambda values: np.full(values.shape, constant, dtype=float)
    calibrator = ProbabilityCalibrator().fit(probabilities, labels)
    payload = {**calibrator.to_dict(), **calibrator.report(probabilities, labels)}
    return payload, calibrator.transform


def _date_range(frames: list[pd.DataFrame]) -> list[str]:
    start = min(pd.Timestamp(frame.index.min()) for frame in frames)
    end = max(pd.Timestamp(frame.index.max()) for frame in frames)
    return [start.isoformat(), end.isoformat()]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _finite(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    return value


def _finite_dict(value: dict[str, object]) -> dict[str, object]:
    return {key: _finite(item) for key, item in value.items()}


def _instrument_backtest(
    prepared: PreparedInstrument,
    normalized_test: pd.DataFrame,
    model: MultiTaskMoE,
    calibration: Callable[[np.ndarray], np.ndarray],
    schema: FeatureSchema,
    spec: PooledTrainingSpec,
    output: Path,
    device: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    loader = _loader(
        [normalized_test],
        schema.feature_names,
        window=spec.window,
        batch_size=spec.batch_size,
        shuffle=False,
        seed=spec.seed,
        cuda=device.startswith("cuda"),
    )
    actual, predicted, raw_probability, predicted_volatility = _evaluate(model, loader, device)
    probability = calibration(raw_probability)
    prediction_index = normalized_test.index[spec.window - 1 :]
    round_trip_cost = (
        2 * (spec.commission_bps + spec.spread_bps + spec.slippage_bps) / 10_000
    )
    signal = np.zeros(len(predicted), dtype=float)
    long_mask = (probability >= spec.probability_threshold) & (predicted > round_trip_cost)
    signal[long_mask] = 1.0
    if spec.allow_short:
        short_mask = (probability <= 1.0 - spec.probability_threshold) & (
            predicted < -round_trip_cost
        )
        signal[short_mask] = -1.0
    predictions = pd.DataFrame(
        {
            "actual_log_return": actual,
            "predicted_log_return": predicted,
            "raw_probability_up": raw_probability,
            "probability_up": probability,
            "predicted_volatility": predicted_volatility,
            "signal": signal,
        },
        index=prediction_index,
    )
    predictions.index.name = "open_time_utc"
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output / "predictions.parquet")

    bars = prepared.bars.set_index("open_time_utc", drop=False)
    bars = bars.loc[(bars.index >= normalized_test.index.min()) & (bars.index <= bars.index.max())]
    signals = predictions["signal"].reindex(bars.index).fillna(0.0)
    result = run_backtest(
        bars.reset_index(drop=True),
        signals,
        BacktestConfig(
            initial_cash=spec.initial_cash,
            base_currency=prepared.instrument.currency,
            periods_per_year=spec.periods_per_year,
            costs=CostModel(
                spec.commission_bps,
                spec.spread_bps,
                spec.slippage_bps,
                spec.funding_bps_per_bar,
                spec.borrow_bps_per_bar,
            ),
            risk=RiskLimits(
                maximum_position_fraction=spec.maximum_position_fraction,
                maximum_drawdown=spec.maximum_drawdown,
                maximum_gap_fraction=spec.maximum_gap_fraction,
                allow_short=spec.allow_short,
            ),
        ),
    )
    write_backtest_report(result, output)
    return (
        {
            "instrument_id": prepared.instrument.instrument_id,
            "prediction_rows": len(predictions),
            "signal_rows": int((predictions["signal"] != 0).sum()),
            "metrics": _finite(result.metrics),
            "warnings": result.warnings,
        },
        predictions,
    )


def train_pooled_model(
    sources: list[tuple[Instrument, pd.DataFrame]],
    spec: PooledTrainingSpec,
    *,
    bundle_path: Path,
    backtest_root: Path,
    registry_root: Path,
    checkpoint_path: Path | None = None,
    resume: bool = True,
    device: str = "cuda",
    log: LogCallback | None = None,
) -> PooledTrainingResult:
    """Train one pooled model, evaluate once on locked test, then backtest it."""

    spec.validate()
    report = log or (lambda _message: None)
    prepared, schema = _prepare(sources, spec)
    report(f"{spec.job_id}: {len(prepared)} instruments passed feature/split checks")
    train_pool = pd.concat([item.train for item in prepared])
    normalization = NormalizationStats.fit(train_pool, schema.feature_names)
    normalized_train = [normalization.transform(item.train) for item in prepared]
    normalized_validation = [normalization.transform(item.validation) for item in prepared]
    normalized_test = [normalization.transform(item.test) for item in prepared]
    using_cuda = device.startswith("cuda")
    train_loader = _loader(
        normalized_train,
        schema.feature_names,
        window=spec.window,
        batch_size=spec.batch_size,
        shuffle=True,
        seed=spec.seed,
        cuda=using_cuda,
    )
    validation_loader = _loader(
        normalized_validation,
        schema.feature_names,
        window=spec.window,
        batch_size=spec.batch_size,
        shuffle=False,
        seed=spec.seed,
        cuda=using_cuda,
    )
    test_loader = _loader(
        normalized_test,
        schema.feature_names,
        window=spec.window,
        batch_size=spec.batch_size,
        shuffle=False,
        seed=spec.seed,
        cuda=using_cuda,
    )
    model = MultiTaskMoE(
        len(schema.feature_names),
        embed_dim=spec.embed_dim,
        router_hidden_dim=spec.router_hidden_dim,
        dropout=spec.dropout,
        branch_dropout_probability=spec.branch_dropout_probability,
    )

    def progress(epoch: int, values: dict[str, float]) -> None:
        report(
            f"{spec.job_id}: epoch {epoch + 1}/{spec.epochs} "
            f"train={values['train_loss']:.6f} val={values['validation_loss']:.6f}"
        )

    training = train_model(
        model,
        train_loader,
        validation_loader,
        TrainingConfig(
            epochs=spec.epochs,
            batch_size=spec.batch_size,
            learning_rate=spec.learning_rate,
            weight_decay=spec.weight_decay,
            patience=spec.patience,
            gradient_clip=spec.gradient_clip,
            seed=spec.seed,
            branch_dropout_start_epoch=spec.branch_dropout_start_epoch,
        ),
        device=device,
        checkpoint_path=checkpoint_path,
        resume=resume,
        progress_callback=progress,
    )
    validation_actual, _validation_predicted, validation_probability, _validation_vol = (
        _evaluate(model, validation_loader, device)
    )
    calibration_payload, calibrate = _fit_calibration(
        validation_probability, validation_actual
    )
    test_actual, test_predicted, test_probability, _test_volatility = _evaluate(
        model, test_loader, device
    )
    calibrated_test = calibrate(test_probability)
    metrics: dict[str, object] = {
        **regression_direction_metrics(test_actual, test_predicted, calibrated_test),
        "zero_return_baseline_mae": float(np.mean(np.abs(test_actual))),
        "best_validation_loss": training.best_validation_loss,
        "best_epoch": training.best_epoch,
        "train_windows": sum(len(frame) - spec.window + 1 for frame in normalized_train),
        "validation_windows": sum(
            len(frame) - spec.window + 1 for frame in normalized_validation
        ),
        "test_windows": sum(len(frame) - spec.window + 1 for frame in normalized_test),
        "eligible_instruments": len(prepared),
        "test_fold_used_for_selection": False,
    }
    manifest = ModelManifest(
        model_id=spec.model_id,
        version=spec.version,
        asset_class=spec.asset_class,
        timeframe=spec.timeframe,
        horizon_bars=spec.horizon_bars,
        feature_schema_hash=schema.schema_hash,
        symbols=[item.instrument.instrument_id for item in prepared],
        date_ranges={
            "train": _date_range([item.train for item in prepared]),
            "validation": _date_range([item.validation for item in prepared]),
            "test": _date_range([item.test for item in prepared]),
            "purge_bars": spec.horizon_bars,
            "test_locked_until_final_evaluation": True,
        },
        provider=",".join(sorted(set(spec.providers))) or "free_public_cache",
        random_seed=spec.seed,
        hyperparameters=asdict(spec),
        limitations=list(spec.limitations),
    )
    clean_metrics = _finite_dict(metrics)
    clean_calibration = _finite_dict(calibration_payload)
    ModelBundle.save(
        bundle_path,
        model,
        manifest,
        schema,
        normalization,
        metrics=clean_metrics,
        calibration=clean_calibration,
        model_card=render_model_card(manifest, metrics),
        training_history=training.history,
    )
    ModelRegistry(registry_root).register(manifest, bundle_path)
    backtests: dict[str, dict[str, object]] = {}
    warnings: list[str] = []
    for item, normalized in zip(prepared, normalized_test, strict=True):
        instrument_output = backtest_root / _safe_name(item.instrument.instrument_id)
        try:
            summary, _predictions = _instrument_backtest(
                item,
                normalized,
                model,
                calibrate,
                schema,
                spec,
                instrument_output,
                device,
            )
            backtests[item.instrument.instrument_id] = summary
            report(f"{spec.job_id}: backtest complete for {item.instrument.instrument_id}")
        except Exception as error:
            message = f"{item.instrument.instrument_id}: backtest failed: {error}"
            warnings.append(message)
            report(f"WARNING {spec.job_id}: {message}")
    backtest_root.mkdir(parents=True, exist_ok=True)
    (backtest_root / "summary.json").write_text(
        json.dumps(
            {
                "model_id": manifest.model_id,
                "version": manifest.version,
                "timeframe": manifest.timeframe,
                "costs_included": True,
                "execution_policy": "signal_at_close_execute_next_open",
                "instruments": backtests,
                "warnings": warnings,
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return PooledTrainingResult(
        bundle_path,
        backtest_root,
        manifest,
        clean_metrics,
        backtests,
        warnings,
    )
