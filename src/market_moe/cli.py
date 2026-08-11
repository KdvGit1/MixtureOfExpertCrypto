"""MarketMoE command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from market_moe.backtest.engine import BacktestConfig, run_backtest
from market_moe.backtest.reports import write_backtest_report
from market_moe.data.universe import discover_universes, load_universe
from market_moe.diagnostics import diagnostics_payload
from market_moe.features.normalization import NormalizationStats
from market_moe.features.pipeline import FeaturePipeline
from market_moe.features.schema import FeatureSchema
from market_moe.models.bundle import ModelBundle, ModelManifest
from market_moe.models.calibration import ProbabilityCalibrator
from market_moe.models.moe import MultiTaskMoE
from market_moe.models.registry import ModelRegistry
from market_moe.portfolio.store import PaperPortfolioStore
from market_moe.services.data_service import DataService
from market_moe.services.scanner import ScannerService
from market_moe.settings import get_settings
from market_moe.training.dataset import WindowDataset
from market_moe.training.evaluator import regression_direction_metrics
from market_moe.training.model_card import render_model_card
from market_moe.training.splits import purged_chronological_split
from market_moe.training.trainer import TrainingConfig, train_model


def _json(value: object) -> None:
    print(json.dumps(value, indent=2, default=str, ensure_ascii=False))


def _instrument(universe_name: str, symbol: str):
    settings = get_settings()
    universes = discover_universes(settings.config_dir)
    if universe_name not in universes:
        raise ValueError(f"unknown universe: {universe_name}")
    universe = load_universe(universes[universe_name])
    for instrument in universe.instruments:
        if instrument.symbol == symbol.upper() or instrument.instrument_id == symbol:
            return instrument
    raise ValueError(f"instrument not found in {universe_name}: {symbol}")


def command_doctor(_args: argparse.Namespace) -> int:
    payload = diagnostics_payload(get_settings())
    _json(payload)
    return 0 if payload["healthy"] else 1


def command_serve(args: argparse.Namespace) -> int:
    import uvicorn

    settings = get_settings()
    host = args.host or settings.host
    if host not in {"127.0.0.1", "localhost"} and not args.allow_network:
        raise ValueError("non-local bind requires explicit --allow-network")
    uvicorn.run(
        "market_moe.api.app:create_app", factory=True, host=host, port=args.port or settings.port
    )
    return 0


def command_universes(_args: argparse.Namespace) -> int:
    settings = get_settings()
    payload = []
    for path in discover_universes(settings.config_dir).values():
        universe = load_universe(path)
        payload.append(
            {
                "id": universe.universe_id,
                "name": universe.display_name,
                "instruments": len(universe.instruments),
                "as_of": universe.as_of,
            }
        )
    _json(payload)
    return 0


def command_fetch(args: argparse.Namespace) -> int:
    instrument = _instrument(args.universe, args.symbol)
    end = datetime.now(UTC)
    start = end - timedelta(days=args.days)
    service = DataService(get_settings())
    frame, report = service.fetch(
        instrument, args.timeframe, start, end, provider_name=args.provider, adjusted=not args.raw
    )
    _json(
        {
            "rows": len(frame),
            "provider": args.provider or service.default_provider_name(instrument),
            "quality": report.model_dump(mode="json"),
        }
    )
    return 0


def command_models(args: argparse.Namespace) -> int:
    registry = ModelRegistry(get_settings().model_dir)
    if args.models_command == "list":
        _json(registry.list(status=args.status))
    else:
        registry.promote(args.model_id, args.version, args.status)
        _json(
            {
                "updated": True,
                "model_id": args.model_id,
                "version": args.version,
                "status": args.status,
            }
        )
    return 0


def command_backtest(args: argparse.Namespace) -> int:
    source = Path(args.input).resolve()
    frame = pd.read_parquet(source) if source.suffix.lower() == ".parquet" else pd.read_csv(source)
    if args.signal_column not in frame:
        raise ValueError(f"signal column missing: {args.signal_column}")
    index = pd.to_datetime(frame["open_time_utc"], utc=True)
    result = run_backtest(
        frame, pd.Series(frame[args.signal_column].to_numpy(), index=index), BacktestConfig()
    )
    paths = write_backtest_report(result, Path(args.output).resolve())
    _json({"metrics": result.metrics, "files": paths})
    return 0


def command_portfolio(args: argparse.Namespace) -> int:
    store = PaperPortfolioStore(get_settings().data_dir / "paper_portfolio.sqlite")
    if args.portfolio_command == "list":
        _json(store.list_trades())
    else:
        _json({"exported": str(store.export(Path(args.output).resolve()))})
    return 0


def command_train_smoke(args: argparse.Namespace) -> int:
    """Exercise the complete model/trainer/bundle path on deterministic synthetic data."""

    generator = np.random.default_rng(20260811)
    count, window, feature_count = 96, 16, 8
    features = torch.tensor(
        generator.normal(size=(count, window, feature_count)), dtype=torch.float32
    )
    target_return = features[:, -1, 0] * 0.01
    target_direction = (target_return > 0).float()
    target_volatility = features[:, :, 1].std(dim=1).abs() * 0.01
    train_loader = DataLoader(
        TensorDataset(
            features[:72], target_return[:72], target_direction[:72], target_volatility[:72]
        ),
        batch_size=24,
    )
    validation_loader = DataLoader(
        TensorDataset(
            features[72:], target_return[72:], target_direction[72:], target_volatility[72:]
        ),
        batch_size=24,
    )
    model = MultiTaskMoE(
        feature_count, embed_dim=16, router_hidden_dim=8, branch_dropout_probability=0.1
    )
    result = train_model(
        model,
        train_loader,
        validation_loader,
        TrainingConfig(epochs=args.epochs, patience=2, batch_size=24),
        device="cpu",
    )
    names = tuple(f"feature_{index}" for index in range(feature_count))
    schema = FeatureSchema("smoke-1", args.asset_class, "1d", names)
    normalization_frame = pd.DataFrame(
        generator.normal(size=(72, feature_count)),
        columns=names,
        index=pd.date_range("2020-01-01", periods=72, tz="UTC"),
    )
    normalization = NormalizationStats.fit(normalization_frame, names)
    version = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    manifest = ModelManifest(
        model_id=f"{args.asset_class}_moe_smoke",
        version=version,
        asset_class=args.asset_class,
        timeframe="1d",
        horizon_bars=1,
        feature_schema_hash=schema.schema_hash,
        status="candidate",
        date_ranges={
            "train": ["synthetic"],
            "validation": ["synthetic"],
            "test": ["locked-not-used"],
        },
        limitations=["Synthetic smoke model; never promote for research decisions."],
    )
    metrics: dict[str, object] = {
        "best_validation_loss": result.best_validation_loss,
        "best_epoch": result.best_epoch,
    }
    path = get_settings().model_dir / manifest.model_id / version
    ModelBundle.save(
        path,
        model,
        manifest,
        schema,
        normalization,
        metrics=metrics,
        model_card=render_model_card(manifest, metrics),
        training_history=result.history,
    )
    ModelRegistry(get_settings().model_dir).register(manifest, path)
    _json({"bundle": str(path), "metrics": metrics})
    return 0


def _evaluate_model(
    model: MultiTaskMoE, loader: DataLoader, *, device: str = "cpu"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actual: list[np.ndarray] = []
    predicted: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    model.to(device).eval()
    with torch.inference_mode():
        for features, target_return, _direction, _volatility in loader:
            output = model(features.to(device))
            actual.append(target_return.numpy())
            predicted.append(output.expected_return.cpu().numpy())
            probabilities.append(torch.sigmoid(output.direction_logit).cpu().numpy())
    return np.concatenate(actual), np.concatenate(predicted), np.concatenate(probabilities)


def command_train(args: argparse.Namespace) -> int:
    instrument = _instrument(args.universe, args.symbol)
    if args.input:
        source = Path(args.input).resolve()
        bars = (
            pd.read_parquet(source) if source.suffix.lower() == ".parquet" else pd.read_csv(source)
        )
        provider_name = "local"
    else:
        end = datetime.now(UTC)
        bars, _report = DataService(get_settings()).fetch(
            instrument,
            args.timeframe,
            end - timedelta(days=args.days),
            end,
            provider_name=args.provider,
        )
        provider_name = args.provider or (
            "ccxt_binance" if instrument.asset_class.value == "crypto" else "yfinance"
        )
    featured = FeaturePipeline().transform(
        bars, instrument, args.timeframe, include_targets=True, horizon_bars=args.horizon
    )
    split = purged_chronological_split(
        pd.DatetimeIndex(featured.frame.index),
        purge_bars=args.horizon,
        embargo_bars=args.horizon,
    )
    train_frame = featured.frame.loc[split.train]
    validation_frame = featured.frame.loc[split.validation]
    test_frame = featured.frame.loc[split.test]
    normalization = NormalizationStats.fit(train_frame, featured.schema.feature_names)
    train_frame = normalization.transform(train_frame)
    validation_frame = normalization.transform(validation_frame)
    test_frame = normalization.transform(test_frame)
    if min(len(train_frame), len(validation_frame), len(test_frame)) < args.window:
        raise ValueError("split is shorter than --window; fetch more history or reduce the window")
    train_loader = DataLoader(
        WindowDataset(train_frame, featured.schema.feature_names, window=args.window),
        batch_size=args.batch_size,
        shuffle=False,
    )
    validation_loader = DataLoader(
        WindowDataset(validation_frame, featured.schema.feature_names, window=args.window),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        WindowDataset(test_frame, featured.schema.feature_names, window=args.window),
        batch_size=args.batch_size,
        shuffle=False,
    )
    model = MultiTaskMoE(
        len(featured.schema.feature_names),
        embed_dim=args.embed_dim,
        router_hidden_dim=max(8, args.embed_dim // 2),
        branch_dropout_probability=0.15,
    )
    training = train_model(
        model,
        train_loader,
        validation_loader,
        TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience,
        ),
        device=args.device,
    )
    validation_actual, _validation_predicted, validation_probabilities = _evaluate_model(
        model, validation_loader, device=args.device
    )
    calibrator = ProbabilityCalibrator().fit(
        validation_probabilities, (validation_actual > 0).astype(int)
    )
    test_actual, test_predicted, test_probabilities = _evaluate_model(
        model, test_loader, device=args.device
    )
    calibrated_test = calibrator.transform(test_probabilities)
    metrics: dict[str, object] = {
        key: value
        for key, value in regression_direction_metrics(
            test_actual, test_predicted, calibrated_test
        ).items()
    }
    metrics.update(
        {
            "best_validation_loss": training.best_validation_loss,
            "best_epoch": training.best_epoch,
            "test_fold_used_for_selection": False,
        }
    )
    version = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    model_id = f"{instrument.asset_class.value}_moe"
    manifest = ModelManifest(
        model_id=model_id,
        version=version,
        asset_class=instrument.asset_class.value,
        timeframe=args.timeframe,
        horizon_bars=args.horizon,
        feature_schema_hash=featured.schema.schema_hash,
        symbols=[instrument.instrument_id],
        date_ranges=split.manifest(),
        provider=provider_name,
        hyperparameters={
            "window": args.window,
            "embed_dim": args.embed_dim,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        },
        limitations=[
            "Single-instrument candidate; validate across instruments and walk-forward folds before promotion.",
            "Free provider data may be delayed, revised or incomplete.",
        ],
    )
    bundle_path = get_settings().model_dir / model_id / version
    ModelBundle.save(
        bundle_path,
        model,
        manifest,
        featured.schema,
        normalization,
        metrics=metrics,
        calibration={
            **calibrator.to_dict(),
            **calibrator.report(test_probabilities, (test_actual > 0).astype(int)),
        },
        model_card=render_model_card(manifest, metrics),
        training_history=training.history,
    )
    ModelRegistry(get_settings().model_dir).register(manifest, bundle_path)
    _json({"bundle": str(bundle_path), "status": "candidate", "metrics": metrics})
    return 0


def command_scan(args: argparse.Namespace) -> int:
    universes = discover_universes(get_settings().config_dir)
    if args.universe not in universes:
        raise ValueError(f"unknown universe: {args.universe}")
    _json(ScannerService(get_settings()).scan(universes[args.universe], args.timeframe))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="market-moe", description="Local-first market research platform"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor = subparsers.add_parser("doctor", help="check installation and safety policy")
    doctor.set_defaults(handler=command_doctor)
    serve = subparsers.add_parser("serve", help="start localhost dashboard")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument("--allow-network", action="store_true")
    serve.set_defaults(handler=command_serve)
    universes = subparsers.add_parser("universes", help="list configured universes")
    universes.set_defaults(handler=command_universes)
    fetch = subparsers.add_parser("fetch", help="download free public market bars")
    fetch.add_argument("universe")
    fetch.add_argument("symbol")
    fetch.add_argument("--timeframe", default="1d")
    fetch.add_argument("--days", type=int, default=365)
    fetch.add_argument("--provider")
    fetch.add_argument("--raw", action="store_true")
    fetch.set_defaults(handler=command_fetch)
    models = subparsers.add_parser("models", help="inspect or promote model bundles")
    model_sub = models.add_subparsers(dest="models_command", required=True)
    model_list = model_sub.add_parser("list")
    model_list.add_argument("--status")
    promote = model_sub.add_parser("promote")
    promote.add_argument("model_id")
    promote.add_argument("version")
    promote.add_argument("status", choices=sorted(ModelRegistry.valid_statuses))
    models.set_defaults(handler=command_models)
    backtest = subparsers.add_parser("backtest", help="run cost-aware simulation on canonical bars")
    backtest.add_argument("input")
    backtest.add_argument("--signal-column", default="signal")
    backtest.add_argument("--output", default="artifacts/backtests/cli")
    backtest.set_defaults(handler=command_backtest)
    portfolio = subparsers.add_parser("portfolio", help="inspect/export the manual paper ledger")
    portfolio_sub = portfolio.add_subparsers(dest="portfolio_command", required=True)
    portfolio_sub.add_parser("list")
    export = portfolio_sub.add_parser("export")
    export.add_argument("output")
    portfolio.set_defaults(handler=command_portfolio)
    smoke = subparsers.add_parser("train-smoke", help="CPU-only trainer and bundle smoke test")
    smoke.add_argument("--asset-class", choices=["crypto", "equity"], default="crypto")
    smoke.add_argument("--epochs", type=int, default=2)
    smoke.set_defaults(handler=command_train_smoke)
    train = subparsers.add_parser("train", help="train a candidate bundle from free/local bars")
    train.add_argument("universe")
    train.add_argument("symbol")
    train.add_argument("--input")
    train.add_argument("--provider")
    train.add_argument("--timeframe", default="1d")
    train.add_argument("--days", type=int, default=2500)
    train.add_argument("--horizon", type=int, default=1)
    train.add_argument("--window", type=int, default=32)
    train.add_argument("--embed-dim", type=int, default=32)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--patience", type=int, default=5)
    train.add_argument("--learning-rate", type=float, default=2e-4)
    train.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    train.set_defaults(handler=command_train)
    scan = subparsers.add_parser("scan", help="rank cached universe data with production models")
    scan.add_argument("universe")
    scan.add_argument("--timeframe", default="1d")
    scan.set_defaults(handler=command_scan)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        return int(args.handler(args))
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
