from __future__ import annotations

import sys
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from market_moe.backtest.accounting import Account
from market_moe.backtest.engine import BacktestConfig
from market_moe.backtest.execution import stop_target_exit
from market_moe.backtest.walk_forward import run_walk_forward_backtests
from market_moe.data.calendars import MarketCalendarService
from market_moe.data.catalog import DataCatalog
from market_moe.data.providers.ccxt_provider import CCXTPublicProvider
from market_moe.data.providers.local_provider import LocalFileProvider
from market_moe.data.providers.stooq_provider import StooqProvider
from market_moe.data.providers.yfinance_provider import YFinanceProvider
from market_moe.domain.errors import DataProviderError
from market_moe.features.normalization import NormalizationStats
from market_moe.features.pipeline import FeaturePipeline
from market_moe.models.bundle import ModelManifest
from market_moe.models.calibration import ProbabilityCalibrator
from market_moe.models.inference import predict
from market_moe.models.moe import MultiTaskMoE
from market_moe.services.data_service import DataService
from market_moe.services.scanner import rank_predictions
from market_moe.settings import Settings
from market_moe.training.baselines import baseline_predictions
from market_moe.training.dataset import WindowDataset
from market_moe.training.evaluator import bootstrap_mean_interval, regression_direction_metrics
from market_moe.training.experiments import run_ablations
from market_moe.training.trainer import TrainingConfig, train_model


def test_accounting_and_intrabar_conservative_policy() -> None:
    account = Account("USD", 1_000, quantity=10)
    assert account.equity(20) == 1_200
    account.apply_split(2)
    account.apply_dividend(1)
    assert account.quantity == 20 and account.cash == 1_020
    decision = stop_target_exit(
        position=10,
        entry_price=100,
        high=120,
        low=80,
        stop_loss_fraction=0.1,
        take_profit_fraction=0.1,
    )
    assert decision.reason == "stop_loss" and decision.intrabar_ambiguous
    short = stop_target_exit(
        position=-2,
        entry_price=100,
        high=105,
        low=85,
        stop_loss_fraction=0.1,
        take_profit_fraction=0.1,
    )
    assert short.reason == "take_profit"
    with pytest.raises(ValueError):
        account.equity(20, 0)


def test_calendar_crypto_and_equity_sessions(crypto_instrument, equity_instrument) -> None:
    service = MarketCalendarService()
    crypto = service.context(crypto_instrument, pd.Timestamp("2026-01-01T00:10:00Z"))
    assert crypto.is_regular_session and crypto.is_opening_window
    equity = service.context(equity_instrument, pd.Timestamp("2026-01-05T15:00:00Z"))
    assert equity.is_regular_session and 0 <= equity.session_progress <= 1
    closed = service.context(equity_instrument, pd.Timestamp("2026-01-04T15:00:00Z"))
    assert not closed.is_regular_session


def test_training_inference_calibration_and_scanner(crypto_bars, crypto_instrument) -> None:
    featured = FeaturePipeline().transform(
        crypto_bars, crypto_instrument, "1d", include_targets=True
    )
    names = featured.schema.feature_names
    normalization = NormalizationStats.fit(featured.frame.iloc[:60], names)
    normalized = normalization.transform(featured.frame)
    dataset = WindowDataset(normalized, names, window=12)
    train_loader = DataLoader(torch.utils.data.Subset(dataset, range(0, 50)), batch_size=25)
    validation_loader = DataLoader(torch.utils.data.Subset(dataset, range(50, 70)), batch_size=20)
    model = MultiTaskMoE(len(names), embed_dim=8, router_hidden_dim=8)
    result = train_model(
        model,
        train_loader,
        validation_loader,
        TrainingConfig(epochs=1, patience=1, branch_dropout_start_epoch=0),
        device="cpu",
    )
    assert result.best_epoch == 0
    manifest = ModelManifest("crypto_moe", "test", "crypto", "1d", 1, featured.schema.schema_hash)
    prediction = predict(
        model,
        featured.frame,
        crypto_instrument,
        manifest,
        featured.schema,
        normalization,
        window=12,
        predicted_at=featured.frame.index[-1].to_pydatetime(),
    )
    assert 0 <= prediction.probability_up <= 1
    ranked = rank_predictions([prediction], estimated_round_trip_cost=0.001)
    assert ranked[0].instrument_id == crypto_instrument.instrument_id

    probabilities = np.array([0.1, 0.25, 0.7, 0.9])
    labels = np.array([0, 0, 1, 1])
    calibrator = ProbabilityCalibrator().fit(probabilities, labels)
    assert calibrator.report(probabilities, labels)["brier_after"] <= 0.5


def test_baselines_evaluation_and_ablations() -> None:
    frame = pd.DataFrame(
        {
            "log_return_1": np.linspace(-0.02, 0.03, 60),
            "close": np.linspace(90, 120, 60),
        }
    )
    baselines = baseline_predictions(frame)
    assert set(baselines) == {
        "zero",
        "naive_momentum",
        "moving_average_trend",
        "random_signal",
        "volatility_scaled_momentum",
    }
    actual = np.array([-0.1, 0.2, 0.1, -0.05])
    metrics = regression_direction_metrics(actual, actual * 0.8, np.array([0.1, 0.8, 0.7, 0.2]))
    assert metrics["sign_accuracy"] == 1
    low, high = bootstrap_mean_interval(actual, samples=100)
    assert low < high
    ablations = run_ablations(lambda flags: {"score": float(sum(flags.values()))})
    assert len(ablations) == 4


def test_local_provider_catalog_and_data_service(tmp_path, equity_instrument, equity_bars) -> None:
    imports = tmp_path / "imports"
    imports.mkdir()
    source = imports / "aapl.csv"
    equity_bars[["open_time_utc", "open", "high", "low", "close", "volume"]].to_csv(
        source, index=False
    )
    provider = LocalFileProvider({equity_instrument.instrument_id: source}, allowed_root=imports)
    frame = provider.fetch_bars(
        equity_instrument,
        "1d",
        datetime(2020, 1, 1, tzinfo=UTC),
        datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert len(frame) > 0 and provider.healthcheck()["authenticated"] is False
    catalog = DataCatalog(tmp_path / "catalog.duckdb")
    assert catalog.list_datasets() == []
    settings = Settings(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        config_dir=tmp_path / "configs",
    )
    service = DataService(settings)
    assert isinstance(service.provider("yfinance"), YFinanceProvider)
    assert isinstance(service.provider("stooq"), StooqProvider)
    assert isinstance(service.provider("ccxt"), CCXTPublicProvider)
    assert service.provider("ccxt_bitget").exchange_name == "bitget"
    with pytest.raises(ValueError):
        service.provider("paid-provider")


def test_free_provider_adapters_with_local_fakes(
    monkeypatch, crypto_instrument, equity_instrument
) -> None:
    class FakeExchange:
        rateLimit = 0

        def fetch_ohlcv(self, _symbol, _timeframe, since, limit):
            del since, limit
            return [[1_700_000_000_000, 100, 102, 99, 101, 10]]

    ccxt = CCXTPublicProvider()
    ccxt._exchange = FakeExchange()
    crypto = ccxt.fetch_bars(
        crypto_instrument,
        "1d",
        datetime(2023, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 1, tzinfo=UTC),
    )
    assert len(crypto) == 1

    history = pd.DataFrame(
        {"Open": [100], "High": [102], "Low": [99], "Close": [101], "Volume": [1000]},
        index=pd.DatetimeIndex(["2025-01-02"], tz="America/New_York"),
    )
    fake_yfinance = SimpleNamespace(
        __version__="test",
        Ticker=lambda _symbol: SimpleNamespace(history=lambda **_kwargs: history),
    )
    monkeypatch.setitem(sys.modules, "yfinance", fake_yfinance)
    equity = YFinanceProvider().fetch_bars(
        equity_instrument,
        "1d",
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2025, 2, 1, tzinfo=UTC),
    )
    assert len(equity) == 1
    with pytest.raises(DataProviderError):
        StooqProvider().fetch_bars(
            equity_instrument,
            "1h",
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2025, 2, 1, tzinfo=UTC),
        )


def test_walk_forward_backtest(crypto_bars) -> None:
    bars = crypto_bars.iloc[:30].copy()
    folds = [
        pd.DatetimeIndex(bars["open_time_utc"].iloc[:15]),
        pd.DatetimeIndex(bars["open_time_utc"].iloc[15:]),
    ]

    def signals(frame: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=pd.to_datetime(frame["open_time_utc"], utc=True))

    results = run_walk_forward_backtests(folds, bars, signals, BacktestConfig())
    assert len(results) == 2
