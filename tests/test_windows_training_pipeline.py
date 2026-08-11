from __future__ import annotations

from pathlib import Path

import torch
from conftest import canonical_bars
from torch.utils.data import DataLoader, TensorDataset

from market_moe.domain.instruments import AssetClass, Instrument
from market_moe.models.bundle import ModelBundle
from market_moe.models.moe import MultiTaskMoE
from market_moe.training.pipeline import PooledTrainingSpec, train_pooled_model
from market_moe.training.trainer import TrainingConfig, train_model


def test_trainer_resumes_epoch_checkpoint(tmp_path: Path) -> None:
    features = torch.randn(32, 8, 6)
    returns = features[:, -1, 0] * 0.01
    dataset = TensorDataset(features, returns, (returns > 0).float(), returns.abs())
    loader = DataLoader(dataset, batch_size=16)
    checkpoint = tmp_path / "checkpoint.pt"
    first = MultiTaskMoE(6, embed_dim=8, router_hidden_dim=4)
    train_model(
        first,
        loader,
        loader,
        TrainingConfig(epochs=1, patience=3, batch_size=16),
        device="cpu",
        checkpoint_path=checkpoint,
    )
    resumed = MultiTaskMoE(6, embed_dim=8, router_hidden_dim=4)
    result = train_model(
        resumed,
        loader,
        loader,
        TrainingConfig(epochs=2, patience=3, batch_size=16),
        device="cpu",
        checkpoint_path=checkpoint,
        resume=True,
    )
    assert checkpoint.exists()
    assert [row["epoch"] for row in result.history] == [0.0, 1.0]


def test_pooled_training_writes_candidate_and_backtests(
    tmp_path: Path, crypto_instrument: Instrument
) -> None:
    second = Instrument.create(
        symbol="ETHUSDT",
        asset_class=AssetClass.CRYPTO,
        exchange_mic="BINANCE",
        currency="USDT",
        timezone="UTC",
        calendar="24/7",
        provider_symbols={"ccxt": "ETH/USDT"},
    )
    sources = [
        (crypto_instrument, canonical_bars(crypto_instrument, rows=520)),
        (second, canonical_bars(second, rows=520)),
    ]
    spec = PooledTrainingSpec(
        job_id="test_crypto_1d",
        model_id="crypto_moe",
        asset_class="crypto",
        timeframe="1d",
        version="test-v1",
        window=8,
        embed_dim=8,
        router_hidden_dim=4,
        batch_size=128,
        epochs=1,
        patience=2,
        branch_dropout_start_epoch=1,
        providers=("fixture",),
    )
    bundle = tmp_path / "models" / "crypto_moe" / "test-v1"
    result = train_pooled_model(
        sources,
        spec,
        bundle_path=bundle,
        backtest_root=tmp_path / "backtests",
        registry_root=tmp_path / "models",
        checkpoint_path=tmp_path / "checkpoint.pt",
        device="cpu",
    )
    assert ModelBundle.required_files <= {path.name for path in bundle.iterdir()}
    assert result.manifest.status == "candidate"
    assert result.metrics["test_fold_used_for_selection"] is False
    assert len(result.backtests) == 2
    assert len(list((tmp_path / "backtests").rglob("report.html"))) == 2
