from __future__ import annotations

import numpy as np

from market_moe.features.normalization import NormalizationStats
from market_moe.features.pipeline import FeaturePipeline


def test_crypto_and_equity_have_domain_features(
    crypto_bars, equity_bars, crypto_instrument, equity_instrument
) -> None:
    pipeline = FeaturePipeline()
    crypto = pipeline.transform(crypto_bars, crypto_instrument, "1d")
    equity = pipeline.transform(equity_bars, equity_instrument, "1d")
    assert "is_weekend" in crypto.frame
    assert "overnight_gap" in equity.frame
    assert crypto.schema.asset_class == "crypto"
    assert equity.schema.asset_class == "equity"
    assert crypto.schema.schema_hash != equity.schema.schema_hash


def test_feature_pipeline_does_not_read_future_rows(crypto_bars, crypto_instrument) -> None:
    pipeline = FeaturePipeline()
    full = pipeline.transform(crypto_bars, crypto_instrument, "1d", drop_incomplete=False).frame
    cutoff = 260
    truncated = pipeline.transform(
        crypto_bars.iloc[:cutoff], crypto_instrument, "1d", drop_incomplete=False
    ).frame
    names = pipeline.feature_names(crypto_instrument)
    np.testing.assert_allclose(
        full.loc[truncated.index[-1], names],
        truncated.iloc[-1][list(names)],
        equal_nan=True,
    )


def test_train_only_normalization_round_trip(tmp_path, crypto_bars, crypto_instrument) -> None:
    result = FeaturePipeline().transform(crypto_bars, crypto_instrument, "1d")
    stats = NormalizationStats.fit(result.frame.iloc[:60], result.schema.feature_names)
    transformed = stats.transform(result.frame.iloc[:60])
    assert float(transformed.mean().abs().max()) < 1e-10
    path = tmp_path / "normalization.json"
    stats.save(path)
    assert NormalizationStats.load(path).normalization_id == stats.normalization_id
