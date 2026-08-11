from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from market_moe.domain.errors import ModelCompatibilityError
from market_moe.features.normalization import NormalizationStats
from market_moe.features.schema import FeatureSchema
from market_moe.models.bundle import ModelBundle, ModelManifest
from market_moe.models.losses import multitask_loss
from market_moe.models.moe import MultiTaskMoE
from market_moe.models.registry import ModelRegistry
from market_moe.training.splits import expanding_walk_forward, purged_chronological_split


def test_router_weights_and_multitask_loss() -> None:
    model = MultiTaskMoE(8, embed_dim=16, router_hidden_dim=8)
    inputs = torch.randn(6, 20, 8)
    output = model(inputs)
    assert output.expected_return.shape == (6,)
    torch.testing.assert_close(output.expert_weights.sum(dim=1), torch.ones(6))
    loss, components = multitask_loss(
        output, torch.randn(6), torch.randint(0, 2, (6,)).float(), torch.rand(6)
    )
    assert loss.isfinite() and components["total"] > 0


def test_purged_split_and_walk_forward_are_disjoint() -> None:
    index = pd.date_range("2020-01-01", periods=400, tz="UTC")
    split = purged_chronological_split(index, purge_bars=20, embargo_bars=5)
    assert split.train.max() < split.validation.min() < split.test.min()
    assert set(split.train).isdisjoint(split.validation)
    assert "test_locked" in split.manifest()
    folds = expanding_walk_forward(
        index, minimum_train=200, validation_size=50, folds=3, purge_bars=20
    )
    assert len(folds) == 3 and len(folds[-1][0]) > len(folds[0][0])


def test_bundle_registry_and_schema_gate(tmp_path) -> None:
    names = tuple(f"f{i}" for i in range(8))
    schema = FeatureSchema("1", "crypto", "1d", names)
    index = pd.date_range("2020", periods=30, tz="UTC")
    normalization = NormalizationStats.fit(
        pd.DataFrame(np.random.default_rng(1).normal(size=(30, 8)), index=index, columns=names),
        names,
    )
    model = MultiTaskMoE(8, embed_dim=16, router_hidden_dim=8)
    manifest = ModelManifest("test", "v1", "crypto", "1d", 1, schema.schema_hash)
    bundle = tmp_path / "models" / "test" / "v1"
    ModelBundle.save(bundle, model, manifest, schema, normalization)
    loaded, loaded_manifest, loaded_schema, _ = ModelBundle.load(bundle)
    assert loaded.input_dim == 8 and loaded_manifest.version == "v1"
    assert loaded_schema.schema_hash == schema.schema_hash
    with pytest.raises(ModelCompatibilityError):
        ModelBundle.load(bundle, expected_schema_hash="wrong")
    registry = ModelRegistry(tmp_path / "models")
    registry.register(manifest, bundle)
    registry.promote("test", "v1", "production")
    assert registry.production("test", "1d") == bundle.resolve()
    payload = json.loads((bundle / "manifest.json").read_text("utf-8"))
    assert payload["feature_schema_hash"] == schema.schema_hash
