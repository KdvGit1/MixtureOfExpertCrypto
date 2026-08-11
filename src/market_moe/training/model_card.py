"""Generate a compact, reproducible model card."""

from __future__ import annotations

from market_moe.models.bundle import ModelManifest


def render_model_card(manifest: ModelManifest, metrics: dict[str, object]) -> str:
    metric_lines = "\n".join(f"- `{name}`: {value}" for name, value in sorted(metrics.items()))
    limitations = "\n".join(f"- {item}" for item in manifest.limitations) or "- None recorded"
    return f"""# {manifest.model_id} / {manifest.version}

## Intended use

Research and decision support for {manifest.asset_class} bars at {manifest.timeframe}.
It does not place orders and is not investment advice.

## Validation metrics

{metric_lines or "- Not evaluated"}

## Data and split

- Provider: {manifest.provider}
- Symbols: {", ".join(manifest.symbols) or "not recorded"}
- Locked chronological test fold: {manifest.date_ranges.get("test", [])}
- Feature schema: `{manifest.feature_schema_hash}`

## Limitations

{limitations}
"""
