"""Versioned YAML market-universe loading."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from market_moe.domain.instruments import Instrument


class Universe(BaseModel):
    model_config = ConfigDict(frozen=True)

    universe_id: str
    display_name: str
    as_of: str
    source: str
    survivorship_warning: bool = True
    instruments: tuple[Instrument, ...]


def load_universe(path: Path) -> Universe:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid universe file: {path}")
    return Universe.model_validate(payload)


def discover_universes(config_dir: Path) -> dict[str, Path]:
    root = config_dir / "universes"
    discovered: dict[str, Path] = {}
    for path in sorted(root.glob("*.yaml")):
        universe = load_universe(path)
        discovered[universe.universe_id] = path
    return discovered
