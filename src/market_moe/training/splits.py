"""Chronological splits with purge, embargo and a locked test fold."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class PurgedSplit:
    train: pd.DatetimeIndex
    validation: pd.DatetimeIndex
    test: pd.DatetimeIndex
    purge_bars: int
    embargo_bars: int

    def manifest(self) -> dict[str, object]:
        def bounds(index: pd.DatetimeIndex) -> list[str]:
            return [] if index.empty else [index.min().isoformat(), index.max().isoformat()]

        return {
            "train": bounds(self.train),
            "validation": bounds(self.validation),
            "test_locked": bounds(self.test),
            "purge_bars": self.purge_bars,
            "embargo_bars": self.embargo_bars,
        }


def purged_chronological_split(
    index: pd.DatetimeIndex,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    purge_bars: int,
    embargo_bars: int,
) -> PurgedSplit:
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("split index must be unique and chronological")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train + validation must leave a test fold")
    size = len(index)
    first = int(size * train_fraction)
    second = int(size * (train_fraction + validation_fraction))
    train_end = max(0, first - purge_bars)
    validation_start = min(size, first + embargo_bars)
    validation_end = max(validation_start, second - purge_bars)
    test_start = min(size, second + embargo_bars)
    split = PurgedSplit(
        train=index[:train_end],
        validation=index[validation_start:validation_end],
        test=index[test_start:],
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
    )
    if any(part.empty for part in (split.train, split.validation, split.test)):
        raise ValueError("not enough rows for requested purge/embargo split")
    return split


def expanding_walk_forward(
    index: pd.DatetimeIndex,
    *,
    minimum_train: int,
    validation_size: int,
    folds: int = 3,
    purge_bars: int = 0,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    if minimum_train + folds * validation_size + purge_bars > len(index):
        raise ValueError("not enough observations for walk-forward folds")
    results = []
    for fold in range(folds):
        validation_start = minimum_train + fold * validation_size
        train = index[: max(0, validation_start - purge_bars)]
        validation = index[validation_start : validation_start + validation_size]
        results.append((train, validation))
    return results
