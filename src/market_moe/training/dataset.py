"""Windowed time-series dataset without future features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset[tuple[torch.Tensor, ...]]):
    def __init__(
        self,
        frame: pd.DataFrame,
        feature_names: tuple[str, ...],
        *,
        window: int,
    ) -> None:
        required = [
            *feature_names,
            "target_log_return",
            "target_direction",
            "target_volatility",
        ]
        missing = [column for column in required if column not in frame]
        if missing:
            raise ValueError(f"dataset columns missing: {missing}")
        if len(frame) < window:
            raise ValueError("dataset shorter than model window")
        self.features = frame.loc[:, feature_names].to_numpy(dtype=np.float32, copy=True)
        self.targets = frame.loc[:, required[-3:]].to_numpy(dtype=np.float32, copy=True)
        self.window = window

    def __len__(self) -> int:
        return len(self.features) - self.window + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        target_index = index + self.window - 1
        window = torch.from_numpy(self.features[index : target_index + 1])
        targets = torch.from_numpy(self.targets[target_index])
        return window, targets[0], targets[1], targets[2]
