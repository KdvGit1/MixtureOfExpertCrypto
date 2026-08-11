"""Prediction metrics and confidence intervals."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, mean_absolute_error


def regression_direction_metrics(
    actual: np.ndarray, predicted: np.ndarray, probability_up: np.ndarray
) -> dict[str, float]:
    labels = (actual > 0).astype(int)
    directions = (probability_up >= 0.5).astype(int)
    pearson = pearsonr(actual, predicted).statistic if len(actual) > 2 else float("nan")
    spearman = spearmanr(actual, predicted).statistic if len(actual) > 2 else float("nan")
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "sign_accuracy": float((np.sign(actual) == np.sign(predicted)).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, directions)),
        "brier_score": float(brier_score_loss(labels, probability_up)),
        "pearson_ic": float(pearson),
        "rank_ic": float(spearman),
    }


def bootstrap_mean_interval(
    values: np.ndarray, *, samples: int = 1_000, seed: int = 20260811
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    generator = np.random.default_rng(seed)
    means = generator.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
