"""Small probability calibration helper fitted on validation data only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss


@dataclass(slots=True)
class ProbabilityCalibrator:
    model: IsotonicRegression | None = None

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> ProbabilityCalibrator:
        self.model = IsotonicRegression(out_of_bounds="clip").fit(probabilities, labels)
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("calibrator has not been fitted")
        return np.asarray(self.model.predict(probabilities), dtype=float)

    def report(self, probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        calibrated = self.transform(probabilities)
        return {
            "brier_before": float(brier_score_loss(labels, probabilities)),
            "brier_after": float(brier_score_loss(labels, calibrated)),
        }

    def to_dict(self) -> dict[str, object]:
        if self.model is None:
            raise RuntimeError("calibrator has not been fitted")
        return {
            "method": "isotonic",
            "x_thresholds": self.model.X_thresholds_.tolist(),
            "y_thresholds": self.model.y_thresholds_.tolist(),
        }
