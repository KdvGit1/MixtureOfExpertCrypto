"""Versioned decision layer kept separate from predictions."""

from __future__ import annotations

import numpy as np

from market_moe.domain.predictions import Prediction
from market_moe.domain.signals import Signal, SignalAction


def prediction_to_signal(
    prediction: Prediction,
    *,
    estimated_round_trip_cost: float,
    minimum_probability: float = 0.58,
    minimum_edge: float = 0.001,
    allow_short: bool = False,
    strategy_version: str = "research_v1",
) -> Signal:
    edge = float(np.expm1(prediction.expected_log_return)) - estimated_round_trip_cost
    uncertainty_penalty = min(1.0, prediction.uncertainty)
    score = float(
        np.clip((prediction.probability_up - 0.5) * 2 - uncertainty_penalty * 0.25, -1, 1)
    )
    reasons = []
    if prediction.data_freshness_seconds > 86_400:
        action = SignalAction.NEUTRAL
        reasons.append("stale_data")
    elif prediction.probability_up >= minimum_probability and edge >= minimum_edge:
        action = SignalAction.STRONG_LONG if score >= 0.65 else SignalAction.LONG
        reasons.extend(("probability_gate_passed", "cost_adjusted_edge_positive"))
    elif (
        allow_short
        and prediction.probability_up <= 1 - minimum_probability
        and edge <= -minimum_edge
    ):
        action = SignalAction.STRONG_SHORT if score <= -0.65 else SignalAction.SHORT
        reasons.extend(("down_probability_gate_passed", "negative_edge"))
    else:
        action = SignalAction.NEUTRAL
        reasons.append("acceptance_gate_not_passed")
    return Signal(
        instrument_id=prediction.instrument_id,
        as_of_utc=prediction.as_of_utc,
        action=action,
        score=score,
        reason_codes=tuple(reasons),
        expected_edge_after_cost=edge,
        risk_level="high" if prediction.uncertainty > 0.5 else "normal",
        prediction_id=prediction.prediction_id,
        strategy_version=strategy_version,
    )
