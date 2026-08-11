"""Deterministic ablation bookkeeping."""

from __future__ import annotations

from collections.abc import Callable


def run_ablations(
    evaluator: Callable[[dict[str, bool]], dict[str, float]],
) -> list[dict[str, object]]:
    variants = [
        {"auxiliary_heads": True, "branch_dropout": True, "router_balance": True},
        {"auxiliary_heads": False, "branch_dropout": True, "router_balance": True},
        {"auxiliary_heads": True, "branch_dropout": False, "router_balance": True},
        {"auxiliary_heads": True, "branch_dropout": True, "router_balance": False},
    ]
    return [{"features": variant, "metrics": evaluator(variant)} for variant in variants]
