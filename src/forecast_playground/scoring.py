"""Proper scoring rules for binary forecasts.

Brier and log loss. Per Bereket & Leskovec / Turtel et al., these are the right
reward signals for forecasting RL — and (separately) do NOT divide GRPO advantages
by the group standard deviation, which breaks the proper-scoring property.
"""

from __future__ import annotations

import math

_EPS = 1e-15


def brier_score(prob: float, outcome: int) -> float:
    """Brier score for a single binary forecast. Lower is better; range [0, 1].

    Args:
        prob: Forecast probability that the event happens (outcome == 1).
        outcome: Actual outcome, 1 or 0.
    """
    return (prob - outcome) ** 2


def log_score(prob: float, outcome: int) -> float:
    """Negative log loss for a single binary forecast. Lower is better.

    Clipped to avoid infinite loss on a confident miss.
    """
    p = min(max(prob, _EPS), 1 - _EPS)
    return -(outcome * math.log(p) + (1 - outcome) * math.log(1 - p))


def mean_brier(forecasts: list[tuple[float, int]]) -> float:
    """Mean Brier score over (prob, outcome) pairs."""
    if not forecasts:
        return float("nan")
    return sum(brier_score(p, o) for p, o in forecasts) / len(forecasts)


def mean_log(forecasts: list[tuple[float, int]]) -> float:
    """Mean log loss over (prob, outcome) pairs."""
    if not forecasts:
        return float("nan")
    return sum(log_score(p, o) for p, o in forecasts) / len(forecasts)
