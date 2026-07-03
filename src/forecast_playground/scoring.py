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


# --- aggregation primitives ------------------------------------------------
# Seams for ensembling: run N samples/scaffolds (your harness's job), then combine
# their probabilities here. These are pure functions — the environment supplies the
# aggregation math, the consumer supplies the orchestration.


def trimmed_mean(probs: list[float], trim: float = 0.1) -> float:
    """Mean of ``probs`` after dropping the lowest and highest ``trim`` fraction.

    Trimming discards outlier forecasts (a common, robust ensemble step). With
    ``trim=0`` this is a plain mean. Raises on an empty list.
    """
    if not probs:
        raise ValueError("trimmed_mean of empty list")
    s = sorted(probs)
    k = int(len(s) * trim)
    kept = s[k: len(s) - k] or s  # never trim everything away
    return sum(kept) / len(kept)


def extremize(prob: float, a: float = 1.5) -> float:
    """Push a probability away from 0.5 by exponent ``a`` (log-odds sharpening).

    Aggregating independent forecasts tends to be underconfident; extremizing with
    ``a > 1`` corrects that (``a = 1`` is a no-op). Operates in log-odds space and
    is symmetric around 0.5. Clipped to keep the result in (0, 1).
    """
    p = min(max(prob, _EPS), 1 - _EPS)
    odds = p / (1 - p)
    ext = odds**a
    return ext / (1 + ext)


def aggregate(probs: list[float], trim: float = 0.1, extremize_a: float = 1.0) -> float:
    """Combine an ensemble of probabilities: trimmed mean, then optional extremize.

    Args:
        probs: One probability per ensemble member (samples/scaffolds).
        trim: Fraction trimmed from each end before averaging.
        extremize_a: Extremizing exponent applied to the mean (1.0 = off).
    """
    return extremize(trimmed_mean(probs, trim=trim), a=extremize_a)
