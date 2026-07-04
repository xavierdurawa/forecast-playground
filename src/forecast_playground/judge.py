"""LLM-as-judge scorer for looser, non-boolean forecasts.

Some questions don't have a knife-edge boolean settlement (an analyst's "how likely
is US–Iran escalation this spring" vs. a market's exact clause). To score those on
the same 0–1, lower-is-better scale as Brier, an LLM judge — given the question, the
forecaster's probability, and the *now-known resolution* — maps what actually
happened onto a soft ``resolved_label`` ∈ [0, 1], and we take the ordinary Brier
distance to it. For a boolean question the label is 0/1 and this *is* exact Brier.

Two important properties (see proposal 0001):

- **Not part of the leak-safety invariant.** The judge scores *after* resolution and
  does no retrieval — it never sees the Clock or a Source, so it cannot leak. It is,
  however, non-deterministic (an LLM) — unlike ``scoring.py``'s pure functions — so
  it lives behind the ``[judge]`` extra and its network call is separate from the
  pure reduction below.
- **Label must be grounded in the OUTCOME, not the forecast's self-consistency.**
  Scoring how well a forecast's own claim was "borne out" inverts (a confident-wrong
  forecast looks perfect). The judge is prompted to label what *happened*.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .scoring import brier_score

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_SYSTEM = """You are scoring a forecaster against a KNOWN outcome.

You are given a question, its resolution criteria, what ACTUALLY happened (the
resolution), and the forecaster's probability + reasoning. Judge based on what
actually happened — NOT on whether the forecaster's own argument was internally
consistent.

Return ONLY a JSON object with these fields, each a number in [0,1]:
  "resolved_label": the degree to which the YES/claim actually came true given the
      resolution (1.0 = fully happened, 0.0 = did not; use the middle only for
      genuinely partial outcomes). This is about the WORLD, not the forecaster.
  "accuracy": how close the forecaster's probability was to what happened.
  "calibration": whether the confidence was warranted.
  "reasoning": quality of the reasoning given what was knowable at forecast time.
  "decision_usefulness": how useful this forecast would have been to a decision-maker.
Output only the JSON, no prose."""

_USER = """QUESTION: {question}

RESOLUTION CRITERIA: {criteria}

WHAT ACTUALLY HAPPENED (resolution): {resolution}

FORECASTER'S PROBABILITY (of YES): {probability}
FORECASTER'S REASONING: {reasoning}"""


def brier_from_judgment(forecaster_probability: float, resolved_label: float) -> float:
    """Brier-compatible score from a judged soft outcome label. Lower is better.

    Pure and deterministic — this is the part all consumers should share. For a
    boolean question ``resolved_label`` is 0/1 and this equals the ordinary Brier
    score; for a loose question the judge supplies the soft label.
    """
    return brier_score(forecaster_probability, resolved_label)


@dataclass
class Judgment:
    """The result of judging one forecast against its known resolution."""

    resolved_label: float
    brier: float  # brier_from_judgment(probability, resolved_label)
    rubric: dict[str, float]  # accuracy / calibration / reasoning / decision_usefulness
    raw: str = ""
    parsed: bool = True  # False if the judge output couldn't be parsed (label defaulted)
    meta: dict[str, Any] = field(default_factory=dict)


def _clip01(x: float) -> float:
    return min(max(float(x), 0.0), 1.0)


def parse_judgment(text: str, probability: float) -> Judgment:
    """Parse the judge's JSON reply into a Judgment (pure; offline-testable).

    On unparseable output, ``resolved_label`` defaults to 0.5 (max-entropy) and
    ``parsed=False`` so callers can see the judge failed rather than trust a guess.
    """
    match = _JSON_RE.search(text or "")
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            data = None
    else:
        data = None

    if not isinstance(data, dict) or "resolved_label" not in data:
        return Judgment(
            resolved_label=0.5,
            brier=brier_from_judgment(probability, 0.5),
            rubric={},
            raw=text,
            parsed=False,
        )

    label = _clip01(data["resolved_label"])
    rubric = {
        k: _clip01(data[k])
        for k in ("accuracy", "calibration", "reasoning", "decision_usefulness")
        if k in data
    }
    return Judgment(
        resolved_label=label,
        brier=brier_from_judgment(probability, label),
        rubric=rubric,
        raw=text,
    )


def judge_forecast(
    driver: Any,
    model: str,
    question: str,
    probability: float,
    resolution: str,
    reasoning: str = "",
    criteria: str = "",
    max_tokens: int = 500,
) -> Judgment:
    """Score a forecast against a known resolution using an LLM judge (via a Driver).

    Args:
        driver: An OpenAIDriver / AnthropicDriver (any provider).
        model: The judge model id.
        question: The forecasting question.
        probability: The forecaster's P(YES) in [0, 1].
        resolution: What actually happened (the now-known outcome, in words or 0/1).
        reasoning: The forecaster's reasoning (optional context for the rubric).
        criteria: The question's resolution criteria (optional).

    Returns a :class:`Judgment`. Non-deterministic (it calls a model); the
    Brier-compatible number comes from the pure :func:`brier_from_judgment`.
    """
    user = _USER.format(
        question=question,
        criteria=criteria or "(none given)",
        resolution=resolution,
        probability=probability,
        reasoning=reasoning or "(none given)",
    )
    turn = driver.step(model, _SYSTEM, driver.start(user), tools=None, max_tokens=max_tokens)
    return parse_judgment(turn.text, probability)
