"""A minimal, provider-agnostic forecasting agent loop.

The model is given a question, a set of time-masked tools, and asked to research and
emit a final probability. The loop runs tool calls until the model stops, hits its
turn budget, or is truncated — then, if no probability was produced, forces one final
tools-disabled answer turn before falling back to a flagged 0.5 default.

The loop talks to models through a :class:`~forecast_playground.drivers.Driver`
(OpenAI-compatible or native Anthropic/Bedrock) and takes its instructions from a
swappable :class:`~forecast_playground.scaffold.Scaffold`. Kept deliberately small —
its purpose is to exercise the harness and provide an honest baseline, not to be a
state-of-the-art forecaster.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .drivers import Driver
from .scaffold import NAIVE, Scaffold
from .toolkit import Toolkit

_PROB_RE = re.compile(r"PROBABILITY:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)


@dataclass
class Forecast:
    """The result of one agent run."""

    probability: float
    turns: int
    transcript_tail: str
    raw_final: str
    n_tool_calls: int = 0
    stop_reason: str | None = None
    defaulted: bool = False  # True if we fell back to 0.5 (no parsable answer)
    forced: bool = False  # True if the answer came from the forced final turn
    meta: dict[str, Any] = field(default_factory=dict)


def _extract_probability(text: str) -> float | None:
    matches = _PROB_RE.findall(text or "")
    if not matches:
        return None
    return min(max(float(matches[-1]), 0.0), 1.0)


def run_forecast(
    driver: Driver,
    model: str,
    question: str,
    toolkit: Toolkit,
    scaffold: Scaffold = NAIVE,
    max_turns: int = 10,
    max_tokens: int = 4000,
) -> Forecast:
    """Drive a model (via ``driver``) to forecast ``question`` using ``toolkit``.

    Args:
        driver: An OpenAIDriver or AnthropicDriver (any provider works via the former).
        model: The model id/name to pass to the driver.
        question: The forecasting question (should ask for P(YES)).
        toolkit: Time-masked tools bound to a Clock (the as-of date the model can't see).
        scaffold: The instruction/methodology strategy (default: NAIVE baseline).

    Returns a Forecast with the extracted probability and run diagnostics. If the
    model never emits a parsable probability, one final tools-disabled turn is forced;
    only if that also fails is the result defaulted to 0.5 (with ``defaulted=True``).
    """
    tools = toolkit.tool_defs()
    system = scaffold.system(max_turns)
    messages = driver.start(question)
    last_text = ""
    stop_reason = None
    turn = 0

    for turn in range(1, max_turns + 1):
        t = driver.step(model, system, messages, tools, max_tokens)
        stop_reason = t.stop_reason
        if t.text:
            last_text = t.text

        if not t.tool_calls:
            break  # model produced its final answer (or was truncated)

        results = [(inv, toolkit.call(inv.name, inv.args)) for inv in t.tool_calls]
        driver.add_tool_results(messages, results)

    prob = _extract_probability(last_text)
    forced = False

    # If research ended without a parsable answer (ran out of turns, or was
    # truncated mid-reasoning), force one final tools-disabled answer turn.
    if prob is None:
        forced = True
        driver.add_user(messages, scaffold.force_answer)
        t = driver.step(model, system, messages, tools=None, max_tokens=200)
        stop_reason = t.stop_reason
        if t.text:
            last_text = t.text
        prob = _extract_probability(last_text)

    defaulted = prob is None
    return Forecast(
        probability=0.5 if defaulted else prob,
        turns=turn,
        transcript_tail=last_text[-1200:],
        raw_final=last_text,
        n_tool_calls=len(toolkit.calls),
        stop_reason=stop_reason,
        defaulted=defaulted,
        forced=forced,
    )
