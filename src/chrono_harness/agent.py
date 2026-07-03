"""A minimal forecasting agent loop driving Claude (via Bedrock) through the toolkit.

The model is given a question, a set of time-masked tools, and asked to research and
emit a final probability. The loop runs tool calls until the model stops, hits its
turn budget, or is truncated — then, if no probability was produced, forces one final
tools-disabled answer turn before falling back to a flagged 0.5 default.

Kept deliberately small — its purpose is to exercise the harness end-to-end and
surface friction, not to be a state-of-the-art forecaster.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .toolkit import Toolkit

# {budget} is filled with the turn budget so the model paces its research.
_SYSTEM = """You are a careful superforecaster. You will be asked a question whose \
answer is not yet known to you. Research it using ONLY the provided tools — they \
return information as it existed on the (hidden) forecast date. Do not rely on \
memory of events, since you cannot be sure of their timing relative to the forecast \
date; ground your reasoning in tool results.

You have about {budget} tool-use turns. Do not exhaust them — once you have enough \
evidence, STOP calling tools and give your answer. It is better to answer with \
partial evidence than to run out of turns.

When done, output your final answer on its own line in EXACTLY this format:
PROBABILITY: 0.XX
where 0.XX is your probability (0 to 1) that the event resolves YES."""

# Sent when the loop must extract an answer from a model that hasn't given one.
_FORCE_ANSWER = (
    "Stop researching now. Based on the evidence you have gathered, give your best "
    "final probability. Output ONLY the line: PROBABILITY: 0.XX"
)

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
    p = float(matches[-1])
    return min(max(p, 0.0), 1.0)


def _text_of(content: list[Any]) -> str:
    return "\n".join(b.text for b in content if getattr(b, "type", None) == "text")


def run_forecast(
    client: Any,
    model: str,
    question: str,
    toolkit: Toolkit,
    max_turns: int = 10,
    max_tokens: int = 4000,
) -> Forecast:
    """Drive ``client`` (an Anthropic/AnthropicBedrock client) to forecast ``question``.

    Returns a Forecast with the extracted probability and run diagnostics. If the
    model never emits a parsable probability, one final tools-disabled turn is forced;
    only if that also fails is the result defaulted to 0.5 (with ``defaulted=True``).
    """
    tools = toolkit.anthropic_tools()
    system = _SYSTEM.format(budget=max_turns)
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    last_text = ""
    stop_reason = None
    turn = 0

    for turn in range(1, max_turns + 1):
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            messages=messages,
        )
        stop_reason = resp.stop_reason
        messages.append({"role": "assistant", "content": resp.content})

        text = _text_of(resp.content)
        if text:
            last_text = text

        if stop_reason != "tool_use":
            break  # model stopped (end_turn, or truncated on max_tokens)

        tool_results = []
        for block in resp.content:
            if block.type == "tool_use":
                result = toolkit.call(block.name, dict(block.input))
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )
        messages.append({"role": "user", "content": tool_results})

    prob = _extract_probability(last_text)
    forced = False

    # If research ended without a parsable answer (ran out of turns, or was
    # truncated mid-reasoning), force one final tools-disabled answer turn.
    if prob is None:
        forced = True
        messages.append({"role": "user", "content": _FORCE_ANSWER})
        resp = client.messages.create(
            model=model,
            max_tokens=200,
            system=system,
            messages=messages,  # no tools -> must answer
        )
        stop_reason = resp.stop_reason
        forced_text = _text_of(resp.content)
        if forced_text:
            last_text = forced_text
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
