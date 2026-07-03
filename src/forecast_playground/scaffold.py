"""Scaffolds: the instructions/methodology wrapped around a fixed model.

A forecaster can be improved two ways: change the weights (RL) or change the
scaffold around fixed weights (prompt + methodology + tool-use strategy). This module
is the second path — and, because a well-scaffolded base model is exactly what you'd
initialize RL from, it also defines the *baseline* the training path is measured
against.

A :class:`Scaffold` is deliberately small: a system prompt (told its turn budget) and
the message used to force a final answer. Swap it to test whether a model is bad at
forecasting or just badly prompted.
"""

from __future__ import annotations

from dataclasses import dataclass

_FORMAT = (
    "\n\nWhen done, output your final answer on its own line in EXACTLY this format:\n"
    "PROBABILITY: 0.XX\n"
    "where 0.XX is your probability (0 to 1) that the event resolves YES."
)

_BUDGET = (
    "\n\nYou have about {budget} tool-use turns. Do not exhaust them — once you have "
    "enough evidence, STOP calling tools and answer. It is better to answer with "
    "partial evidence than to run out of turns."
)


@dataclass
class Scaffold:
    """A named forecasting strategy: system instructions + force-answer prompt.

    Attributes:
        name: Short identifier used in study output.
        instructions: The core system prompt (methodology). The turn-budget and
            output-format boilerplate are appended automatically by :meth:`system`.
        force_answer: Message sent to extract an answer if the model never gave one.
    """

    name: str
    instructions: str
    force_answer: str = (
        "Stop researching now. Based on the evidence you have gathered, give your "
        "best final probability. Output ONLY the line: PROBABILITY: 0.XX"
    )

    def system(self, budget: int) -> str:
        """The full system prompt for a run with ``budget`` tool-use turns."""
        return self.instructions + _BUDGET.format(budget=budget) + _FORMAT


# The current default: a light touch. This is roughly what produced the honest
# baseline (base model loses to the market on uncertain questions).
NAIVE = Scaffold(
    name="naive",
    instructions=(
        "You are a careful superforecaster. You will be asked a question whose answer "
        "is not yet known to you. Research it using ONLY the provided tools — they "
        "return information as it existed on the (hidden) forecast date. Do not rely "
        "on memory of events, since you cannot be sure of their timing relative to "
        "the forecast date; ground your reasoning in tool results."
    ),
)

# A methodology-heavy scaffold applying superforecasting practice. The hypothesis:
# structured technique lifts a base model without touching its weights.
SUPERFORECASTER = Scaffold(
    name="superforecaster",
    instructions=(
        "You are an elite superforecaster. Answer by following this method, and only "
        "using the provided tools (which return information as of the hidden forecast "
        "date — do not trust your memory of event timing):\n"
        "1. BASE RATE: What is the outside-view base rate for this class of event? "
        "Look up reference cases before considering specifics.\n"
        "2. EVIDENCE: Gather the current situation from the tools. Note what is known "
        "as of the forecast date and, crucially, what is still uncertain.\n"
        "3. BOTH SIDES: State the strongest case for YES and the strongest case for "
        "NO. Actively look for disconfirming evidence.\n"
        "4. UPDATE: Start from the base rate and adjust for the specific evidence. "
        "Make small, justified updates — avoid overconfidence.\n"
        "5. CALIBRATE: If genuinely uncertain, stay near the base rate rather than "
        "committing to an extreme probability. A confident wrong answer is heavily "
        "penalized by the score."
    ),
)

# Registry for CLI/study selection by name.
SCAFFOLDS = {s.name: s for s in (NAIVE, SUPERFORECASTER)}
