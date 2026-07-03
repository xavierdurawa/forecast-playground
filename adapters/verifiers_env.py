"""verifiers adapter: expose ForecastPlayground as an RL/eval Environment.

``load_environment()`` returns a ``ToolEnv`` whose retrieval tools are time-masked
per dataset row. The as-of date travels in each row's ``info["as_of"]`` and is
turned into a Clock by the env — never a model-supplied tool argument, so the model
cannot widen its own time window. Scoring uses the Brier score (a proper scoring
rule) against the row's known outcome.

Requires the ``verifiers`` extra:  pip install -e ".[verifiers]"

Design note: verifiers' ToolEnv calls tools with only model-supplied args, so we
subclass it and inject the row's Clock in ``call_tool`` (the idiomatic extension
point). This keeps the time-mask out of the model's hands while staying compatible
with verifiers' fixed-tool model.
"""

from __future__ import annotations

import re
from typing import Any

try:
    import verifiers as vf
    from datasets import Dataset
except ImportError as e:  # pragma: no cover - guidance for a missing optional dep
    raise ImportError(
        "The verifiers adapter needs the 'verifiers' extra: "
        'pip install -e ".[verifiers]"'
    ) from e

from forecast_playground import (
    Clock,
    PageviewsSource,
    PolymarketSource,
    WikipediaSource,
    brier_score,
)
from forecast_playground.toolkit import Toolkit

_PROB_RE = re.compile(r"PROBABILITY:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)

_SYSTEM = (
    "You are a careful superforecaster. Research the question using ONLY the "
    "provided tools, which return information as it existed on the (hidden) forecast "
    "date. End with a line 'PROBABILITY: 0.XX' giving P(event resolves YES)."
)


def _extract_probability(text: str) -> float:
    m = _PROB_RE.findall(text or "")
    if not m:
        return 0.5  # no parsable answer -> max-entropy guess
    return min(max(float(m[-1]), 0.0), 1.0)


class TimeMaskedToolEnv(vf.ToolEnv):
    """ToolEnv that binds a per-row Clock (from ``info['as_of']``) to each rollout.

    The Clock is stored on state in ``setup_state`` and injected into every tool
    call, so retrieval is masked to the row's as-of date regardless of what the
    model passes.
    """

    def __init__(self, source_specs: list[dict[str, Any]], **kwargs: Any) -> None:
        self._source_specs = source_specs
        # Placeholder tools just so verifiers can build schemas; real dispatch is
        # overridden in call_tool. We expose one search tool per source + python.
        self._tool_names = [s["tool_name"] for s in source_specs] + ["run_python"]
        super().__init__(tools=[], **kwargs)
        # Build tool_defs/tool_map ourselves from a throwaway Toolkit.
        probe = self._make_toolkit(Clock.at("2024-01-01"))
        self.tool_defs = probe.anthropic_tools_as_openai()
        self.tool_map = {d["name"]: d["name"] for d in self.tool_defs}

    def _make_toolkit(self, clock: Clock) -> Toolkit:
        sources = [spec["factory"]() for spec in self._source_specs]
        return Toolkit(clock=clock, sources=sources)

    async def setup_state(self, state: vf.State) -> None:
        info = state.get("info", {}) or {}
        as_of = info.get("as_of")
        state["fp_clock"] = Clock.at(as_of) if as_of else Clock.at("2024-01-01")

    async def call_tool(
        self, tool_name: str, tool_args: dict, tool_call_id: str, **kwargs: Any
    ) -> Any:
        clock = kwargs.get("state", {}).get("fp_clock") or Clock.at("2024-01-01")
        toolkit = self._make_toolkit(clock)
        result = toolkit.call(tool_name, tool_args)
        return vf.types.ToolMessage(
            role="tool", content=result, tool_call_id=tool_call_id
        )


def _brier_reward(completion, answer, **kwargs) -> float:
    """Reward = 1 - Brier (so higher is better, in [0, 1])."""
    text = completion[-1]["content"] if isinstance(completion, list) else str(completion)
    prob = _extract_probability(text)
    outcome = int(answer)
    return 1.0 - brier_score(prob, outcome)


def load_environment(
    dataset: Dataset | None = None,
    max_turns: int = 6,
    **kwargs: Any,
) -> vf.Environment:
    """Build the time-masked forecasting environment.

    Args:
        dataset: HF Dataset with columns ``question`` (prompt), ``answer`` (0/1
            outcome), and ``info`` (dict carrying ``as_of`` ISO date and optionally
            ``token_id_yes`` for the market tool). If None, a tiny demo set is used.
        max_turns: Max tool-use turns per rollout.
    """
    if dataset is None:
        dataset = _demo_dataset()

    source_specs = [
        {"tool_name": "wikipedia_search", "factory": lambda: WikipediaSource(mode="search")},
        {"tool_name": "pageviews_search", "factory": PageviewsSource},
        {"tool_name": "polymarket_search", "factory": PolymarketSource},
    ]
    rubric = vf.Rubric(funcs=[_brier_reward], weights=[1.0])
    return TimeMaskedToolEnv(
        source_specs=source_specs,
        eval_dataset=dataset,
        system_prompt=_SYSTEM,
        rubric=rubric,
        max_turns=max_turns,
        **kwargs,
    )


def _demo_dataset() -> Dataset:
    return Dataset.from_list(
        [
            {
                "question": "Will Donald Trump win the 2024 US Presidential Election?",
                "answer": 1,
                "info": {"as_of": "2024-10-06"},
            }
        ]
    )
