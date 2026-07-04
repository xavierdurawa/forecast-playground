"""Offline tests for the LLM-as-judge scorer (pure parts + fake-driver flow)."""

from forecast_playground import (
    brier_from_judgment,
    judge_forecast,
    parse_judgment,
)
from forecast_playground.drivers import Turn


# --- pure reduction --------------------------------------------------------

def test_brier_from_judgment_matches_brier_on_boolean_label():
    # resolved_label 0/1 -> exact Brier.
    assert brier_from_judgment(0.8, 1) == brier_from_judgment(0.8, 1.0) == (0.8 - 1) ** 2
    assert brier_from_judgment(0.0, 0) == 0.0


def test_reduction_penalizes_confident_wrong():
    # The whole point: grounded in the OUTCOME, a confident-wrong forecast scores
    # badly (not "perfect because self-consistent").
    confident_wrong = brier_from_judgment(0.02, 1.0)  # said ~no, it happened
    assert confident_wrong > 0.9


# --- parser ----------------------------------------------------------------

def test_parse_valid_judgment():
    text = '{"resolved_label": 1.0, "accuracy": 0.9, "calibration": 0.8}'
    j = parse_judgment(text, probability=0.7)
    assert j.parsed and j.resolved_label == 1.0
    assert j.brier == (0.7 - 1.0) ** 2
    assert j.rubric["accuracy"] == 0.9


def test_parse_extracts_json_from_surrounding_prose():
    text = 'Here is my judgment:\n{"resolved_label": 0.0}\nThanks!'
    j = parse_judgment(text, probability=0.3)
    assert j.parsed and j.resolved_label == 0.0


def test_parse_clips_out_of_range():
    j = parse_judgment('{"resolved_label": 1.5}', probability=0.5)
    assert j.resolved_label == 1.0


def test_parse_unparseable_defaults_and_flags():
    j = parse_judgment("no json here", probability=0.6)
    assert j.parsed is False
    assert j.resolved_label == 0.5  # max-entropy fallback
    assert j.brier == (0.6 - 0.5) ** 2


# --- judge_forecast flow (fake driver, no network) -------------------------

class _FakeDriver:
    def __init__(self, reply):
        self._reply = reply

    def start(self, question):
        return [{"role": "user", "content": question}]

    def step(self, model, system, messages, tools, max_tokens):
        return Turn(text=self._reply, tool_calls=[], stop_reason="stop")


def test_judge_forecast_end_to_end():
    driver = _FakeDriver('{"resolved_label": 1.0, "accuracy": 0.95}')
    j = judge_forecast(
        driver, "judge-model",
        question="Will X happen?",
        probability=0.9,
        resolution="Yes, X happened on 2024-05-01.",
        reasoning="Evidence pointed to yes.",
    )
    assert j.parsed and j.resolved_label == 1.0
    assert abs(j.brier - (0.9 - 1.0) ** 2) < 1e-9
