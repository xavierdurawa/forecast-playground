"""Offline tests for the agent loop's answer-forcing and defaulting behavior.

A fake Anthropic-style client returns scripted responses so we can exercise the
loop (force-final-turn, defaulting) with no network or model.
"""

from types import SimpleNamespace

from chrono_harness import Clock, Toolkit
from chrono_harness.agent import run_forecast


def _block(**kw):
    return SimpleNamespace(**kw)


class _FakeClient:
    """Returns queued responses; each is (stop_reason, content_blocks)."""

    def __init__(self, script):
        self._script = list(script)
        self.messages = self
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        stop_reason, content = self._script.pop(0)
        return SimpleNamespace(stop_reason=stop_reason, content=content)


def _tk():
    return Toolkit(clock=Clock.at("2024-01-01"), sources=[], enable_python=False)


def test_direct_answer_no_force():
    client = _FakeClient([("end_turn", [_block(type="text", text="PROBABILITY: 0.7")])])
    fc = run_forecast(client, "m", "Q?", _tk())
    assert fc.probability == 0.7
    assert not fc.forced and not fc.defaulted


def test_forces_final_turn_when_no_probability():
    # First turn: model talks but gives no probability and stops.
    # Forced turn: model complies with a probability.
    client = _FakeClient(
        [
            ("end_turn", [_block(type="text", text="I need more data.")]),
            ("end_turn", [_block(type="text", text="PROBABILITY: 0.42")]),
        ]
    )
    fc = run_forecast(client, "m", "Q?", _tk())
    assert fc.probability == 0.42
    assert fc.forced and not fc.defaulted


def test_defaults_to_half_when_force_also_fails():
    client = _FakeClient(
        [
            ("end_turn", [_block(type="text", text="hmm")]),
            ("end_turn", [_block(type="text", text="still no number")]),
        ]
    )
    fc = run_forecast(client, "m", "Q?", _tk())
    assert fc.probability == 0.5
    assert fc.defaulted and fc.forced
