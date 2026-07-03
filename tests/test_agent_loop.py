"""Offline tests for the agent loop's answer-forcing and defaulting behavior.

A fake Driver returns scripted turns so we can exercise the loop (tool calls,
force-final-turn, defaulting) with no network, model, or provider SDK.
"""

from forecast_playground import Clock, Toolkit
from forecast_playground.agent import run_forecast
from forecast_playground.drivers import ToolInvocation, Turn


class _FakeDriver:
    """Replays a script of Turns; records how many steps and force-turns happened."""

    def __init__(self, script):
        self._script = list(script)
        self.steps = 0

    def start(self, question):
        return [{"role": "user", "content": question}]

    def step(self, model, system, messages, tools, max_tokens):
        self.steps += 1
        turn = self._script.pop(0)
        messages.append({"role": "assistant", "content": turn.text})
        return turn

    def add_tool_results(self, messages, results):
        messages.append({"role": "tool", "content": [r for _, r in results]})

    def add_user(self, messages, text):
        messages.append({"role": "user", "content": text})


def _tk():
    return Toolkit(clock=Clock.at("2024-01-01"), sources=[], enable_python=False)


def test_direct_answer_no_force():
    drv = _FakeDriver([Turn(text="PROBABILITY: 0.7", tool_calls=[], stop_reason="stop")])
    fc = run_forecast(drv, "m", "Q?", _tk())
    assert fc.probability == 0.7
    assert not fc.forced and not fc.defaulted


def test_forces_final_turn_when_no_probability():
    drv = _FakeDriver(
        [
            Turn(text="I need more data.", tool_calls=[], stop_reason="stop"),
            Turn(text="PROBABILITY: 0.42", tool_calls=[], stop_reason="stop"),
        ]
    )
    fc = run_forecast(drv, "m", "Q?", _tk())
    assert fc.probability == 0.42
    assert fc.forced and not fc.defaulted


def test_defaults_to_half_when_force_also_fails():
    drv = _FakeDriver(
        [
            Turn(text="hmm", tool_calls=[], stop_reason="stop"),
            Turn(text="still no number", tool_calls=[], stop_reason="stop"),
        ]
    )
    fc = run_forecast(drv, "m", "Q?", _tk())
    assert fc.probability == 0.5
    assert fc.defaulted and fc.forced


def test_tool_call_is_dispatched_then_answered():
    # First turn requests run_python; second turn answers.
    drv = _FakeDriver(
        [
            Turn(
                text="",
                tool_calls=[ToolInvocation(id="1", name="run_python", args={"code": "print(1)"})],
                stop_reason="tool_use",
            ),
            Turn(text="PROBABILITY: 0.6", tool_calls=[], stop_reason="stop"),
        ]
    )
    tk = Toolkit(clock=Clock.at("2024-01-01"), sources=[])  # python enabled
    fc = run_forecast(drv, "m", "Q?", tk)
    assert fc.probability == 0.6
    assert fc.n_tool_calls == 1 and tk.calls[0].name == "run_python"
