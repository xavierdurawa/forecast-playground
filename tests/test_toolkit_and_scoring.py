"""Offline tests for scoring, the toolkit dispatcher, and probability extraction."""

from datetime import datetime, timezone

from chrono_harness import (
    AsOfGuarantee,
    Clock,
    Document,
    Toolkit,
    brier_score,
    log_score,
    mean_brier,
)
from chrono_harness.agent import _extract_probability


# --- scoring ---------------------------------------------------------------

def test_brier_perfect_and_worst():
    assert brier_score(1.0, 1) == 0.0
    assert brier_score(0.0, 1) == 1.0


def test_log_score_clipped_not_inf():
    # Confident miss should be large but finite.
    assert log_score(0.0, 1) < 100 and log_score(0.0, 1) > 0


def test_mean_brier():
    assert mean_brier([(0.5, 1), (0.5, 0)]) == 0.25


# --- probability extraction ------------------------------------------------

def test_extract_probability_takes_last():
    text = "draft 0.2 ... PROBABILITY: 0.3\nfinal PROBABILITY: 0.71"
    assert _extract_probability(text) == 0.71


def test_extract_probability_none_when_absent():
    assert _extract_probability("no answer here") is None


# --- toolkit dispatch ------------------------------------------------------

class _FakeSource:
    name = "fake:test"
    guarantee = AsOfGuarantee.HARD

    def __init__(self, ts):
        self._ts = ts

    def fetch(self, query, clock, **kwargs):
        ts = clock.guard(self._ts, source=self.name)  # may raise
        return [Document(content=f"result for {query}", timestamp=ts, source=self.name)]


def test_toolkit_dispatch_and_trace():
    clock = Clock.at("2024-01-01")
    tk = Toolkit(clock=clock, sources=[_FakeSource(datetime(2023, 6, 1, tzinfo=timezone.utc))])
    defs = tk.anthropic_tools()
    names = {d["name"] for d in defs}
    assert "fake_search" in names and "run_python" in names

    out = tk.call("fake_search", {"query": "spacex"})
    assert "result for spacex" in out
    assert len(tk.calls) == 1 and tk.calls[0].ok


def test_toolkit_converts_lookahead_to_error_not_crash():
    clock = Clock.at("2024-01-01")
    # source returns a doc stamped AFTER as_of -> Clock raises -> toolkit catches.
    tk = Toolkit(clock=clock, sources=[_FakeSource(datetime(2024, 6, 1, tzinfo=timezone.utc))])
    out = tk.call("fake_search", {"query": "x"})
    assert "lookahead blocked" in out.lower()
    assert tk.calls[0].ok is False


def test_run_python_tool():
    clock = Clock.at("2024-01-01")
    tk = Toolkit(clock=clock, sources=[])
    out = tk.call("run_python", {"code": "print(6*7)"})
    assert out.strip() == "42"
