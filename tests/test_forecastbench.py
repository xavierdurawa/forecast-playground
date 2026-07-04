"""Offline tests for the ForecastBench question loader (HTTP mocked)."""

from unittest.mock import MagicMock

from forecast_playground import fetch_forecastbench_questions


def _session(questions, resolutions):
    """Fake session: first GET returns the question set, second the resolution set."""
    session = MagicMock()
    q_resp = MagicMock()
    q_resp.raise_for_status.return_value = None
    q_resp.json.return_value = {"questions": questions}
    r_resp = MagicMock()
    r_resp.raise_for_status.return_value = None
    r_resp.json.return_value = {"resolutions": resolutions}
    session.get.side_effect = [q_resp, r_resp]
    return session


def _q(id, source, combination_of="N/A", prob=0.4):
    return {
        "id": id, "source": source, "question": f"Q {id}?",
        "freeze_datetime": "2025-03-02T00:00:00+00:00",
        "freeze_datetime_value": prob, "combination_of": combination_of,
        "resolution_criteria": "crit", "background": "bg",
    }


def _r(id, source, date, resolved_to, direction=None):
    return {"id": id, "source": source, "resolution_date": date,
            "resolved_to": resolved_to, "direction": direction}


def test_joins_question_to_resolution():
    q = [_q("a", "metaculus", prob=0.6)]
    r = [_r("a", "metaculus", "2025-04-01", 1)]
    out = fetch_forecastbench_questions(session=_session(q, r))
    assert len(out) == 1
    assert out[0].outcome == 1 and out[0].market_prob == 0.6
    assert out[0].resolution_criteria == "crit"


def test_takes_earliest_resolution():
    q = [_q("a", "metaculus")]
    r = [_r("a", "metaculus", "2025-06-01", 0), _r("a", "metaculus", "2025-04-01", 1)]
    out = fetch_forecastbench_questions(session=_session(q, r))
    assert out[0].resolution_date.date().isoformat() == "2025-04-01"
    assert out[0].outcome == 1  # from the earliest


def test_skips_combination_questions():
    # combo via question field...
    q = [_q("a", "metaculus", combination_of=[{"id": "x"}])]
    r = [_r("a", "metaculus", "2025-04-01", 1)]
    assert fetch_forecastbench_questions(session=_session(q, r)) == []


def test_skips_resolution_with_direction():
    # A combo-component resolution (direction set) must not be joined.
    q = [_q("a", "metaculus")]
    r = [_r("a", "metaculus", "2025-04-01", 1, direction=[1, -1])]
    assert fetch_forecastbench_questions(session=_session(q, r)) == []


def test_sources_filter():
    q = [_q("a", "metaculus"), _q("b", "manifold")]
    r = [_r("a", "metaculus", "2025-04-01", 1), _r("b", "manifold", "2025-04-01", 0)]
    out = fetch_forecastbench_questions(sources=("metaculus",), session=_session(q, r))
    assert [x.source for x in out] == ["metaculus"]


def test_nonnumeric_market_prob_becomes_none():
    # Some freeze_datetime_value fields carry junk strings; don't crash.
    q = [_q("a", "metaculus", prob="50m backstroke")]
    r = [_r("a", "metaculus", "2025-04-01", 1)]
    out = fetch_forecastbench_questions(session=_session(q, r))
    assert out[0].market_prob is None


def test_question_as_of_is_before_resolution():
    # Leak-safety property for a question source: freeze < resolution, always.
    q = [_q("a", "metaculus")]
    r = [_r("a", "metaculus", "2025-04-01", 1)]
    out = fetch_forecastbench_questions(session=_session(q, r))
    assert out[0].as_of < out[0].resolution_date
