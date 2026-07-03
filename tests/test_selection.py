"""Offline tests for uncertainty-based question selection (leak-free)."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

from forecast_playground import Clock, ResolvedMarket
from forecast_playground.sources.polymarket import market_prob_at, select_uncertain


def _hist_session(prob_points):
    """Fake session whose CLOB history returns the given (unix_ts, prob) points."""
    session = MagicMock()
    resp = MagicMock()
    resp.json.return_value = {"history": [{"t": t, "p": p} for t, p in prob_points]}
    resp.raise_for_status.return_value = None
    session.get.return_value = resp
    return session


def _market(token="tok", end="2024-11-05", outcome=1):
    return ResolvedMarket(
        question="Q?",
        token_id_yes=token,
        end_date=datetime.fromisoformat(end).replace(tzinfo=timezone.utc),
        outcome=outcome,
        volume=1e6,
    )


def test_market_prob_at_returns_last_before_clock():
    # Points across time; clock at 2024-10-06 admits only the first two.
    pts = [
        (int(datetime(2024, 10, 1, tzinfo=timezone.utc).timestamp()), 0.4),
        (int(datetime(2024, 10, 5, tzinfo=timezone.utc).timestamp()), 0.55),
        (int(datetime(2024, 10, 20, tzinfo=timezone.utc).timestamp()), 0.9),  # after clock
    ]
    p = market_prob_at("tok", Clock.at("2024-10-06"), session=_hist_session(pts))
    assert p == 0.55  # latest admissible, the 0.9 is masked out


def test_select_uncertain_keeps_only_midrange():
    # Two markets: one uncertain (0.5), one already-decided (0.95) at as-of.
    uncertain_pts = [(int(datetime(2024, 10, 1, tzinfo=timezone.utc).timestamp()), 0.5)]
    decided_pts = [(int(datetime(2024, 10, 1, tzinfo=timezone.utc).timestamp()), 0.95)]

    m_unc, m_dec = _market(token="unc"), _market(token="dec")

    def _session_for(*args, **kwargs):
        # Route by token in the request params.
        token = kwargs.get("params", {}).get("market", "")
        pts = uncertain_pts if token == "unc" else decided_pts
        resp = MagicMock()
        resp.json.return_value = {"history": [{"t": t, "p": p} for t, p in pts]}
        resp.raise_for_status.return_value = None
        return resp

    session = MagicMock()
    session.get.side_effect = _session_for

    kept = select_uncertain(
        [m_unc, m_dec],
        clock_for=lambda m: Clock.at("2024-10-06"),
        session=session,
    )
    assert [m.token_id_yes for m, _ in kept] == ["unc"]
    assert kept[0][1] == 0.5
