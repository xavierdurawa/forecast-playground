"""Offline tests for the FRED/ALFRED source (network mocked)."""

from unittest.mock import MagicMock

import pytest

from forecast_playground import Clock, FREDSource, LookaheadError


def _obs_response(observations):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"observations": observations}
    return resp


def _obs(date, value, vintage="2024-06-01"):
    # ALFRED observation shape: period `date`, `value`, and the realtime window.
    return {"realtime_start": vintage, "realtime_end": vintage, "date": date, "value": value}


def test_fetch_returns_observations_as_of_vintage():
    session = MagicMock()
    session.get.return_value = _obs_response([
        _obs("2024-04-01", "28000.0"),
        _obs("2024-01-01", "27500.0"),
    ])
    src = FREDSource(api_key="x" * 32, session=session)
    docs = src.fetch("GDP", Clock.at("2024-06-01"))
    assert len(docs) == 1
    assert "28000.0" in docs[0].content
    assert docs[0].meta["series_id"] == "GDP"
    assert docs[0].meta["vintage"] == "2024-06-01"
    # Vintage params were sent so revised values can't leak.
    params = session.get.call_args.kwargs["params"]
    assert params["realtime_start"] == "2024-06-01" == params["realtime_end"]
    assert params["observation_end"] == "2024-06-01"


def test_missing_values_skipped():
    session = MagicMock()
    session.get.return_value = _obs_response([
        _obs("2024-04-01", "."),   # FRED missing marker
        _obs("2024-01-01", "27500.0"),
    ])
    src = FREDSource(api_key="x" * 32, session=session)
    docs = src.fetch("GDP", Clock.at("2024-06-01"))
    assert "27500.0" in docs[0].content and docs[0].meta["points"] == 1


def test_no_key_returns_empty():
    # No key configured -> no data (leak-safe: nothing rather than a live call).
    src = FREDSource(api_key="", session=MagicMock())
    assert src.fetch("GDP", Clock.at("2024-06-01")) == []


def test_future_observation_period_is_guarded():
    session = MagicMock()
    session.get.return_value = _obs_response([_obs("2024-12-01", "99999.0")])  # after as_of
    src = FREDSource(api_key="x" * 32, session=session)
    with pytest.raises(LookaheadError):
        src.fetch("GDP", Clock.at("2024-06-01"))
