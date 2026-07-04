"""Offline tests for the NOAA weather source (network mocked)."""

from unittest.mock import MagicMock

import pytest

from forecast_playground import Clock, LookaheadError, NOAASource


def _results_response(results):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"results": results}
    return resp


def _obs(date, datatype, value):
    return {"date": f"{date}T00:00:00", "datatype": datatype, "value": value, "station": "S"}


def test_fetch_returns_observations_up_to_as_of():
    session = MagicMock()
    session.get.return_value = _results_response([
        _obs("2024-05-18", "TMAX", 22.0),
        _obs("2024-05-19", "TMIN", 11.0),
    ])
    src = NOAASource(token="tok", session=session)
    docs = src.fetch("GHCND:USW00094728", Clock.at("2024-05-20"))
    assert len(docs) == 1
    assert "TMAX=22.0" in docs[0].content
    assert docs[0].meta["station"] == "GHCND:USW00094728"
    # enddate capped at as_of (params is a list of tuples); token sent as header.
    params = session.get.call_args.kwargs["params"]
    assert ("enddate", "2024-05-20") in params
    assert session.get.call_args.kwargs["headers"]["token"] == "tok"


def test_empty_query_uses_default_station():
    session = MagicMock()
    session.get.return_value = _results_response([_obs("2024-05-19", "PRCP", 0.0)])
    src = NOAASource(token="tok", session=session)
    docs = src.fetch("", Clock.at("2024-05-20"))
    assert docs and "GHCND:" in docs[0].meta["station"]


def test_no_token_returns_empty():
    src = NOAASource(token="", session=MagicMock())
    assert src.fetch("GHCND:USW00094728", Clock.at("2024-05-20")) == []


def test_future_observation_is_guarded():
    session = MagicMock()
    session.get.return_value = _results_response([_obs("2024-12-01", "TMAX", 5.0)])
    src = NOAASource(token="tok", session=session)
    with pytest.raises(LookaheadError):
        src.fetch("GHCND:USW00094728", Clock.at("2024-05-20"))
