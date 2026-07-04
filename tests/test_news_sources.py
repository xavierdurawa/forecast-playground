"""Offline tests for the news sources (Current Events + GDELT). Network mocked."""

import io
import zipfile
from unittest.mock import MagicMock

import pytest

from forecast_playground import (
    Clock,
    CurrentEventsSource,
    GDELTNewsSource,
    LookaheadError,
)


# --- Current Events --------------------------------------------------------

def _day_page(title, ts, content):
    return {"query": {"pages": [{"title": title, "revisions": [
        {"revid": 1, "timestamp": ts, "slots": {"main": {"content": content}}}
    ]}]}}


def test_current_events_fetches_daily_digests_as_of():
    # Every day's fetch returns a digest stamped before the as-of date.
    session = MagicMock()

    def _get(url, **kwargs):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = _day_page(
            "Portal:Current events/2024 May 20", "2024-05-20T23:00:00Z", "war news"
        )
        return resp

    session.get.side_effect = _get
    src = CurrentEventsSource(lookback_days=3, session=session)
    docs = src.fetch("", Clock.at("2024-05-21"))
    assert len(docs) == 3  # three days in the window
    assert all("war news" in d.content for d in docs)
    assert all(d.timestamp <= Clock.at("2024-05-21").as_of for d in docs)


def test_current_events_skips_missing_days():
    session = MagicMock()
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"query": {"pages": [{"title": "x", "missing": True}]}}
    session.get.return_value = resp
    src = CurrentEventsSource(lookback_days=3, session=session)
    assert src.fetch("", Clock.at("2024-05-21")) == []


# --- GDELT -----------------------------------------------------------------

def _gkg_zip(rows):
    """Build a zipped tab-delimited GKG payload from (domain, url) rows."""
    lines = []
    for domain, url in rows:
        fields = [""] * 27
        fields[3] = domain
        fields[4] = url
        lines.append("\t".join(fields))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("x.gkg.csv", "\n".join(lines))
    return buf.getvalue()


def _gkg_session(payload):
    session = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.content = payload
    resp.raise_for_status.return_value = None
    session.get.return_value = resp
    return session


def test_gdelt_filters_by_query_terms():
    payload = _gkg_zip([
        ("reuters.com", "https://reuters.com/spacex-starship-orbit-success"),
        ("bbc.com", "https://bbc.com/cooking-recipes-summer"),
    ])
    src = GDELTNewsSource(slots=1, session=_gkg_session(payload))
    docs = src.fetch("starship orbit", Clock.at("2024-05-20T12:05:00"))
    assert len(docs) == 1
    assert "starship" in docs[0].url
    assert docs[0].meta["domain"] == "reuters.com"


def test_gdelt_slot_timestamps_floored_and_before_as_of():
    src = GDELTNewsSource(slots=3)
    stamps = src._slot_stamps(Clock.at("2024-05-20T12:07:00"))
    # 12:07 floors to 12:00, then 11:45, 11:30 — all <= as_of, newest first.
    assert stamps[0].strftime("%H:%M") == "12:00"
    assert stamps[1].strftime("%H:%M") == "11:45"
    assert all(s <= Clock.at("2024-05-20T12:07:00").as_of for s in stamps)


def test_gdelt_empty_before_epoch():
    # 2010 predates GKG 2.0 (2015-02) -> no slots -> empty.
    src = GDELTNewsSource(slots=5, session=_gkg_session(_gkg_zip([])))
    assert src.fetch("anything", Clock.at("2010-01-01")) == []


def test_gdelt_missing_slot_404_skipped():
    session = MagicMock()
    resp = MagicMock()
    resp.status_code = 404
    session.get.return_value = resp
    src = GDELTNewsSource(slots=2, session=session)
    assert src.fetch("x", Clock.at("2024-05-20T12:00:00")) == []


def test_gdelt_bad_backend_rejected():
    with pytest.raises(ValueError):
        GDELTNewsSource(backend="nonsense")


# --- GDELT BigQuery backend (mocked client — no GCP, no network) -----------

class _FakeScalarParam:
    def __init__(self, name, type_, value):
        self.name, self.type_, self.value = name, type_, value


class _FakeJobConfig:
    def __init__(self, query_parameters=None):
        self.query_parameters = query_parameters or []


def _install_fake_bigquery(monkeypatch, rows, capture):
    """Inject a fake google.cloud.bigquery module; record the SQL + params."""
    import sys
    import types

    bq = types.ModuleType("google.cloud.bigquery")

    class _Client:
        def __init__(self, project=None):
            capture["project"] = project

        def query(self, sql, job_config=None):
            capture["sql"] = sql
            capture["params"] = {p.name: p.value for p in job_config.query_parameters}
            job = MagicMock()
            job.result.return_value = rows
            return job

    bq.Client = _Client
    bq.ScalarQueryParameter = _FakeScalarParam
    bq.QueryJobConfig = _FakeJobConfig
    google = types.ModuleType("google")
    cloud = types.ModuleType("google.cloud")
    cloud.bigquery = bq
    google.cloud = cloud
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.bigquery", bq)


def test_gdelt_bigquery_builds_query_and_parses_rows(monkeypatch):
    rows = [
        {"DATE": 20240519120000, "SourceCommonName": "reuters.com",
         "DocumentIdentifier": "https://reuters.com/spacex-starship"},
    ]
    cap = {}
    _install_fake_bigquery(monkeypatch, rows, cap)
    src = GDELTNewsSource(backend="bigquery", bq_project="my-proj", max_results=10)
    docs = src.fetch("starship orbit", Clock.at("2024-05-20T00:00:00"))

    assert len(docs) == 1
    assert docs[0].meta["domain"] == "reuters.com"
    assert docs[0].timestamp <= Clock.at("2024-05-20T00:00:00").as_of
    # Query construction: capped by as_of, project passed, per-term LIKE params bound.
    assert cap["project"] == "my-proj"
    assert cap["params"]["as_of"] == 20240520000000
    assert cap["params"]["t0"] == "%starship%" and cap["params"]["t1"] == "%orbit%"
    assert "DATE <= @as_of" in cap["sql"]


def test_gdelt_bigquery_guards_future_row(monkeypatch):
    # A row dated after as_of (shouldn't happen given the WHERE, but the guard is
    # the last line of defense) must raise rather than leak.
    rows = [{"DATE": 20241201000000, "SourceCommonName": "x.com",
             "DocumentIdentifier": "https://x.com/future"}]
    _install_fake_bigquery(monkeypatch, rows, {})
    src = GDELTNewsSource(backend="bigquery", bq_project="p")
    with pytest.raises(LookaheadError):
        src.fetch("future", Clock.at("2024-05-20"))
