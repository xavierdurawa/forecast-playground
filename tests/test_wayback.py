"""Tests for the Wayback source, focused on the at-or-before leak guarantee."""

from unittest.mock import MagicMock

from chrono_harness import Clock, WaybackSource


def _session(cdx_rows=None, avail=None, page_text="<html>archived</html>", cdx_raises=False):
    """Fake session: first call is the snapshot lookup, second is the page fetch."""
    session = MagicMock()
    calls = {"n": 0}

    def _get(url, **kwargs):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        calls["n"] += 1
        if "cdx" in url:
            if cdx_raises:
                import requests
                raise requests.RequestException("cdx down")
            resp.json.return_value = cdx_rows
        elif "available" in url:
            resp.json.return_value = avail
        else:  # the page fetch
            resp.text = page_text
        return resp

    session.get.side_effect = _get
    return session


def test_cdx_returns_latest_at_or_before():
    rows = [["timestamp"], ["20231201000000"], ["20231215120000"]]
    src = WaybackSource(session=_session(cdx_rows=rows))
    docs = src.fetch("https://example.com", Clock.at("2024-01-01"))
    assert len(docs) == 1
    assert docs[0].meta["snapshot"] == "20231215120000"
    assert docs[0].timestamp <= Clock.at("2024-01-01").as_of


def test_cdx_no_snapshot_before_as_of_returns_empty():
    rows = [["timestamp"]]  # header only -> nothing at or before
    src = WaybackSource(session=_session(cdx_rows=rows))
    assert src.fetch("https://example.com", Clock.at("2024-01-01")) == []


def test_availability_fallback_rejects_snapshot_after_as_of():
    """The core leak fix: Availability 'closest' AFTER as_of must NOT be returned."""
    avail = {
        "archived_snapshots": {
            "closest": {"available": True, "timestamp": "20240601000000"}  # after as_of
        }
    }
    src = WaybackSource(session=_session(avail=avail, cdx_raises=True))
    # CDX raises -> falls back to Availability -> closest is after as_of -> empty.
    assert src.fetch("https://example.com", Clock.at("2024-01-01")) == []


def test_availability_fallback_accepts_snapshot_before_as_of():
    avail = {
        "archived_snapshots": {
            "closest": {"available": True, "timestamp": "20231101000000"}  # before
        }
    }
    src = WaybackSource(session=_session(avail=avail, cdx_raises=True))
    docs = src.fetch("https://example.com", Clock.at("2024-01-01"))
    assert len(docs) == 1
    assert docs[0].meta["snapshot"] == "20231101000000"
