"""Tests for the Wikipedia source, including the central no-leak property.

Network is mocked so the suite runs offline and deterministically.
"""

from unittest.mock import MagicMock

import pytest

from forecast_playground import Clock, LookaheadError, WikipediaSource


def _mock_session(api_json):
    session = MagicMock()
    resp = MagicMock()
    resp.json.return_value = api_json
    resp.raise_for_status.return_value = None
    session.get.return_value = resp
    return session


def test_fetch_returns_revision_at_or_before_as_of():
    api = {
        "query": {
            "pages": [
                {
                    "title": "SpaceX",
                    "revisions": [
                        {
                            "revid": 111,
                            "timestamp": "2023-12-15T10:00:00Z",
                            "comment": "edit",
                            "slots": {"main": {"content": "Founded 2002. Cached text."}},
                        }
                    ],
                }
            ]
        }
    }
    src = WikipediaSource(session=_mock_session(api))
    docs = src.fetch("SpaceX", Clock.at("2024-01-01"))
    assert len(docs) == 1
    assert "Cached text" in docs[0].content
    assert docs[0].meta["revid"] == 111
    assert docs[0].source == "wikipedia:en"


def test_missing_article_returns_empty():
    api = {"query": {"pages": [{"title": "Nonexistent", "missing": True}]}}
    src = WikipediaSource(session=_mock_session(api))
    assert src.fetch("Nonexistent", Clock.at("2024-01-01")) == []


def test_lookahead_revision_is_rejected():
    """The central correctness property: a revision newer than the Clock must raise.

    A well-behaved API won't return one (we pass rvstart), but a buggy/misbehaving
    backend could — the Clock chokepoint is the last line of defense.
    """
    api = {
        "query": {
            "pages": [
                {
                    "title": "SpaceX",
                    "revisions": [
                        {
                            "revid": 999,
                            "timestamp": "2024-06-01T10:00:00Z",  # AFTER as_of
                            "comment": "future edit",
                            "slots": {"main": {"content": "Leaked future info."}},
                        }
                    ],
                }
            ]
        }
    }
    src = WikipediaSource(session=_mock_session(api))
    with pytest.raises(LookaheadError):
        src.fetch("SpaceX", Clock.at("2024-01-01"))


def _rev_page(title, revid, ts, content):
    return {"query": {"pages": [{"title": title, "revisions": [
        {"revid": revid, "timestamp": ts, "comment": "", "slots": {"main": {"content": content}}}
    ]}]}}


def test_search_mode_discovers_titles_then_fetches_as_of():
    """Search mode: list=search yields titles, each fetched via leak-safe as-of rev."""
    search_resp = {"query": {"search": [{"title": "SpaceX"}, {"title": "Starship"}]}}
    # Per-call responses: first the search, then one revision fetch per title.
    responses = [
        search_resp,
        _rev_page("SpaceX", 1, "2023-12-15T10:00:00Z", "SpaceX makes rockets. Orbit reached."),
        _rev_page("Starship", 2, "2023-11-01T10:00:00Z", "Starship is a launch vehicle."),
    ]
    session = MagicMock()

    def _get(*args, **kwargs):
        resp = MagicMock()
        resp.json.return_value = responses.pop(0)
        resp.raise_for_status.return_value = None
        return resp

    session.get.side_effect = _get
    src = WikipediaSource(mode="search", max_titles=2, session=session)
    docs = src.fetch("rockets reaching orbit", Clock.at("2024-01-01"))
    assert {d.meta["title"] for d in docs} == {"SpaceX", "Starship"}
    assert all("ranked_chunks" in d.meta for d in docs)


def test_search_mode_drops_articles_not_yet_existing():
    """A title with no revision at/before as_of (didn't exist yet) is dropped."""
    search_resp = {"query": {"search": [{"title": "FutureThing"}]}}
    missing = {"query": {"pages": [{"title": "FutureThing", "missing": True}]}}
    session = MagicMock()
    responses = [search_resp, missing]

    def _get(*args, **kwargs):
        resp = MagicMock()
        resp.json.return_value = responses.pop(0)
        resp.raise_for_status.return_value = None
        return resp

    session.get.side_effect = _get
    src = WikipediaSource(mode="search", session=session)
    assert src.fetch("future thing", Clock.at("2024-01-01")) == []
