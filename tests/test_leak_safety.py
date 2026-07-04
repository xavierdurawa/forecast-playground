"""The standing guarantee: NO source may surface data newer than the as-of Clock.

This is the one invariant the whole package exists to enforce. Rather than trust
each source's hand-written checks, this parametrizes every registered source and
asserts the property directly against its live API, across several as-of dates.

Adding a new source? Add one row to ``LIVE_SOURCES`` below and it's covered.

Run with:  pytest -m integration
(Live/network; skipped by default. A companion offline test guards the Toolkit
chokepoint with a deliberately-leaky fake source — no network needed.)
"""

from datetime import datetime, timezone

import pytest

from forecast_playground import (
    AsOfGuarantee,
    Clock,
    CurrentEventsSource,
    Document,
    GDELTNewsSource,
    PageviewsSource,
    PolymarketSource,
    Toolkit,
    WaybackSource,
    WikipediaSource,
    fetch_resolved_markets,
)

# (label, factory, sample query that should return results). One row per source.
# A new source is covered by adding a row here.
LIVE_SOURCES = [
    ("wikipedia", lambda: WikipediaSource(mode="search"), "SpaceX Starship"),
    ("pageviews", lambda: PageviewsSource(lookback_days=14), "ChatGPT"),
    ("wayback", WaybackSource, "https://www.nasa.gov"),
    ("current_events", lambda: CurrentEventsSource(lookback_days=2), ""),
    ("gdelt", lambda: GDELTNewsSource(slots=2, max_results=5), "government"),
]

# As-of dates spanning a range so the guarantee is checked, not a single lucky point.
AS_OF_DATES = ["2023-03-15T12:00:00", "2024-05-20T18:00:00", "2024-11-01T09:00:00"]


@pytest.mark.integration
@pytest.mark.parametrize("label,factory,query", LIVE_SOURCES, ids=[s[0] for s in LIVE_SOURCES])
@pytest.mark.parametrize("as_of", AS_OF_DATES)
def test_source_never_leaks_future(label, factory, query, as_of):
    """No Document from any source is stamped after the as-of instant."""
    clock = Clock.at(as_of)
    docs = factory().fetch(query, clock)
    for d in docs:  # empty is fine (some URLs/dates have no data); a leak is not
        assert d.timestamp <= clock.as_of, (
            f"{label} leaked: {d.timestamp.isoformat()} > as_of {clock.as_of.isoformat()}"
        )


@pytest.mark.integration
@pytest.mark.parametrize("as_of", AS_OF_DATES)
def test_polymarket_never_leaks_future(as_of):
    """Polymarket needs a live token id, so it gets its own row (same invariant)."""
    clock = Clock.at(as_of)
    markets = fetch_resolved_markets(limit=1)
    if not markets:
        pytest.skip("no resolved markets available")
    docs = PolymarketSource().fetch(markets[0].token_id_yes, clock)
    for d in docs:
        assert d.timestamp <= clock.as_of


# --- offline companion: the Toolkit chokepoint catches a leaky source ------
# This does NOT need network — it proves the structural guarantee holds even for a
# source that violates the contract, which is the property BYO-tool authors rely on.

class _LeakySource:
    name = "leaky"
    guarantee = AsOfGuarantee.HARD

    def fetch(self, query, clock, **kwargs):
        return [Document(
            content="future", timestamp=datetime(2099, 1, 1, tzinfo=timezone.utc),
            source=self.name,
        )]


def test_toolkit_blocks_leaky_source_offline():
    tk = Toolkit(clock=Clock.at("2024-01-01"), sources=[_LeakySource()], enable_python=False)
    out = tk.call("leaky_search", {"query": "x"})
    assert "future" not in out and "lookahead blocked" in out.lower()
    assert tk.calls[0].ok is False
