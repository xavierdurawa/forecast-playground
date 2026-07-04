"""Integration tests: real APIs, real leak-safety, the full forecast path.

Run with:  pytest -m integration
Skipped by default (see conftest). These need network; the model test also needs a
local Ollama (auto-skipped if absent). They assert the ONE property that mocks can't
verify: that live services actually respect the as-of Clock end to end.
"""

import urllib.request

import pytest

from forecast_playground import (
    Clock,
    CurrentEventsSource,
    GDELTNewsSource,
    OpenAIDriver,
    PageviewsSource,
    PolymarketSource,
    ResultCache,
    Toolkit,
    WaybackSource,
    WikipediaSource,
    fetch_resolved_markets,
    run_forecast,
)

pytestmark = pytest.mark.integration

AS_OF = Clock.at("2024-05-20T18:00:00")
OLLAMA = "http://localhost:11434/v1"


def _ollama_up() -> bool:
    try:
        urllib.request.urlopen(OLLAMA.replace("/v1", "") + "/api/tags", timeout=2)
        return True
    except Exception:
        return False


# --- each live source respects the Clock -----------------------------------

def _assert_leakfree(docs, clock):
    assert docs, "expected at least one document from the live source"
    for d in docs:
        assert d.timestamp <= clock.as_of, f"LEAK: {d.source} {d.timestamp} > {clock.as_of}"


def test_wikipedia_live_asof():
    docs = WikipediaSource(mode="search").fetch("SpaceX Starship", AS_OF)
    _assert_leakfree(docs, AS_OF)


def test_pageviews_live_asof():
    docs = PageviewsSource(lookback_days=14).fetch("ChatGPT", AS_OF)
    _assert_leakfree(docs, AS_OF)


def test_wayback_live_asof():
    docs = WaybackSource().fetch("https://www.nasa.gov", AS_OF)
    # Wayback may legitimately return empty for a URL/date; only check if present.
    for d in docs:
        assert d.timestamp <= AS_OF.as_of


def test_current_events_live_asof():
    docs = CurrentEventsSource(lookback_days=2).fetch("", AS_OF)
    _assert_leakfree(docs, AS_OF)


def test_gdelt_live_asof():
    docs = GDELTNewsSource(slots=2, max_results=5).fetch("israel", AS_OF)
    _assert_leakfree(docs, AS_OF)


def test_polymarket_live_resolved_markets():
    markets = fetch_resolved_markets(limit=3)
    assert markets and all(m.outcome in (0, 1) for m in markets)
    # Market-implied history up to as-of must not include post-as_of points.
    docs = PolymarketSource().fetch(markets[0].token_id_yes, Clock.at("2024-11-01"))
    for d in docs:
        assert d.timestamp <= Clock.at("2024-11-01").as_of


# --- toolkit end-to-end (real source, cache, trace) ------------------------

def test_toolkit_live_dispatch_and_cache(tmp_path):
    cache = ResultCache(directory=tmp_path)
    tk = Toolkit(clock=AS_OF, sources=[WikipediaSource(mode="search")], cache=cache)
    out1 = tk.call("wikipedia_search", {"query": "SpaceX Starship"})
    out2 = tk.call("wikipedia_search", {"query": "SpaceX Starship"})
    assert out1 == out2
    assert tk.calls[0].cached is False and tk.calls[1].cached is True  # 2nd from cache


# --- full forecast path via a real (local) model ---------------------------

def test_full_forecast_via_ollama():
    if not _ollama_up():
        pytest.skip("local Ollama not running on :11434")
    driver = OpenAIDriver(base_url=OLLAMA, api_key="ollama")
    tk = Toolkit(clock=AS_OF, sources=[WikipediaSource(mode="search")], enable_python=False)
    fc = run_forecast(
        driver, "qwen2.5-coder:14b",
        "Will SpaceX Starship reach orbit before 2026? Forecast P(YES).",
        tk, max_turns=3, max_tokens=2500,
    )
    assert 0.0 <= fc.probability <= 1.0  # a valid probability came out the far end
