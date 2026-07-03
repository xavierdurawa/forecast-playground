"""Tests for the on-disk result cache and its integration with the Toolkit."""

from datetime import datetime, timezone

from forecast_playground import AsOfGuarantee, Clock, Document, ResultCache, Toolkit


def test_cache_roundtrip_and_miss(tmp_path):
    cache = ResultCache(directory=tmp_path)
    assert cache.get("k") is None  # miss
    cache.put("k", "value")
    assert cache.get("k") == "value"


def test_cache_disabled_is_noop(tmp_path):
    cache = ResultCache(directory=tmp_path, enabled=False)
    cache.put("k", "value")
    assert cache.get("k") is None


class _CountingSource:
    name = "count:test"
    guarantee = AsOfGuarantee.HARD

    def __init__(self):
        self.fetches = 0

    def fetch(self, query, clock, **kwargs):
        self.fetches += 1
        return [Document(content=f"r:{query}", timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc), source=self.name)]


def test_toolkit_caches_repeated_query(tmp_path):
    src = _CountingSource()
    cache = ResultCache(directory=tmp_path)
    tk = Toolkit(clock=Clock.at("2024-01-01"), sources=[src], enable_python=False, cache=cache)

    r1 = tk.call("count_search", {"query": "x"})
    r2 = tk.call("count_search", {"query": "x"})
    assert r1 == r2
    assert src.fetches == 1  # second call served from cache
    assert tk.calls[0].cached is False and tk.calls[1].cached is True


def test_cache_key_includes_as_of(tmp_path):
    src = _CountingSource()
    cache = ResultCache(directory=tmp_path)
    # Same query, different as_of -> different key -> two real fetches.
    Toolkit(clock=Clock.at("2024-01-01"), sources=[src], enable_python=False, cache=cache).call(
        "count_search", {"query": "x"}
    )
    Toolkit(clock=Clock.at("2023-01-01"), sources=[src], enable_python=False, cache=cache).call(
        "count_search", {"query": "x"}
    )
    assert src.fetches == 2


def test_run_python_not_cached(tmp_path):
    cache = ResultCache(directory=tmp_path)
    tk = Toolkit(clock=Clock.at("2024-01-01"), sources=[], cache=cache)
    tk.call("run_python", {"code": "print(1)"})
    tk.call("run_python", {"code": "print(1)"})
    assert all(not c.cached for c in tk.calls)  # never served from cache
