# Contributing

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Before committing

```bash
python -m pytest -q          # offline suite: fast, network-mocked, must pass
ruff check src/ tests/ examples/ adapters/
```

Integration tests hit live APIs (and a local model) and are skipped by default:

```bash
CHRONO_CONTACT=you@example.com pytest -m integration   # live; needs network
```

The leak-safety matrix (`tests/test_leak_safety.py`) checks every source against
its real API across several dates — this is the invariant that must never regress.

## Writing a new source

Start from `examples/custom_source_template.py` — a runnable, annotated skeleton.
Then add one row to `LIVE_SOURCES` in `tests/test_leak_safety.py` so the no-leak
guarantee is checked for it automatically.

## The one rule that matters: don't leak the future

Every retrieval flows through a `Clock`, and no source may return data stamped after
`clock.as_of`. The Toolkit re-guards every `Document` a source returns, so the
guarantee is enforced structurally — a source that forgets still cannot leak. Your
job when adding a **new source** is to make that guarantee *true and tight*:

1. Stamp each `Document.timestamp` with when that content actually became available
   (revision time, snapshot time, filing/trade date). This is the one thing only you
   can get right — the Toolkit trusts your timestamp, then guards it.
2. Query the backend with a native "at or before" filter so you don't fetch the
   future in the first place (see `WikipediaSource` `rvstart`, `WaybackSource` CDX
   `to=`). Calling `clock.guard(ts, source=...)` yourself is recommended (fail fast).
3. Declare an honest `AsOfGuarantee`: `HARD` (timestamp is a true upper bound),
   `SOFT` (date-filterable but values may be revised after the date), or `NONE`.
4. Add a test asserting a query at `T` cannot surface a fact created after `T` — the
   central correctness property. See `tests/test_wayback.py` for the pattern.

## Style

Match the surrounding code: typed functions with a one-line-summary docstring
(the toolkit and schema generator rely on that first docstring line). Keep sources
small and single-purpose.
