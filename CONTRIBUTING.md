# Contributing

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Before committing

```bash
python -m pytest -q          # tests must pass (network is mocked; runs offline)
ruff check src/ tests/ examples/ adapters/
```

## The one rule that matters: don't leak the future

Every retrieval flows through a `Clock`, and no source may return data stamped after
`clock.as_of`. When adding a **new source**:

1. Route every candidate timestamp through `clock.guard(ts, source=...)` — it raises
   `LookaheadError` on lookahead. This is the last line of defense.
2. Prefer querying the backend with a native "at or before" filter so the guard
   never has to fire (see `WikipediaSource` `rvstart`, `WaybackSource` CDX `to=`).
3. Declare an honest `AsOfGuarantee`: `HARD` (timestamp is a true upper bound),
   `SOFT` (date-filterable but values may be revised after the date), or `NONE`.
4. Add a test asserting a query at `T` cannot surface a fact created after `T` — the
   central correctness property. See `tests/test_wayback.py` for the pattern.

## Style

Match the surrounding code: typed functions with a one-line-summary docstring
(the toolkit and schema generator rely on that first docstring line). Keep sources
small and single-purpose.
