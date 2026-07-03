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
