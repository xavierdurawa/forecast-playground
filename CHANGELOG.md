# Changelog

All notable changes to ForecastPlayground. This project is pre-1.0; the public API
(everything in `forecast_playground.__all__`) is stabilizing but may still change.

## [Unreleased]

### Added
- **Leak-safety property test** — a parametrized integration test asserting every
  registered source respects the as-of Clock across a range of dates. Adding a
  source means adding one row; the invariant is then guarded automatically.
- **Custom-source template** (`examples/custom_source_template.py`) — a runnable,
  annotated skeleton for writing your own time-masked `Source`.
- **Integration test suite** (`pytest -m integration`) — live API + full-forecast
  checks, skipped by default so the normal run stays offline and fast.
- **News sources**: `CurrentEventsSource` (Wikipedia's curated daily digest) and
  `GDELTNewsSource` (global article stream, back to 2015) — both free and leak-safe.
- **Calibration reporting** (`calibration_report` → reliability curve + ECE/MCE).
- **Ensemble aggregation primitives** (`trimmed_mean`, `extremize`, `aggregate`).
- **Provider-agnostic drivers** (`OpenAIDriver`, `AnthropicDriver`) + swappable
  `Scaffold` (`NAIVE`, `SUPERFORECASTER`).
- **On-disk result cache** (`ResultCache`).

### Changed
- The `Toolkit` now re-guards every returned Document's timestamp, so the
  no-lookahead guarantee is structural — a bring-your-own source that forgets to
  guard still cannot leak.
- Public API in `__all__` is now grouped by role and documented as the stable surface.

### Notes
- News: free news APIs (NewsAPI, Mediastack) are intentionally unsupported — their
  free tiers are recent-only (a rolling window anchored to now) and cannot reach a
  past as-of date.
