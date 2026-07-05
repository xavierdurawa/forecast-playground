# Changelog

All notable changes to ForecastPlayground. This project is pre-1.0; the public API
(everything in `forecast_playground.__all__`) is stabilizing but may still change.

## [Unreleased]

### Added
- **Parametric leakage guard** — `is_leak_safe(resolution_date, model)` +
  `training_cutoff` / `min_safe_resolution` / `register_cutoff`. The Clock masks
  retrieval; this catches the model *knowing* an old outcome from pretraining (keep
  only questions resolving safely after the model's training cutoff + margin).
  Cutoffs from models.dev (multi-provider, cached) with consumer overrides;
  unknown-model → reject (fail-safe). `fetch_forecastbench_questions(..., model=...)`
  gates on it. Pure/deterministic. Proposed in `proposals/0003`.
- **LLM-as-judge scorer** — `judge_forecast()` (`[judge]` extra) scores looser,
  non-boolean forecasts: a Driver-based model maps a known resolution onto a soft
  label in [0, 1] and `brier_from_judgment` (pure) takes the Brier distance — equals
  Brier on boolean questions, verified to agree with true Brier to 4 dp on known
  outcomes. Non-deterministic and outside the leak-safety invariant (scores after
  resolution, no retrieval). Includes a live quality-gate integration test. Proposed
  in `proposals/0001`.
- **ForecastBench question source** — `fetch_forecastbench_questions()` loads resolved
  questions (Metaculus / INFER / ACLED + markets; CC BY-SA 4.0, keyless) as
  `ResolvedQuestion(text, as_of=freeze_datetime, outcome, market_prob, ...)`. Adds
  geopolitics/security + looser analyst-style questions beyond Polymarket. Joins
  questions↔resolutions by `(source, id)`, takes the earliest horizon, skips
  combination questions, and tolerates junk `freeze_datetime_value` fields. Proposed
  in `proposals/0002`.
- **GDELT BigQuery backend** — `GDELTNewsSource(backend="bigquery")` queries GDELT's
  day-partitioned public BigQuery table for efficient multi-year news search (needs
  a GCP project + the `bigquery` extra; keyless auth via ADC). The keyless
  `rawfiles` backend remains the default. Same leak-safe DATE filtering and
  `Document` shape either way. Cost-guarded: a hard `maximum_bytes_billed` cap
  (default 2 GB — over-budget queries are rejected unbilled) plus a
  `bigquery_dry_run()` estimator. Uses the `_partitioned` table so a date window
  prunes to ~0.4 GB rather than scanning ~200 GB of the unpartitioned table.
- **Leak-safety property test** — a parametrized integration test asserting every
  registered source respects the as-of Clock across a range of dates. Adding a
  source means adding one row; the invariant is then guarded automatically.
- **Custom-source template** (`examples/custom_source_template.py`) — a runnable,
  annotated skeleton for writing your own time-masked `Source`.
- **Integration test suite** (`pytest -m integration`) — live API + full-forecast
  checks, skipped by default so the normal run stays offline and fast.
- **News sources**: `CurrentEventsSource` (Wikipedia's curated daily digest) and
  `GDELTNewsSource` (global article stream, back to 2015) — both free and leak-safe.
- **`FREDSource`** — economic series (GDP, CPI, unemployment, ...) via ALFRED
  vintages, so values are point-in-time-*known*, not later-revised. Needs a free
  `FRED_API_KEY`.
- **`NOAASource`** — historical daily weather (temp/precip) for a station via NOAA
  CDO v2. Needs a free `NOAA_TOKEN`.
- **MCP server** (`adapters/mcp_server.py`) — exposes every source to interactive
  MCP clients as `<source>(query, as_of)` tools. `as_of` is user-controlled here
  (exploration); scored-eval leak-safety stays in the Toolkit/verifiers paths.
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
- Scope: this is the RL *environment*, not the trainer. Training loops belong in a
  separate package that consumes the `verifiers` adapter (see README).
