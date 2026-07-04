# Proposal 0002 — ForecastBench as a question source (geopolitics/security + looser questions)

**From:** the `superforecaster` harness (a downstream consumer of this env)
**Status:** ACCEPTED & implemented (2026-07-04) — see `src/forecast_playground/sources/forecastbench.py`.

## Summary

Add a **ForecastBench question source** so the harness isn't limited to Polymarket-style
market questions. The ForecastBench datasets repo
(`github.com/forecastingresearch/forecastbench-datasets`, **CC BY-SA 4.0, no auth**)
publishes biweekly `question_sets/` + matching `resolution_sets/` covering **Metaculus,
INFER/RFI, and ACLED conflict data** alongside markets — i.e. real **geopolitics/security**
questions and **looser, analyst-style** phrasing, with clean freeze + resolution dates.

This complements the existing `PolymarketSource`/`fetch_resolved_markets` path. Today a
consumer that wants non-market questions has to load and join these files itself (we do,
downstream); promoting it here means every harness gets leak-free geopolitical questions
for free, and it lives next to the other data sources where data belongs.

## Why it's a clean fit for this repo

It maps directly onto your "Writing a new source" checklist in `CONTRIBUTING.md`:

1. **Timestamp = `freeze_datetime`.** Each question carries a `freeze_datetime` (the
   as-of instant) and `freeze_datetime_value` (the frozen market/community probability —
   a ready-made baseline). The Clock is set to `freeze_datetime`; nothing after it may be
   surfaced. Resolutions carry `resolution_date` + `resolved_to` (0/1 for binary).
2. **Native "at or before":** the data is static and pre-dated, so there's no live
   backend to over-fetch — the freeze date *is* the filter. (This is a *question* source
   that supplies `(text, as_of, outcome)`; retrieval masking still flows through the
   existing HARD sources.)
3. **AsOfGuarantee: HARD** for the question/outcome metadata (freeze and resolution
   timestamps are true and fixed).
4. **Leak-safety test:** add a row asserting a question's `as_of` (freeze) is strictly
   before its `resolution_date`, and that we never expose the resolution to the model.

## Shape (verified against the live repo)

- Files: `datasets/question_sets/<date>-llm.json`, `datasets/resolution_sets/<date>_resolution_set.json`.
- Join **questions → resolutions by `(source, id)`**; a question has multiple horizon
  resolutions — take the earliest resolved one.
- Skip **combination questions** (`combination_of != "N/A"`) and non-string `id`/`source`
  (combo resolution records have list ids).
- Yield: `id, source, question, resolution_criteria, background, as_of (=freeze_datetime),
  resolution_date, outcome (0/1), market_prob (=freeze_datetime_value)`.

Verified yield from one dated set (`2025-03-02`): **384** standalone resolved binary
questions all-sources; **75** from the geopolitics-leaning sources (acled/metaculus/infer).

## Caveats worth designing around (found while prototyping)

- **ACLED dominates and is formulaic.** In `2025-03-02` the geo split was 63 acled / 8
  metaculus / 4 infer. ACLED questions are auto-generated conflict templates ("≥X
  fatalities in region Y by date Z") — genuinely security, but not *looser analytic*
  phrasing. The looser, more interesting questions are the Metaculus/INFER ones (~12/set),
  so a `sources=` filter is important, and accumulating across dates matters.
- **Cross-date duplication.** Long-horizon questions recur (same `id`) across biweekly
  sets, so multi-date loading needs de-dup by `(source, id)`.
- **Outcome skew.** The geo subset skews NO, so an uncertainty filter (via `market_prob`)
  or balancing is wanted for a fair eval.
- **License:** CC BY-SA 4.0 — attribution + share-alike (fine to consume; note in docs).

## Reference implementation

Working loader + tests downstream: `superforecaster/src/superforecaster/sources/forecastbench.py`
(+ `tests/test_forecastbench_loader.py`, mocked HTTP). Ran end-to-end through this env's
Clock/Toolkit on 17 Metaculus/INFER geopolitical questions with no leakage. Happy to port
to this repo's source style and add the leak-safety row.

## Follow-ups (separate, lower priority)

- **News tier** (already on your roadmap, deferred) would lift the ceiling on
  surprise-event questions that Wikipedia/pageviews can't foresee.
- IARPA HFC + GJP (Harvard Dataverse, **CC0**) and Autocast (ships a date-partitioned news
  corpus) are strong additional geopolitical sources if you want more after ForecastBench.
