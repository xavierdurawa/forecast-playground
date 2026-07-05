# Proposal 0003 — Parametric (pretraining) leakage guard: training-cutoff filter

**From:** the `superforecaster` harness (a downstream consumer of this env)
**Status:** ACCEPTED & implemented (2026-07-05) — see `src/forecast_playground/leakage.py`. Built provider-agnostic with cutoffs sourced live from models.dev (multi-provider) rather than a hardcoded Claude-only table, plus `register_cutoff` overrides.

## Summary

Add a **training-cutoff leak filter** so studies can drop questions whose outcome the
*model already knows from pretraining*. The `Clock` masks **retrieval** — a tool can't
return anything after `as_of`. It does nothing about the model's **weights**: if a question
resolved before the model's training cutoff, the model may simply *know* the answer, no
retrieval required. That is a leak, and it's invisible to the current leak-safety machinery.

The proposed primitive is small and deterministic: `is_leak_safe(resolution_date, model)` —
true iff the resolution date is safely after `model`'s training cutoff (plus a margin). It
belongs next to the `Clock` because it's the same *kind* of guarantee — "the model cannot
have seen this" — just enforced on the parametric side instead of the retrieval side.

## Why this belongs in the env

Every element is already env-shaped:

- **It's a leak-safety guarantee**, which is this repo's entire reason to exist. The `Clock`
  enforces "retrieval ≤ `as_of`"; this enforces "resolution > model training cutoff." Two
  halves of the same invariant. A question that passes the Clock but fails this is still
  contaminated.
- **The join point already exists.** `ResolvedQuestion.resolution_date` is exactly the field
  the filter reads. A study that iterates resolved questions can gate on `is_leak_safe(
  q.resolution_date, model)` with no new plumbing.
- **It's deterministic, dependency-free, no model calls** — unlike the judge (proposal 0001),
  so there's no `[extra]`/nondeterminism caveat. It's a pure function over a small table.
- **Every consumer needs it.** Any harness evaluating on *historical* resolved questions (the
  whole point of the time-mask) is exposed to parametric leakage. Right now each consumer
  must re-derive the cutoff table and the filter — and get the subtle parts right (below).

## Why it matters — a live leak we hit

Downstream, before this filter existed, an exception-mining run scored the question **"Will
Israel invade Syria in 2024?"** at **agent probability 0.93** — near-perfect. The agent
didn't forecast it; a 2024 event is deep inside Sonnet 4.6's training cutoff (Jan 2026), so
the model *remembered* the outcome. Retrieval was correctly masked; the leak was entirely
parametric. Any historical-question eval on a recent model has this exposure.

## The subtle parts (get these right once, centrally)

1. **Filter on the TRAINING cutoff, not the "reliable knowledge" cutoff.** Anthropic publishes
   both (e.g. Sonnet 4.6: reliable = Aug 2025, training = Jan 2026). Training is the later,
   broader date — the conservative choice. Filtering on the reliable date would admit leaked
   questions.
2. **Add a safety margin past the stated cutoff.** Models often know events slightly past
   their nominal cutoff (post-training / RLHF data). We default to **90 days**, and this is
   *grounded, not arbitrary*: for models where both dates are known (dated-snapshot IDs +
   Fable 5 GA), release-date − training-cutoff is consistently ≥ ~90 days (Sonnet 4.5 ~90,
   Haiku 4.5 ~92, Opus 4.5 ~92, Fable 5 ~159; median ~92). Release date is a hard upper bound
   on model knowledge, so a 90-day margin sits at the lenient-but-defensible end.
3. **Unknown model ⇒ not certifiable ⇒ reject.** If the model isn't in the cutoff table, the
   safe default is "cannot certify leak-safe" (return False), not "assume clean."
4. **Substring match on the model id** so region/prefix variants resolve
   (`us.anthropic.claude-sonnet-4-6`, `global.…`, `anthropic.…` → same cutoff).

## Proposed API

```python
def training_cutoff(model: str) -> datetime | None: ...        # None if unknown
def min_safe_resolution(model, margin_days=90) -> datetime | None: ...
def is_leak_safe(resolution_date, model, margin_days=90) -> bool: ...
```

The cutoff table lives centrally (sourced from the Anthropic model overview,
platform.claude.com/docs/en/about-claude/models/overview) so it's maintained in one place as
new models ship. Optionally, `fetch_forecastbench_questions(..., model=...)` could gate on it
directly so questions come back leak-safe by construction — mirroring how `select_uncertain`
takes a `clock_for`.

## Implementation note (accepted with one change)

Accepted as proposed, with the cutoff **table** sourced differently: instead of a
hardcoded Claude-only table, cutoffs are pulled live (and cached) from **models.dev**
(MIT, multi-provider — OpenAI/Google/Mistral/DeepSeek/Llama/… as well as Claude),
because "others use this env too" and a hardcoded table goes stale and is Claude-only.
`register_cutoff(model, date)` gives the same explicit-override control the proposal's
table implied; the unknown-model→reject default is unchanged. Caveat surfaced during
impl: models.dev exposes one `knowledge` field that leans toward the training cutoff
but doesn't formally separate training vs. reliable-knowledge — use an override when a
model needs specific semantics. Live-verified: correctly flags a 2024 question as
unsafe for Sonnet 4.6, and gates ForecastBench geo questions 75 → 4 leak-safe for a
Jan-2026-cutoff model.

## Reference implementation

Working, tested implementation downstream: `superforecaster/src/superforecaster/leakage.py`
(+ `tests/test_leakage.py`, no network). Its cutoff table (verified 2026-07-05):
Sonnet 4.6 / Sonnet 5 / Opus 4.7 / Opus 4.8 / Fable 5 = Jan 2026; Opus 4.6 = Aug 2025;
Haiku 4.5 = Jul 2025. Happy to port to this repo's style and add a leak-safety-style test row.

## Caveats / what this does NOT do

- The stated cutoff is a **floor**, not a guarantee — the margin is the hedge, but a
  determined leak (an event that made it into post-training data) can still slip through. The
  cutoff filter is the cheap, deterministic first line, not the whole defense.
- Month-granularity cutoffs carry ~±30 days of slop; the margin absorbs it.

## Future companion (NOT part of this proposal)

The complete defense pairs the cutoff filter with an **empirical contamination probe**: run
the model with retrieval disabled on a resolved question and flag suspicious
confident-and-correct answers (contamination the *stated* cutoff misses — e.g. a model that
knows a post-cutoff event via post-training). That is nondeterministic, costs tokens, and —
critically — is **not yet validated** (a cold-confident-correct answer might mean leakage, or
just an easy question; disentangling the two is unsolved). We're prototyping and validating it
downstream first; if it proves out, the *probe primitive* is a natural env addition (the
*gating policy* — thresholds, drop-vs-downweight — stays in the consuming harness, exactly as
with the judge). Flagging it here so the full leak-safety picture is visible, but proposing
only the deterministic cutoff filter for now.
