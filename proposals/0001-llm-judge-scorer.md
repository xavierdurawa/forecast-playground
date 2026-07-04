# Proposal 0001 — LLM-as-judge scorer for looser, non-boolean forecasts

**From:** the `superforecaster` harness (a downstream consumer of this env)
**Status:** ACCEPTED & implemented (2026-07-04) — see `src/forecast_playground/judge.py`.

## Summary

Add an optional **LLM-as-judge scoring rule** that scores a forecast against a
*now-known* resolution and reduces to a **Brier-compatible** number, so that looser,
non-knife-edge questions (the kind intelligence analysts answer — "how likely is
US–Iran escalation this spring," not a market's exact settlement clause) can be scored
on the same 0–1, lower-is-better scale as boolean questions.

This sits next to `scoring.py` / `calibration.py` as a *scoring capability*. It is
proposed here (not kept downstream) for the same reason `aggregate()` and
`calibration_report()` landed here: **any** harness that wants to use loose/geopolitical
questions needs to score them, and a source of loose questions is only useful if the env
can also score them — source and scorer are a matched pair.

## Why it belongs in the env (and the caveat that says "maybe behind an extra")

`brier_score`/`log_score` are pure, deterministic, dependency-free. **An LLM judge is
none of those** — it is nondeterministic, needs a model/driver, costs tokens, and can
fail to parse. That is a real departure from this repo's "structural guarantee" ethos,
so we are *proposing*, not assuming. Suggested shape that respects it:

- Ship it behind an optional extra (like `[anthropic]`/`[verifiers]`), e.g. `[judge]`,
  so the core stays pure.
- Mark it clearly as **non-deterministic** and **not** part of the leak-safety
  invariant (it scores *after* resolution; it does no retrieval, so it cannot leak).
- Keep the reduction (below) a pure function so *that* part is testable offline.

## The design (validated downstream)

The judge, given the question + the forecaster's probability & reasoning + the known
resolution, returns a rubric (accuracy / calibration / reasoning / decision-usefulness,
each 0–1) **and** a `resolved_label` ∈ [0,1]: the probability the YES/claim actually
came true *given the resolution* — a soft outcome label, independent of the forecaster.
The Brier-compatible score is then:

```
brier_from_judgment = (forecaster_probability - resolved_label) ** 2
```

For a boolean question `resolved_label` is 0 or 1 and this **is exact Brier**. For a
loose question the judge maps "what happened" onto [0,1] so the same proper score
applies. Report the rich rubric and the consolidated scalar separately or together.

### The failure mode we already found and fixed (so you don't have to)

Our first version asked the judge how well the forecast's *own claim* was borne out and
scored against that. It **inverted**: a confident-wrong forecast (p=0.02 on an event that
happened) scored ~perfect, because it was self-consistent. On a known-outcome validation
set the judge Brier was **anti-correlated** with true Brier (corr −0.25). Grounding the
label in the **outcome** (not the forecast's self-consistency) is the fix.

### Validation evidence (this is the point of proposing a *proven* prototype)

We validated the fixed judge where truth is known — scoring boolean questions with known
outcomes and checking the judge Brier against the real Brier:

| Metric | v1 (self-consistency) | v2 (outcome-grounded) |
|---|---|---|
| MAE(true Brier, judge Brier) | 0.318 | **0.0000** |
| corr(true, judge) | −0.247 | **1.000** |
| `mean\|resolved_label − outcome\|` | 0.27 | **0.0000** |

Then, separately, on **geopolitical** binary questions (Metaculus/INFER, n=17): judge
Brier == proper Brier to 4 dp, every row exact. So the judge is a faithful drop-in for
Brier where truth is known, across two domains — which is the evidence that lets us trust
it where truth is only expressible in words.

## What we're NOT asking you to own

Which judge model to use, judge ensembling, rubric weights, prompt variants — those are
*experiment configuration* and stay in the consuming harness. We're asking only for the
scoring primitive + the pure reduction, so all consumers share one validated scorer
instead of each re-deriving (and mis-deriving — see the inversion above) their own.

## Reference implementation

A working, tested implementation lives in the `superforecaster` repo
(`src/superforecaster/judge.py`, `experiments/validate_judge.py`). Happy to port it to
this repo's style (typed, one-line-summary docstrings) and add offline tests for the
pure reduction if you want it here.

## Open questions for you

1. Behind a `[judge]` extra, or out of scope for a leak-safety-focused env?
2. If in scope: a generic `Driver`-based judge, or provider-specific?
3. Should a "judge-vs-known-truth agreement" check live in the leak-safety-style test
   suite (as a *quality* gate, not a leak gate)?
