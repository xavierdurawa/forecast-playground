# ChronoHarness

A **time-masked retrieval harness** for AI forecasting.

Give a model tools that, as of a frozen date, return only information that existed
*before* that date — so you can train or evaluate a forecaster on questions whose
answers are now known, without lookahead leakage. A "cached internet."

> Status: early but working. Ships the core (Clock + Source protocol), five
> time-masked sources, a model-callable toolkit, a minimal agent loop, scoring, an
> on-disk cache, and a verifiers RL-environment adapter. See [DESIGN.md](DESIGN.md).

## Why

To get training signal you want to ask a model about events that have already
resolved, letting it research using only what was knowable at the time. Future-
dating questions is leak-free but yields tiny data volume. Time-masking reuses every
historically-resolved question — **but only if the time-mask is airtight.** That
airtight mask, applied uniformly across heterogeneous sources, is what this provides.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                # core + test tooling
pip install -e ".[dev,anthropic,verifiers]"   # + Bedrock agent loop + RL env
```

Set `CHRONO_CONTACT` (an email or URL) so public-archive requests are polite per
Wikimedia's User-Agent policy; without it a neutral fallback is used.

## Quickstart

```python
from chrono_harness import Clock, WikipediaSource

clock = Clock.at("2024-01-01")          # nothing after this instant is visible
wiki = WikipediaSource(mode="search")   # full-text search -> as-of article text
for doc in wiki.fetch("SpaceX Starship orbital test", clock):
    print(doc.timestamp, "-", doc.content[:120])   # guaranteed <= as_of
```

The `Clock` is the single chokepoint: any source that tries to surface data newer
than `as_of` raises `LookaheadError`. Every source declares an **as-of guarantee**
(`HARD` / `SOFT` / `NONE`) so a run can restrict itself to leak-free sources.

## Sources (v1 — all free, no API keys)

| Source | Returns | Guarantee |
|---|---|---|
| `WikipediaSource` | Article text as-of a date (exact-title or full-text search) | HARD |
| `PageviewsSource` | Daily pageview counts (leak-safe attention signal) | HARD |
| `WaybackSource` | Any URL as archived at-or-before a date | HARD |
| `PolymarketSource` | Prediction-market YES-probability history up to a date | HARD |

Plus `run_python` (a sandboxed compute tool) for distributions / Monte Carlo / scoring.

## Driving a model

`Toolkit` binds sources + a Clock into model-callable tools (the Clock is injected,
never a model argument), dispatches calls, caches results, and traces every call:

```python
from chrono_harness import Clock, Toolkit, WikipediaSource, PolymarketSource, ResultCache
from chrono_harness.agent import run_forecast          # needs [anthropic]
from anthropic import AnthropicBedrock

tk = Toolkit(
    clock=Clock.at("2024-10-06"),
    sources=[WikipediaSource(mode="search"), PolymarketSource()],
    cache=ResultCache(),                                # optional on-disk cache
)
fc = run_forecast(AnthropicBedrock(), "global.anthropic.claude-sonnet-4-6",
                  "Will X happen?", tk)
print(fc.probability)     # a calibrated 0..1 forecast
```

## RL environment (verifiers)

`adapters/verifiers_env.py` exposes `load_environment()` → a `ToolEnv` whose tools
are time-masked per dataset row (the as-of date travels in `info["as_of"]`, bound by
the env, never model-controlled). Scoring uses the Brier score (a proper scoring
rule). Needs the `verifiers` extra.

## Study

`examples/leak_ab_study.py` runs a forecasting study over resolved Polymarket
questions with configurable arms (`full`, `nomarket`, `unmasked`) and optional
uncertainty-based question selection. See its module docstring for flags.

```bash
CHRONO_CONTACT=you@example.com python examples/leak_ab_study.py \
    --n 15 --arms full nomarket --uncertain-only
```

## Layout

```
src/chrono_harness/
  clock.py        # the no-lookahead chokepoint
  http.py         # retry/backoff session + polite User-Agent
  cache.py        # simple on-disk result cache
  retrieval.py    # chunk + rank (keeps results relevant and context-sized)
  schema.py       # plain functions -> OpenAI tool defs
  scoring.py      # Brier / log score (proper scoring rules)
  toolkit.py      # binds sources + Clock into model-callable tools
  agent.py        # minimal Claude (Bedrock) forecasting loop
  sources/        # wikipedia, pageviews, wayback, polymarket
adapters/         # verifiers_env.py (RL/eval)
examples/         # runnable demos
tests/            # incl. the central no-leak property
```

## License

MIT — see [LICENSE](LICENSE). Architecture and prior-art credit in [DESIGN.md](DESIGN.md).
