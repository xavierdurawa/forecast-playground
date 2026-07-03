# ForecastPlayground — Design

A time-masked retrieval harness for AI forecasting.

## Problem

To train or evaluate a forecasting model, you want to ask it questions whose
answers are *now* known but were *unknown* at some past "as-of" date `T`, and let
it research the question using only information that existed before `T`. If any
tool leaks information from after `T`, the signal is contaminated and the model
learns nothing real ("lookahead bias").

Future-dating questions (the ForecastBench approach) avoids leakage trivially —
the answer doesn't exist yet — but generates tiny training volume, because you
must wait for real time to pass. **Time-masked retrieval** instead reuses every
historically-resolved question, giving huge volume and fast RL iteration — *but
only if the time-mask is airtight.*

The airtight time-mask, applied uniformly across heterogeneous data sources, is
the thing this project provides and the thing nobody has open-sourced.

## Core abstractions

### `Clock`
A frozen "current time." Every retrieval flows through it. The Clock is the
single chokepoint that forbids lookahead. A tool cannot return data stamped after
the Clock's `as_of` instant. There is no per-source date-juggling — sources hand
the Clock their data's timestamp and the Clock decides admissibility.

### `Source` (protocol)
An adapter over one data backend (Wikipedia, Wayback, a prediction market, ...).
Every source declares an **as-of guarantee**:

| Guarantee | Meaning | Examples |
|---|---|---|
| `HARD` | Leak-free by construction; the timestamp is a true upper bound on availability. | Wikipedia revisions, Wayback snapshots, FRED vintages, SEC filing dates, arXiv submit dates, prediction-market price history |
| `SOFT` | Date-filterable, but may return values revised after `T` (the *value* is point-in-time-true, not point-in-time-*known*). Usable with a documented caveat. | World Bank indicators, Wikidata P585 qualifiers |
| `NONE` | Latest-only; no historical query. Rejected by the Clock unless explicitly allowed. | (used to mark backends we refuse to time-mask) |

The harness surfaces the guarantee to the caller so a training run can choose its
own risk tolerance (e.g. "HARD sources only").

### `Tool`
A plain Python function with type hints + a docstring. Tools are the unit a model
calls. A retrieval tool takes its query plus the Clock (injected, not model-
supplied — the model never sets its own as-of date). The same function bodies
power every downstream surface.

## Layered surfaces (one core, many consumers)

```
        ┌──────────────────────────────────────────────┐
        │  Core: Clock + Source protocol + Tool funcs    │
        └──────────────────────────────────────────────┘
             │              │                  │
   OpenAI/JSON schema   verifiers ToolEnv   MCP server
   (any framework)      (RL + eval)         (interactive agents)
```

- **OpenAI/JSON tool schema** — `tools_to_openai_schema()` emits the cross-provider
  `{name, description, parameters}` form. Usable directly by any framework.
- **verifiers `load_environment()`** — wraps the tools in `vf.ToolEnv` with a
  Brier/log-score rubric; plugs into prime-rl/GRPO. Nearly free because `ToolEnv`
  auto-converts plain Python functions.
- **MCP server** — optional FastMCP wrapper for interactive agents (Claude
  Desktop, Cursor, ChatGPT). Not the RL path.

## v1 toolset (all free, no API keys)

- `wiki_read(title, clock)` — Wikipedia article text as of the Clock. **HARD.**
- `wiki_pageviews(title, start, clock)` — daily pageview counts (leak-safe
  attention signal, replaces fragile Google Trends). **HARD.**
- `web_fetch(url, clock)` — closest Wayback snapshot at-or-before the Clock. **HARD.**
- `market_history(market, clock)` — prediction-market probability history up to
  the Clock (Polymarket, keyless). **HARD.**
- `run_python(code)` — sandboxed compute for distributions / Monte Carlo / scoring.

Fast-follow: FRED/ALFRED vintages (free key), news (CC-NEWS / GDELT / paid AskNews).

## Non-goals (v1)

- We do not ship the RL trainer — only the environment. Bring your own (prime-rl,
  verifiers RLTrainer, TRL).
- We do not host data. We query public APIs / archives at request time, with
  optional local caching.
- News retrieval is deliberately deferred — no free source is as clean as the
  HARD-tier ones, and it needs separate design.

## Leakage discipline (the part that must be airtight)

1. The model never supplies its own as-of date; the Clock is injected by the harness.
2. Every source maps its result to a timestamp and the Clock rejects anything `> as_of`.
3. Sources with `SOFT`/`NONE` guarantees are opt-in and logged, never silent.
4. A test suite asserts, per source, that a query at `T` cannot surface a fact
   created after `T` (the central correctness property).

## Prior art / credit

Inspired by amitlevy49's "cached internet" (LessWrong, 2026). Architecture
informed by Halawi et al.'s `llm_forecasting` (ideas only — that repo is
unlicensed). RL reward design follows Bereket & Leskovec / Turtel et al.: use
proper scoring rules and do **not** divide GRPO rewards by group std-dev.
