"""Forecasting study over resolved Polymarket questions, with a leak-free time-mask.

Each question is forecast under one or more ARMS:
  - full:     masked, all tools (Wikipedia + Pageviews + Polymarket market odds)
  - nomarket: masked, NO market tool (Wikipedia + Pageviews only) — isolates genuine
              reasoning from "just read the market price"
  - unmasked: clock = now, all tools — the future leaks in; the leak sanity check.
              If the mask works, unmasked should score far better (it can cheat).

Question selection: with --uncertain-only, keep only questions whose market-implied
probability at the as-of date was in [0.25, 0.75] — i.e. genuinely live at the
forecast date, so a good score reflects skill, not stating the obvious. This uses
the same time-mask (leak-free), never the resolved outcome.

Uses Claude via Bedrock. API calls ≈ N * (number of arms).

Run:  FORECAST_CONTACT=you@example.com python examples/leak_ab_study.py \
          --n 15 --arms full nomarket --uncertain-only
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

from forecast_playground import (
    NAIVE,
    SUPERFORECASTER,
    AnthropicDriver,
    Clock,
    OpenAIDriver,
    PageviewsSource,
    PolymarketSource,
    ResultCache,
    Toolkit,
    WikipediaSource,
    brier_score,
    calibration_report,
    fetch_resolved_markets,
    mean_brier,
    run_forecast,
    select_uncertain,
)

# Sonnet 4.6 via Bedrock: capable, not as costly as Opus; good for testing.
MODEL = "global.anthropic.claude-sonnet-4-6"
SCAFFOLDS = {"naive": NAIVE, "superforecaster": SUPERFORECASTER}


def build_toolkit(clock: Clock, *, market: bool, cache: ResultCache) -> Toolkit:
    sources = [WikipediaSource(mode="search"), PageviewsSource()]
    if market:
        sources.append(PolymarketSource())
    return Toolkit(clock=clock, sources=sources, cache=cache)


def make_question(m, *, market: bool) -> str:
    q = f"{m.question}\n\nForecast the probability this resolves YES."
    if market:
        q += (
            f" (The market's YES outcome token id, for the polymarket tool, is "
            f"{m.token_id_yes}.)"
        )
    return q


def main() -> None:
    # Line-buffer stdout so progress is visible during long sequential runs
    # (otherwise output only appears when the process exits).
    sys.stdout.reconfigure(line_buffering=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="number of questions to forecast")
    ap.add_argument("--lead-days", type=int, default=30, help="masked clock lead time")
    ap.add_argument(
        "--arms",
        nargs="+",
        choices=["full", "nomarket", "unmasked"],
        default=["full", "unmasked"],
        help="which conditions to run per question",
    )
    ap.add_argument(
        "--uncertain-only",
        action="store_true",
        help="keep only questions with as-of market prob in [0.25, 0.75]",
    )
    ap.add_argument("--pool", type=int, default=40, help="candidate pool to select from")
    ap.add_argument(
        "--provider",
        choices=["bedrock", "openai"],
        default="bedrock",
        help="bedrock = native Anthropic/Bedrock; openai = any OpenAI-compatible endpoint",
    )
    ap.add_argument("--model", default=MODEL, help="model id (default is Bedrock Sonnet 4.6)")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible base URL (vLLM/Ollama/...)")
    ap.add_argument("--region", default="us-west-2", help="AWS region for Bedrock")
    ap.add_argument(
        "--scaffold",
        choices=list(SCAFFOLDS),
        default="naive",
        help="forecasting strategy/instructions to wrap the model in",
    )
    ap.add_argument("--no-cache", action="store_true", help="disable the retrieval cache")
    args = ap.parse_args()

    if args.provider == "openai":
        driver = OpenAIDriver(base_url=args.base_url)
    else:
        driver = AnthropicDriver(aws_region=args.region)
    scaffold = SCAFFOLDS[args.scaffold]
    now = datetime.now(timezone.utc)
    cache = ResultCache(enabled=not args.no_cache)
    print(f"provider={args.provider} model={args.model} scaffold={scaffold.name}\n")

    def as_of_of(m):
        return m.end_date - timedelta(days=args.lead_days)

    # Fetch a candidate pool, then optionally filter to genuinely-uncertain ones.
    pool = fetch_resolved_markets(limit=max(args.pool, args.n))
    if args.uncertain_only:
        pairs = select_uncertain(pool, clock_for=lambda m: Clock.at(as_of_of(m)))
        selected = [(m, p) for m, p in pairs][: args.n]
        print(
            f"Selected {len(selected)} genuinely-uncertain questions "
            f"(as-of market prob in [0.25, 0.75]) from a pool of {len(pool)}.\n"
        )
    else:
        selected = [(m, None) for m in pool[: args.n]]
        print(f"Loaded {len(selected)} resolved markets.\n")

    # Per-arm accumulators.
    scores: dict[str, list[tuple[float, int]]] = {a: [] for a in args.arms}

    for i, (m, asof_prob) in enumerate(selected, 1):
        as_of = as_of_of(m)
        base = f"    resolved {m.outcome} on {m.end_date.date()}; masked as-of {as_of.date()}"
        if asof_prob is not None:
            base += f"; market@as-of={asof_prob:.2f}"
        print(f"[{i}/{len(selected)}] {m.question[:70]}")
        print(base)

        for arm in args.arms:
            has_market = arm != "nomarket"
            clock = Clock.at(now) if arm == "unmasked" else Clock.at(as_of)
            tk = build_toolkit(clock, market=has_market, cache=cache)
            fc = run_forecast(
                driver, args.model, make_question(m, market=has_market), tk,
                scaffold=scaffold,
            )
            scores[arm].append((fc.probability, m.outcome))
            b = brier_score(fc.probability, m.outcome)
            ok = sum(c.ok for c in tk.calls)
            hits = sum(c.cached for c in tk.calls)
            flag = " DEFAULTED" if fc.defaulted else (" forced" if fc.forced else "")
            print(
                f"    {arm:9s} p={fc.probability:.2f}  brier={b:.3f}  "
                f"tools={len(tk.calls)}(ok {ok}, cached {hits})  turns={fc.turns}  "
                f"stop={fc.stop_reason}{flag}"
            )
        print()

    print("=" * 64)
    for arm in args.arms:
        note = {
            "full": "masked, all tools",
            "nomarket": "masked, no market tool (pure reasoning)",
            "unmasked": "clock=now, can see the future",
        }[arm]
        print(f"{arm:9s} mean Brier: {mean_brier(scores[arm]):.4f}  ({note})")
    # A resolved-market study also has an obvious baseline: the market itself.
    if args.uncertain_only:
        market_pairs = [(p, m.outcome) for m, p in selected]
        print(f"{'market':9s} mean Brier: {mean_brier(market_pairs):.4f}  "
              f"(market-implied prob at as-of — the bar to beat)")
    print(f"{'always.5':9s} mean Brier: 0.2500  (max-entropy reference)")

    # Calibration matters as much as mean Brier: is a "0.7" really ~70%? Show the
    # reliability curve per arm (needs enough forecasts per bin to be meaningful).
    print()
    for arm in args.arms:
        print(f"[{arm}] {calibration_report(scores[arm]).table()}")


if __name__ == "__main__":
    main()
