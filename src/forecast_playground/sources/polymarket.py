"""Polymarket source: market-implied probability for a question up to the Clock.

Given a CLOB token id, returns the YES-probability time series from the market's
price history, masked to <= as_of. HARD guarantee: each price point carries its own
trade timestamp. Keyless.

Also provides ``ResolvedMarket`` + ``fetch_resolved_markets`` to source a dataset of
already-resolved binary questions with ground-truth outcomes for studies.

Docs: https://docs.polymarket.com (Gamma + CLOB)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import requests

from ..clock import Clock
from ..http import make_session, user_agent
from .base import AsOfGuarantee, Document, Source

_GAMMA = "https://gamma-api.polymarket.com/markets"
_CLOB_HISTORY = "https://clob.polymarket.com/prices-history"


class PolymarketSource:
    """Fetch a market's YES-probability history up to the as-of date.

    The ``query`` is a CLOB token id (the YES outcome token). Returns one Document
    whose content summarizes the probability series ending at the Clock.
    """

    guarantee = AsOfGuarantee.HARD

    def __init__(
        self,
        session: requests.Session | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.name = "polymarket"
        self.timeout = timeout
        self._session = session or make_session(user_agent=user_agent())

    def fetch(self, query: str, clock: Clock, **kwargs: Any) -> list[Document]:
        """Return the YES-probability series for CLOB token ``query`` up to the Clock."""
        resp = self._session.get(
            _CLOB_HISTORY,
            params={"market": query, "interval": "max", "fidelity": 1440},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        hist = resp.json().get("history", [])
        rows: list[tuple[datetime, float]] = []
        for pt in hist:
            ts = datetime.fromtimestamp(int(pt["t"]), tz=timezone.utc)
            if not clock.admits(ts):
                continue  # mask out post-as_of trading
            rows.append((ts, float(pt["p"])))
        if not rows:
            return []
        latest_ts, latest_p = max(rows, key=lambda r: r[0])
        clock.guard(latest_ts, source=self.name, detail=f"token={query[:12]}...")
        table = "\n".join(f"{d.date().isoformat()}: {p:.3f}" for d, p in rows)
        return [
            Document(
                content=f"Market-implied YES probability ({len(rows)} days, latest "
                f"{latest_p:.3f} on {latest_ts.date().isoformat()}):\n{table}",
                timestamp=latest_ts,
                source=self.name,
                url=None,
                meta={"latest_prob": latest_p, "points": len(rows)},
            )
        ]


_: type[Source] = PolymarketSource


@dataclass
class ResolvedMarket:
    """A resolved binary Polymarket question, for use as study ground truth."""

    question: str
    token_id_yes: str
    end_date: datetime
    outcome: int  # 1 if YES resolved true, 0 if NO
    volume: float
    slug: str | None = None


def fetch_resolved_markets(
    limit: int = 20,
    min_volume: float = 1_000_000,
    session: requests.Session | None = None,
    timeout: float = 30.0,
    max_pages: int = 20,
) -> list[ResolvedMarket]:
    """Fetch high-volume, cleanly-resolved binary (Yes/No) markets.

    Filters to markets that resolved to a clean 0/1 outcome (excludes mid-range /
    ambiguous resolutions) so they make unambiguous study labels. Paginates the
    Gamma API (which caps ~100 rows/page) until ``limit`` markets are collected or
    the pages run out.
    """
    session = session or make_session(user_agent=user_agent())
    out: list[ResolvedMarket] = []
    page_size = 100
    for page in range(max_pages):
        resp = session.get(
            _GAMMA,
            params={
                "closed": "true",
                "limit": page_size,
                "offset": page * page_size,
                "order": "volumeNum",
                "ascending": "false",
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            break  # no more pages
        _collect_markets(rows, out, min_volume, limit)
        if len(out) >= limit:
            break
    return out


def _collect_markets(rows, out, min_volume, limit) -> None:
    """Append clean binary resolved markets from ``rows`` into ``out`` (in place)."""
    for m in rows:
        outcomes = m.get("outcomes", "")
        if "Yes" not in outcomes or "No" not in outcomes:
            continue
        try:
            prices = json.loads(m.get("outcomePrices", "[]"))
            tokens = json.loads(m.get("clobTokenIds", "[]"))
        except (json.JSONDecodeError, TypeError):
            continue
        if len(prices) != 2 or len(tokens) != 2:
            continue
        yes_price = float(prices[0])
        if yes_price not in (0.0, 1.0):  # require a clean resolution
            continue
        vol = float(m.get("volumeNum") or 0)
        if vol < min_volume:
            continue
        end = m.get("endDate")
        if not end:
            continue
        out.append(
            ResolvedMarket(
                question=m["question"],
                token_id_yes=tokens[0],
                end_date=datetime.fromisoformat(end.replace("Z", "+00:00")),
                outcome=int(yes_price),
                volume=vol,
                slug=m.get("slug"),
            )
        )
        if len(out) >= limit:
            break


def market_prob_at(
    token_id_yes: str,
    clock: Clock,
    session: requests.Session | None = None,
    timeout: float = 30.0,
) -> float | None:
    """Leak-free market-implied YES probability at ``clock.as_of``.

    Returns the last traded probability at or before the as-of instant, or None if
    the market had no trades yet. Used for question SELECTION (was this question
    genuinely uncertain at the as-of date?), never as a label — computing it goes
    through the same time-mask as everything else, so it cannot see the future.
    """
    src = PolymarketSource(session=session, timeout=timeout)
    docs = src.fetch(token_id_yes, clock)
    if not docs:
        return None
    return float(docs[0].meta["latest_prob"])


def select_uncertain(
    markets: list[ResolvedMarket],
    clock_for: "Callable[[ResolvedMarket], Clock]",
    low: float = 0.25,
    high: float = 0.75,
    session: requests.Session | None = None,
) -> list[tuple[ResolvedMarket, float]]:
    """Keep only markets that were genuinely uncertain at their as-of date.

    A question whose market-implied probability at the as-of date was already near
    0 or 1 was effectively decided — scoring well on it measures nothing. We keep
    markets whose implied prob sat in ``[low, high]`` (default 0.25–0.75), which is
    a leak-free proxy for "the outcome was still live at the forecast date."

    Returns (market, implied_prob_at_as_of) pairs so callers can see the baseline
    the model must beat.
    """
    session = session or make_session(user_agent=user_agent())
    kept: list[tuple[ResolvedMarket, float]] = []
    for m in markets:
        p = market_prob_at(m.token_id_yes, clock_for(m), session=session)
        if p is not None and low <= p <= high:
            kept.append((m, p))
    return kept
