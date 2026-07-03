"""Wikipedia Pageviews source: daily view counts up to the Clock's as-of instant.

A leak-safe "public attention" signal — pageviews for a day strictly precede any
outcome on a later day. Replaces fragile Google Trends. HARD guarantee: the API is
queried only up to ``as_of`` and each daily bucket is guarded.

Docs: https://wikimedia.org/api/rest_v1/#/Pageviews%20data
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from ..clock import Clock
from ..http import make_session, user_agent
from .base import AsOfGuarantee, Document, Source

_API = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "{project}/all-access/all-agents/{article}/daily/{start}/{end}"
)


class PageviewsSource:
    """Fetch daily Wikipedia pageview counts for an article up to the as-of date.

    Args:
        project: Wikimedia project domain (default ``"en.wikipedia"``).
        lookback_days: How many days of history to fetch before the as-of date.
    """

    guarantee = AsOfGuarantee.HARD

    def __init__(
        self,
        project: str = "en.wikipedia",
        lookback_days: int = 60,
        session: requests.Session | None = None,
        timeout: float = 20.0,
    ) -> None:
        self.project = project
        self.lookback_days = lookback_days
        self.name = f"pageviews:{project}"
        self.timeout = timeout
        self._session = session or make_session(user_agent=user_agent())

    def fetch(self, query: str, clock: Clock, **kwargs: Any) -> list[Document]:
        """Return a daily pageview series for article ``query`` ending at the Clock.

        The returned Document's ``content`` is a compact "date: views" table; its
        ``timestamp`` is the most recent (last admissible) day.
        """
        # Pageviews data starts ~2015-07-01; clamp the window to <= as_of.
        end = clock.as_of
        start = end - timedelta(days=self.lookback_days)
        url = _API.format(
            project=self.project,
            article=query.replace(" ", "_"),
            start=start.strftime("%Y%m%d"),
            end=end.strftime("%Y%m%d"),
        )
        resp = self._session.get(url, timeout=self.timeout)
        if resp.status_code == 404:
            return []  # no data for this article/range
        resp.raise_for_status()
        items = resp.json().get("items", [])
        rows: list[tuple[datetime, int]] = []
        for it in items:
            # Pageviews timestamps look like "2024010100" (YYYYMMDDHH).
            day = datetime.strptime(it["timestamp"], "%Y%m%d%H").replace(
                tzinfo=timezone.utc
            )
            if not clock.admits(day):
                continue  # defensive: never include a day after as_of
            rows.append((day, int(it.get("views", 0))))
        if not rows:
            return []
        latest = max(r[0] for r in rows)
        clock.guard(latest, source=self.name, detail=f"article={query!r}")
        table = "\n".join(f"{d.date().isoformat()}: {v}" for d, v in rows)
        total = sum(v for _, v in rows)
        return [
            Document(
                content=f"Daily pageviews for {query!r} ({len(rows)} days, "
                f"total {total}):\n{table}",
                timestamp=latest,
                source=self.name,
                url=f"https://{self.project}.org/wiki/{query.replace(' ', '_')}",
                meta={"article": query, "days": len(rows), "total_views": total},
            )
        ]


_: type[Source] = PageviewsSource
