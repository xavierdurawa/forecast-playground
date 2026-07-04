"""Wikipedia Current Events portal: a curated, leak-safe news summary as of a date.

Wikipedia maintains a daily human-curated digest of world news at
``Portal:Current events/<YYYY Month D>`` — categorized items (conflicts, politics,
business, disasters, science, sports) each with external source citations. Each
daily subpage is fetched at its revision active at-or-before the Clock, so this is
the *most* leak-safe news tier: revision history is immutable and the as-of fetch
cannot surface a later edit.

It's a summary layer, not full article text — ideal grounding for "what was known
by date T", complementary to a broad source like GDELT.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from ..clock import Clock
from ..http import make_session, user_agent
from .base import AsOfGuarantee, Document, Source

_API = "https://en.wikipedia.org/w/api.php"
# Daily subpage title format, e.g. "Portal:Current events/2024 May 20".
_TITLE = "Portal:Current events/{year} {month} {day}"


class CurrentEventsSource:
    """Fetch the Wikipedia Current Events digest for the days up to the as-of date.

    Args:
        lookback_days: How many days of digests to fetch, ending at the as-of date.
        session: Optional pre-configured ``requests.Session``.
        timeout: Per-request timeout in seconds.
    """

    guarantee = AsOfGuarantee.HARD

    def __init__(
        self,
        lookback_days: int = 7,
        session: requests.Session | None = None,
        timeout: float = 20.0,
    ) -> None:
        self.lookback_days = lookback_days
        self.name = "current_events"
        self.timeout = timeout
        self._session = session or make_session(user_agent=user_agent())

    def _day_title(self, day: datetime) -> str:
        return _TITLE.format(
            year=day.year, month=day.strftime("%B"), day=day.day
        )

    def _fetch_day(self, day: datetime, clock: Clock) -> Document | None:
        """Fetch the digest for ``day`` at its revision active at-or-before the Clock."""
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "revisions",
            "titles": self._day_title(day),
            "rvlimit": "1",
            "rvdir": "older",
            "rvstart": clock.as_of.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "rvprop": "ids|timestamp|content",
            "rvslots": "main",
        }
        resp = self._session.get(_API, params=params, timeout=self.timeout)
        resp.raise_for_status()
        for page in resp.json().get("query", {}).get("pages", []):
            if page.get("missing") or "revisions" not in page:
                continue  # that day's digest didn't exist at/before as_of
            rev = page["revisions"][0]
            ts = datetime.strptime(rev["timestamp"], "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
            ts = clock.guard(ts, source=self.name, detail=self._day_title(day))
            content = rev.get("slots", {}).get("main", {}).get("content", "")
            return Document(
                content=content,
                timestamp=ts,
                source=self.name,
                url=f"https://en.wikipedia.org/wiki/{self._day_title(day).replace(' ', '_')}",
                meta={"date": day.date().isoformat(), "revid": rev["revid"]},
            )
        return None

    def fetch(self, query: str, clock: Clock, **kwargs: Any) -> list[Document]:
        """Return daily news digests for the window ending at ``clock.as_of``.

        ``query`` is currently unused (the digest is date-driven, not searched) —
        callers filter the returned text themselves. Days with no digest yet are
        skipped; the newest days come first.
        """
        docs: list[Document] = []
        for offset in range(self.lookback_days):
            day = clock.as_of - timedelta(days=offset)
            doc = self._fetch_day(day, clock)
            if doc is not None:
                docs.append(doc)
        return docs


_: type[Source] = CurrentEventsSource
