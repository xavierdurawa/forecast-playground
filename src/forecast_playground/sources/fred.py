"""FRED / ALFRED economic-data source: macro series as *published* on a past date.

FRED (Federal Reserve Economic Data) serves economic time series. Its ALFRED layer
adds *vintages*: with ``realtime_start = realtime_end = T`` the API returns the
series exactly as it was published on date T — so a revised GDP/CPI figure released
after T is invisible. That vintage mechanism is what makes this a HARD source: the
numbers a forecaster sees are the numbers that were actually knowable at T, revisions
and all excluded.

Needs a free API key (https://fred.stlouisfed.org/docs/api/api_key.html) in the
``FRED_API_KEY`` env var. Query with a series id (e.g. "GDP", "CPIAUCSL", "UNRATE").

Docs: https://fred.stlouisfed.org/docs/api/fred/series_observations.html
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import requests

from ..clock import Clock
from ..http import make_session, user_agent
from .base import AsOfGuarantee, Document, Source

_API = "https://api.stlouisfed.org/fred/series/observations"


class FREDSource:
    """Fetch a FRED economic series as it was published at-or-before the as-of date.

    Args:
        api_key: FRED API key. Defaults to the ``FRED_API_KEY`` env var.
        max_points: Cap on the most-recent observations returned.
        session: Optional pre-configured ``requests.Session``.
        timeout: Per-request timeout in seconds.
    """

    guarantee = AsOfGuarantee.HARD

    def __init__(
        self,
        api_key: str | None = None,
        max_points: int = 24,
        session: requests.Session | None = None,
        timeout: float = 20.0,
    ) -> None:
        self.name = "fred"
        self.api_key = api_key or os.environ.get("FRED_API_KEY", "")
        self.max_points = max_points
        self.timeout = timeout
        self._session = session or make_session(user_agent=user_agent())

    def fetch(self, query: str, clock: Clock, **kwargs: Any) -> list[Document]:
        """Return the ``query`` series (a FRED series id) as published at the Clock.

        One Document holding the recent observations as of the as-of date. Empty if
        no key is configured or the series has no data at/before the as-of date.
        """
        if not self.api_key:
            return []  # no key -> silently no data (leak-safe: returns nothing)
        as_of_date = clock.as_of.strftime("%Y-%m-%d")
        params = {
            "series_id": query,
            "api_key": self.api_key,
            "file_type": "json",
            # Vintage: the series as it stood on the as-of date (no later revisions).
            "realtime_start": as_of_date,
            "realtime_end": as_of_date,
            # Only observation periods up to the as-of date.
            "observation_end": as_of_date,
            "sort_order": "desc",  # newest first
            "limit": self.max_points,
        }
        resp = self._session.get(_API, params=params, timeout=self.timeout)
        resp.raise_for_status()
        obs = resp.json().get("observations", [])
        rows: list[tuple[datetime, str]] = []
        for o in obs:
            if o.get("value") in (None, ".", ""):
                continue  # FRED marks missing values with "."
            period = datetime.strptime(o["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            # The observation's period is its timestamp; guard it against the Clock.
            period = clock.guard(period, source=self.name, detail=f"series={query}")
            rows.append((period, o["value"]))
        if not rows:
            return []
        rows.sort(key=lambda r: r[0])  # chronological for readability
        latest_ts = rows[-1][0]
        table = "\n".join(f"{d.date().isoformat()}: {v}" for d, v in rows)
        return [
            Document(
                content=f"FRED series {query} (as published {as_of_date}, "
                f"{len(rows)} observations):\n{table}",
                timestamp=latest_ts,
                source=self.name,
                url=f"https://fred.stlouisfed.org/series/{query}",
                meta={"series_id": query, "vintage": as_of_date, "points": len(rows)},
            )
        ]


_: type[Source] = FREDSource
