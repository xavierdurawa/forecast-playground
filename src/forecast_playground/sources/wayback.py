"""Wayback Machine source: any web page as it was archived at-or-before the Clock.

HARD guarantee: the Internet Archive's snapshot timestamp is when that capture was
made public. We ask for the closest snapshot at-or-before ``as_of`` and guard it.

Docs: https://archive.org/help/wayback_api.php
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

import requests

from ..clock import Clock
from ..http import make_session, user_agent
from .base import AsOfGuarantee, Document, Source

_AVAIL = "https://archive.org/wayback/available"
_CDX = "https://web.archive.org/cdx/search/cdx"

# Strip tags crudely; the harness consumer can parse further if needed.
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\n\s*\n+")


class WaybackSource:
    """Fetch a URL as archived at-or-before the as-of date.

    The ``query`` is the URL to retrieve. Returns the closest snapshot whose
    capture time is <= as_of, with HTML reduced to rough text.

    Args:
        max_chars: Truncate extracted text to this many characters.
    """

    guarantee = AsOfGuarantee.HARD

    def __init__(
        self,
        session: requests.Session | None = None,
        timeout: float = 30.0,
        max_chars: int = 8000,
    ) -> None:
        self.name = "wayback"
        self.timeout = timeout
        self.max_chars = max_chars
        self._session = session or make_session(user_agent=user_agent())

    def fetch(self, query: str, clock: Clock, **kwargs: Any) -> list[Document]:
        """Return the latest Wayback snapshot of URL ``query`` at-or-before the Clock.

        Empty if no snapshot exists at or before the as-of instant.
        """
        snap_ts_str = self._latest_snapshot_at_or_before(query, clock)
        if snap_ts_str is None:
            return []
        snap_ts = datetime.strptime(snap_ts_str, "%Y%m%d%H%M%S").replace(
            tzinfo=timezone.utc
        )
        # Belt-and-suspenders: the chokepoint still guards (should never fire now,
        # since both lookup paths already constrain to <= as_of).
        snap_ts = clock.guard(snap_ts, source=self.name, detail=f"url={query!r}")
        snap_url = f"https://web.archive.org/web/{snap_ts_str}/{query}"
        raw_url = f"https://web.archive.org/web/{snap_ts_str}id_/{query}"
        page = self._session.get(raw_url, timeout=self.timeout)
        page.raise_for_status()
        text = _WS_RE.sub("\n\n", _TAG_RE.sub(" ", page.text)).strip()
        return [
            Document(
                content=text[: self.max_chars],
                timestamp=snap_ts,
                source=self.name,
                url=snap_url,
                meta={"original_url": query, "snapshot": snap_ts_str},
            )
        ]

    def _latest_snapshot_at_or_before(self, url: str, clock: Clock) -> str | None:
        """Timestamp (YYYYMMDDhhmmss) of the newest capture at-or-before as_of.

        Prefers the CDX API with ``to=`` (exact "at or before" semantics). Falls
        back to the Availability API, whose "closest" result can be AFTER as_of —
        in which case we return None (no valid snapshot) rather than leaking.
        """
        as_of_str = clock.as_of.strftime("%Y%m%d%H%M%S")
        try:
            resp = self._session.get(
                _CDX,
                params={
                    "url": url,
                    "to": as_of_str,
                    "output": "json",
                    "limit": "-1",  # newest matching row
                    "filter": "statuscode:200",
                    "fl": "timestamp",
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            rows = resp.json()
            # rows[0] is the header; rows[1:] are data.
            if len(rows) > 1:
                return rows[-1][0]
            return None
        except (requests.RequestException, ValueError):
            # CDX flaky/unavailable — fall back to Availability with a hard guard.
            resp = self._session.get(
                _AVAIL,
                params={"url": url, "timestamp": as_of_str},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            snap = resp.json().get("archived_snapshots", {}).get("closest")
            if not snap or not snap.get("available"):
                return None
            ts = snap["timestamp"]
            # "closest" may be after as_of; only accept it if it's <= as_of.
            return ts if ts <= as_of_str else None


_: type[Source] = WaybackSource
