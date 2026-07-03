"""The Clock: the single chokepoint that forbids lookahead.

Every retrieval in the harness flows through a Clock. A tool may not return data
stamped after the Clock's ``as_of`` instant. Sources hand the Clock the timestamp
of the data they found and the Clock decides admissibility; sources never compare
dates themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


class LookaheadError(Exception):
    """Raised when a source tries to surface data created after the as-of instant."""


def _as_utc(dt: datetime) -> datetime:
    """Normalize a datetime to timezone-aware UTC.

    Naive datetimes are *assumed* to already be UTC. Mixing timezones across
    sources is the most common way a time-mask silently leaks, so everything is
    coerced to UTC at the boundary.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass(frozen=True)
class Clock:
    """A frozen 'current time' for a retrieval session.

    Args:
        as_of: The instant the model is pretending to be at. All retrieval is
            masked to information available at or before this instant.
    """

    as_of: datetime

    def __post_init__(self) -> None:
        # Stored normalized to UTC so every downstream comparison is apples-to-apples.
        object.__setattr__(self, "as_of", _as_utc(self.as_of))

    @classmethod
    def at(cls, when: str | datetime) -> "Clock":
        """Build a Clock from an ISO-8601 string or a datetime.

        A bare date (``"2024-01-01"``) is interpreted as the start of that day in
        UTC, so a query "as of 2024-01-01" cannot see anything from Jan 1 onward.
        """
        if isinstance(when, datetime):
            return cls(as_of=when)
        return cls(as_of=datetime.fromisoformat(when))

    def admits(self, ts: datetime) -> bool:
        """Return True iff data stamped ``ts`` is visible at this Clock."""
        return _as_utc(ts) <= self.as_of

    def guard(self, ts: datetime, *, source: str, detail: str = "") -> datetime:
        """Assert that ``ts`` is admissible, else raise LookaheadError.

        Returns the (UTC-normalized) timestamp so callers can use it directly.
        """
        ts = _as_utc(ts)
        if ts > self.as_of:
            suffix = f" ({detail})" if detail else ""
            raise LookaheadError(
                f"{source}: data stamped {ts.isoformat()} is after as_of "
                f"{self.as_of.isoformat()}{suffix}"
            )
        return ts
