"""The Source protocol and its as-of guarantee levels.

A Source adapts one data backend (Wikipedia, Wayback, a prediction market, ...).
Each Source declares how trustworthy its time-mask is via an ``AsOfGuarantee`` so
callers can choose their own risk tolerance (e.g. "HARD sources only").
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from ..clock import Clock


class AsOfGuarantee(enum.Enum):
    """How leak-safe a source's point-in-time query is."""

    HARD = "hard"
    """Leak-free by construction: the timestamp is a true upper bound on
    availability (revision time, snapshot time, filing date, vintage)."""

    SOFT = "soft"
    """Date-filterable but may return values revised after the as-of instant.
    The value is point-in-time *true*, not point-in-time *known*. Use with care."""

    NONE = "none"
    """Latest-only; no historical query. Rejected unless explicitly allowed."""


@dataclass
class Document:
    """A single retrieved item, carrying the timestamp the Clock guards against.

    Attributes:
        content: The retrieved text/data payload.
        timestamp: When this content became available (its as-of upper bound).
        source: Name of the source that produced it.
        url: Canonical/origin URL if any.
        meta: Source-specific extra fields (revision id, snapshot id, ...).
    """

    content: str
    timestamp: datetime
    source: str
    url: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Source(Protocol):
    """A time-masked data backend.

    Implementations must guarantee that :meth:`fetch` never returns a Document
    whose ``timestamp`` is after ``clock.as_of`` — and should route every
    candidate timestamp through ``clock.guard(...)`` so the chokepoint enforces it.
    """

    name: str
    guarantee: AsOfGuarantee

    def fetch(self, query: str, clock: Clock, **kwargs: Any) -> list[Document]:
        """Return documents matching ``query`` visible at ``clock``."""
        ...
