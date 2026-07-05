"""Parametric (pretraining) leakage guard: a training-cutoff filter.

The :class:`~forecast_playground.clock.Clock` masks *retrieval* — a tool can't return
anything after ``as_of``. It does nothing about the model's *weights*: if a question
resolved before the model's training cutoff, the model may simply *know* the answer,
no retrieval required. That's a leak too, and it's invisible to the Clock. (Observed
live: a model scored ~0.93 on a 2024 event well inside its training window — it
remembered, it didn't forecast.)

This module is the parametric half of the same invariant: "the model cannot have seen
this." It's pure and deterministic — a cutoff lookup + date arithmetic, no model
calls. Cutoffs come from models.dev (a maintained, multi-provider source) with a
consumer override that always wins; an unknown model is *not certifiable* and is
rejected (fail-safe), never assumed clean.

Cutoff data: models.dev (https://models.dev, MIT). Note models.dev exposes a single
``knowledge`` field that leans toward the *training* cutoff (the conservative choice
for this filter) but doesn't formally separate training vs. reliable-knowledge — so
pass an explicit override when you need a specific semantics for a model.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import requests

from .http import make_session, user_agent

_MODELS_DEV = "https://models.dev/api.json"
# Default safety margin past the stated cutoff: models often know events slightly
# past their nominal cutoff (post-training/RLHF). ~90d is where release_date minus
# training-cutoff consistently sits for models where both are known.
_DEFAULT_MARGIN_DAYS = 90

# Consumer overrides: {model_substring: "YYYY-MM-DD" | datetime}. Checked first.
_OVERRIDES: dict[str, object] = {}
# Cached flat {bare_model_id: "YYYY-MM[-DD]"} pulled from models.dev, lazily.
_CUTOFF_CACHE: dict[str, str] | None = None


def register_cutoff(model: str, cutoff: str | datetime) -> None:
    """Register/override a training cutoff for models matching ``model`` (substring).

    An override always wins over the models.dev lookup — use it for models not in
    models.dev, or when you want a specific (e.g. training-vs-knowledge) semantics.
    """
    _OVERRIDES[model] = cutoff


def _to_dt(value: str | datetime) -> datetime:
    """Parse a YYYY-MM or YYYY-MM-DD (or datetime) into UTC-aware datetime."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    parts = str(value).split("-")
    y, m = int(parts[0]), int(parts[1]) if len(parts) > 1 else 1
    d = int(parts[2]) if len(parts) > 2 else 1
    return datetime(y, m, d, tzinfo=timezone.utc)


def _load_cutoffs(session: requests.Session | None = None) -> dict[str, str]:
    """Return (and cache) a flat {model_id: knowledge_date} map from models.dev.

    Network failure yields an empty map — every lookup then misses and the guard
    rejects (fail-safe), rather than raising.
    """
    global _CUTOFF_CACHE
    if _CUTOFF_CACHE is not None:
        return _CUTOFF_CACHE
    flat: dict[str, str] = {}
    try:
        sess = session or make_session(user_agent=user_agent())
        data = sess.get(_MODELS_DEV, timeout=30).json()
        for provider in data.values():
            for mid, meta in provider.get("models", {}).items():
                if meta.get("knowledge"):
                    flat[mid] = meta["knowledge"]
    except (requests.RequestException, ValueError, AttributeError):
        flat = {}  # offline / bad payload -> empty -> everything rejects
    _CUTOFF_CACHE = flat
    return flat


def _resolve(model: str, table: dict[str, str]) -> str | object | None:
    """Find the cutoff for ``model``: overrides first, then longest matching id.

    Model ids vary by region/prefix (``global.anthropic.claude-sonnet-4-6`` vs the
    bare ``claude-sonnet-4-6``), so we match a table id that is a substring of the
    query, preferring the longest (most specific) match.
    """
    for sub, cut in _OVERRIDES.items():
        if sub in model:
            return cut
    if model in table:
        return table[model]
    matches = [mid for mid in table if mid in model]
    return table[max(matches, key=len)] if matches else None


def training_cutoff(
    model: str, session: requests.Session | None = None
) -> datetime | None:
    """The training cutoff for ``model`` (override or models.dev), or None if unknown."""
    raw = _resolve(model, _load_cutoffs(session))
    return _to_dt(raw) if raw is not None else None


def min_safe_resolution(
    model: str, margin_days: int = _DEFAULT_MARGIN_DAYS, session: requests.Session | None = None
) -> datetime | None:
    """Earliest resolution date considered leak-safe for ``model``, or None if unknown.

    That's the training cutoff plus ``margin_days``: a question resolving on or after
    this is safely outside the model's parametric knowledge.
    """
    cut = training_cutoff(model, session=session)
    return cut + timedelta(days=margin_days) if cut is not None else None


def is_leak_safe(
    resolution_date: str | datetime,
    model: str,
    margin_days: int = _DEFAULT_MARGIN_DAYS,
    session: requests.Session | None = None,
) -> bool:
    """True iff ``resolution_date`` is safely after ``model``'s training cutoff.

    Fail-safe: an unknown model (no override, not in models.dev) returns False — we
    cannot certify it leak-safe, so we don't. ``resolution_date`` may be a datetime or
    a YYYY-MM(-DD) string.
    """
    floor = min_safe_resolution(model, margin_days=margin_days, session=session)
    if floor is None:
        return False  # not certifiable -> reject
    return _to_dt(resolution_date) >= floor
