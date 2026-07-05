"""ForecastBench question source: resolved forecasting questions with freeze dates.

ForecastBench (github.com/forecastingresearch/forecastbench-datasets, CC BY-SA 4.0,
keyless) publishes biweekly ``question_sets`` + matching ``resolution_sets`` drawn
from Metaculus, INFER/RFI, ACLED conflict data, and markets. Each question carries a
``freeze_datetime`` (the as-of instant) and a resolution dated strictly later, so it
is a HARD, leak-free supply of ``(text, as_of, outcome)`` — the analog of
``fetch_resolved_markets`` for non-market, geopolitics/analyst-style questions.

This is a *question* source (it supplies questions + ground-truth outcomes to
forecast); retrieval masking still flows through the HARD retrieval sources with the
Clock set to each question's ``as_of``.

License: CC BY-SA 4.0 — attribute ForecastBench and share alike.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import requests

from ..http import make_session, user_agent

_BASE = "https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets/main"
_QUESTIONS = _BASE + "/datasets/question_sets/{date}-llm.json"
_RESOLUTIONS = _BASE + "/datasets/resolution_sets/{date}_resolution_set.json"

# The geopolitics/security + analyst-style sources (vs. markets); the interesting,
# looser questions are the metaculus/infer ones (acled is formulaic conflict templates).
GEO_SOURCES = ("metaculus", "infer", "acled")


@dataclass
class ResolvedQuestion:
    """A resolved ForecastBench question, for use as study ground truth.

    Attributes:
        id, source: ForecastBench identity (join key).
        question: The question text.
        as_of: Freeze datetime — the instant a forecaster is masked to.
        resolution_date: When it resolved (strictly after ``as_of``).
        outcome: 1 if the YES/claim came true, 0 if not.
        market_prob: The frozen market/community probability at ``as_of`` (a baseline).
        resolution_criteria, background: Optional context for the forecaster.
    """

    id: str
    source: str
    question: str
    as_of: datetime
    resolution_date: datetime
    outcome: int
    market_prob: float | None = None
    resolution_criteria: str = ""
    background: str = ""


def _parse_dt(value: str) -> datetime:
    """Parse an ISO datetime or bare date into UTC-aware datetime."""
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _to_float(value: object) -> float | None:
    """Best-effort float; None on missing/unparseable (some rows carry junk strings)."""
    if value in (None, ""):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def fetch_forecastbench_questions(
    date: str = "2025-03-02",
    sources: tuple[str, ...] | None = None,
    model: str | None = None,
    margin_days: int = 90,
    session: requests.Session | None = None,
    timeout: float = 30.0,
) -> list[ResolvedQuestion]:
    """Load resolved standalone binary questions from one ForecastBench dated set.

    Args:
        date: The dated set to load (``YYYY-MM-DD``), matching a published set.
        sources: Restrict to these sources (e.g. ``GEO_SOURCES``). None = all.
        model: If given, drop questions whose ``resolution_date`` is NOT safely after
            this model's training cutoff — i.e. keep only *parametrically* leak-safe
            questions (see ``leakage.is_leak_safe``). Off by default.
        margin_days: Safety margin past the training cutoff for the ``model`` gate.

    Joins questions to resolutions by ``(source, id)`` and takes the earliest
    resolved horizon. Skips combination questions (``combination_of != "N/A"`` or a
    non-None resolution ``direction``) and non-binary outcomes. De-dup within a set
    is inherent (one record per id); de-dup *across* dates is the caller's job.
    """
    session = session or make_session(user_agent=user_agent())

    q_raw = session.get(_QUESTIONS.format(date=date), timeout=timeout)
    q_raw.raise_for_status()
    questions = q_raw.json()
    questions = questions.get("questions", questions)

    r_raw = session.get(_RESOLUTIONS.format(date=date), timeout=timeout)
    r_raw.raise_for_status()
    resolutions = r_raw.json()
    resolutions = resolutions.get("resolutions", resolutions)

    # Earliest standalone resolution per (source, id).
    earliest: dict[tuple[str, str], dict] = {}
    for rec in resolutions:
        sid, src = rec.get("id"), rec.get("source")
        if not isinstance(sid, str) or not isinstance(src, str):
            continue  # combo resolutions have list ids
        if rec.get("direction") is not None:
            continue  # combination question component
        if rec.get("resolved_to") not in (0, 1, 0.0, 1.0):
            continue  # non-binary / unresolved
        key = (src, sid)
        if key not in earliest or rec["resolution_date"] < earliest[key]["resolution_date"]:
            earliest[key] = rec

    out: list[ResolvedQuestion] = []
    for q in questions:
        sid, src = q.get("id"), q.get("source")
        if not isinstance(sid, str) or not isinstance(src, str):
            continue
        if q.get("combination_of", "N/A") != "N/A":
            continue
        if sources and src not in sources:
            continue
        res = earliest.get((src, sid))
        if res is None:
            continue
        out.append(
            ResolvedQuestion(
                id=sid,
                source=src,
                question=q["question"],
                as_of=_parse_dt(q["freeze_datetime"]),
                resolution_date=_parse_dt(res["resolution_date"]),
                outcome=int(res["resolved_to"]),
                market_prob=_to_float(q.get("freeze_datetime_value")),
                resolution_criteria=q.get("resolution_criteria", "") or "",
                background=q.get("background", "") or "",
            )
        )

    if model is not None:
        # Keep only questions the model can't already know from pretraining.
        from ..leakage import is_leak_safe

        out = [
            q for q in out
            if is_leak_safe(q.resolution_date, model, margin_days=margin_days)
        ]
    return out
