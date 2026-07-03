"""Shared HTTP session factory with polite retry/backoff.

Public archives (Wikipedia, Wayback, ...) rate-limit aggressively, and an RL
training loop hits them at volume. A harness that doesn't retry on 429/503 will
drop documents mid-rollout and silently degrade the time-mask's coverage, so
backoff is a correctness concern, not just politeness.
"""

from __future__ import annotations

import os

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_PROJECT_URL = "https://github.com/xavierdurawa/forecast-playground"


def user_agent() -> str:
    """A polite User-Agent string for public archive APIs.

    Wikimedia's policy asks for a contact; set the ``FORECAST_CONTACT`` env var (an
    email or URL) for real use. Without it, a neutral non-personal fallback is used
    so nothing personal is baked into the package.
    """
    contact = os.environ.get("FORECAST_CONTACT", "set FORECAST_CONTACT env var")
    return f"forecast-playground (forecasting retrieval; {_PROJECT_URL}; {contact})"


def make_session(
    *,
    total_retries: int = 5,
    backoff_factor: float = 1.0,
    user_agent: str | None = None,
) -> requests.Session:
    """Build a Session that retries on 429/5xx with exponential backoff.

    ``Retry`` honors a server-sent ``Retry-After`` header and backs off
    ``backoff_factor * (2 ** (attempt - 1))`` seconds otherwise.
    """
    retry = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    if user_agent:
        session.headers["User-Agent"] = user_agent
    return session
