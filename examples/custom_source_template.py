"""Template: write your own time-masked Source.

Copy this file, fill in the two marked spots, and your source drops straight into a
Toolkit as a model-callable, cached, traced tool. The leak-safety contract is short:

    Return only Documents whose `timestamp` is when that content became public,
    and query your backend for data at-or-before `clock.as_of`.

You do NOT have to enforce the cutoff yourself — the Toolkit re-guards every
Document's timestamp, so a mistake fails loudly instead of leaking. Your job is just
to (a) fetch historically and (b) stamp each Document with an honest timestamp.

Run this file to see it work against an in-memory example backend:
    python examples/custom_source_template.py
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from forecast_playground import AsOfGuarantee, Clock, Document, Toolkit
from forecast_playground.http import make_session, user_agent  # if you need HTTP


class MySource:
    """One-line summary shown to the model as the tool description's basis.

    Args:
        ...: your source's options (an API base url, a lookback window, etc.).
    """

    # (1) DECLARE YOUR AS-OF GUARANTEE ---------------------------------------
    # HARD: the timestamp is a true upper bound on availability (revision/snapshot/
    #       filing/trade time). Prefer this — it's what makes results trustworthy.
    # SOFT: date-filterable, but values may be revised after the date (e.g. an
    #       economic series that gets restated). Usable, but say so.
    # NONE: latest-only, no history. Don't time-mask with this.
    guarantee = AsOfGuarantee.HARD

    def __init__(self, timeout: float = 20.0) -> None:
        # `name` becomes the tool name "<name>_search" (split on ':').
        self.name = "mysource"
        self.timeout = timeout
        # Use the shared session for polite retry/backoff + a proper User-Agent:
        self._session = make_session(user_agent=user_agent())

    def fetch(self, query: str, clock: Clock, **kwargs: Any) -> list[Document]:
        """Return documents matching `query` that existed at/before `clock.as_of`."""
        docs: list[Document] = []

        # (2) FETCH HISTORICALLY -------------------------------------------------
        # Query your backend using its native "as of / at-or-before" capability
        # (a date param, a revision id, a snapshot timestamp, WHERE ts <= T, ...).
        # NEVER fetch "latest" and hope — that's how leaks happen.
        for content, became_public_at in self._query_backend(query, clock.as_of):
            # Recommended: fail fast with a clear label. Not required (the Toolkit
            # re-guards), but it pinpoints a buggy source immediately.
            ts = clock.guard(became_public_at, source=self.name)
            docs.append(
                Document(
                    content=content,
                    timestamp=ts,        # <-- the honest "when did this become public"
                    source=self.name,
                    url=None,            # canonical/origin URL if you have one
                    meta={},             # any extra fields (ids, domain, ...)
                )
            )
        return docs

    # --- your backend; replace with real HTTP/DB calls ---------------------
    def _query_backend(self, query: str, as_of: datetime):
        """Toy in-memory backend. Real sources hit an API/DB filtered by date."""
        rows = [
            ("Old fact about " + query, datetime(2023, 1, 1, tzinfo=timezone.utc)),
            ("Future fact", datetime(2099, 1, 1, tzinfo=timezone.utc)),  # must be masked
        ]
        return [(c, ts) for c, ts in rows if ts <= as_of]


if __name__ == "__main__":
    tk = Toolkit(clock=Clock.at("2024-06-01"), sources=[MySource()], enable_python=False)
    print("tool exposed as:", [d["name"] for d in tk.tool_defs()])
    print("result:", tk.call("mysource_search", {"query": "rockets"}))
    # The 2099 row is filtered by _query_backend; even if it weren't, the Toolkit
    # would block it. Try returning it unconditionally to see "lookahead blocked".
