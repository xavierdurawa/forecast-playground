"""Wikipedia source: article text as it existed at the Clock's as-of instant.

Two modes:
  - ``mode="title"`` (default): treat the query as an exact article title and fetch
    the revision active at-or-before the as-of instant.
  - ``mode="search"``: treat the query as a full-text search; discover candidate
    titles, then fetch each one's as-of revision and return the most relevant
    chunks.

Both are HARD: a revision's timestamp is the exact moment that text became public.

Leak note on search: ``list=search`` queries the *current* index, so it can name
articles that didn't exist yet and its snippets reflect today's content. We
therefore use search ONLY to discover candidate titles — never its snippets — and
each title's content comes from the leak-safe as-of revision. Titles with no
revision at-or-before the as-of instant are dropped (they didn't exist yet).

Docs: https://www.mediawiki.org/wiki/API:Revisions , API:Search
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import requests

from ..clock import Clock
from ..http import make_session, user_agent
from ..retrieval import top_chunks
from .base import AsOfGuarantee, Document, Source

_API = "https://{lang}.wikipedia.org/w/api.php"


class WikipediaSource:
    """Fetch Wikipedia article text as of a point in time.

    Args:
        lang: Wikipedia language edition (default ``"en"``).
        mode: ``"title"`` (exact-title lookup) or ``"search"`` (full-text discovery
            of candidate titles, then as-of fetch + relevance ranking).
        max_titles: In search mode, how many candidate titles to fetch.
        top_k: In search mode, how many relevant chunks to return across articles.
        chunk_chars: Passage size for chunking.
        session: Optional pre-configured ``requests.Session`` (for caching/retries).
        timeout: Per-request timeout in seconds.
    """

    guarantee = AsOfGuarantee.HARD

    def __init__(
        self,
        lang: str = "en",
        mode: str = "title",
        max_titles: int = 3,
        top_k: int = 4,
        chunk_chars: int = 1500,
        session: requests.Session | None = None,
        timeout: float = 20.0,
    ) -> None:
        if mode not in ("title", "search"):
            raise ValueError("mode must be 'title' or 'search'")
        self.lang = lang
        self.mode = mode
        self.max_titles = max_titles
        self.top_k = top_k
        self.chunk_chars = chunk_chars
        self.name = f"wikipedia:{lang}"
        self.timeout = timeout
        # A fresh Session ships with a "python-requests/x" UA that Wikipedia 403s
        # on content requests, so we always set a contactful one (see http.py).
        self._session = session or make_session(user_agent=user_agent())

    def search(self, query: str) -> list[str]:
        """Return candidate article titles for ``query`` (full-text search).

        Titles only — snippets/timestamps from the current index are not leak-safe
        and are discarded. The as-of filtering happens when each title is fetched.
        """
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "list": "search",
            "srsearch": query,
            "srlimit": str(self.max_titles),
            "srprop": "",  # titles only
        }
        resp = self._session.get(
            _API.format(lang=self.lang), params=params, timeout=self.timeout
        )
        resp.raise_for_status()
        return [s["title"] for s in resp.json().get("query", {}).get("search", [])]

    def _fetch_title(self, title: str, clock: Clock) -> Document | None:
        """Fetch the revision of ``title`` active at ``clock.as_of`` (leak-safe)."""
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "revisions",
            "titles": title,
            "rvlimit": "1",
            "rvdir": "older",
            "rvstart": clock.as_of.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "rvprop": "ids|timestamp|content|comment",
            "rvslots": "main",
        }
        resp = self._session.get(
            _API.format(lang=self.lang), params=params, timeout=self.timeout
        )
        resp.raise_for_status()
        for page in resp.json().get("query", {}).get("pages", []):
            if page.get("missing") or "revisions" not in page:
                continue  # article didn't exist at/before as_of
            rev = page["revisions"][0]
            ts = datetime.strptime(rev["timestamp"], "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
            # The chokepoint: never surface a revision newer than the Clock.
            ts = clock.guard(ts, source=self.name, detail=f"title={title!r}")
            content = rev.get("slots", {}).get("main", {}).get("content", "")
            return Document(
                content=content,
                timestamp=ts,
                source=self.name,
                url=f"https://{self.lang}.wikipedia.org/?oldid={rev['revid']}",
                meta={
                    "title": page.get("title", title),
                    "revid": rev["revid"],
                    "comment": rev.get("comment", ""),
                },
            )
        return None

    def fetch(self, query: str, clock: Clock, **kwargs: Any) -> list[Document]:
        """Return relevant article text for ``query`` as of ``clock.as_of``.

        In ``title`` mode, ``query`` is an exact title and the whole as-of article
        is returned (ranked to the top chunks to stay context-sized). In ``search``
        mode, candidate titles are discovered, each fetched as-of, and the most
        relevant chunks across them returned. Empty if nothing existed yet.
        """
        titles = (
            self.search(query)[: self.max_titles]
            if self.mode == "search"
            else [query]
        )
        docs: list[Document] = []
        for title in titles:
            doc = self._fetch_title(title, clock)
            if doc is None:
                continue
            # Rank to the most query-relevant passages instead of blind truncation,
            # so a big article doesn't overflow context or bury the answer.
            chunks = top_chunks(
                doc.content, query, k=self.top_k, chunk_chars=self.chunk_chars
            )
            doc.content = "\n\n[…]\n\n".join(chunks)
            doc.meta["ranked_chunks"] = len(chunks)
            docs.append(doc)
        return docs


# Static type-check that the class satisfies the Source protocol.
_: type[Source] = WikipediaSource
