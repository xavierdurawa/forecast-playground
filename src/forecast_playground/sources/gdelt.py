"""GDELT news source: article URLs from the global news stream, as of a date.

GDELT publishes a Global Knowledge Graph (GKG) file every 15 minutes, keyless and
free, back to 2015-02-18. Each file is named by its capture time
(``YYYYMMDDHHMMSS.gkg.csv.zip``) and every row carries that capture time, so
filtering to files at-or-before the Clock is leak-safe *by construction* — there is
no way to pull post-as_of coverage.

Two backends, same leak-safe contract and Document shape:

- ``"rawfiles"`` (default, keyless): scan a bounded window of the most recent 15-min
  slots at-or-before the as-of instant. Great for "the hours before date T"; slow
  for broad, long-range search (you'd download many files).
- ``"bigquery"`` (opt-in): query GDELT's public BigQuery table
  (``gdelt-bq.gdeltv2.gkg``) with ``WHERE DATE <= T``. Same corpus, searched
  server-side — efficient across years. Needs a GCP project + credentials and the
  ``google-cloud-bigquery`` package (the ``bigquery`` extra). Queries count against
  BigQuery's free monthly tier.

Leak note: GDELT's timestamp is when it *first saw* the article, a hard upper bound
on public discoverability (it may lag true publish time slightly, never lead it).
Both backends filter on that same timestamp, so both are leak-safe by construction.

Docs: https://www.gdeltproject.org/data.html  (GKG 2.0)
"""

from __future__ import annotations

import io
import os
import zipfile
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from ..clock import Clock
from ..http import make_session, user_agent
from .base import AsOfGuarantee, Document, Source

_BASE = "http://data.gdeltproject.org/gdeltv2/{stamp}.gkg.csv.zip"
_GKG_EPOCH = datetime(2015, 2, 18, 23, 0, 0, tzinfo=timezone.utc)  # first GKG 2.0 file
# GKG is tab-delimited; these 0-based columns hold the domain and the article URL.
_COL_DOMAIN = 3
_COL_URL = 4
# BigQuery public GDELT GKG 2.0 table (same DATE/DocumentIdentifier fields).
_BQ_TABLE = "gdelt-bq.gdeltv2.gkg"


class GDELTNewsSource:
    """Fetch news article URLs from GDELT's GKG stream up to the as-of date.

    Args:
        backend: ``"rawfiles"`` (default, keyless) or ``"bigquery"`` (opt-in, needs a
            GCP project + the ``bigquery`` extra).
        slots: rawfiles only — how many 15-min GKG files to scan, working backward
            from the as-of instant (each ~6 MB / ~1500 articles). Default ~3 hours.
        max_results: Cap on returned article Documents.
        bq_project: bigquery only — GCP project id for billing/quota. Defaults to the
            ``GOOGLE_CLOUD_PROJECT`` env var.
        bq_lookback_days: bigquery only — how far back before the as-of date to search.
        session: rawfiles only — optional pre-configured ``requests.Session``.
        timeout: Per-request timeout in seconds (rawfiles).
    """

    guarantee = AsOfGuarantee.HARD

    def __init__(
        self,
        backend: str = "rawfiles",
        slots: int = 12,
        max_results: int = 40,
        bq_project: str | None = None,
        bq_lookback_days: int = 7,
        session: requests.Session | None = None,
        timeout: float = 30.0,
    ) -> None:
        if backend not in ("rawfiles", "bigquery"):
            raise ValueError("backend must be 'rawfiles' or 'bigquery'")
        self.backend = backend
        self.slots = slots
        self.max_results = max_results
        self.bq_project = bq_project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.bq_lookback_days = bq_lookback_days
        self.name = "gdelt"
        self.timeout = timeout
        self._session = session or make_session(user_agent=user_agent())

    def _slot_stamps(self, clock: Clock) -> list[datetime]:
        """The 15-min slot timestamps at-or-before the as-of instant, newest first.

        GKG files land on :00/:15/:30/:45; we floor the as-of instant to a slot and
        walk backward, never before the GKG 2.0 epoch.
        """
        floored_min = (clock.as_of.minute // 15) * 15
        slot = clock.as_of.replace(minute=floored_min, second=0, microsecond=0)
        stamps = []
        for _ in range(self.slots):
            if slot < _GKG_EPOCH:
                break
            stamps.append(slot)
            slot = slot - timedelta(minutes=15)
        return stamps

    def _fetch_slot(self, stamp: datetime, terms: list[str]) -> list[Document]:
        """Fetch one GKG file and return rows whose URL matches all query terms."""
        url = _BASE.format(stamp=stamp.strftime("%Y%m%d%H%M%S"))
        resp = self._session.get(url, timeout=self.timeout)
        if resp.status_code == 404:
            return []  # missing slot (gaps happen); skip
        resp.raise_for_status()
        try:
            zf = zipfile.ZipFile(io.BytesIO(resp.content))
            raw = zf.read(zf.namelist()[0]).decode("utf-8", "replace")
        except (zipfile.BadZipFile, IndexError):
            return []
        docs: list[Document] = []
        for line in raw.split("\n"):
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) <= _COL_URL:
                continue
            article_url = fields[_COL_URL]
            hay = article_url.lower()
            if terms and not all(t in hay for t in terms):
                continue
            docs.append(
                Document(
                    content=article_url,
                    timestamp=stamp,  # file capture time = leak-safe upper bound
                    source=self.name,
                    url=article_url,
                    meta={"domain": fields[_COL_DOMAIN]},
                )
            )
        return docs

    def fetch(self, query: str, clock: Clock, **kwargs: Any) -> list[Document]:
        """Return article URLs matching ``query`` from GKG up to the Clock.

        Matching is a case-insensitive AND over the query's words against the
        article URL (which embeds the headline slug). Stops once ``max_results`` are
        collected. Empty if the as-of date predates GDELT GKG 2.0 (2015-02).
        """
        terms = [w for w in query.lower().split() if len(w) > 2]
        if self.backend == "bigquery":
            return self._fetch_bigquery(terms, clock)
        docs: list[Document] = []
        for stamp in self._slot_stamps(clock):
            for doc in self._fetch_slot(stamp, terms):
                clock.guard(doc.timestamp, source=self.name)  # belt-and-suspenders
                docs.append(doc)
                if len(docs) >= self.max_results:
                    return docs
        return docs

    def _bq_client(self):
        """Build a BigQuery client (lazy import so the dep is optional)."""
        try:
            from google.cloud import bigquery
        except ImportError as e:  # pragma: no cover - guidance for missing extra
            raise ImportError(
                'The bigquery backend needs the "bigquery" extra: '
                'pip install -e ".[bigquery]"'
            ) from e
        return bigquery.Client(project=self.bq_project)

    def _fetch_bigquery(self, terms: list[str], clock: Clock) -> list[Document]:
        """Query the public GDELT GKG table for matching articles up to the Clock.

        Uses a parameterized ``WHERE DATE <= @as_of`` (GKG DATE is an int
        ``YYYYMMDDHHMMSS``), plus a lower bound for cost control and a LIKE per term.
        The DATE column is the same leak-safe timestamp as the raw files.
        """
        from google.cloud import bigquery

        end = clock.as_of
        start = end - timedelta(days=self.bq_lookback_days)
        as_of_int = int(end.strftime("%Y%m%d%H%M%S"))
        start_int = int(start.strftime("%Y%m%d%H%M%S"))

        where = ["DATE <= @as_of", "DATE >= @start"]
        params = [
            bigquery.ScalarQueryParameter("as_of", "INT64", as_of_int),
            bigquery.ScalarQueryParameter("start", "INT64", start_int),
            bigquery.ScalarQueryParameter("lim", "INT64", self.max_results),
        ]
        for i, term in enumerate(terms):
            where.append(f"LOWER(DocumentIdentifier) LIKE @t{i}")
            params.append(
                bigquery.ScalarQueryParameter(f"t{i}", "STRING", f"%{term}%")
            )
        sql = (
            "SELECT DATE, SourceCommonName, DocumentIdentifier "
            f"FROM `{_BQ_TABLE}` WHERE {' AND '.join(where)} "
            "ORDER BY DATE DESC LIMIT @lim"
        )
        job = self._bq_client().query(
            sql, job_config=bigquery.QueryJobConfig(query_parameters=params)
        )
        docs: list[Document] = []
        for row in job.result():
            ts = datetime.strptime(str(row["DATE"]), "%Y%m%d%H%M%S").replace(
                tzinfo=timezone.utc
            )
            ts = clock.guard(ts, source=self.name)  # belt-and-suspenders
            docs.append(
                Document(
                    content=row["DocumentIdentifier"],
                    timestamp=ts,
                    source=self.name,
                    url=row["DocumentIdentifier"],
                    meta={"domain": row["SourceCommonName"]},
                )
            )
        return docs


_: type[Source] = GDELTNewsSource
