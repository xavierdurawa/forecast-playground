"""NOAA weather source: historical daily observations up to the as-of date.

NOAA's Climate Data Online (CDO) v2 API serves historical weather/climate
observations. Each observation carries its observation ``date``, and we cap the
query's ``enddate`` at the as-of instant — so this is a HARD source: a measurement
is dated when it was taken, a true upper bound on availability.

Needs a free CDO token (https://www.ncdc.noaa.gov/cdo-web/token) in the
``NOAA_TOKEN`` env var. The ``query`` is a NOAA station id (default: a major
station); ``datatypes`` selects which measurements (default TMAX/TMIN/PRCP).

Docs: https://www.ncei.noaa.gov/cdo-web/webservices/v2
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from ..clock import Clock
from ..http import make_session, user_agent
from .base import AsOfGuarantee, Document, Source

_API = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
# GHCND station for NYC Central Park — a reasonable default when none is given.
_DEFAULT_STATION = "GHCND:USW00094728"


class NOAASource:
    """Fetch NOAA daily weather observations for a station up to the as-of date.

    Args:
        token: CDO API token. Defaults to the ``NOAA_TOKEN`` env var.
        dataset: CDO dataset id (default ``"GHCND"`` — daily summaries).
        datatypes: Measurement types to request (default max/min temp + precip).
        lookback_days: How many days of history to fetch before the as-of date.
        default_station: Station id used when the query is empty.
    """

    guarantee = AsOfGuarantee.HARD

    def __init__(
        self,
        token: str | None = None,
        dataset: str = "GHCND",
        datatypes: tuple[str, ...] = ("TMAX", "TMIN", "PRCP"),
        lookback_days: int = 30,
        default_station: str = _DEFAULT_STATION,
        session: requests.Session | None = None,
        timeout: float = 20.0,
    ) -> None:
        self.name = "noaa"
        self.token = token or os.environ.get("NOAA_TOKEN", "")
        self.dataset = dataset
        self.datatypes = datatypes
        self.lookback_days = lookback_days
        self.default_station = default_station
        self.timeout = timeout
        self._session = session or make_session(user_agent=user_agent())

    def fetch(self, query: str, clock: Clock, **kwargs: Any) -> list[Document]:
        """Return daily observations for station ``query`` up to ``clock.as_of``.

        ``query`` is a NOAA station id (empty -> the default station). Empty result
        if no token is configured or no data exists at/before the as-of date.
        """
        if not self.token:
            return []  # no token -> no data (leak-safe: returns nothing)
        station = query.strip() or self.default_station
        end = clock.as_of
        start = end - timedelta(days=self.lookback_days)
        params = [
            ("datasetid", self.dataset),
            ("stationid", station),
            ("startdate", start.strftime("%Y-%m-%d")),
            ("enddate", end.strftime("%Y-%m-%d")),  # capped at as_of
            ("units", "metric"),
            ("limit", "1000"),
        ]
        params += [("datatypeid", dt) for dt in self.datatypes]
        resp = self._session.get(
            _API, params=params, headers={"token": self.token}, timeout=self.timeout
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        rows: list[tuple[datetime, str, Any]] = []
        for r in results:
            # CDO dates look like "2024-05-20T00:00:00".
            obs = datetime.fromisoformat(r["date"]).replace(tzinfo=timezone.utc)
            obs = clock.guard(obs, source=self.name, detail=f"station={station}")
            rows.append((obs, r.get("datatype", "?"), r.get("value")))
        if not rows:
            return []
        rows.sort(key=lambda x: x[0])
        latest = rows[-1][0]
        table = "\n".join(
            f"{d.date().isoformat()}  {dt}={v}" for d, dt, v in rows
        )
        return [
            Document(
                content=f"NOAA {self.dataset} observations for {station} "
                f"({len(rows)} readings up to {end.date().isoformat()}):\n{table}",
                timestamp=latest,
                source=self.name,
                url=f"https://www.ncdc.noaa.gov/cdo-web/datasets/{self.dataset}/stations/{station}/detail",
                meta={"station": station, "readings": len(rows)},
            )
        ]


_: type[Source] = NOAASource
