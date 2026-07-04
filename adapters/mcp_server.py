"""MCP server: expose the time-masked tools to interactive assistants.

Adds ForecastPlayground's sources to any MCP client (Claude Desktop, Cursor,
VS Code, ...) as tools of the form ``<source>(query, as_of)``. A user can then ask,
in plain language, "what did Wikipedia say about X as of 2024-01-01?" and the
assistant calls the leak-free tool directly — no Python.

Framing note (important): here the ``as_of`` date is a USER-CONTROLLED tool
parameter, not an enforced cage. MCP is for interactive *exploration*; the model
chooses what date to look at. The leak-free *evaluation/training* guarantee lives in
the Toolkit / verifiers paths, where the environment binds the Clock and the model
can't pick its own date. So: great for browsing the past, not a scored eval harness.

Requires the ``mcp`` extra:  pip install -e ".[mcp]"

Run:  python adapters/mcp_server.py            # stdio (for Claude Desktop etc.)
Then point your MCP client's config at that command.
"""

from __future__ import annotations

from typing import Any

try:
    from fastmcp import FastMCP
except ImportError as e:  # pragma: no cover - guidance for a missing optional dep
    raise ImportError(
        'The MCP server needs the "mcp" extra: pip install -e ".[mcp]"'
    ) from e

from forecast_playground import (
    Clock,
    CurrentEventsSource,
    FREDSource,
    GDELTNewsSource,
    NOAASource,
    PageviewsSource,
    PolymarketSource,
    Toolkit,
    WaybackSource,
    WikipediaSource,
)

mcp = FastMCP("forecast-playground")


def _run(source: Any, query: str, as_of: str) -> str:
    """Fetch from one source at ``as_of`` (YYYY-MM-DD) via a per-call Toolkit.

    Routing through the Toolkit keeps the leak-guard chokepoint in play even here.
    """
    tk = Toolkit(clock=Clock.at(as_of), sources=[source], enable_python=False)
    tool_name = f"{source.name.split(':')[0]}_search"
    return tk.call(tool_name, {"query": query})


@mcp.tool()
def wikipedia(query: str, as_of: str) -> str:
    """Wikipedia article text as it existed on `as_of` (YYYY-MM-DD). Full-text search."""
    return _run(WikipediaSource(mode="search"), query, as_of)


@mcp.tool()
def pageviews(article: str, as_of: str) -> str:
    """Daily Wikipedia pageview counts for `article`, up to `as_of` (YYYY-MM-DD)."""
    return _run(PageviewsSource(), article, as_of)


@mcp.tool()
def web_archive(url: str, as_of: str) -> str:
    """A web page as archived at-or-before `as_of` (YYYY-MM-DD), via the Wayback Machine."""
    return _run(WaybackSource(), url, as_of)


@mcp.tool()
def prediction_market(token_id: str, as_of: str) -> str:
    """Polymarket YES-probability history for a CLOB token id, up to `as_of`."""
    return _run(PolymarketSource(), token_id, as_of)


@mcp.tool()
def current_events(ignored: str, as_of: str) -> str:
    """Wikipedia's curated daily news digest for the days up to `as_of` (YYYY-MM-DD)."""
    return _run(CurrentEventsSource(), "", as_of)


@mcp.tool()
def news(query: str, as_of: str) -> str:
    """Global news article URLs (GDELT) matching `query`, up to `as_of` (YYYY-MM-DD)."""
    return _run(GDELTNewsSource(), query, as_of)


@mcp.tool()
def economic_series(series_id: str, as_of: str) -> str:
    """A FRED economic series (e.g. GDP, CPIAUCSL) as published on `as_of`. Needs FRED_API_KEY."""
    return _run(FREDSource(), series_id, as_of)


@mcp.tool()
def weather(station_id: str, as_of: str) -> str:
    """NOAA daily weather for a station id (blank = default), up to `as_of`. Needs NOAA_TOKEN."""
    return _run(NOAASource(), station_id, as_of)


if __name__ == "__main__":
    mcp.run()
