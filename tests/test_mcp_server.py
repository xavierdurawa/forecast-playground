"""Smoke test for the MCP server adapter (offline — no network, no live model).

Verifies the server module imports and registers a tool per source. Skipped if the
optional ``fastmcp`` dependency isn't installed.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastmcp")

# The adapter lives outside the package (in adapters/), so load it by path.
_ADAPTER = Path(__file__).resolve().parents[1] / "adapters" / "mcp_server.py"


def _load_server():
    spec = importlib.util.spec_from_file_location("mcp_server", _ADAPTER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mcp_server"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_server_registers_a_tool_per_source():
    import asyncio

    mod = _load_server()
    assert mod.mcp.name == "forecast-playground"
    # Assert the tools are actually registered with FastMCP (what a client sees),
    # not merely defined as functions.
    tools = asyncio.run(mod.mcp.list_tools())
    names = {t.name for t in tools}
    assert names == {
        "wikipedia", "pageviews", "web_archive", "prediction_market",
        "current_events", "news", "economic_series", "weather",
    }
