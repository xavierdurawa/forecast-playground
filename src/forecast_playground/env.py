"""Optional .env loading for local runs.

Sources read secrets (FRED_API_KEY, NOAA_TOKEN, FORECAST_CONTACT) from the
environment. ``load_env()`` pulls them from a local ``.env`` file if one exists and
``python-dotenv`` is installed — a no-op otherwise. Entry points (the study, the MCP
server) call it so keys "just work" without a manual ``export``. It never overrides
variables already set in the real environment.
"""

from __future__ import annotations


def load_env() -> bool:
    """Load a local ``.env`` into os.environ if possible. Returns True if loaded."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    return load_dotenv(override=False)
