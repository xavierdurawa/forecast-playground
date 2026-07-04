"""Shared pytest config: keep the default run offline.

Integration tests hit live APIs (and optionally a model). They are skipped unless
explicitly selected with ``-m integration``, so the normal ``pytest`` run stays
fast, deterministic, and network-free.
"""

import pytest

# Auto-load a local .env (gitignored) so integration tests pick up API keys /
# FORECAST_CONTACT without any manual `export`. Never overrides existing env vars.
from forecast_playground import load_env

load_env()


def pytest_collection_modifyitems(config, items):
    # If the user asked for integration tests (-m integration), run them as-is.
    if "integration" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason="integration test; run with -m integration")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
