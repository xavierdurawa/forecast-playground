"""A dead-simple on-disk cache for tool results.

Time-masked retrieval is deterministic: the same (tool, query, as_of) always yields
the same historical data, so results can be cached forever with no TTL or eviction.
This is a plain JSON-file-per-key store — no database, no dependencies, no expiry.
Its only job is to stop a study from re-fetching the same Wikipedia article dozens
of times.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class ResultCache:
    """Content-addressed cache of string results under a directory.

    Args:
        directory: Where to store cache files (created if missing).
        enabled: If False, every get() misses and put() is a no-op — a single flag
            to turn caching off without changing call sites.
    """

    def __init__(self, directory: str | Path = ".cache/chrono", enabled: bool = True):
        self.dir = Path(directory)
        self.enabled = enabled
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        return self.dir / f"{digest}.json"

    def get(self, key: str) -> str | None:
        """Return the cached result for ``key``, or None on a miss."""
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())["result"]
        except (json.JSONDecodeError, KeyError, OSError):
            return None  # corrupt entry -> treat as a miss

    def put(self, key: str, result: str) -> None:
        """Store ``result`` under ``key``. Records the readable key for debugging."""
        if not self.enabled:
            return
        try:
            self._path(key).write_text(json.dumps({"key": key, "result": result}))
        except OSError:
            pass  # caching is best-effort; never fail a run over it
