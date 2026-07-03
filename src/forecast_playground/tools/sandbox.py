"""A minimal Python execution tool for the forecasting agent.

Lets a model compute base rates, distributions, Monte Carlo estimates, and the
arithmetic of combining evidence into a probability. Runs code in a separate
process with a wall-clock timeout.

NOTE: this is *isolation by subprocess + timeout*, not a security sandbox. It is
fine for trusted research use. Do not expose it to untrusted input without a real
sandbox (e.g. a container, seccomp, or a hosted code-exec service).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def run_python(code: str, timeout_s: float = 10.0) -> str:
    """Execute Python code and return its stdout (and stderr on error).

    Use this to compute probabilities, run simulations, or do arithmetic. Print
    the result you care about — only stdout is returned.

    Args:
        code: Python source to execute.
        timeout_s: Wall-clock limit before the run is killed.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", textwrap.dedent(code)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: execution exceeded {timeout_s}s timeout"
    out = proc.stdout.strip()
    if proc.returncode != 0:
        err = proc.stderr.strip()
        return f"ERROR (exit {proc.returncode}):\n{err}\n--- stdout ---\n{out}".strip()
    return out or "(no output — did you print() the result?)"
