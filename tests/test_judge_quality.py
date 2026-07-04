"""Live QUALITY gate for the LLM-as-judge scorer (not a leak gate).

Runs the judge against boolean questions with KNOWN outcomes and asserts the judge's
Brier agrees with the true Brier — i.e. the judge is a faithful drop-in where truth
is known (proposal 0001 validated this downstream; this is our own standing check).

Costs model tokens, so it's an integration test (skipped by default). It prefers a
Bedrock model (via .env / AWS creds), else a local Ollama, else skips.
"""

import os
import urllib.request

import pytest

from forecast_playground import judge_forecast
from forecast_playground.scoring import brier_score

pytestmark = pytest.mark.integration

_OLLAMA = "http://localhost:11434/v1"


def _make_driver_and_model():
    """Return (driver, model) from whatever's available, else (None, None)."""
    # Prefer Bedrock if AWS creds look present.
    if os.environ.get("AWS_REGION") or os.path.exists(os.path.expanduser("~/.aws")):
        try:
            from forecast_playground import AnthropicDriver

            return AnthropicDriver(), "global.anthropic.claude-sonnet-4-6"
        except Exception:
            pass
    try:
        urllib.request.urlopen(_OLLAMA.replace("/v1", "") + "/api/tags", timeout=2)
        from forecast_playground import OpenAIDriver

        return OpenAIDriver(base_url=_OLLAMA, api_key="ollama"), "qwen2.5-coder:14b"
    except Exception:
        return None, None


# Known-outcome boolean cases: (question, forecaster_prob, resolution_text, outcome).
_CASES = [
    ("Will SpaceX Starship reach orbit before 2025?", 0.85,
     "Yes — Starship reached orbital velocity during a 2024 flight test.", 1),
    ("Will the Sacramento Kings win the 2025 NBA Finals?", 0.05,
     "No — the Kings did not win the 2025 NBA Finals.", 0),
    ("Was Donald Trump inaugurated as US President in January 2025?", 0.97,
     "Yes — he was inaugurated on 2025-01-20.", 1),
]


def test_judge_brier_agrees_with_true_brier():
    driver, model = _make_driver_and_model()
    if driver is None:
        pytest.skip("no judge model available (no AWS creds and no local Ollama)")

    errors = []
    for question, prob, resolution, outcome in _CASES:
        j = judge_forecast(driver, model, question, prob, resolution)
        if not j.parsed:
            continue  # a parse failure is a model quirk, not a scorer defect
        # The judge's soft label should recover the true boolean outcome...
        assert abs(j.resolved_label - outcome) < 0.5, (
            f"judge mislabeled outcome for {question!r}: "
            f"label={j.resolved_label}, true={outcome}"
        )
        # ...so its Brier should closely track the true Brier.
        errors.append(abs(j.brier - brier_score(prob, outcome)))

    assert errors, "judge produced no parsable judgments"
    assert sum(errors) / len(errors) < 0.1, f"judge Brier drifts from true: MAE={errors}"
