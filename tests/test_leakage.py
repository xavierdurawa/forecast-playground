"""Offline tests for the parametric leakage guard (no network — cache injected)."""

from datetime import datetime, timezone

import pytest

import forecast_playground.leakage as leak
from forecast_playground import is_leak_safe, min_safe_resolution, training_cutoff


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    # Inject a fake models.dev cache and reset overrides so tests are hermetic.
    monkeypatch.setattr(leak, "_CUTOFF_CACHE", {
        "claude-sonnet-4-6": "2025-08-31",
        "gpt-5": "2024-09-30",
    })
    monkeypatch.setattr(leak, "_OVERRIDES", {})
    yield


def test_training_cutoff_exact_and_substring():
    assert training_cutoff("gpt-5") == datetime(2024, 9, 30, tzinfo=timezone.utc)
    # Region/prefix variant resolves via the bare id substring.
    assert training_cutoff("global.anthropic.claude-sonnet-4-6") == datetime(
        2025, 8, 31, tzinfo=timezone.utc
    )


def test_unknown_model_is_none_and_rejected():
    assert training_cutoff("qwen2.5-coder:14b") is None
    # Fail-safe: unknown model cannot be certified -> reject.
    assert is_leak_safe("2020-01-01", "qwen2.5-coder:14b") is False


def test_min_safe_resolution_adds_margin():
    # cutoff 2024-09-30 + 90d.
    assert min_safe_resolution("gpt-5", margin_days=90) == datetime(
        2024, 12, 29, tzinfo=timezone.utc
    )


def test_is_leak_safe_true_after_margin():
    # Well after cutoff+margin -> safe.
    assert is_leak_safe("2025-06-01", "gpt-5") is True


def test_is_leak_safe_false_inside_window():
    # Resolved before the cutoff -> the model may know it -> not safe.
    assert is_leak_safe("2024-01-01", "gpt-5") is False
    # Just inside the margin -> still not safe.
    assert is_leak_safe("2024-11-01", "gpt-5") is False


def test_override_wins(monkeypatch):
    leak.register_cutoff("qwen2.5-coder", "2024-01-01")
    # Now the previously-unknown model is certifiable via the override.
    assert training_cutoff("qwen2.5-coder:14b") == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert is_leak_safe("2025-01-01", "qwen2.5-coder:14b") is True


def test_override_beats_table(monkeypatch):
    leak.register_cutoff("gpt-5", "2099-01-01")  # absurd override to prove precedence
    assert training_cutoff("gpt-5") == datetime(2099, 1, 1, tzinfo=timezone.utc)


def test_accepts_datetime_and_string_dates():
    dt = datetime(2025, 6, 1, tzinfo=timezone.utc)
    assert is_leak_safe(dt, "gpt-5") is True
    assert is_leak_safe("2025-06", "gpt-5") is True  # YYYY-MM granularity
