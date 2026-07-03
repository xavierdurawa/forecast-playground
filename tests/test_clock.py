"""Tests for the Clock — the no-lookahead chokepoint."""

from datetime import datetime, timezone

import pytest

from forecast_playground import Clock, LookaheadError


def test_at_parses_bare_date_as_utc_midnight():
    clock = Clock.at("2024-01-01")
    assert clock.as_of == datetime(2024, 1, 1, tzinfo=timezone.utc)


def test_admits_before_and_at_boundary():
    clock = Clock.at("2024-01-01T12:00:00")
    assert clock.admits(datetime(2024, 1, 1, 11, 59, tzinfo=timezone.utc))
    assert clock.admits(datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc))  # inclusive


def test_rejects_future():
    clock = Clock.at("2024-01-01")
    assert not clock.admits(datetime(2024, 1, 2, tzinfo=timezone.utc))


def test_guard_raises_on_lookahead():
    clock = Clock.at("2024-01-01")
    with pytest.raises(LookaheadError):
        clock.guard(datetime(2024, 6, 1, tzinfo=timezone.utc), source="test")


def test_naive_datetime_treated_as_utc():
    clock = Clock.at("2024-01-01T00:00:00")
    # A naive timestamp equal to as_of is admitted (assumed UTC).
    assert clock.admits(datetime(2024, 1, 1, 0, 0, 0))


def test_timezone_normalization_no_leak():
    # 2024-01-01T01:00 in +02:00 is 2023-12-31T23:00 UTC -> admissible at UTC midnight.
    from datetime import timedelta

    clock = Clock.at("2024-01-01T00:00:00")
    tz_plus2 = timezone(timedelta(hours=2))
    assert clock.admits(datetime(2024, 1, 1, 1, 0, tzinfo=tz_plus2))
