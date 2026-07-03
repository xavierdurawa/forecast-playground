"""Tests for ensemble aggregation primitives and the calibration report."""

import math

from forecast_playground import (
    aggregate,
    calibration_report,
    extremize,
    trimmed_mean,
)


# --- aggregation -----------------------------------------------------------

def test_trimmed_mean_drops_outliers():
    # 0.0 and 1.0 are trimmed; middle three average to 0.5.
    assert trimmed_mean([0.0, 0.4, 0.5, 0.6, 1.0], trim=0.2) == 0.5


def test_trimmed_mean_trim_zero_is_plain_mean():
    assert abs(trimmed_mean([0.2, 0.4, 0.6], trim=0.0) - 0.4) < 1e-9


def test_trimmed_mean_never_trims_everything():
    # Small list + trim would remove all; falls back to full mean.
    assert trimmed_mean([0.3, 0.7], trim=0.5) == 0.5


def test_extremize_pushes_away_from_half():
    assert extremize(0.7, a=1.5) > 0.7
    assert extremize(0.3, a=1.5) < 0.3
    assert abs(extremize(0.5, a=2.0) - 0.5) < 1e-9  # 0.5 is a fixed point


def test_extremize_identity_at_a_one():
    assert abs(extremize(0.73, a=1.0) - 0.73) < 1e-9


def test_aggregate_combines_trim_and_extremize():
    out = aggregate([0.55, 0.6, 0.65], trim=0.0, extremize_a=1.0)
    assert abs(out - 0.6) < 1e-9  # plain mean when extremize is off


# --- calibration -----------------------------------------------------------

def test_perfectly_calibrated_has_low_ece():
    # 100 forecasts at p=0.7 where exactly 70% resolve YES -> near-zero ECE.
    fc = [(0.7, 1)] * 70 + [(0.7, 0)] * 30
    rep = calibration_report(fc)
    assert rep.n == 100
    assert rep.ece < 0.01
    # Single populated bin: predicted 0.70, observed 0.70.
    assert len(rep.bins) == 1
    assert abs(rep.bins[0].frac_positive - 0.7) < 1e-9


def test_overconfident_has_high_ece():
    # Says 0.95 but only half happen -> big calibration gap.
    fc = [(0.95, 1)] * 50 + [(0.95, 0)] * 50
    rep = calibration_report(fc)
    assert rep.ece > 0.4
    assert rep.mce > 0.4


def test_empty_report_is_nan():
    rep = calibration_report([])
    assert rep.n == 0 and math.isnan(rep.ece)


def test_prob_one_lands_in_top_bin():
    rep = calibration_report([(1.0, 1)], n_bins=10)
    assert rep.bins[0].low == 0.9 and rep.bins[0].high == 1.0


def test_table_renders():
    rep = calibration_report([(0.7, 1), (0.7, 0), (0.2, 0)])
    assert "reliability" in rep.table() and "ECE" in rep.table()
