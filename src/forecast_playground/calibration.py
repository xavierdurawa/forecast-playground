"""Calibration reporting for binary forecasts.

Mean Brier alone hides *why* a forecaster scores as it does: a well-calibrated but
uncertain model and an overconfident-but-lucky one can share a Brier. A reliability
curve answers the question that matters — "when this forecaster says 70%, does it
happen ~70% of the time?" — and expected calibration error (ECE) summarizes the gap.

Pure functions over ``(prob, outcome)`` pairs; no plotting dependency (the report
is data you can print or feed to any plotter).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CalibrationBin:
    """One bucket of the reliability curve.

    Attributes:
        low, high: Probability range of this bin, e.g. [0.6, 0.7).
        count: How many forecasts fell in the bin.
        mean_pred: Average predicted probability among them (x on the curve).
        frac_positive: Fraction that actually resolved YES (y on the curve).
    """

    low: float
    high: float
    count: int
    mean_pred: float
    frac_positive: float

    @property
    def gap(self) -> float:
        """|predicted - observed| for this bin (0 = perfectly calibrated)."""
        return abs(self.mean_pred - self.frac_positive)


@dataclass
class CalibrationReport:
    """A reliability curve plus summary calibration metrics.

    Attributes:
        bins: Non-empty reliability bins, low to high.
        ece: Expected calibration error — count-weighted mean bin gap. 0 is perfect.
        mce: Maximum calibration error — the worst single bin gap.
        n: Total number of forecasts scored.
    """

    bins: list[CalibrationBin]
    ece: float
    mce: float
    n: int

    def table(self) -> str:
        """A compact text reliability table for printing in studies/logs."""
        lines = [f"reliability ({self.n} forecasts, ECE={self.ece:.3f}, MCE={self.mce:.3f}):"]
        lines.append(f"  {'bin':>11}  {'n':>4}  {'pred':>5}  {'actual':>6}  gap")
        for b in self.bins:
            lines.append(
                f"  [{b.low:.2f},{b.high:.2f})  {b.count:>4}  "
                f"{b.mean_pred:>5.2f}  {b.frac_positive:>6.2f}  {b.gap:.2f}"
            )
        return "\n".join(lines)


def calibration_report(
    forecasts: list[tuple[float, int]], n_bins: int = 10
) -> CalibrationReport:
    """Build a reliability curve + ECE/MCE from ``(prob, outcome)`` pairs.

    Forecasts are bucketed into ``n_bins`` equal-width probability bins over [0, 1]
    (1.0 falls in the top bin). Empty bins are omitted from the curve but do not
    affect ECE. Returns a report with ``ece == nan`` if there are no forecasts.
    """
    if not forecasts:
        return CalibrationReport(bins=[], ece=float("nan"), mce=float("nan"), n=0)

    buckets: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, o in forecasts:
        idx = min(int(p * n_bins), n_bins - 1)  # p == 1.0 -> top bin
        buckets[idx].append((p, o))

    bins: list[CalibrationBin] = []
    ece = 0.0
    mce = 0.0
    total = len(forecasts)
    for i, bucket in enumerate(buckets):
        if not bucket:
            continue
        mean_pred = sum(p for p, _ in bucket) / len(bucket)
        frac_pos = sum(o for _, o in bucket) / len(bucket)
        b = CalibrationBin(
            low=i / n_bins,
            high=(i + 1) / n_bins,
            count=len(bucket),
            mean_pred=mean_pred,
            frac_positive=frac_pos,
        )
        bins.append(b)
        ece += (b.count / total) * b.gap
        mce = max(mce, b.gap)

    return CalibrationReport(bins=bins, ece=ece, mce=mce, n=total)
