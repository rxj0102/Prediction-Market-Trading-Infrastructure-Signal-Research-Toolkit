"""Scoring and calibration diagnostics for probability forecasts.

* Brier score: mean squared error between forecast probability and the 0/1
  outcome (lower is better; a constant 0.5 forecast scores 0.25).
* Murphy decomposition over forecast bins:
  ``Brier = reliability - resolution + uncertainty`` (exact when forecasts are
  constant within each bin; otherwise there is a small within-bin term,
  reported as ``residual``).
* Calibration curve: mean forecast vs. observed YES frequency per bucket.
* Expected calibration error (ECE): count-weighted mean |forecast - observed|.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def _check(p: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if p.shape != y.shape or p.ndim != 1 or len(p) == 0:
        raise ValueError("p and y must be non-empty 1-D arrays of equal length")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("outcomes must be 0/1")
    return p, y


def brier_score(p: FloatArray, y: FloatArray) -> float:
    p, y = _check(p, y)
    return float(np.mean((p - y) ** 2))


def log_loss(p: FloatArray, y: FloatArray, eps: float = 1e-12) -> float:
    p, y = _check(p, y)
    q = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(q) + (1.0 - y) * np.log(1.0 - q)))


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_forecast: float
    observed_freq: float

    @property
    def gap(self) -> float:
        return self.mean_forecast - self.observed_freq


def _bin_index(p: FloatArray, n_bins: int) -> npt.NDArray[np.int64]:
    return np.minimum((p * n_bins).astype(np.int64), n_bins - 1)


def calibration_curve(p: FloatArray, y: FloatArray, n_bins: int = 10) -> list[CalibrationBin]:
    """Equal-width buckets on [0, 1]; empty buckets are omitted."""
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    p, y = _check(p, y)
    idx = _bin_index(p, n_bins)
    bins: list[CalibrationBin] = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        bins.append(
            CalibrationBin(
                b / n_bins, (b + 1) / n_bins, n, float(p[mask].mean()), float(y[mask].mean())
            )
        )
    return bins


def expected_calibration_error(bins: list[CalibrationBin]) -> float:
    total = sum(b.count for b in bins)
    if total == 0:
        raise ValueError("no observations in bins")
    return sum(b.count * abs(b.gap) for b in bins) / total


@dataclass(frozen=True)
class BrierDecomposition:
    brier: float
    reliability: float
    resolution: float
    uncertainty: float

    @property
    def residual(self) -> float:
        """Within-bin variance term: brier - (reliability - resolution + uncertainty)."""
        return self.brier - (self.reliability - self.resolution + self.uncertainty)


def brier_decomposition(p: FloatArray, y: FloatArray, n_bins: int = 10) -> BrierDecomposition:
    p, y = _check(p, y)
    bins = calibration_curve(p, y, n_bins)
    n = len(p)
    base = float(y.mean())
    rel = sum(b.count * (b.mean_forecast - b.observed_freq) ** 2 for b in bins) / n
    res = sum(b.count * (b.observed_freq - base) ** 2 for b in bins) / n
    unc = base * (1.0 - base)
    return BrierDecomposition(brier_score(p, y), rel, res, unc)
