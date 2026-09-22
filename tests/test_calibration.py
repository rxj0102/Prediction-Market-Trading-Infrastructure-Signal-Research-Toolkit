"""Brier score / calibration math verified against hand-computed cases."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest

from signals.calibration import (
    CalibrationBin,
    brier_decomposition,
    brier_score,
    calibration_curve,
    expected_calibration_error,
    log_loss,
)


def arr(*v: float) -> npt.NDArray[np.float64]:
    return np.array(v, dtype=np.float64)


def test_brier_hand_computed() -> None:
    # ((0.9-1)^2 + (0.2-0)^2 + (0.6-0)^2) / 3 = (0.01 + 0.04 + 0.36) / 3
    assert brier_score(arr(0.9, 0.2, 0.6), arr(1, 0, 0)) == pytest.approx(0.41 / 3)


def test_brier_reference_points() -> None:
    y = arr(1, 0, 1, 0)
    assert brier_score(arr(0.5, 0.5, 0.5, 0.5), y) == pytest.approx(0.25)
    assert brier_score(y.copy(), y) == 0.0
    assert brier_score(1 - y, y) == 1.0


def test_log_loss_hand_computed() -> None:
    assert log_loss(arr(0.8), arr(1)) == pytest.approx(-math.log(0.8))
    assert log_loss(arr(0.8, 0.3), arr(1, 0)) == pytest.approx(-(math.log(0.8) + math.log(0.7)) / 2)
    # Clipping keeps a confident miss finite.
    assert math.isfinite(log_loss(arr(0.0), arr(1)))


def test_calibration_curve_hand_computed() -> None:
    bins = calibration_curve(arr(0.05, 0.15, 0.15, 0.95), arr(0, 0, 1, 1), n_bins=10)
    assert [b.count for b in bins] == [1, 2, 1]
    assert [b.mean_forecast for b in bins] == pytest.approx([0.05, 0.15, 0.95])
    assert [b.observed_freq for b in bins] == [0.0, 0.5, 1.0]
    assert bins[1].lower == pytest.approx(0.1)
    assert bins[1].upper == pytest.approx(0.2)
    # ECE = (1*0.05 + 2*0.35 + 1*0.05) / 4 = 0.2
    assert expected_calibration_error(bins) == pytest.approx(0.2)
    assert bins[1].gap == pytest.approx(-0.35)


def test_probability_one_lands_in_last_bin() -> None:
    bins = calibration_curve(arr(1.0, 0.0), arr(1, 0), n_bins=4)
    assert bins[-1].lower == 0.75
    assert bins[0].lower == 0.0


def test_brier_decomposition_exact_for_bin_constant_forecasts() -> None:
    p = arr(0.15, 0.15, 0.15, 0.85, 0.85)
    y = arr(0, 0, 1, 1, 1)
    d = brier_decomposition(p, y, n_bins=10)
    # Hand computed: base rate 0.6
    assert d.uncertainty == pytest.approx(0.24)
    assert d.reliability == pytest.approx((3 * (0.15 - 1 / 3) ** 2 + 2 * 0.15**2) / 5)
    assert d.resolution == pytest.approx((3 * (1 / 3 - 0.6) ** 2 + 2 * 0.4**2) / 5)
    assert d.brier == pytest.approx(0.1625)
    assert d.residual == pytest.approx(0.0, abs=1e-12)


def test_brier_decomposition_perfectly_calibrated_has_zero_reliability() -> None:
    rng = np.random.default_rng(0)
    p = np.repeat([0.05, 0.25, 0.45, 0.65, 0.85], 20_000)
    y = (rng.random(len(p)) < p).astype(float)
    d = brier_decomposition(p, y)
    assert d.reliability < 1e-4
    assert abs(d.residual) < 1e-9


@pytest.mark.parametrize(
    ("p", "y"),
    [
        (arr(0.5), arr(1, 0)),
        (arr(), arr()),
        (arr(1.2), arr(1)),
        (arr(0.5), arr(0.5)),
    ],
)
def test_input_validation(p: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        brier_score(p, y)


def test_curve_and_ece_validation() -> None:
    with pytest.raises(ValueError, match="n_bins"):
        calibration_curve(arr(0.5), arr(1), n_bins=0)
    with pytest.raises(ValueError, match="no observations"):
        expected_calibration_error([])
    assert CalibrationBin(0.0, 0.1, 1, 0.1, 0.0).gap == pytest.approx(0.1)
