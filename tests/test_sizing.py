"""Position sizing and risk-limit logic."""

from __future__ import annotations

from typing import Any

import pytest

from execution.sizing import (
    PositionSizer,
    RiskLimits,
    drawdown_throttle,
    kelly_fraction_binary,
    worst_case_exposure,
)

BIG = RiskLimits(max_position_per_market=1_000_000, max_gross_exposure=1e12)


def test_worst_case_exposure() -> None:
    assert worst_case_exposure(100, 0.30) == pytest.approx(30.0)  # long loses the price paid
    assert worst_case_exposure(-100, 0.30) == pytest.approx(70.0)  # short loses 1 - price
    assert worst_case_exposure(0, 0.5) == 0.0


def test_kelly_binary() -> None:
    assert kelly_fraction_binary(0.6, 0.5) == pytest.approx(0.2)
    assert kelly_fraction_binary(0.4, 0.5) == pytest.approx(-0.2)
    assert kelly_fraction_binary(0.5, 0.5) == 0.0
    assert kelly_fraction_binary(0.9, 0.8) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="price"):
        kelly_fraction_binary(0.5, 1.0)


def test_drawdown_throttle_is_linear_between_bounds() -> None:
    lim = RiskLimits(throttle_start=0.04, throttle_full=0.10, min_throttle=0.25)
    assert drawdown_throttle(0.0, lim) == 1.0
    assert drawdown_throttle(0.04, lim) == 1.0
    assert drawdown_throttle(0.07, lim) == pytest.approx(0.625)
    assert drawdown_throttle(0.10, lim) == 0.25
    assert drawdown_throttle(0.50, lim) == 0.25


def test_kelly_sizing_long_and_short() -> None:
    sizer = PositionSizer(BIG)  # kelly_fraction 0.25
    # f = 0.2 * 0.25 = 0.05 of $10k = $500 at $0.50 per contract -> 1000 contracts
    assert sizer.target_position(0.6, 0.5, 10_000, 0.0, 0, 0.0) == 1000
    # short: f = 0.05, cost per contract 1 - 0.5
    assert sizer.target_position(0.4, 0.5, 10_000, 0.0, 0, 0.0) == -1000


def test_per_market_cap() -> None:
    sizer = PositionSizer(RiskLimits(max_position_per_market=400, max_gross_exposure=1e9))
    assert sizer.target_position(0.6, 0.5, 10_000, 0.0, 0, 0.0) == 400
    assert sizer.target_position(0.4, 0.5, 10_000, 0.0, 0, 0.0) == -400


def test_gross_exposure_cap_counts_other_markets() -> None:
    sizer = PositionSizer(RiskLimits(max_position_per_market=10_000, max_gross_exposure=4_000))
    # $100 of room left at $0.50 per contract -> 200 contracts
    assert sizer.target_position(0.6, 0.5, 10_000, 0.0, 0, 3_900.0) == 200
    assert sizer.target_position(0.6, 0.5, 10_000, 0.0, 0, 5_000.0) == 0


def test_drawdown_throttle_reduces_size() -> None:
    sizer = PositionSizer(BIG)
    full = sizer.target_position(0.6, 0.5, 10_000, 0.0, 0, 0.0)
    throttled = sizer.target_position(0.6, 0.5, 10_000, 0.10, 0, 0.0)
    assert throttled == full * 0.25


def test_hysteresis_band() -> None:
    sizer = PositionSizer(BIG)  # entry 0.05, exit 0.01
    assert sizer.target_position(0.505, 0.5, 10_000, 0.0, 300, 0.0) == 0  # below exit
    assert sizer.target_position(0.53, 0.5, 10_000, 0.0, 300, 0.0) == 300  # hold
    assert sizer.target_position(0.53, 0.5, 10_000, 0.0, 0, 0.0) == 0  # no new entry
    assert sizer.target_position(0.53, 0.5, 10_000, 0.0, -300, 0.0) == 0  # wrong side
    assert sizer.target_position(0.5, 0.5, 10_000, 0.0, 300, 0.0) == 0


def test_hold_branch_respects_caps() -> None:
    sizer = PositionSizer(RiskLimits(max_position_per_market=10_000, max_gross_exposure=4_000))
    # Inside the band we would hold 1000, but only $100 of room is left -> 200 contracts.
    assert sizer.target_position(0.53, 0.5, 10_000, 0.0, 1000, 3_900.0) == 200


def test_non_positive_equity_sizes_to_zero() -> None:
    assert PositionSizer(BIG).target_position(0.9, 0.5, -50.0, 0.0, 0, 0.0) == 0


def test_should_trade() -> None:
    sizer = PositionSizer(RiskLimits(min_trade_size=5, rebalance_band=0.25))
    assert not sizer.should_trade(100, 97)  # below min size
    assert not sizer.should_trade(100, 80)  # 20 < 25% of 100
    assert sizer.should_trade(100, 70)
    assert sizer.should_trade(0, 30)  # closing out
    assert sizer.should_trade(-40, 30)  # flipping
    assert sizer.should_trade(40, 0)  # opening
    assert not sizer.should_trade(90, 100)  # small reduction inside the band...
    assert sizer.should_trade(90, 100, over_limit=True)  # ...unless the book is over a limit
    assert not sizer.should_trade(110, 100, over_limit=True)  # increases never forced


def test_clip_to_participation() -> None:
    sizer = PositionSizer(RiskLimits(max_book_participation=0.5))
    assert sizer.clip_to_participation(100, 150) == 75
    assert sizer.clip_to_participation(10, 150) == 10
    assert sizer.clip_to_participation(10, 0) == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_position_per_market": -1},
        {"kelly_fraction": 0.0},
        {"entry_edge": 0.01, "exit_edge": 0.02},
        {"throttle_start": 0.2, "throttle_full": 0.1},
        {"min_throttle": 1.5},
        {"max_book_participation": 0.0},
        {"rebalance_band": 1.0},
        {"min_trade_size": 0},
    ],
)
def test_risk_limits_validation(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        RiskLimits(**kwargs)
