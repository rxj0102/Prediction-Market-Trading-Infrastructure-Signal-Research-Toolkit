"""Metrics, the tick-by-tick risk monitor and the circuit breaker (unit level)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from monitoring.metrics import drawdown_series, max_drawdown, sharpe_ratio, simple_returns
from monitoring.monitor import (
    Action,
    AlertLevel,
    CircuitBreaker,
    MonitorConfig,
    RiskMonitor,
)


# ---------------------------------------------------------------------- metrics
def test_simple_returns() -> None:
    np.testing.assert_allclose(simple_returns([100, 110, 99]), [0.10, -0.10])
    assert len(simple_returns([100])) == 0
    with pytest.raises(ValueError, match="positive"):
        simple_returns([100, 0, 50])


def test_sharpe_hand_computed() -> None:
    r = [0.01, -0.01, 0.02]
    mean, sd = 0.02 / 3, float(np.std(r, ddof=1))
    assert sharpe_ratio(r, 365) == pytest.approx(mean / sd * math.sqrt(365))
    assert math.isnan(sharpe_ratio([0.01], 365))
    assert sharpe_ratio([0.01, 0.01, 0.01], 365) == 0.0


def test_drawdown_series_hand_computed() -> None:
    np.testing.assert_allclose(drawdown_series([100, 110, 99, 120, 90]), [0, 0, 0.1, 0, 0.25])
    assert max_drawdown([100, 110, 99, 120, 90]) == pytest.approx(0.25)
    assert max_drawdown([]) == 0.0
    assert len(drawdown_series([])) == 0


# ------------------------------------------------------------- circuit breaker
def test_breaker_trips_once_and_halts_permanently() -> None:
    cb = CircuitBreaker(max_drawdown=0.10)
    assert not cb.update(0, 100.0)
    assert not cb.update(1, 120.0)
    assert not cb.update(2, 109.0)  # 9.2% below the 120 peak
    assert cb.update(3, 107.0)  # 10.8% -> trip
    assert cb.trips == [3]
    assert cb.is_halted(4)
    assert not cb.update(4, 50.0)  # already halted: no second trip
    assert cb.is_halted(10_000)


def test_breaker_cooldown_resumes_and_rebases_peak() -> None:
    cb = CircuitBreaker(max_drawdown=0.10, cooldown_steps=5)
    cb.update(0, 100.0)
    assert cb.update(1, 85.0)
    assert cb.is_halted(3)
    assert not cb.update(3, 80.0)  # still cooling down
    assert not cb.is_halted(6)
    assert not cb.update(6, 80.0)  # resumes with peak rebased to 80
    assert cb.drawdown == 0.0
    assert cb.update(7, 71.0)  # 11.25% below the new peak
    assert cb.trips == [1, 7]


def test_breaker_validation() -> None:
    with pytest.raises(ValueError, match="max_drawdown"):
        CircuitBreaker(0.0)


# ------------------------------------------------------------------- monitor
def test_monitor_tracks_rolling_metrics_and_exposure() -> None:
    mon = RiskMonitor(MonitorConfig(window=3), initial_equity=100.0)
    mon.on_tick(0, 101.0, {"A": 10.0})
    mon.on_tick(1, 103.0, {"A": 10.0, "B": 30.0})
    res = mon.on_tick(2, 102.0, {"A": 5.0, "B": 0.0})
    s = res.snapshot
    assert s.pnl == pytest.approx(2.0)
    assert s.rolling_pnl == pytest.approx(2.0)  # window of 3 returns starts at 100
    assert s.gross_exposure == pytest.approx(5.0)
    assert s.n_positions == 1
    assert s.largest_market == "A"
    assert s.drawdown == pytest.approx(1 - 102 / 103)
    assert mon.max_drawdown == pytest.approx(1 - 102 / 103)
    assert not math.isnan(s.rolling_sharpe)
    mon.on_tick(3, 104.0, {})
    assert mon.snapshots[-1].rolling_pnl == pytest.approx(3.0)  # 104 - 101
    assert mon.snapshots[-1].largest_market is None
    assert len(mon.exposure_history) == 4
    assert res.action is Action.NONE
    assert "step    2" in mon.format_snapshot(s)


def test_monitor_drawdown_warning_fires_on_crossing_and_rearms() -> None:
    mon = RiskMonitor(MonitorConfig(max_drawdown_limit=0.2, drawdown_warn_fraction=0.5), 100.0)
    assert mon.on_tick(0, 95.0, {}).alerts == ()
    first = mon.on_tick(1, 89.0, {}).alerts  # 11% >= 10% warning level
    assert [a.rule for a in first] == ["drawdown_warning"]
    assert first[0].level is AlertLevel.WARNING
    assert mon.on_tick(2, 88.0, {}).alerts == ()  # still in warning: no repeat
    mon.on_tick(3, 100.0, {})  # recovered: re-arm
    assert [a.rule for a in mon.on_tick(4, 85.0, {}).alerts] == ["drawdown_warning"]


def test_monitor_exposure_warning() -> None:
    mon = RiskMonitor(MonitorConfig(exposure_limit=1000.0, exposure_warn_fraction=0.9), 100.0)
    assert mon.on_tick(0, 100.0, {"A": 500.0}).alerts == ()
    alerts = mon.on_tick(1, 100.0, {"A": 950.0}).alerts
    assert [a.rule for a in alerts] == ["exposure_warning"]
    assert mon.on_tick(2, 100.0, {"A": 950.0}).alerts == ()
    assert mon.on_tick(3, 100.0, {"A": 100.0}).alerts == ()
    assert [a.rule for a in mon.on_tick(4, 100.0, {"A": 990.0}).alerts] == ["exposure_warning"]


def test_monitor_breaker_returns_flatten_action() -> None:
    mon = RiskMonitor(MonitorConfig(max_drawdown_limit=0.10), initial_equity=1000.0)
    mon.on_tick(0, 1000.0, {"A": 300.0})
    res = mon.on_tick(1, 880.0, {"A": 300.0})
    assert res.action is Action.FLATTEN_ALL
    assert any(a.rule == "circuit_breaker" and a.level is AlertLevel.CRITICAL for a in res.alerts)
    assert res.snapshot.halted
    assert mon.is_halted(2)
    assert "HALTED" in mon.format_snapshot(res.snapshot)
    assert mon.on_tick(2, 700.0, {}).action is Action.NONE  # does not re-fire


@pytest.mark.parametrize(
    "kwargs", [{"window": 1}, {"max_drawdown_limit": 1.0}, {"breaker_cooldown": 0}]
)
def test_monitor_config_validation(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        MonitorConfig(**kwargs)


def test_monitor_requires_positive_equity() -> None:
    with pytest.raises(ValueError, match="initial_equity"):
        RiskMonitor(MonitorConfig(), 0.0)
