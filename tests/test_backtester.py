"""Event-driven backtester: accounting, risk limits and the drawdown circuit breaker."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

import numpy as np
import pytest

from execution.backtester import (
    BacktestConfig,
    Backtester,
    BacktestResult,
    Event,
    EventKind,
    ExecutionStyle,
    Portfolio,
)
from execution.costs import ExecutionRecord
from execution.sizing import RiskLimits
from market_sim.markets import LiquidityTier, MarketConfig, SyntheticMarket, generate_universe
from market_sim.order_book import Side
from monitoring.monitor import AlertLevel, MonitorConfig
from signals.research import MarketSignal
from tests.helpers import make_market, make_signal


@pytest.fixture(scope="module")
def universe() -> list[SyntheticMarket]:
    return generate_universe(
        MarketConfig(n_markets=8, seed=5, min_length=60, max_length=90, horizon=120)
    )


def informed_signals(markets: list[SyntheticMarket]) -> dict[str, MarketSignal]:
    """A signal that leans toward the true outcome - enough to make trades happen."""
    return {
        m.market_id: make_signal(m, np.clip(m.true_prob * 0.5 + m.outcome * 0.5, 0.02, 0.98))
        for m in markets
    }


def replay_positions(records: list[ExecutionRecord]) -> dict[str, list[int]]:
    path: dict[str, list[int]] = defaultdict(list)
    pos: dict[str, int] = defaultdict(int)
    for r in records:
        pos[r.market_id] += r.signed_qty
        path[r.market_id].append(pos[r.market_id])
    return path


# ------------------------------------------------------------------- portfolio
def test_portfolio_accounting() -> None:
    pf = Portfolio(1000.0)
    buy = ExecutionRecord(0, "A", "deep", "aggressive", Side.BUY, 10, 0.40, 0.39, 0.40, fee=0.5)
    pf.apply(buy)
    assert pf.cash == pytest.approx(1000 - 4.0 - 0.5)
    assert pf.position("A") == 10
    assert pf.equity({"A": 0.50}) == pytest.approx(1000 - 4.5 + 5.0)
    assert pf.exposures({"A": 0.50}) == {"A": pytest.approx(5.0)}
    assert pf.settle("A", 1) == pytest.approx(10 * (1 - 0.40) - 0.5)
    assert pf.position("A") == 0
    short = ExecutionRecord(1, "B", "thin", "aggressive", Side.SELL, 20, 0.70, 0.70, 0.70)
    pf.apply(short)
    assert pf.exposures({"B": 0.70}) == {"B": pytest.approx(20 * 0.30)}
    assert pf.settle("B", 0) == pytest.approx(14.0)
    assert pf.settle("never-traded", 1) == 0.0
    close = ExecutionRecord(2, "C", "deep", "aggressive", Side.BUY, 5, 0.5, 0.5, 0.5)
    pf.apply(close)
    pf.apply(replace(close, side=Side.SELL))
    assert "C" not in pf.positions


def test_event_ordering() -> None:
    evs = sorted(
        [
            Event(3, EventKind.END_OF_STEP, 0),
            Event(3, EventKind.MARKET_DATA, 1, "A"),
            Event(3, EventKind.RESOLUTION, 2, "B"),
            Event(2, EventKind.END_OF_STEP, 3),
        ]
    )
    assert [(e.step, e.kind) for e in evs] == [
        (2, EventKind.END_OF_STEP),
        (3, EventKind.RESOLUTION),
        (3, EventKind.MARKET_DATA),
        (3, EventKind.END_OF_STEP),
    ]


# ---------------------------------------------------------------- invariants
@pytest.mark.parametrize("style", list(ExecutionStyle))
def test_backtest_invariants(universe: list[SyntheticMarket], style: ExecutionStyle) -> None:
    cfg = BacktestConfig(style=style)
    res = Backtester(universe, informed_signals(universe), cfg).run()
    assert len(res.equity) == len(res.steps) == len(res.gross_exposure)
    assert res.records, "the informed signal should trade"
    assert res.final_positions == {}
    # Every dollar of PnL is attributed to a settled market.
    assert sum(res.market_pnl.values()) == pytest.approx(res.total_pnl, abs=1e-6)
    # Per-market position cap is never exceeded.
    for path in replay_positions(res.records).values():
        assert max(abs(p) for p in path) <= cfg.limits.max_position_per_market
    # No new risk in the final steps before resolution.
    by_id = {m.market_id: m for m in universe}
    for r in res.records:
        m = by_id[r.market_id]
        assert m.open_step <= r.step < m.close_step
        if style is ExecutionStyle.AGGRESSIVE:
            assert m.close_step - r.step > cfg.no_new_risk_last_steps
        assert r.markout_mid is not None
    if style is ExecutionStyle.PASSIVE:
        ps = res.passive_stats
        assert 0 < ps.contracts_filled <= ps.contracts_ordered
        assert ps.fully_filled + ps.cancelled == ps.orders
        assert all(r.slippage_cost == 0.0 for r in res.records)
        assert res.cost_summary("passive").spread_cost < 0  # passive fills earn the spread
    else:
        assert res.cost_summary("aggressive").spread_cost > 0
    assert res.market_tier.keys() == by_id.keys()
    assert res.monitor.snapshots[-1].step == res.steps[-1]


def test_gross_exposure_limit_is_respected(universe: list[SyntheticMarket]) -> None:
    lim = RiskLimits(max_gross_exposure=300.0)
    res = Backtester(universe, informed_signals(universe), BacktestConfig(limits=lim)).run()
    # Exposure is checked at decision time at the mid; allow a margin for later price moves.
    assert max(res.gross_exposure) <= 300.0 * 1.25


def test_invert_signal_changes_direction(universe: list[SyntheticMarket]) -> None:
    sigs = informed_signals(universe)
    good = Backtester(universe, sigs, BacktestConfig()).run()
    bad = Backtester(universe, sigs, BacktestConfig(invert_signal=True)).run()
    assert good.total_pnl > 0 > bad.total_pnl


def test_market_impact_raises_costs() -> None:
    m = make_market([0.50] * 40, outcome=1, tier=LiquidityTier.MEDIUM)
    sig = {m.market_id: make_signal(m, 0.80)}
    limits = RiskLimits(rebalance_band=0.0, max_book_participation=0.1)
    no_imp = Backtester([m], sig, BacktestConfig(limits=limits, impact_ticks_per_depth=0.0)).run()
    imp = Backtester([m], sig, BacktestConfig(limits=limits, impact_ticks_per_depth=5.0)).run()
    assert imp.cost_summary().cost_per_contract > no_imp.cost_summary().cost_per_contract


def test_backtester_validation(universe: list[SyntheticMarket]) -> None:
    with pytest.raises(ValueError, match="at least one market"):
        Backtester([], {})
    with pytest.raises(ValueError, match="missing signals"):
        Backtester(universe, {})
    with pytest.raises(ValueError, match="initial_capital"):
        BacktestConfig(initial_capital=0)
    with pytest.raises(ValueError, match="markout"):
        BacktestConfig(markout_horizon=0)
    with pytest.raises(ValueError, match="impact"):
        BacktestConfig(impact_decay=1.0)


def test_empty_result_defaults(universe: list[SyntheticMarket]) -> None:
    res = Backtester(universe, informed_signals(universe)).run()
    empty = BacktestResult(
        res.config, [], [], [], [], {}, {}, res.monitor, [], res.passive_stats, {}
    )
    assert empty.final_equity == res.config.initial_capital
    assert empty.markout_per_contract() == 0.0


# ------------------------------------------------------- circuit breaker (forced)
def crash_markets() -> list[SyntheticMarket]:
    """Three markets whose prices collapse while the signal insists they are cheap."""
    path = list(np.linspace(0.50, 0.10, 60)) + [0.10] * 20
    return [
        make_market(path, outcome=0, market_id="CRASH-A", tier=LiquidityTier.DEEP),
        make_market(path, outcome=0, market_id="CRASH-B", tier=LiquidityTier.MEDIUM),
        make_market(path, outcome=0, market_id="CRASH-C", tier=LiquidityTier.MEDIUM, open_step=2),
    ]


def test_forced_drawdown_trips_breaker_and_flattens_everything() -> None:
    markets = crash_markets()
    sigs = {m.market_id: make_signal(m, 0.90) for m in markets}
    cfg = BacktestConfig(initial_capital=2_000.0, monitor=MonitorConfig(max_drawdown_limit=0.05))
    logs: list[str] = []
    res = Backtester(markets, sigs, cfg, log=logs.append).run()

    # 1. The breaker fired.
    assert len(res.breaker_events) == 1
    ev = res.breaker_events[0]
    assert ev.drawdown >= 0.05
    assert res.monitor.breaker.trips == [ev.step]
    assert any(
        a.rule == "circuit_breaker" and a.level is AlertLevel.CRITICAL for a in res.monitor.alerts
    )
    # 2. It had real positions to flatten, in more than one market.
    assert len(ev.positions_before) >= 2
    assert all(v > 0 for v in ev.positions_before.values())
    # 3. Every position was closed by flatten orders, and nothing remains open.
    assert ev.completed_step is not None
    assert ev.positions_after == {}
    flattened = {r.market_id for r in ev.flatten_records}
    assert flattened == set(ev.positions_before)
    assert all(r.style == "flatten" and r.side is Side.SELL for r in ev.flatten_records)
    assert res.final_positions == {}
    pos_at_flat = replay_positions([r for r in res.records if r.step <= ev.completed_step])
    assert all(path[-1] == 0 for path in pos_at_flat.values())
    # 4. The book stays halted: no strategy trades after the trip.
    assert not [r for r in res.records if r.step > ev.step and r.style != "flatten"]
    assert all(s.halted for s in res.monitor.snapshots if s.step >= ev.step)
    assert any("all positions flat" in line for line in logs)
    # 5. Flattening stopped the bleeding: final loss is far below holding to resolution.
    held_loss = sum(ev.positions_before.values()) * 0.40
    assert -res.total_pnl < held_loss


def test_flatten_spans_multiple_steps_when_book_is_thin() -> None:
    path = list(np.linspace(0.50, 0.20, 80)) + [0.20] * 20
    m = make_market(path, outcome=0, market_id="THIN", tier=LiquidityTier.THIN, top_depth=6.0)
    sigs = {m.market_id: make_signal(m, 0.90)}
    cfg = BacktestConfig(
        initial_capital=300.0,
        monitor=MonitorConfig(max_drawdown_limit=0.05),
        limits=RiskLimits(max_book_participation=1.0, participation_levels=15, rebalance_band=0.0),
    )
    res = Backtester([m], sigs, cfg).run()
    ev = res.breaker_events[0]
    assert ev.positions_before["THIN"] > 0
    assert ev.completed_step is not None
    assert ev.completed_step > ev.step  # needed more than one step of liquidity
    assert len(ev.flatten_records) >= 2
    assert ev.positions_after == {}


def test_breaker_cooldown_allows_trading_to_resume() -> None:
    markets = crash_markets()
    rebound = [0.10] * 40 + list(np.linspace(0.10, 0.60, 40))
    markets.append(make_market(rebound, outcome=1, market_id="LATE", open_step=60))
    sigs = {m.market_id: make_signal(m, 0.90) for m in markets}
    cfg = BacktestConfig(
        initial_capital=2_000.0, monitor=MonitorConfig(max_drawdown_limit=0.05, breaker_cooldown=10)
    )
    res = Backtester(markets, sigs, cfg).run()
    trip = res.breaker_events[0].step
    assert any(
        r.market_id == "LATE" and r.style == "aggressive" and r.step >= trip + 10
        for r in res.records
    )
