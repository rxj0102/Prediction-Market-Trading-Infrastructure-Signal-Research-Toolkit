"""Cost accounting and the aggressive / passive executors."""

from __future__ import annotations

import math

import pytest

from execution.costs import CostSummary, ExecutionRecord
from execution.executors import (
    AggressiveExecutor,
    PassiveExecutor,
    PassiveOrder,
    fill_probability,
    poisson_tail,
)
from market_sim.book_builder import DepthProfile, build_book
from market_sim.order_book import LimitOrderBook, Side


def rec(
    side: Side, avg: float, touch: float, mid: float = 0.50, qty: int = 10, fee: float = 0.0
) -> ExecutionRecord:
    return ExecutionRecord(1, "M", "deep", "aggressive", side, qty, avg, mid, touch, fee)


# ------------------------------------------------------------------------ costs
def test_record_cost_split_buy() -> None:
    r = rec(Side.BUY, avg=0.53, touch=0.52, fee=0.1)
    assert r.signed_qty == 10
    assert r.spread_cost == pytest.approx(0.20)  # (0.52 - 0.50) * 10
    assert r.slippage_cost == pytest.approx(0.10)  # (0.53 - 0.52) * 10
    assert r.total_cost == pytest.approx(0.40)
    assert r.cost_per_contract == pytest.approx(0.04)


def test_record_cost_split_sell_and_passive_rebate() -> None:
    sell = rec(Side.SELL, avg=0.47, touch=0.48)
    assert sell.signed_qty == -10
    assert sell.spread_cost == pytest.approx(0.20)
    assert sell.slippage_cost == pytest.approx(0.10)
    passive_buy = rec(Side.BUY, avg=0.49, touch=0.49)  # resting bid below mid
    assert passive_buy.spread_cost == pytest.approx(-0.10)
    assert passive_buy.slippage_cost == 0.0


def test_markout() -> None:
    r = rec(Side.BUY, avg=0.52, touch=0.52)
    assert r.markout_pnl is None
    assert r.with_markout(0.60).markout_pnl == pytest.approx(0.8)
    assert rec(Side.SELL, 0.48, 0.48).with_markout(0.60).markout_pnl == pytest.approx(-1.2)


def test_cost_summary() -> None:
    s = CostSummary.from_records(
        [rec(Side.BUY, 0.53, 0.52, fee=0.1), rec(Side.SELL, 0.48, 0.48, qty=30)]
    )
    assert s.n_fills == 2
    assert s.contracts == 40
    assert s.notional == pytest.approx(0.53 * 10 + 0.48 * 30)
    assert s.spread_cost == pytest.approx(0.2 + 0.6)
    assert s.slippage_cost == pytest.approx(0.1)
    assert s.total_cost == pytest.approx(1.0)
    assert s.cost_per_contract == pytest.approx(0.025)
    assert s.avg_slippage_per_trade == pytest.approx(0.05)
    assert s.cost_bps_of_notional == pytest.approx(1e4 * 1.0 / s.notional)
    empty = CostSummary.from_records([])
    assert empty.cost_per_contract == 0.0
    assert empty.avg_slippage_per_trade == 0.0
    assert empty.cost_bps_of_notional == 0.0


# -------------------------------------------------------------------- aggressive
def test_aggressive_walks_book() -> None:
    book = build_book(0.50, 1, DepthProfile(top_depth=10, decay=1.0, n_levels=5))
    r = AggressiveExecutor(fee_per_contract=0.01).execute(book, "M", "thin", 3, Side.BUY, 25, 0.50)
    assert r is not None
    assert r.qty == 25
    assert r.touch_price == pytest.approx(0.51)
    assert r.avg_price == pytest.approx((10 * 51 + 10 * 52 + 5 * 53) / 25 / 100)
    assert r.fee == pytest.approx(0.25)
    assert r.slippage_cost > 0
    assert r.style == "aggressive"


def test_aggressive_no_liquidity() -> None:
    ex = AggressiveExecutor()
    assert ex.execute(LimitOrderBook(), "M", "thin", 0, Side.BUY, 5, 0.5) is None
    book = build_book(0.5, 1, DepthProfile(10, 1.0, 2))
    assert ex.execute(book, "M", "thin", 0, Side.BUY, 0, 0.5) is None


# ------------------------------------------------------------ fill probability
def test_poisson_tail_known_values() -> None:
    assert poisson_tail(0, 3.0) == 1.0
    assert poisson_tail(2, 0.0) == 0.0
    assert poisson_tail(1, 2.0) == pytest.approx(1 - math.exp(-2))
    assert poisson_tail(3, 1.0) == pytest.approx(1 - math.exp(-1) * 2.5)


def test_fill_probability_monotone_in_queue() -> None:
    near = fill_probability(0, 10, 20.0, 1)
    far = fill_probability(100, 10, 20.0, 1)
    assert near > 0.99
    assert far < 1e-6
    assert fill_probability(0, 10, 2.0, 5) > fill_probability(0, 10, 2.0, 1)


# ---------------------------------------------------------------------- passive
def book_at(mid: float, half: int = 1, depth: float = 100) -> LimitOrderBook:
    return build_book(mid, half, DepthProfile(top_depth=depth, decay=1.0, n_levels=5))


def test_passive_place_joins_back_of_queue() -> None:
    order = PassiveExecutor().place(book_at(0.50), "M", "deep", 0, Side.BUY, 20, 0.50, 30.0)
    assert order is not None
    assert order.limit_price == 49
    assert order.queue_ahead == 100
    assert 0.0 <= order.expected_fill_prob <= 1.0
    assert order.signed_remaining == 20


def test_passive_place_improves_wide_spread() -> None:
    order = PassiveExecutor(improve_if_spread_ge=3).place(
        book_at(0.50, half=2), "M", "thin", 0, Side.SELL, 20, 0.50, 5.0
    )
    assert order is not None
    assert order.limit_price == 51  # ask 52 improved by one tick
    assert order.queue_ahead == 0
    assert order.signed_remaining == -20


def test_passive_place_needs_two_sided_book() -> None:
    assert PassiveExecutor().place(LimitOrderBook(), "M", "deep", 0, Side.BUY, 5, 0.5, 1) is None


def _order(qty: int = 20, ahead: int = 100) -> PassiveOrder:
    return PassiveOrder("M", "deep", Side.BUY, 49, qty, qty, ahead, 0, 0.50, 0.5)


def test_passive_queue_position_is_respected() -> None:
    ex = PassiveExecutor(max_wait=5)
    order = _order()
    # 110 contracts of sell flow: 100 ahead of us, then 10 of ours.
    r1 = ex.process_step(order, book_at(0.50), 1, opposing_flow=110)
    assert r1 is not None
    assert r1.qty == 10
    assert r1.avg_price == pytest.approx(0.49)
    assert r1.style == "passive"
    assert order.remaining == 10
    assert order.queue_ahead == 0
    # Next step we are at the front of the rebuilt queue.
    r2 = ex.process_step(order, book_at(0.50), 2, opposing_flow=15)
    assert r2 is not None
    assert r2.qty == 10
    assert order.remaining == 0
    assert order.filled == 20


def test_passive_no_flow_no_fill_and_expiry() -> None:
    ex = PassiveExecutor(max_wait=2)
    order = _order()
    assert ex.process_step(order, book_at(0.50), 1, opposing_flow=0) is None
    assert not ex.expired(order)
    assert ex.process_step(order, book_at(0.50), 2, opposing_flow=50) is None  # still queued
    assert order.queue_ahead == 50
    assert ex.expired(order)


def test_passive_price_through_fills_at_limit() -> None:
    order = _order(ahead=500)
    r = PassiveExecutor().process_step(order, book_at(0.46), 1, opposing_flow=0)
    assert r is not None
    assert r.qty == 20
    assert r.avg_price == pytest.approx(0.49)
    assert order.remaining == 0


def test_passive_behind_better_prices() -> None:
    # Market moved up: best bid is now 0.53, our bid at 0.49 is four levels deep.
    order = _order(ahead=0)
    r = PassiveExecutor().process_step(order, book_at(0.54), 1, opposing_flow=150)
    assert r is None  # 150 contracts only consume the better-priced levels


def test_passive_validation() -> None:
    with pytest.raises(ValueError, match="max_wait"):
        PassiveExecutor(max_wait=0)
