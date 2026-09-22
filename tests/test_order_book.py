"""Unit tests for the price-time priority limit order book."""

from __future__ import annotations

import pytest

from market_sim.order_book import (
    LimitOrderBook,
    OrderResult,
    Side,
    price_to_ticks,
    ticks_to_price,
)


@pytest.fixture
def book() -> LimitOrderBook:
    b = LimitOrderBook()
    b.submit_limit(Side.BUY, 48, 10, owner="b1")
    b.submit_limit(Side.BUY, 47, 20, owner="b2")
    b.submit_limit(Side.SELL, 52, 10, owner="s1")
    b.submit_limit(Side.SELL, 53, 20, owner="s2")
    return b


def test_side_opposite() -> None:
    assert Side.BUY.opposite is Side.SELL
    assert Side.SELL.opposite is Side.BUY


def test_tick_conversions() -> None:
    assert ticks_to_price(37) == pytest.approx(0.37)
    assert price_to_ticks(0.37) == 37


def test_empty_book_queries() -> None:
    b = LimitOrderBook()
    assert b.best_bid() is None
    assert b.best_ask() is None
    assert b.mid() is None
    assert b.spread_ticks() is None
    assert b.depth(Side.BUY) == []
    assert len(b) == 0


def test_top_of_book(book: LimitOrderBook) -> None:
    assert book.best_bid() == 48
    assert book.best_ask() == 52
    assert book.mid() == pytest.approx(0.50)
    assert book.spread_ticks() == 4
    assert book.depth(Side.BUY) == [(48, 10), (47, 20)]
    assert book.depth(Side.SELL, levels=1) == [(52, 10)]
    assert book.total_depth(Side.SELL) == 30
    assert book.level_qty(Side.BUY, 47) == 20
    assert book.level_qty(Side.BUY, 10) == 0
    assert len(book) == 4


def test_market_buy_single_level_full_fill(book: LimitOrderBook) -> None:
    res = book.submit_market(Side.BUY, 4, owner="t")
    assert res.filled_qty == 4
    assert res.remaining_qty == 0
    assert not res.resting
    assert res.avg_price == pytest.approx(0.52)
    assert book.level_qty(Side.SELL, 52) == 6


def test_market_order_walks_book_with_slippage(book: LimitOrderBook) -> None:
    res = book.submit_market(Side.BUY, 25)
    assert [(f.price, f.qty) for f in res.fills] == [(52, 10), (53, 15)]
    # VWAP = (10*52 + 15*53) / 25 = 52.6 ticks
    assert res.avg_price == pytest.approx(0.526)
    assert book.best_ask() == 53
    assert book.level_qty(Side.SELL, 53) == 5


def test_market_order_partial_fill_when_book_exhausted(book: LimitOrderBook) -> None:
    res = book.submit_market(Side.SELL, 50)
    assert res.filled_qty == 30
    assert res.remaining_qty == 20
    assert not res.resting
    assert book.best_bid() is None


def test_market_order_on_empty_side() -> None:
    res = LimitOrderBook().submit_market(Side.BUY, 5)
    assert res.filled_qty == 0
    assert res.avg_price is None


def test_price_time_priority_fifo_within_level() -> None:
    b = LimitOrderBook()
    first = b.submit_limit(Side.SELL, 60, 5, owner="first")
    second = b.submit_limit(Side.SELL, 60, 5, owner="second")
    better = b.submit_limit(Side.SELL, 59, 3, owner="better")
    res = b.submit_market(Side.BUY, 10)
    owners = [(f.maker_owner, f.price, f.qty) for f in res.fills]
    # Best price first, then FIFO at 60.
    assert owners == [("better", 59, 3), ("first", 60, 5), ("second", 60, 2)]
    assert b.get_order(first.order_id) is None
    assert b.get_order(better.order_id) is None
    remaining = b.get_order(second.order_id)
    assert remaining is not None
    assert remaining.qty == 3
    assert remaining.original_qty == 5


def test_crossing_limit_fills_then_rests_remainder(book: LimitOrderBook) -> None:
    res = book.submit_limit(Side.BUY, 52, 15, owner="x")
    assert res.filled_qty == 10
    assert res.remaining_qty == 5
    assert res.resting
    assert book.best_bid() == 52
    assert book.best_ask() == 53
    order = book.get_order(res.order_id)
    assert order is not None
    assert order.qty == 5


def test_limit_does_not_trade_through_its_price(book: LimitOrderBook) -> None:
    res = book.submit_limit(Side.SELL, 48, 25)
    # Only the 48 bid is at-or-above the limit; the 47 bid is not touched.
    assert res.filled_qty == 10
    assert all(f.price == 48 for f in res.fills)
    assert book.best_ask() == 48
    assert book.level_qty(Side.BUY, 47) == 20


def test_non_crossing_limit_rests_without_fills(book: LimitOrderBook) -> None:
    res = book.submit_limit(Side.BUY, 50, 7)
    assert res.fills == ()
    assert res.resting
    assert book.best_bid() == 50


def test_cancel(book: LimitOrderBook) -> None:
    res = book.submit_limit(Side.BUY, 49, 3)
    assert book.cancel(res.order_id)
    assert book.best_bid() == 48
    assert not book.cancel(res.order_id)
    assert not book.cancel(999_999)


def test_cancel_keeps_level_when_other_orders_remain() -> None:
    b = LimitOrderBook()
    a = b.submit_limit(Side.BUY, 40, 2)
    b.submit_limit(Side.BUY, 40, 3)
    assert b.cancel(a.order_id)
    assert b.level_qty(Side.BUY, 40) == 3


def test_fill_attribution_and_dollar_price(book: LimitOrderBook) -> None:
    res = book.submit_market(Side.SELL, 1, owner="taker")
    fill = res.fills[0]
    assert fill.taker_owner == "taker"
    assert fill.maker_owner == "b1"
    assert fill.side is Side.SELL
    assert fill.price_dollars == pytest.approx(0.48)
    assert fill.taker_order_id == res.order_id


@pytest.mark.parametrize("price", [0, 100, -3])
def test_rejects_out_of_range_price(price: int) -> None:
    with pytest.raises(ValueError, match="price"):
        LimitOrderBook().submit_limit(Side.BUY, price, 1)


@pytest.mark.parametrize("qty", [0, -1])
def test_rejects_non_positive_qty(qty: int) -> None:
    with pytest.raises(ValueError, match="quantity"):
        LimitOrderBook().submit_market(Side.BUY, qty)


def test_queue_ahead_and_insert_at_queue_position() -> None:
    b = LimitOrderBook()
    b.submit_limit(Side.BUY, 45, 10, owner="mm")
    b.submit_limit(Side.BUY, 45, 10, owner="mm")
    b.submit_limit(Side.SELL, 50, 10, owner="mm")
    oid = b.insert_at_queue_position(Side.BUY, 45, 4, qty_ahead=15, owner="me")
    assert b.queue_ahead(oid) == 15
    assert b.level_qty(Side.BUY, 45) == 24
    # A 17-lot sell consumes the 15 ahead, then 2 of ours.
    res = b.submit_market(Side.SELL, 17)
    mine = [f for f in res.fills if f.maker_owner == "me"]
    assert sum(f.qty for f in mine) == 2
    assert b.queue_ahead(oid) == 0


def test_insert_at_queue_position_edges() -> None:
    b = LimitOrderBook()
    b.submit_limit(Side.SELL, 60, 5)
    b.submit_limit(Side.BUY, 40, 5)
    # New level: we are alone.
    oid = b.insert_at_queue_position(Side.SELL, 58, 3, qty_ahead=100)
    assert b.queue_ahead(oid) == 0
    # qty_ahead larger than the level: goes to the back.
    back = b.insert_at_queue_position(Side.BUY, 40, 2, qty_ahead=999)
    assert b.queue_ahead(back) == 5
    # qty_ahead == 0: front of the queue.
    front = b.insert_at_queue_position(Side.BUY, 40, 2, qty_ahead=0)
    assert b.queue_ahead(front) == 0
    # Exactly on an order boundary.
    boundary = b.insert_at_queue_position(Side.BUY, 40, 1, qty_ahead=2)
    assert b.queue_ahead(boundary) == 2
    with pytest.raises(ValueError, match="non-crossing"):
        b.insert_at_queue_position(Side.BUY, 58, 1, qty_ahead=0)
    with pytest.raises(ValueError, match="non-crossing"):
        b.insert_at_queue_position(Side.SELL, 40, 1, qty_ahead=0)
    with pytest.raises(ValueError, match="qty_ahead"):
        b.insert_at_queue_position(Side.BUY, 39, 1, qty_ahead=-1)
    with pytest.raises(KeyError):
        b.queue_ahead(123_456)


def test_order_result_avg_price_none_when_unfilled() -> None:
    assert OrderResult(1, (), 0, 5, True).avg_price is None
