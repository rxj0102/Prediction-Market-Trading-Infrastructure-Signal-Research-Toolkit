"""Price-time priority limit order book for a binary-outcome event contract.

A binary contract pays $1.00 if the event resolves YES and $0.00 otherwise, so
every valid price lies strictly between the two. Prices are stored as integer
ticks (1 tick = $0.01, valid range 1..99) to avoid floating point comparisons in
the matching logic.

Matching rules:
  * Best price first; within a price level, earliest order first (FIFO).
  * Trades execute at the resting (maker) order's price.
  * Limit orders that cross the book match immediately; any remainder rests.
  * Market orders walk the book until filled or the opposite side is empty;
    any unfilled remainder is cancelled (never rests).
"""

from __future__ import annotations

import bisect
import itertools
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

MIN_TICK = 1
MAX_TICK = 99
TICK_SIZE = 0.01


class Side(Enum):
    """Order side. ``value`` is the sign applied to a position change."""

    BUY = 1
    SELL = -1

    @property
    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY


def ticks_to_price(ticks: int) -> float:
    """Convert integer ticks to a dollar price."""
    return ticks * TICK_SIZE


def price_to_ticks(price: float) -> int:
    """Convert a dollar price to the nearest integer tick."""
    return round(price / TICK_SIZE)


@dataclass
class Order:
    """A resting or incoming order. ``qty`` is the *remaining* quantity."""

    order_id: int
    side: Side
    price: int | None  # None for market orders
    qty: int
    timestamp: int
    owner: str
    original_qty: int = field(init=False)

    def __post_init__(self) -> None:
        self.original_qty = self.qty


@dataclass(frozen=True)
class Fill:
    """A single execution between an aggressing (taker) and resting (maker) order."""

    taker_order_id: int
    maker_order_id: int
    taker_owner: str
    maker_owner: str
    side: Side  # side of the aggressor
    price: int  # ticks
    qty: int

    @property
    def price_dollars(self) -> float:
        return ticks_to_price(self.price)


@dataclass(frozen=True)
class OrderResult:
    """Outcome of submitting an order."""

    order_id: int
    fills: tuple[Fill, ...]
    filled_qty: int
    remaining_qty: int
    resting: bool

    @property
    def avg_price(self) -> float | None:
        """Volume-weighted average fill price in dollars (None if unfilled)."""
        if self.filled_qty == 0:
            return None
        notional = sum(f.price * f.qty for f in self.fills)
        return ticks_to_price(1) * notional / self.filled_qty


class LimitOrderBook:
    """Single-instrument limit order book with price-time priority."""

    def __init__(self) -> None:
        self._levels: dict[Side, dict[int, deque[Order]]] = {Side.BUY: {}, Side.SELL: {}}
        # Ascending sorted price lists per side.
        self._prices: dict[Side, list[int]] = {Side.BUY: [], Side.SELL: []}
        self._orders: dict[int, Order] = {}
        self._ids = itertools.count(1)
        self._clock = itertools.count()

    # ------------------------------------------------------------------ queries
    def best_bid(self) -> int | None:
        prices = self._prices[Side.BUY]
        return prices[-1] if prices else None

    def best_ask(self) -> int | None:
        prices = self._prices[Side.SELL]
        return prices[0] if prices else None

    def best_price(self, side: Side) -> int | None:
        """Best price on ``side`` of the book (bid for BUY, ask for SELL)."""
        return self.best_bid() if side is Side.BUY else self.best_ask()

    def mid(self) -> float | None:
        bid, ask = self.best_bid(), self.best_ask()
        if bid is None or ask is None:
            return None
        return ticks_to_price(bid + ask) / 2.0

    def spread_ticks(self) -> int | None:
        bid, ask = self.best_bid(), self.best_ask()
        if bid is None or ask is None:
            return None
        return ask - bid

    def depth(self, side: Side, levels: int | None = None) -> list[tuple[int, int]]:
        """Aggregated (price, qty) levels on ``side``, best price first."""
        prices = self._prices[side]
        ordered = reversed(prices) if side is Side.BUY else iter(prices)
        out: list[tuple[int, int]] = []
        for price in ordered:
            if levels is not None and len(out) >= levels:
                break
            out.append((price, sum(o.qty for o in self._levels[side][price])))
        return out

    def total_depth(self, side: Side, levels: int | None = None) -> int:
        return sum(qty for _, qty in self.depth(side, levels))

    def level_qty(self, side: Side, price: int) -> int:
        return sum(o.qty for o in self._levels[side].get(price, ()))

    def get_order(self, order_id: int) -> Order | None:
        return self._orders.get(order_id)

    def queue_ahead(self, order_id: int) -> int:
        """Quantity resting ahead of ``order_id`` at its price level."""
        order = self._orders.get(order_id)
        if order is None or order.price is None:
            raise KeyError(f"order {order_id} is not resting in the book")
        ahead = 0
        for resting in self._levels[order.side][order.price]:
            if resting.order_id == order_id:
                return ahead
            ahead += resting.qty
        raise KeyError(f"order {order_id} missing from its level")  # pragma: no cover

    def __len__(self) -> int:
        return len(self._orders)

    # --------------------------------------------------------------- mutations
    def submit_limit(self, side: Side, price: int, qty: int, owner: str = "anon") -> OrderResult:
        """Submit a limit order; crosses the book first, then rests any remainder."""
        _validate_price(price)
        _validate_qty(qty)
        order = Order(next(self._ids), side, price, qty, next(self._clock), owner)
        fills = self._match(order, limit=price)
        resting = order.qty > 0
        if resting:
            self._rest(order)
        return OrderResult(
            order.order_id, tuple(fills), order.original_qty - order.qty, order.qty, resting
        )

    def submit_market(self, side: Side, qty: int, owner: str = "anon") -> OrderResult:
        """Submit a market order; unfilled remainder is cancelled."""
        _validate_qty(qty)
        order = Order(next(self._ids), side, None, qty, next(self._clock), owner)
        fills = self._match(order, limit=None)
        return OrderResult(
            order.order_id, tuple(fills), order.original_qty - order.qty, order.qty, False
        )

    def insert_at_queue_position(
        self, side: Side, price: int, qty: int, qty_ahead: int, owner: str = "anon"
    ) -> int:
        """Rest a non-crossing order with exactly ``qty_ahead`` contracts ahead of it.

        Used to carry a passive order's queue position across book rebuilds. If
        a resting order straddles the requested position it is split in two, and
        both halves keep their original timestamp. ``qty_ahead`` is capped at the
        level's current size. Returns the new order id.
        """
        _validate_price(price)
        _validate_qty(qty)
        if qty_ahead < 0:
            raise ValueError("qty_ahead must be non-negative")
        opposite = self.best_price(side.opposite)
        if opposite is not None and (
            (side is Side.BUY and price >= opposite) or (side is Side.SELL and price <= opposite)
        ):
            raise ValueError("insert_at_queue_position requires a non-crossing price")
        order = Order(next(self._ids), side, price, qty, next(self._clock), owner)
        if price not in self._levels[side]:
            self._rest(order)
            return order.order_id
        level = self._levels[side][price]
        new_level: deque[Order] = deque()
        remaining_ahead = qty_ahead
        inserted = False
        for resting in level:
            if inserted:
                new_level.append(resting)
            elif resting.qty <= remaining_ahead:
                new_level.append(resting)
                remaining_ahead -= resting.qty
            else:
                if remaining_ahead > 0:
                    front = Order(
                        next(self._ids),
                        resting.side,
                        resting.price,
                        remaining_ahead,
                        resting.timestamp,
                        resting.owner,
                    )
                    resting.qty -= remaining_ahead
                    resting.original_qty = resting.qty
                    self._orders[front.order_id] = front
                    new_level.append(front)
                new_level.append(order)
                new_level.append(resting)
                inserted = True
        if not inserted:
            new_level.append(order)
        self._levels[side][price] = new_level
        self._orders[order.order_id] = order
        return order.order_id

    def cancel(self, order_id: int) -> bool:
        """Cancel a resting order. Returns False if it is not in the book."""
        order = self._orders.pop(order_id, None)
        if order is None or order.price is None:
            return False
        level = self._levels[order.side][order.price]
        level.remove(order)
        if not level:
            self._remove_level(order.side, order.price)
        return True

    # ---------------------------------------------------------------- internals
    def _rest(self, order: Order) -> None:
        assert order.price is not None
        levels = self._levels[order.side]
        if order.price not in levels:
            levels[order.price] = deque()
            bisect.insort(self._prices[order.side], order.price)
        levels[order.price].append(order)
        self._orders[order.order_id] = order

    def _remove_level(self, side: Side, price: int) -> None:
        del self._levels[side][price]
        prices = self._prices[side]
        prices.pop(bisect.bisect_left(prices, price))

    def _match(self, taker: Order, limit: int | None) -> list[Fill]:
        book_side = taker.side.opposite
        fills: list[Fill] = []
        while taker.qty > 0:
            best = self.best_price(book_side)
            if best is None:
                break
            if limit is not None and (
                (taker.side is Side.BUY and best > limit)
                or (taker.side is Side.SELL and best < limit)
            ):
                break
            level = self._levels[book_side][best]
            while taker.qty > 0 and level:
                maker = level[0]
                traded = min(taker.qty, maker.qty)
                taker.qty -= traded
                maker.qty -= traded
                fills.append(
                    Fill(
                        taker.order_id,
                        maker.order_id,
                        taker.owner,
                        maker.owner,
                        taker.side,
                        best,
                        traded,
                    )
                )
                if maker.qty == 0:
                    level.popleft()
                    del self._orders[maker.order_id]
            if not level:
                self._remove_level(book_side, best)
        return fills


def _validate_price(price: int) -> None:
    if not isinstance(price, int) or not MIN_TICK <= price <= MAX_TICK:
        raise ValueError(f"price must be an integer tick in [{MIN_TICK}, {MAX_TICK}], got {price}")


def _validate_qty(qty: int) -> None:
    if not isinstance(qty, int) or qty <= 0:
        raise ValueError(f"quantity must be a positive integer, got {qty}")
