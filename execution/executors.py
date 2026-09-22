"""Execution styles that trade against the Module 1 order book.

Aggressive
    A marketable order sent straight into the book. It pays the half spread
    and walks deeper levels if the order is larger than the touch
    (depth-dependent slippage). It fills immediately, up to the visible depth.

Passive
    A limit order that joins the best price on our side (or improves it by one
    tick when the spread is wide). Fills are simulated each subsequent step:

    * The book is rebuilt from the market's next snapshot and our order is
      re-inserted with the queue position it had earned (price-time priority:
      only size that arrived before us is ahead).
    * That step's synthetic taker flow on the opposite side is sent into the
      book as a market order; our order fills only after the resting size
      ahead of it has been consumed.
    * If the market moves through our limit (the opposite touch reaches our
      price), the remainder fills at our limit. This is where adverse
      selection shows up: fills are more likely just before the price moves
      against us.
    * Orders not complete after ``max_wait`` steps are cancelled.

    An ex-ante fill probability is computed at placement from a Poisson model
    of opposing flow, so the model can be checked against realised fill rates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from execution.costs import ExecutionRecord
from market_sim.order_book import LimitOrderBook, Side, ticks_to_price

STRATEGY = "strategy"
TAKER = "taker"


def poisson_tail(k: int, lam: float) -> float:
    """P(N >= k) for N ~ Poisson(lam)."""
    if k <= 0:
        return 1.0
    if lam <= 0:
        return 0.0
    term = math.exp(-lam)
    cdf = term
    for i in range(1, k):
        term *= lam / i
        cdf += term
    return max(0.0, 1.0 - cdf)


def fill_probability(queue_ahead: int, qty: int, flow_per_step: float, horizon: int) -> float:
    """Probability that opposing flow over ``horizon`` steps clears the queue and our order.

    Assumes Poisson taker arrivals at the touch and a static queue; ignores
    price moving through the order (which in practice adds fills, mostly
    adverse ones).
    """
    return poisson_tail(queue_ahead + qty, flow_per_step * horizon)


class AggressiveExecutor:
    style = "aggressive"

    def __init__(self, fee_per_contract: float = 0.0) -> None:
        self.fee_per_contract = fee_per_contract

    def execute(
        self,
        book: LimitOrderBook,
        market_id: str,
        tier: str,
        step: int,
        side: Side,
        qty: int,
        decision_mid: float,
        style: str | None = None,
    ) -> ExecutionRecord | None:
        touch = book.best_price(side.opposite)
        if touch is None or qty <= 0:
            return None
        res = book.submit_market(side, qty, owner=STRATEGY)
        if res.filled_qty == 0 or res.avg_price is None:  # pragma: no cover - touch exists
            return None
        return ExecutionRecord(
            step=step,
            market_id=market_id,
            tier=tier,
            style=style or self.style,
            side=side,
            qty=res.filled_qty,
            avg_price=res.avg_price,
            decision_mid=decision_mid,
            touch_price=ticks_to_price(touch),
            fee=self.fee_per_contract * res.filled_qty,
        )


@dataclass
class PassiveOrder:
    market_id: str
    tier: str
    side: Side
    limit_price: int  # ticks
    original_qty: int
    remaining: int
    queue_ahead: int
    placed_step: int
    decision_mid: float
    expected_fill_prob: float
    age: int = 0

    @property
    def filled(self) -> int:
        return self.original_qty - self.remaining

    @property
    def signed_remaining(self) -> int:
        return self.side.value * self.remaining


class PassiveExecutor:
    style = "passive"

    def __init__(
        self, fee_per_contract: float = 0.0, max_wait: int = 5, improve_if_spread_ge: int = 3
    ) -> None:
        if max_wait < 1:
            raise ValueError("max_wait must be >= 1")
        self.fee_per_contract = fee_per_contract
        self.max_wait = max_wait
        self.improve_if_spread_ge = improve_if_spread_ge

    def place(
        self,
        book: LimitOrderBook,
        market_id: str,
        tier: str,
        step: int,
        side: Side,
        qty: int,
        decision_mid: float,
        flow_per_step: float,
    ) -> PassiveOrder | None:
        same, opp = book.best_price(side), book.best_price(side.opposite)
        if same is None or opp is None or qty <= 0:
            return None
        if abs(opp - same) >= self.improve_if_spread_ge:
            price, ahead = same + side.value, 0  # step inside the spread: new level, first in queue
        else:
            price, ahead = same, book.level_qty(side, same)  # join the back of the touch queue
        return PassiveOrder(
            market_id=market_id,
            tier=tier,
            side=side,
            limit_price=price,
            original_qty=qty,
            remaining=qty,
            queue_ahead=ahead,
            placed_step=step,
            decision_mid=decision_mid,
            expected_fill_prob=fill_probability(ahead, qty, flow_per_step, self.max_wait),
        )

    def process_step(
        self, order: PassiveOrder, book: LimitOrderBook, step: int, opposing_flow: int
    ) -> ExecutionRecord | None:
        """Advance a resting order by one step against ``book`` and taker flow."""
        order.age += 1
        opp = book.best_price(order.side.opposite)
        traded_through = opp is not None and (
            (order.side is Side.BUY and opp <= order.limit_price)
            or (order.side is Side.SELL and opp >= order.limit_price)
        )
        if traded_through:
            fill_qty = order.remaining
            order.queue_ahead = 0
        else:
            ahead = min(order.queue_ahead, book.level_qty(order.side, order.limit_price))
            oid = book.insert_at_queue_position(
                order.side, order.limit_price, order.remaining, ahead, owner=STRATEGY
            )
            fill_qty = 0
            if opposing_flow > 0:
                res = book.submit_market(order.side.opposite, opposing_flow, owner=TAKER)
                fill_qty = sum(f.qty for f in res.fills if f.maker_order_id == oid)
            order.queue_ahead = book.queue_ahead(oid) if book.get_order(oid) else 0
        if fill_qty == 0:
            return None
        order.remaining -= fill_qty
        px = ticks_to_price(order.limit_price)
        return ExecutionRecord(
            step=step,
            market_id=order.market_id,
            tier=order.tier,
            style=self.style,
            side=order.side,
            qty=fill_qty,
            avg_price=px,
            decision_mid=order.decision_mid,
            touch_price=px,
            fee=self.fee_per_contract * fill_qty,
        )

    def expired(self, order: PassiveOrder) -> bool:
        return order.age >= self.max_wait
