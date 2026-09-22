"""Position sizing under risk limits.

A YES contract bought at price ``c`` risks ``c`` to win ``1 - c``; selling YES
at ``c`` risks ``1 - c`` to win ``c``. Worst-case loss is therefore the
natural exposure measure for a binary contract and is what every limit below
is expressed in.

Sizing steps for a target position:
1. Hysteresis on the edge ``prob - price``: open/flip only if ``|edge| >=
   entry_edge``; keep an existing same-direction position while ``|edge| >=
   exit_edge``; otherwise go flat.
2. Fractional Kelly for a binary bet: ``f = (q - c) / (1 - c)`` long,
   ``f = (c - q) / c`` short, scaled by ``kelly_fraction``.
3. Drawdown throttle: the Kelly fraction is scaled down linearly from 1 at
   ``throttle_start`` drawdown to ``min_throttle`` at ``throttle_full``.
4. Hard caps: ``max_position_per_market`` contracts and the portfolio's
   ``max_gross_exposure`` in worst-case dollars.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RiskLimits:
    max_position_per_market: int = 400  # contracts
    max_gross_exposure: float = 4_000.0  # $ worst-case loss across all markets
    kelly_fraction: float = 0.25
    entry_edge: float = 0.05
    exit_edge: float = 0.01
    throttle_start: float = 0.04  # drawdown at which sizing starts to shrink
    throttle_full: float = 0.10  # drawdown at which sizing reaches min_throttle
    min_throttle: float = 0.25
    max_book_participation: float = 0.5  # max share of visible opposite depth per order
    participation_levels: int = 3
    min_trade_size: int = 5
    rebalance_band: float = 0.25  # skip trades smaller than this fraction of the target

    def __post_init__(self) -> None:
        if self.max_position_per_market < 0 or self.max_gross_exposure < 0:
            raise ValueError("position and exposure caps must be non-negative")
        if not 0.0 < self.kelly_fraction <= 1.0:
            raise ValueError("kelly_fraction must be in (0, 1]")
        if not 0.0 <= self.exit_edge <= self.entry_edge:
            raise ValueError("require 0 <= exit_edge <= entry_edge")
        if not 0.0 <= self.throttle_start < self.throttle_full:
            raise ValueError("require 0 <= throttle_start < throttle_full")
        if not 0.0 <= self.min_throttle <= 1.0:
            raise ValueError("min_throttle must be in [0, 1]")
        if not 0.0 < self.max_book_participation <= 1.0:
            raise ValueError("max_book_participation must be in (0, 1]")
        if not 0.0 <= self.rebalance_band < 1.0:
            raise ValueError("rebalance_band must be in [0, 1)")
        if self.min_trade_size < 1 or self.participation_levels < 1:
            raise ValueError("min_trade_size and participation_levels must be >= 1")


def worst_case_exposure(position: int | float, price: float) -> float:
    """Dollars lost if the contract resolves against ``position`` (marked at ``price``)."""
    return position * price if position >= 0 else -position * (1.0 - price)


def kelly_fraction_binary(prob: float, price: float) -> float:
    """Signed full-Kelly fraction of capital for a binary contract at ``price``."""
    if not 0.0 < price < 1.0:
        raise ValueError("price must be in (0, 1)")
    if prob > price:
        return (prob - price) / (1.0 - price)
    if prob < price:
        return -(price - prob) / price
    return 0.0


def drawdown_throttle(drawdown: float, limits: RiskLimits) -> float:
    """Sizing multiplier in [min_throttle, 1] as a function of current drawdown."""
    if drawdown <= limits.throttle_start:
        return 1.0
    if drawdown >= limits.throttle_full:
        return limits.min_throttle
    frac = (drawdown - limits.throttle_start) / (limits.throttle_full - limits.throttle_start)
    return 1.0 - frac * (1.0 - limits.min_throttle)


class PositionSizer:
    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()

    def target_position(
        self,
        prob: float,
        price: float,
        equity: float,
        drawdown: float,
        current_position: int,
        other_gross_exposure: float,
    ) -> int:
        """Target signed contract position for one market.

        ``other_gross_exposure`` is the worst-case exposure of every *other*
        market, so the gross cap applies to the whole portfolio.
        """
        lim = self.limits
        edge = prob - price
        direction = 1 if edge > 0 else -1 if edge < 0 else 0
        held = 1 if current_position > 0 else -1 if current_position < 0 else 0
        if abs(edge) < lim.exit_edge or direction == 0:
            return 0
        cost_per_contract = price if direction > 0 else 1.0 - price
        room = max(lim.max_gross_exposure - other_gross_exposure, 0.0)
        cap = min(lim.max_position_per_market, math.floor(room / cost_per_contract + 1e-9))
        if abs(edge) < lim.entry_edge:
            # Inside the hysteresis band: keep a same-direction position (within the
            # caps), otherwise go flat.
            return direction * min(abs(current_position), cap) if held == direction else 0

        f = abs(kelly_fraction_binary(prob, price)) * lim.kelly_fraction
        f *= drawdown_throttle(drawdown, lim)
        contracts = math.floor(max(equity, 0.0) * f / cost_per_contract + 1e-9)
        return direction * min(contracts, cap)

    def should_trade(self, target: int, working: int, over_limit: bool = False) -> bool:
        """Trade only if the change is at least ``min_trade_size`` and, unless we are
        closing out, flipping, or reducing while the book is over a risk limit, at
        least ``rebalance_band`` of the target (cuts churn)."""
        delta = abs(target - working)
        if delta < self.limits.min_trade_size:
            return False
        if target == 0 or working == 0 or (target > 0) != (working > 0):
            return True
        if over_limit and abs(target) < abs(working):
            return True
        return delta >= self.limits.rebalance_band * abs(target)

    def clip_to_participation(self, qty: int, visible_depth: int) -> int:
        """Cap an order at ``max_book_participation`` of visible opposite-side depth."""
        cap = math.floor(self.limits.max_book_participation * visible_depth)
        return max(0, min(qty, cap))
