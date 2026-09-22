"""Build a synthetic limit order book snapshot from a market state.

The simulator does not model individual market makers. Instead, at each step a
book is reconstructed around the synthetic mid price: the touch sits
``half_spread_ticks`` either side of the mid, and each deeper level holds
geometrically less size than the one before it. This is deliberately simple
(see README "Known limitations"): real prediction-market books are lumpy,
often one-sided near 0/1, and change in response to the trader's own orders.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from market_sim.order_book import MAX_TICK, MIN_TICK, LimitOrderBook, Side

MARKET_MAKER = "mm"


@dataclass(frozen=True)
class DepthProfile:
    """Shape of the synthetic resting liquidity on each side of the book."""

    top_depth: float  # contracts resting at the best level
    decay: float  # size multiplier per level away from the touch (0 < decay <= 1)
    n_levels: int = 15

    def __post_init__(self) -> None:
        if self.top_depth <= 0:
            raise ValueError("top_depth must be positive")
        if not 0.0 < self.decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        if self.n_levels < 1:
            raise ValueError("n_levels must be >= 1")

    def level_sizes(self, rng: np.random.Generator | None = None) -> list[int]:
        """Integer size per level, optionally with multiplicative lognormal noise."""
        sizes = self.top_depth * self.decay ** np.arange(self.n_levels)
        if rng is not None:
            sizes = sizes * rng.lognormal(0.0, 0.25, size=self.n_levels)
        return [max(1, round(float(s))) for s in sizes]


def touch_prices(mid: float, half_spread_ticks: int) -> tuple[int, int]:
    """Best bid and ask ticks for a given mid (dollars) and half spread (ticks)."""
    if half_spread_ticks < 1:
        raise ValueError("half_spread_ticks must be >= 1")
    centre = round(mid * 100)
    bid = max(MIN_TICK, min(MAX_TICK - 1, centre - half_spread_ticks))
    ask = min(MAX_TICK, max(bid + 1, centre + half_spread_ticks))
    return bid, ask


def build_book(
    mid: float,
    half_spread_ticks: int,
    profile: DepthProfile,
    rng: np.random.Generator | None = None,
) -> LimitOrderBook:
    """Construct a two-sided book centred on ``mid`` with the given depth profile."""
    bid, ask = touch_prices(mid, half_spread_ticks)
    book = LimitOrderBook()
    bid_sizes = profile.level_sizes(rng)
    ask_sizes = profile.level_sizes(rng)
    for k in range(profile.n_levels):
        if bid - k >= MIN_TICK:
            book.submit_limit(Side.BUY, bid - k, bid_sizes[k], owner=MARKET_MAKER)
        if ask + k <= MAX_TICK:
            book.submit_limit(Side.SELL, ask + k, ask_sizes[k], owner=MARKET_MAKER)
    return book
