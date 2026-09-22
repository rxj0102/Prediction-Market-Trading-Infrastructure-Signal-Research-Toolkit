"""Module 1: synthetic prediction-market generator and limit order book."""

from market_sim.book_builder import DepthProfile, build_book, touch_prices
from market_sim.markets import (
    DEFAULT_SEED,
    TIER_PARAMS,
    LiquidityTier,
    MarketConfig,
    SyntheticMarket,
    generate_universe,
)
from market_sim.order_book import Fill, LimitOrderBook, Order, OrderResult, Side

__all__ = [
    "DEFAULT_SEED",
    "TIER_PARAMS",
    "DepthProfile",
    "Fill",
    "LimitOrderBook",
    "LiquidityTier",
    "MarketConfig",
    "Order",
    "OrderResult",
    "Side",
    "SyntheticMarket",
    "build_book",
    "generate_universe",
    "touch_prices",
]
