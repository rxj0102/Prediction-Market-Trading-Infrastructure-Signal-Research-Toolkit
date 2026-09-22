"""Hand-built deterministic markets and signals for execution/monitoring tests."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from market_sim.markets import TIER_PARAMS, LiquidityTier, SyntheticMarket
from signals.research import MarketSignal


def make_market(
    mids: Sequence[float],
    outcome: int,
    market_id: str = "T-000",
    open_step: int = 0,
    tier: LiquidityTier = LiquidityTier.DEEP,
    half_spread: int = 1,
    top_depth: float | None = None,
    flow: int = 50,
) -> SyntheticMarket:
    n = len(mids)
    depth = TIER_PARAMS[tier].top_depth if top_depth is None else top_depth
    mid = np.asarray(mids, dtype=np.float64)
    return SyntheticMarket(
        market_id=market_id,
        question=f"[SYNTHETIC] test market {market_id}",
        tier=tier,
        open_step=open_step,
        outcome=outcome,
        true_prob=mid.copy(),
        mid=mid,
        half_spread_ticks=np.full(n, half_spread, dtype=np.int64),
        top_depth=np.full(n, depth),
        depth_decay=TIER_PARAMS[tier].decay,
        volume=np.full(n, 2 * flow, dtype=np.int64),
        buy_flow=np.full(n, flow, dtype=np.int64),
        sell_flow=np.full(n, flow, dtype=np.int64),
        fundamental_z=np.zeros(n),
        poll_z=np.zeros(n),
    )


def make_signal(market: SyntheticMarket, probs: npt.ArrayLike) -> MarketSignal:
    p = np.broadcast_to(np.asarray(probs, dtype=np.float64), market.mid.shape).copy()
    return MarketSignal(market.market_id, market.mid.copy(), p.copy(), p)
