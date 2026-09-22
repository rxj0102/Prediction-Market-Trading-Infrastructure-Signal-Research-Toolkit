"""Synthetic binary-outcome event markets.

EVERYTHING GENERATED HERE IS SYNTHETIC. No data is scraped from, or calibrated
to, any real prediction market.

Model (per market, ``L`` trading steps, step index ``i = 0..L-1``):

* Latent information process. ``W_i = W_0 + sum_{s<=i} sqrt(v_s) * xi_s`` with
  ``xi_s ~ N(0, 1)``. The per-step information variance ``v_s`` is mostly
  small, with occasional "news" steps where it jumps by ``news_scale**2``.
  The variances are normalised so the total over the market's life is 1.
* Resolution. The final increment ``v_L`` arrives after the last trading step
  (index ``L-1``); the market resolves YES iff ``W_L > 0``.
* True probability. Because the future increments are Gaussian with known
  variance ``V_i = sum_{s>i} v_s``, the exact conditional probability is
  ``p_i = Phi(W_i / sqrt(V_i))``. It starts near 0.5 (``W_0`` is small) and is
  pulled toward 0 or 1 as ``V_i`` shrinks approaching resolution. ``p_i`` is a
  diagnostic only - no strategy code reads it.
* Market price. The market's log-odds track ``logit(p_i)`` with partial
  adjustment (speed ``kappa`` < 1, i.e. under-reaction) plus an AR(1)
  "sentiment" noise term. The mid is ``sigmoid`` of that, clipped to
  [0.02, 0.98].
* Observable information features (what a researcher could build):
  ``fundamental_z`` - a noisy, persistently-biased (AR(1) error) read of
  ``W_i`` scaled by the remaining uncertainty; ``poll_z`` - a lagged read of
  ``W`` with independent noise.
* Liquidity. Each market has a tier (thin/medium/deep) setting its top-of-book
  depth; depth also wanders over time (lognormal AR(1)). The half spread is
  wider when depth is low and when the mid has just moved.
* Order flow. Per-step taker volume is Poisson with rate proportional to depth
  and the size of the price move, split into buy/sell flow tilted toward the
  direction of the move over the same step (a crude adverse-selection
  mechanism: a resting bid is most likely to be hit while the price falls).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import numpy.typing as npt

from market_sim.book_builder import DepthProfile, build_book, touch_prices
from market_sim.order_book import LimitOrderBook

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

DEFAULT_SEED = 42


class LiquidityTier(str, Enum):
    THIN = "thin"
    MEDIUM = "medium"
    DEEP = "deep"


@dataclass(frozen=True)
class TierParams:
    top_depth: float  # median contracts at the touch
    decay: float  # per-level size decay


TIER_PARAMS: dict[LiquidityTier, TierParams] = {
    LiquidityTier.THIN: TierParams(top_depth=40.0, decay=0.75),
    LiquidityTier.MEDIUM: TierParams(top_depth=200.0, decay=0.82),
    LiquidityTier.DEEP: TierParams(top_depth=800.0, decay=0.88),
}


@dataclass(frozen=True)
class MarketConfig:
    """All knobs of the synthetic market generator (documented in README)."""

    n_markets: int = 50
    seed: int = DEFAULT_SEED
    horizon: int = 480  # global clock length (1 step == 1 synthetic hour)
    min_length: int = 160
    max_length: int = 320
    initial_info_sd: float = 0.12  # sd of W_0 (total variance is 1)
    news_prob: float = 0.03
    news_scale: float = 4.0
    market_kappa: float = 0.08  # speed the price absorbs new information
    sentiment_phi: float = 0.92
    sentiment_sigma: float = 0.12
    fundamental_noise_sd: float = 0.10
    fundamental_noise_phi: float = 0.97
    poll_noise_sd: float = 0.20
    poll_lag: int = 6
    liquidity_phi: float = 0.95
    liquidity_sigma: float = 0.08
    spread_coef: float = 2.0
    spread_move_sensitivity: float = 8.0
    max_half_spread: int = 6
    volume_rate: float = 0.25
    volume_move_sensitivity: float = 20.0
    flow_tilt: float = 6.0
    tier_weights: tuple[float, float, float] = (0.3, 0.4, 0.3)

    def __post_init__(self) -> None:
        if self.n_markets < 1:
            raise ValueError("n_markets must be >= 1")
        if not 2 <= self.min_length <= self.max_length <= self.horizon:
            raise ValueError("require 2 <= min_length <= max_length <= horizon")
        if abs(sum(self.tier_weights) - 1.0) > 1e-9:
            raise ValueError("tier_weights must sum to 1")


@dataclass
class SyntheticMarket:
    """One synthetic event market. Arrays are indexed by local step."""

    market_id: str
    question: str
    tier: LiquidityTier
    open_step: int
    outcome: int
    true_prob: FloatArray
    mid: FloatArray
    half_spread_ticks: IntArray
    top_depth: FloatArray
    depth_decay: float
    volume: IntArray
    buy_flow: IntArray
    sell_flow: IntArray
    fundamental_z: FloatArray
    poll_z: FloatArray
    synthetic: bool = field(default=True, init=False)

    @property
    def length(self) -> int:
        return len(self.mid)

    @property
    def close_step(self) -> int:
        """Global step at which the market resolves (no trading at or after it)."""
        return self.open_step + self.length

    def is_active(self, global_step: int) -> bool:
        return self.open_step <= global_step < self.close_step

    def local_index(self, global_step: int) -> int:
        if not self.is_active(global_step):
            raise IndexError(f"{self.market_id} not active at global step {global_step}")
        return global_step - self.open_step

    def touch(self, i: int) -> tuple[int, int]:
        """(best bid, best ask) ticks at local step ``i``."""
        return touch_prices(float(self.mid[i]), int(self.half_spread_ticks[i]))

    def depth_profile(self, i: int) -> DepthProfile:
        return DepthProfile(top_depth=float(self.top_depth[i]), decay=self.depth_decay)

    def build_book(self, i: int, rng: np.random.Generator | None = None) -> LimitOrderBook:
        return build_book(
            float(self.mid[i]), int(self.half_spread_ticks[i]), self.depth_profile(i), rng
        )


def norm_cdf(x: FloatArray) -> FloatArray:
    """Standard normal CDF (vectorised, no SciPy dependency)."""
    return np.array(
        [0.5 * (1.0 + math.erf(v / math.sqrt(2.0))) for v in np.ravel(x)], dtype=np.float64
    ).reshape(np.shape(x))


def _sigmoid(x: FloatArray) -> FloatArray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: FloatArray) -> FloatArray:
    return np.log(p / (1.0 - p))


def _ar1(rng: np.random.Generator, n: int, phi: float, stationary_sd: float) -> FloatArray:
    """AR(1) series with the given stationary standard deviation."""
    innov_sd = stationary_sd * math.sqrt(1.0 - phi**2)
    out = np.empty(n, dtype=np.float64)
    out[0] = rng.normal(0.0, stationary_sd)
    for t in range(1, n):
        out[t] = phi * out[t - 1] + rng.normal(0.0, innov_sd)
    return out


def generate_market(index: int, config: MarketConfig, rng: np.random.Generator) -> SyntheticMarket:
    """Generate a single synthetic market from its own RNG stream."""
    length = int(rng.integers(config.min_length, config.max_length + 1))
    open_step = int(rng.integers(0, config.horizon - length + 1))
    tier_names = list(LiquidityTier)
    tier = tier_names[int(rng.choice(len(tier_names), p=np.array(config.tier_weights)))]
    params = TIER_PARAMS[tier]

    # --- latent information process ------------------------------------------
    n_inc = length  # increments 1..L; increment L arrives after the last trading step
    step_var = np.ones(n_inc)
    news = rng.random(n_inc) < config.news_prob
    step_var[news] *= config.news_scale**2
    step_var /= step_var.sum()
    increments = rng.normal(0.0, 1.0, n_inc) * np.sqrt(step_var)
    w0 = rng.normal(0.0, config.initial_info_sd)
    w_path = w0 + np.cumsum(increments)  # w_path[k] = W_{k+1}
    w = np.concatenate(([w0], w_path[:-1]))  # W_i for i = 0..L-1
    w_final = float(w_path[-1])
    remaining_var = np.cumsum(step_var[::-1])[::-1]  # V_i = sum_{s>i} v_s
    true_prob = norm_cdf(w / np.sqrt(remaining_var))
    outcome = int(w_final > 0.0)

    # --- market price: partial adjustment + AR(1) sentiment ----------------------
    target = np.clip(_logit(np.clip(true_prob, 1e-6, 1 - 1e-6)), -5.0, 5.0)
    anchor = np.empty(length)
    anchor[0] = target[0]
    for t in range(1, length):
        anchor[t] = anchor[t - 1] + config.market_kappa * (target[t] - anchor[t - 1])
    sentiment = _ar1(
        rng,
        length,
        config.sentiment_phi,
        config.sentiment_sigma / math.sqrt(1.0 - config.sentiment_phi**2),
    )
    mid = np.clip(_sigmoid(anchor + sentiment), 0.02, 0.98)

    # --- observable information features ----------------------------------------
    fund_err = _ar1(rng, length, config.fundamental_noise_phi, config.fundamental_noise_sd)
    fundamental_z = (w + fund_err) / np.sqrt(remaining_var + config.fundamental_noise_sd**2)
    lagged_w = np.concatenate((np.full(config.poll_lag, w0), w))[:length]
    poll_err = rng.normal(0.0, config.poll_noise_sd, length)
    poll_z = (lagged_w + poll_err) / np.sqrt(remaining_var + config.poll_noise_sd**2)

    # --- liquidity, spread, volume ----------------------------------------------
    liq = np.exp(
        _ar1(
            rng,
            length,
            config.liquidity_phi,
            config.liquidity_sigma / math.sqrt(1.0 - config.liquidity_phi**2),
        )
    )
    top_depth = params.top_depth * liq
    abs_move = np.abs(np.diff(mid, prepend=mid[0]))
    raw_half = (
        config.spread_coef
        / np.sqrt(top_depth / 100.0)
        * (1.0 + config.spread_move_sensitivity * abs_move)
    )
    half_spread = np.clip(np.round(raw_half), 1, config.max_half_spread).astype(np.int64)
    rate = top_depth * config.volume_rate * (1.0 + config.volume_move_sensitivity * abs_move)
    volume = rng.poisson(rate).astype(np.int64)
    # Flow at step i is the taker flow during (i-1, i]; it leans in the direction
    # of the price move over that same interval (sellers dominate while it falls).
    move = np.diff(_logit(mid), prepend=_logit(mid[:1]))
    p_sell = _sigmoid(-config.flow_tilt * move)
    sell_flow = rng.binomial(volume, p_sell).astype(np.int64)
    buy_flow = (volume - sell_flow).astype(np.int64)

    market_id = f"SYN-{index:03d}"
    question = (
        f"[SYNTHETIC] Will synthetic event #{index:03d} resolve YES by "
        f"global step {open_step + length}?"
    )
    return SyntheticMarket(
        market_id=market_id,
        question=question,
        tier=tier,
        open_step=open_step,
        outcome=outcome,
        true_prob=true_prob,
        mid=mid,
        half_spread_ticks=half_spread,
        top_depth=top_depth,
        depth_decay=params.decay,
        volume=volume,
        buy_flow=buy_flow,
        sell_flow=sell_flow,
        fundamental_z=fundamental_z,
        poll_z=poll_z,
    )


def generate_universe(config: MarketConfig | None = None) -> list[SyntheticMarket]:
    """Generate ``config.n_markets`` independent synthetic markets (deterministic in seed)."""
    config = config or MarketConfig()
    children = np.random.SeedSequence(config.seed).spawn(config.n_markets)
    return [
        generate_market(k, config, np.random.default_rng(child)) for k, child in enumerate(children)
    ]
