"""Causal price/volume features.

Every function returns an array aligned with its input where element ``t``
depends only on inputs at indices ``<= t`` (no look-ahead). This is asserted
in the test suite by perturbing future values.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

PROB_CLIP = 1e-4


def logit(p: FloatArray) -> FloatArray:
    q = np.clip(np.asarray(p, dtype=np.float64), PROB_CLIP, 1.0 - PROB_CLIP)
    return np.log(q / (1.0 - q))


def sigmoid(x: FloatArray) -> FloatArray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def momentum(mid: FloatArray, lookback: int = 12) -> FloatArray:
    """Change in market log-odds over ``lookback`` steps (shorter history at the start)."""
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    lo = logit(mid)
    lagged = np.concatenate((np.full(lookback, lo[0]), lo[:-lookback]))[: len(lo)]
    out: FloatArray = lo - lagged
    return out


def _trailing_sum(x: FloatArray, window: int) -> FloatArray:
    """Sum of x[max(0, t-window+1) .. t] for every t (causal)."""
    c = np.concatenate(([0.0], np.cumsum(x)))
    idx = np.arange(len(x))
    lo = np.maximum(0, idx - window + 1)
    out: FloatArray = c[idx + 1] - c[lo]
    return out


def _trailing_count(n: int, window: int) -> FloatArray:
    idx = np.arange(n)
    out: FloatArray = (idx - np.maximum(0, idx - window + 1) + 1).astype(np.float64)
    return out


def mean_reversion(mid: FloatArray, window: int = 24, min_std: float = 0.05) -> FloatArray:
    """Negative z-score of log-odds versus its trailing mean.

    Positive values mean the price is below its recent average (a reversion
    feature expects it to move back up). Element 0 is defined as 0.
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    lo = logit(mid)
    n = _trailing_count(len(lo), window)
    mean = _trailing_sum(lo, window) / n
    var = np.maximum(_trailing_sum(lo**2, window) / n - mean**2, 0.0)
    sd = np.maximum(np.sqrt(var), min_std)
    out: FloatArray = -(lo - mean) / sd
    out[0] = 0.0
    return out


def volume_weighted_momentum(
    mid: FloatArray, volume: npt.NDArray[np.int64] | FloatArray, lookback: int = 12
) -> FloatArray:
    """Volume-weighted sum of per-step log-odds changes over ``lookback`` steps.

    Moves that happen on heavier volume get more weight; the weights are
    normalised by trailing mean volume so the feature is scale-free.
    """
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    lo = logit(mid)
    dlo = np.diff(lo, prepend=lo[0])
    vol = np.asarray(volume, dtype=np.float64)
    mean_v = _trailing_sum(vol, lookback) / _trailing_count(len(lo), lookback)
    num = _trailing_sum(dlo * vol, lookback)
    out: FloatArray = np.divide(num, mean_v, out=np.zeros_like(num), where=mean_v > 0)
    return out
