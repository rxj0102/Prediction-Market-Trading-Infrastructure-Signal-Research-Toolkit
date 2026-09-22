"""Performance metric primitives shared by the live monitor and the tear sheet."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def simple_returns(equity: Sequence[float] | FloatArray) -> FloatArray:
    """Per-step simple returns of an equity curve (length n-1)."""
    eq = np.asarray(equity, dtype=np.float64)
    if len(eq) < 2:
        return np.zeros(0)
    if np.any(eq[:-1] <= 0):
        raise ValueError("equity must stay positive to compute returns")
    out: FloatArray = eq[1:] / eq[:-1] - 1.0
    return out


def sharpe_ratio(returns: Sequence[float] | FloatArray, periods_per_year: float) -> float:
    """Annualised Sharpe ratio (zero risk-free rate).

    Returns NaN with fewer than two observations and 0.0 for zero variance.
    """
    r = np.asarray(returns, dtype=np.float64)
    if len(r) < 2:
        return float("nan")
    sd = float(r.std(ddof=1))
    if sd == 0.0:
        return 0.0
    return float(r.mean()) / sd * math.sqrt(periods_per_year)


def drawdown_series(equity: Sequence[float] | FloatArray) -> FloatArray:
    """Fractional drawdown from the running peak at each step (>= 0)."""
    eq = np.asarray(equity, dtype=np.float64)
    if len(eq) == 0:
        return np.zeros(0)
    peak = np.maximum.accumulate(eq)
    out: FloatArray = 1.0 - eq / peak
    return out


def max_drawdown(equity: Sequence[float] | FloatArray) -> float:
    dd = drawdown_series(equity)
    return float(dd.max()) if len(dd) else 0.0
