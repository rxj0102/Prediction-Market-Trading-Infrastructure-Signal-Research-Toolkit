"""Minimal L2-regularised logistic regression (Newton / IRLS), NumPy only.

Supports an optional fixed per-row ``offset`` added to the linear predictor
(``logit p = offset + intercept + x @ coef``). The composite signal uses the
market's own log-odds as the offset, so the regression only learns a
*correction* to the market price.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from signals.features import sigmoid

FloatArray = npt.NDArray[np.float64]


class LogisticModel:
    """Binary logistic regression with an unpenalised intercept."""

    def __init__(
        self, l2: float = 1e-3, fit_intercept: bool = True, max_iter: int = 50, tol: float = 1e-9
    ) -> None:
        if l2 < 0:
            raise ValueError("l2 must be non-negative")
        self.l2 = l2
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.coef_: FloatArray | None = None
        self.intercept_: float = 0.0
        self.n_iter_: int = 0

    def fit(
        self,
        x: FloatArray,
        y: FloatArray,
        sample_weight: FloatArray | None = None,
        offset: FloatArray | None = None,
    ) -> LogisticModel:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim != 2 or len(x) != len(y):
            raise ValueError("x must be 2-D with one row per label")
        if not np.all((y == 0) | (y == 1)):
            raise ValueError("y must be binary 0/1")
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        w = w / w.mean()
        off = np.zeros(len(y)) if offset is None else np.asarray(offset, dtype=np.float64)
        intercept_col = np.ones(len(x)) if self.fit_intercept else np.zeros(len(x))
        design = np.column_stack((intercept_col, x))
        beta = np.zeros(design.shape[1])
        penalty = np.full(design.shape[1], self.l2)
        penalty[0] = 0.0 if self.fit_intercept else 1.0
        for it in range(1, self.max_iter + 1):
            p = sigmoid(off + design @ beta)
            grad = design.T @ (w * (p - y)) + penalty * beta
            hess = (design * (w * p * (1.0 - p))[:, None]).T @ design + np.diag(penalty)
            hess += 1e-10 * np.eye(len(beta))
            step = np.linalg.solve(hess, grad)
            beta -= step
            self.n_iter_ = it
            if float(np.max(np.abs(step))) < self.tol:
                break
        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:]
        return self

    def decision_function(self, x: FloatArray, offset: FloatArray | None = None) -> FloatArray:
        if self.coef_ is None:
            raise RuntimeError("model is not fitted")
        out: FloatArray = np.asarray(x, dtype=np.float64) @ self.coef_ + self.intercept_
        if offset is not None:
            out = out + np.asarray(offset, dtype=np.float64)
        return out

    def predict_proba(self, x: FloatArray, offset: FloatArray | None = None) -> FloatArray:
        return sigmoid(self.decision_function(x, offset))
