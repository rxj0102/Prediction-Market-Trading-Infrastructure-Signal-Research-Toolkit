"""Mispricing signal research: model-implied vs. market-implied probability.

Pipeline
--------
1. Split markets (not time steps) into train/test sets, so no market used to
   fit a model is used to evaluate it.
2. ``info model``: logistic regression of the resolved outcome on the two
   synthetic information features (``fundamental_z``, ``poll_z``). Its output is
   a model-implied probability that never looks at the market price.
3. ``composite model``: a market-anchored correction model,
   ``logit p = logit(mid) + b1*(info_logit - logit(mid)) + b2*momentum
   + b3*mean_reversion + b4*vw_momentum``. The market log-odds enter as a fixed
   offset, so with all ``b = 0`` the composite equals the market price. Its
   output is the probability the strategy trades on.

4. Mispricing / edge = model probability - market mid.
5. Out-of-sample evaluation on the test markets: Brier score, log loss,
   calibration curve, Murphy decomposition, ECE, information coefficient and a
   simple "PnL if traded" check (1 contract, hold to resolution).

Neither model fits an intercept by default: the features are centred, and with
only a few dozen resolved markets an intercept mostly learns the training
set's YES/NO imbalance (sampling noise) rather than anything that generalises.
The L2 penalty is chosen by K-fold cross-validation *grouped by market* on the
training markets only (the test markets are never used for model selection).

The training rows within a market are heavily autocorrelated, which is why the
training set is subsampled with a stride and why the test statistics are also
reported per market (one trade per market).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from market_sim.markets import SyntheticMarket
from signals.calibration import (
    BrierDecomposition,
    CalibrationBin,
    brier_decomposition,
    brier_score,
    calibration_curve,
    expected_calibration_error,
    log_loss,
)
from signals.features import (
    logit,
    mean_reversion,
    momentum,
    sigmoid,
    volume_weighted_momentum,
)
from signals.model import LogisticModel

FloatArray = npt.NDArray[np.float64]

# Model probabilities are bounded away from 0/1: near resolution the info
# log-odds can be very large, and a forecast of exactly 0 or 1 is never useful
# for sizing (and is infinitely penalised by log loss).
PROB_BOUNDS = (1e-3, 1.0 - 1e-3)

INFO_FEATURES: tuple[str, ...] = ("fundamental_z", "poll_z")
COMPOSITE_FEATURES: tuple[str, ...] = (
    "info_gap",
    "momentum",
    "mean_reversion",
    "vw_momentum",
)


@dataclass(frozen=True)
class SignalConfig:
    momentum_lookback: int = 12
    reversion_window: int = 24
    vw_lookback: int = 12
    train_fraction: float = 0.5
    split_seed: int = 0
    train_stride: int = 4
    eval_stride: int = 4
    l2_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    cv_folds: int = 5
    fit_intercept: bool = False
    edge_threshold: float = 0.05
    n_bins: int = 10

    def __post_init__(self) -> None:
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be in (0, 1)")
        if self.train_stride < 1 or self.eval_stride < 1:
            raise ValueError("strides must be >= 1")
        if not self.l2_grid or any(v < 0 for v in self.l2_grid):
            raise ValueError("l2_grid must be a non-empty tuple of non-negative values")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be >= 2")


def market_features(market: SyntheticMarket, config: SignalConfig) -> dict[str, FloatArray]:
    """All causal features for one market, each aligned with its local steps."""
    return {
        "market_logit": logit(market.mid),
        "fundamental_z": market.fundamental_z.astype(np.float64),
        "poll_z": market.poll_z.astype(np.float64),
        "momentum": momentum(market.mid, config.momentum_lookback),
        "mean_reversion": mean_reversion(market.mid, config.reversion_window),
        "vw_momentum": volume_weighted_momentum(market.mid, market.volume, config.vw_lookback),
    }


def _matrix(feats: dict[str, FloatArray], names: Sequence[str]) -> FloatArray:
    return np.column_stack([feats[n] for n in names])


def split_markets(
    markets: Sequence[SyntheticMarket], train_fraction: float, seed: int
) -> tuple[list[SyntheticMarket], list[SyntheticMarket]]:
    """Deterministic shuffled split by market."""
    order = np.random.default_rng(seed).permutation(len(markets))
    n_train = max(1, min(len(markets) - 1, round(train_fraction * len(markets))))
    train = [markets[i] for i in sorted(order[:n_train])]
    test = [markets[i] for i in sorted(order[n_train:])]
    return train, test


@dataclass
class MarketSignal:
    """Per-step model outputs for one market (arrays aligned with local steps)."""

    market_id: str
    mid: FloatArray
    info_prob: FloatArray
    composite_prob: FloatArray

    @property
    def mispricing(self) -> FloatArray:
        """Info-model probability minus market-implied probability."""
        out: FloatArray = self.info_prob - self.mid
        return out

    @property
    def edge(self) -> FloatArray:
        """Composite probability minus market-implied probability (the traded signal)."""
        out: FloatArray = self.composite_prob - self.mid
        return out


class SignalModel:
    """Fits the info and composite models and produces per-market signals."""

    def __init__(self, config: SignalConfig | None = None) -> None:
        self.config = config or SignalConfig()
        self.info_model = self._new_model(self.config.l2_grid[0])
        self.composite_model = self._new_model(self.config.l2_grid[0])
        self.selected_l2: dict[str, float] = {}
        self.cv_scores: dict[str, dict[float, float]] = {}
        self.fitted = False
        self._cache: dict[int, tuple[SyntheticMarket, dict[str, FloatArray]]] = {}

    def _new_model(self, l2: float) -> LogisticModel:
        return LogisticModel(l2=l2, fit_intercept=self.config.fit_intercept)

    def _base(self, market: SyntheticMarket) -> dict[str, FloatArray]:
        cached = self._cache.get(id(market))
        if cached is None or cached[0] is not market:
            cached = (market, market_features(market, self.config))
            self._cache[id(market)] = cached
        return dict(cached[1])

    def _features(
        self, market: SyntheticMarket, info_model: LogisticModel | None = None
    ) -> dict[str, FloatArray]:
        """Base features plus ``info_logit``/``info_gap`` from a fitted info model."""
        feats = self._base(market)
        model = info_model or self.info_model
        info_logit = model.decision_function(_matrix(feats, INFO_FEATURES))
        feats["info_logit"] = info_logit
        feats["info_gap"] = info_logit - feats["market_logit"]
        return feats

    def _fit_pair(
        self, markets: Sequence[SyntheticMarket], l2_info: float, l2_comp: float
    ) -> tuple[LogisticModel, LogisticModel]:
        idxs = [np.arange(0, m.length, self.config.train_stride) for m in markets]
        y = np.concatenate(
            [np.full(len(i), float(m.outcome)) for m, i in zip(markets, idxs, strict=True)]
        )
        # Each market carries equal total weight regardless of its length.
        w = np.concatenate([np.full(len(i), 1.0 / len(i)) for i in idxs])
        base = [self._base(m) for m in markets]
        x_info = np.vstack([_matrix(f, INFO_FEATURES)[i] for f, i in zip(base, idxs, strict=True)])
        info = self._new_model(l2_info).fit(x_info, y, w)
        # Stage 2 uses the stage-1 info model's (in-sample) log-odds.
        feats = [self._features(m, info) for m in markets]
        x_comp = np.vstack(
            [_matrix(f, COMPOSITE_FEATURES)[i] for f, i in zip(feats, idxs, strict=True)]
        )
        offset = np.concatenate([f["market_logit"][i] for f, i in zip(feats, idxs, strict=True)])
        comp = self._new_model(l2_comp).fit(x_comp, y, w, offset=offset)
        return info, comp

    def _cv_select(self, markets: Sequence[SyntheticMarket]) -> tuple[float, float]:
        """Pick L2 for each stage by grouped (by-market) K-fold CV Brier score."""
        folds = np.arange(len(markets)) % min(self.config.cv_folds, len(markets))
        info_scores: dict[float, float] = {}
        comp_scores: dict[float, float] = {}
        for l2 in self.config.l2_grid:
            info_err: list[FloatArray] = []
            comp_err: list[FloatArray] = []
            for k in np.unique(folds):
                train = [m for m, f in zip(markets, folds, strict=True) if f != k]
                held = [m for m, f in zip(markets, folds, strict=True) if f == k]
                info, comp = self._fit_pair(train, l2, l2)
                for m in held:
                    feats = self._features(m, info)
                    idx = np.arange(0, m.length, self.config.eval_stride)
                    p_info = sigmoid(feats["info_logit"][idx])
                    p_comp = comp.predict_proba(
                        _matrix(feats, COMPOSITE_FEATURES)[idx], offset=feats["market_logit"][idx]
                    )
                    info_err.append((p_info - m.outcome) ** 2)
                    comp_err.append((p_comp - m.outcome) ** 2)
            info_scores[l2] = float(np.mean(np.concatenate(info_err)))
            comp_scores[l2] = float(np.mean(np.concatenate(comp_err)))
        self.cv_scores = {"info": info_scores, "composite": comp_scores}
        return min(info_scores, key=info_scores.__getitem__), min(
            comp_scores, key=comp_scores.__getitem__
        )

    def fit(self, markets: Sequence[SyntheticMarket]) -> SignalModel:
        if not markets:
            raise ValueError("need at least one training market")
        if len(self.config.l2_grid) > 1 and len(markets) >= 2:
            l2_info, l2_comp = self._cv_select(markets)
        else:
            l2_info = l2_comp = self.config.l2_grid[0]
        self.selected_l2 = {"info": l2_info, "composite": l2_comp}
        self.info_model, self.composite_model = self._fit_pair(markets, l2_info, l2_comp)
        self.fitted = True
        return self

    def compute(self, market: SyntheticMarket) -> MarketSignal:
        if not self.fitted:
            raise RuntimeError("SignalModel.fit must be called first")
        feats = self._features(market)
        return MarketSignal(
            market_id=market.market_id,
            mid=market.mid.astype(np.float64),
            info_prob=np.clip(sigmoid(feats["info_logit"]), *PROB_BOUNDS),
            composite_prob=np.clip(
                self.composite_model.predict_proba(
                    _matrix(feats, COMPOSITE_FEATURES), offset=feats["market_logit"]
                ),
                *PROB_BOUNDS,
            ),
        )

    def coefficients(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for name, model, feats in (
            ("info", self.info_model, INFO_FEATURES),
            ("composite", self.composite_model, COMPOSITE_FEATURES),
        ):
            assert model.coef_ is not None
            coefs = {"intercept": model.intercept_} if model.fit_intercept else {}
            coefs.update({f: float(c) for f, c in zip(feats, model.coef_, strict=True)})
            out[name] = coefs
        return out


# ---------------------------------------------------------------------- evaluation
@dataclass(frozen=True)
class ForecastScore:
    name: str
    brier: float
    log_loss: float
    ece: float
    decomposition: BrierDecomposition
    bins: tuple[CalibrationBin, ...]


def score_forecast(name: str, p: FloatArray, y: FloatArray, n_bins: int = 10) -> ForecastScore:
    bins = calibration_curve(p, y, n_bins)
    return ForecastScore(
        name=name,
        brier=brier_score(p, y),
        log_loss=log_loss(p, y),
        ece=expected_calibration_error(bins),
        decomposition=brier_decomposition(p, y, n_bins),
        bins=tuple(bins),
    )


@dataclass(frozen=True)
class TradeEval:
    """Hold-to-resolution, 1-contract PnL of trading in the direction of the edge."""

    label: str
    n_trades: int
    n_markets: int
    hit_rate: float
    mean_pnl_mid: float  # per contract, filled at mid (frictionless)
    mean_pnl_touch: float  # per contract, buying at ask / selling at bid
    total_pnl_touch: float
    hit_rate_touch: float


def trade_eval(
    label: str,
    direction: FloatArray,
    mid: FloatArray,
    bid: FloatArray,
    ask: FloatArray,
    outcome: FloatArray,
    market_idx: npt.NDArray[np.int64],
) -> TradeEval:
    """Evaluate +1/-1 (0 = no trade) directions against resolved outcomes."""
    mask = direction != 0
    n = int(mask.sum())
    if n == 0:
        return TradeEval(label, 0, 0, float("nan"), float("nan"), float("nan"), 0.0, float("nan"))
    d = direction[mask]
    pnl_mid = d * (outcome[mask] - mid[mask])
    fill = np.where(d > 0, ask[mask], bid[mask])
    pnl_touch = d * (outcome[mask] - fill)
    return TradeEval(
        label=label,
        n_trades=n,
        n_markets=len(np.unique(market_idx[mask])),
        hit_rate=float(np.mean(pnl_mid > 0)),
        mean_pnl_mid=float(pnl_mid.mean()),
        mean_pnl_touch=float(pnl_touch.mean()),
        total_pnl_touch=float(pnl_touch.sum()),
        hit_rate_touch=float(np.mean(pnl_touch > 0)),
    )


@dataclass
class SignalReport:
    n_train_markets: int
    n_test_markets: int
    n_obs: int
    base_rate: float
    scores: dict[str, ForecastScore]
    information_coefficient: dict[str, float]
    trade_evals: list[TradeEval]
    coefficients: dict[str, dict[str, float]] = field(default_factory=dict)


def evaluate(
    model: SignalModel,
    train_markets: Sequence[SyntheticMarket],
    test_markets: Sequence[SyntheticMarket],
) -> SignalReport:
    """Out-of-sample forecast and trading-signal diagnostics on ``test_markets``."""
    if not test_markets:
        raise ValueError("need at least one test market")
    cfg = model.config
    cols: dict[str, list[FloatArray]] = {
        k: [] for k in ("mid", "info", "comp", "true", "y", "bid", "ask", "midx")
    }
    first_dir: list[FloatArray] = []
    for j, m in enumerate(test_markets):
        sig = model.compute(m)
        idx = np.arange(0, m.length, cfg.eval_stride)
        touch = np.array([m.touch(int(i)) for i in idx], dtype=np.float64) / 100.0
        cols["mid"].append(sig.mid[idx])
        cols["info"].append(sig.info_prob[idx])
        cols["comp"].append(sig.composite_prob[idx])
        cols["true"].append(m.true_prob[idx])
        cols["y"].append(np.full(len(idx), float(m.outcome)))
        cols["bid"].append(touch[:, 0])
        cols["ask"].append(touch[:, 1])
        cols["midx"].append(np.full(len(idx), float(j)))
        # One trade per market: the first sampled step where |edge| clears the bar.
        edge = sig.edge[idx]
        fd = np.zeros(len(idx))
        hits = np.flatnonzero(np.abs(edge) > cfg.edge_threshold)
        if len(hits):
            fd[hits[0]] = np.sign(edge[hits[0]])
        first_dir.append(fd)
    c = {k: np.concatenate(v) for k, v in cols.items()}
    y, mid = c["y"], c["mid"]
    midx = c["midx"].astype(np.int64)
    scores = {
        name: score_forecast(name, p, y, cfg.n_bins)
        for name, p in (
            ("market_mid", mid),
            ("info_model", c["info"]),
            ("composite", c["comp"]),
            ("oracle_true_prob", c["true"]),
        )
    }
    realized_move = y - mid
    ic = {
        "mispricing(info - mid)": float(np.corrcoef(c["info"] - mid, realized_move)[0, 1]),
        "edge(composite - mid)": float(np.corrcoef(c["comp"] - mid, realized_move)[0, 1]),
    }
    edge_all = c["comp"] - mid
    dir_all = np.where(np.abs(edge_all) > cfg.edge_threshold, np.sign(edge_all), 0.0)
    evals = [
        trade_eval("all sampled steps |edge|>thr", dir_all, mid, c["bid"], c["ask"], y, midx),
        trade_eval(
            "first signal per market", np.concatenate(first_dir), mid, c["bid"], c["ask"], y, midx
        ),
    ]
    return SignalReport(
        n_train_markets=len(train_markets),
        n_test_markets=len(test_markets),
        n_obs=len(y),
        base_rate=float(np.mean([m.outcome for m in test_markets])),
        scores=scores,
        information_coefficient=ic,
        trade_evals=evals,
        coefficients=model.coefficients(),
    )
