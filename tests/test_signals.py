"""Tests for causal features, the logistic model and the research pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from market_sim.markets import MarketConfig, SyntheticMarket, generate_universe
from signals.features import (
    logit,
    mean_reversion,
    momentum,
    sigmoid,
    volume_weighted_momentum,
)
from signals.model import LogisticModel
from signals.research import (
    COMPOSITE_FEATURES,
    SignalConfig,
    SignalModel,
    evaluate,
    market_features,
    split_markets,
    trade_eval,
)

SMALL = MarketConfig(n_markets=12, seed=3, min_length=40, max_length=60, horizon=80)


@pytest.fixture(scope="module")
def markets() -> list[SyntheticMarket]:
    return generate_universe(SMALL)


# -------------------------------------------------------------------- features
def test_logit_sigmoid_roundtrip() -> None:
    p = np.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(sigmoid(logit(p)), p)
    assert np.isfinite(logit(np.array([0.0, 1.0]))).all()


def test_momentum_known_values() -> None:
    mid = sigmoid(np.array([0.0, 0.5, 1.0, 1.5]))
    np.testing.assert_allclose(momentum(mid, lookback=2), [0.0, 0.5, 1.0, 1.0])


def test_mean_reversion_sign_and_zero_start() -> None:
    mid = sigmoid(np.array([0.0, 0.0, 0.0, 2.0]))
    mr = mean_reversion(mid, window=4)
    assert mr[0] == 0.0
    assert mr[-1] < 0  # price jumped above its trailing mean -> expect reversion down
    # Brute-force check of one element.
    lo = logit(mid)
    hist = lo[0:4]
    assert mr[3] == pytest.approx(-(lo[3] - hist.mean()) / max(hist.std(), 0.05))


def test_vw_momentum_weights_by_volume() -> None:
    mid = sigmoid(np.array([0.0, 1.0, 1.0, 2.0]))
    vol = np.array([10, 10, 10, 30])
    out = volume_weighted_momentum(mid, vol, lookback=2)
    # t=3: dlo = [0, 1], vol = [10, 30], mean vol 20 -> (0*10 + 1*30) / 20
    assert out[3] == pytest.approx(1.5)
    assert volume_weighted_momentum(mid, np.zeros(4), lookback=2)[3] == 0.0


@pytest.mark.parametrize(
    "fn",
    [
        lambda m, v: momentum(m, 5),
        lambda m, v: mean_reversion(m, 6),
        lambda m, v: volume_weighted_momentum(m, v, 5),
    ],
)
def test_features_are_causal(fn) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(1)
    mid = rng.uniform(0.2, 0.8, 50)
    vol = rng.integers(1, 100, 50)
    base = fn(mid, vol)
    mid2, vol2 = mid.copy(), vol.copy()
    mid2[30:] = rng.uniform(0.2, 0.8, 20)
    vol2[30:] = rng.integers(1, 100, 20)
    np.testing.assert_allclose(fn(mid2, vol2)[:30], base[:30])


@pytest.mark.parametrize(
    "call",
    [
        lambda: momentum(np.array([0.5]), 0),
        lambda: mean_reversion(np.array([0.5]), 1),
        lambda: volume_weighted_momentum(np.array([0.5]), np.array([1]), 0),
    ],
)
def test_feature_validation(call) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):  # noqa: PT011
        call()


# ----------------------------------------------------------------------- model
def test_logistic_recovers_known_coefficients() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20_000, 2))
    y = (rng.random(20_000) < sigmoid(0.5 + x @ np.array([1.5, -2.0]))).astype(float)
    model = LogisticModel(l2=0.0).fit(x, y)
    assert model.coef_ is not None
    np.testing.assert_allclose(model.coef_, [1.5, -2.0], atol=0.08)
    assert model.intercept_ == pytest.approx(0.5, abs=0.08)
    assert 1 <= model.n_iter_ < 50


def test_logistic_offset_and_no_intercept() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20_000, 1))
    off = rng.normal(size=20_000)
    y = (rng.random(20_000) < sigmoid(off + 0.8 * x[:, 0])).astype(float)
    model = LogisticModel(l2=0.0, fit_intercept=False).fit(x, y, offset=off)
    assert model.intercept_ == 0.0
    assert model.coef_ is not None
    assert model.coef_[0] == pytest.approx(0.8, abs=0.06)
    p = model.predict_proba(x[:3], offset=off[:3])
    np.testing.assert_allclose(p, sigmoid(off[:3] + model.coef_[0] * x[:3, 0]))


def test_logistic_l2_shrinks() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(500, 1))
    y = (rng.random(500) < sigmoid(2 * x[:, 0])).astype(float)
    weak = LogisticModel(l2=0.0).fit(x, y).coef_
    strong = LogisticModel(l2=1e4).fit(x, y).coef_
    assert weak is not None
    assert strong is not None
    assert abs(strong[0]) < abs(weak[0])


def test_logistic_validation() -> None:
    with pytest.raises(ValueError, match="l2"):
        LogisticModel(l2=-1)
    m = LogisticModel()
    with pytest.raises(RuntimeError, match="not fitted"):
        m.predict_proba(np.zeros((1, 1)))
    with pytest.raises(ValueError, match="2-D"):
        m.fit(np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError, match="binary"):
        m.fit(np.zeros((2, 1)), np.array([0.0, 0.5]))


# -------------------------------------------------------------------- research
def test_split_is_disjoint_and_deterministic(markets: list[SyntheticMarket]) -> None:
    tr, te = split_markets(markets, 0.5, seed=0)
    tr2, _ = split_markets(markets, 0.5, seed=0)
    assert {m.market_id for m in tr}.isdisjoint({m.market_id for m in te})
    assert len(tr) + len(te) == len(markets)
    assert [m.market_id for m in tr] == [m.market_id for m in tr2]
    tr_a, te_a = split_markets(markets[:2], 0.99, seed=0)
    assert len(tr_a) == 1
    assert len(te_a) == 1


def test_market_features_shapes(markets: list[SyntheticMarket]) -> None:
    feats = market_features(markets[0], SignalConfig())
    assert all(len(v) == markets[0].length for v in feats.values())


def test_signal_model_fit_compute_evaluate(markets: list[SyntheticMarket]) -> None:
    tr, te = split_markets(markets, 0.5, seed=1)
    model = SignalModel(SignalConfig(cv_folds=3, l2_grid=(0.1, 10.0)))
    with pytest.raises(RuntimeError, match="fit"):
        model.compute(te[0])
    model.fit(tr)
    assert set(model.selected_l2) == {"info", "composite"}
    assert set(model.cv_scores["composite"]) == {0.1, 10.0}
    sig = model.compute(te[0])
    assert np.all((sig.composite_prob > 0) & (sig.composite_prob < 1))
    np.testing.assert_allclose(sig.edge, sig.composite_prob - sig.mid)
    np.testing.assert_allclose(sig.mispricing, sig.info_prob - sig.mid)
    report = evaluate(model, tr, te)
    assert report.n_test_markets == len(te)
    assert set(report.scores) == {"market_mid", "info_model", "composite", "oracle_true_prob"}
    for score in report.scores.values():
        assert 0.0 <= score.brier <= 1.0
        assert sum(b.count for b in score.bins) == report.n_obs
    assert set(report.coefficients["composite"]) == set(COMPOSITE_FEATURES)
    assert len(report.trade_evals) == 2


def test_signal_model_single_l2_and_intercept(markets: list[SyntheticMarket]) -> None:
    model = SignalModel(SignalConfig(l2_grid=(1.0,), fit_intercept=True)).fit(markets[:4])
    assert model.selected_l2 == {"info": 1.0, "composite": 1.0}
    assert "intercept" in model.coefficients()["info"]


def test_signal_errors(markets: list[SyntheticMarket]) -> None:
    with pytest.raises(ValueError, match="training"):
        SignalModel().fit([])
    model = SignalModel(SignalConfig(l2_grid=(1.0,))).fit(markets[:3])
    with pytest.raises(ValueError, match="test"):
        evaluate(model, markets[:3], [])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train_fraction": 1.0},
        {"train_stride": 0},
        {"l2_grid": ()},
        {"l2_grid": (-1.0,)},
        {"cv_folds": 1},
    ],
)
def test_signal_config_validation(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        SignalConfig(**kwargs)  # type: ignore[arg-type]


def test_trade_eval_hand_computed() -> None:
    direction = np.array([1.0, -1.0, 0.0])
    mid = np.array([0.40, 0.70, 0.5])
    bid = np.array([0.39, 0.69, 0.49])
    ask = np.array([0.41, 0.71, 0.51])
    outcome = np.array([1.0, 1.0, 0.0])
    ev = trade_eval("x", direction, mid, bid, ask, outcome, np.array([0, 1, 2]))
    assert ev.n_trades == 2
    assert ev.n_markets == 2
    # Buy at 0.41 -> +0.59; sell at 0.69 -> -0.31
    assert ev.mean_pnl_touch == pytest.approx((0.59 - 0.31) / 2)
    assert ev.mean_pnl_mid == pytest.approx((0.60 - 0.30) / 2)
    assert ev.hit_rate == 0.5
    empty = trade_eval("none", np.zeros(3), mid, bid, ask, outcome, np.array([0, 1, 2]))
    assert empty.n_trades == 0
    assert np.isnan(empty.hit_rate)
