"""Tests for the synthetic market generator, book builder and data script."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from market_sim import generate_data
from market_sim.book_builder import DepthProfile, build_book, touch_prices
from market_sim.markets import (
    LiquidityTier,
    MarketConfig,
    SyntheticMarket,
    generate_universe,
    norm_cdf,
)
from market_sim.order_book import Side


@pytest.fixture(scope="module")
def universe() -> list[SyntheticMarket]:
    return generate_universe(MarketConfig(n_markets=40, seed=7))


def test_universe_is_deterministic_in_seed() -> None:
    a = generate_universe(MarketConfig(n_markets=3, seed=11))
    b = generate_universe(MarketConfig(n_markets=3, seed=11))
    c = generate_universe(MarketConfig(n_markets=3, seed=12))
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x.mid, y.mid)
        assert x.outcome == y.outcome
    assert not np.array_equal(a[0].mid, c[0].mid)


def test_markets_are_labelled_synthetic(universe: list[SyntheticMarket]) -> None:
    for m in universe:
        assert m.synthetic
        assert m.question.startswith("[SYNTHETIC]")
        assert m.market_id.startswith("SYN-")


def test_price_paths_start_near_half_and_converge(universe: list[SyntheticMarket]) -> None:
    starts = np.array([m.mid[0] for m in universe])
    assert np.all(np.abs(starts - 0.5) < 0.25)
    # Late in life, true probabilities are much closer to the realised outcome.
    early = np.mean([abs(m.true_prob[0] - m.outcome) for m in universe])
    late = np.mean([abs(m.true_prob[-1] - m.outcome) for m in universe])
    assert late < early
    for m in universe:
        assert np.all((m.mid >= 0.02) & (m.mid <= 0.98))
        assert np.all((m.true_prob >= 0.0) & (m.true_prob <= 1.0))


def test_arrays_consistent(universe: list[SyntheticMarket]) -> None:
    cfg = MarketConfig()
    for m in universe:
        n = m.length
        assert cfg.min_length <= n <= cfg.max_length
        for arr in (
            m.true_prob,
            m.half_spread_ticks,
            m.top_depth,
            m.volume,
            m.buy_flow,
            m.sell_flow,
            m.fundamental_z,
            m.poll_z,
        ):
            assert len(arr) == n
        np.testing.assert_array_equal(m.buy_flow + m.sell_flow, m.volume)
        assert m.close_step <= cfg.horizon
        assert m.outcome in (0, 1)


def test_thin_markets_have_wider_spreads_and_less_depth(universe: list[SyntheticMarket]) -> None:
    by_tier = {t: [m for m in universe if m.tier is t] for t in LiquidityTier}
    assert all(by_tier.values())
    spread = {t: np.mean([m.half_spread_ticks.mean() for m in ms]) for t, ms in by_tier.items()}
    depth = {t: np.mean([m.top_depth.mean() for m in ms]) for t, ms in by_tier.items()}
    assert spread[LiquidityTier.THIN] > spread[LiquidityTier.MEDIUM] >= spread[LiquidityTier.DEEP]
    assert depth[LiquidityTier.THIN] < depth[LiquidityTier.MEDIUM] < depth[LiquidityTier.DEEP]


def test_active_window_and_local_index(universe: list[SyntheticMarket]) -> None:
    m = universe[0]
    assert m.is_active(m.open_step)
    assert not m.is_active(m.close_step)
    assert m.local_index(m.open_step + 3) == 3
    with pytest.raises(IndexError):
        m.local_index(m.close_step)


def test_market_book_matches_touch(universe: list[SyntheticMarket]) -> None:
    m = universe[0]
    book = m.build_book(5, np.random.default_rng(0))
    assert (book.best_bid(), book.best_ask()) == m.touch(5)


def test_true_prob_is_calibrated_in_aggregate() -> None:
    # p_i is an exact conditional probability, so across many markets the
    # mean of p at open should be close to the realised YES frequency.
    ms = generate_universe(
        MarketConfig(n_markets=300, seed=3, min_length=20, max_length=30, horizon=40)
    )
    p0 = np.mean([m.true_prob[0] for m in ms])
    freq = np.mean([m.outcome for m in ms])
    assert abs(p0 - freq) < 0.1


def test_norm_cdf_known_values() -> None:
    vals = norm_cdf(np.array([0.0, 1.959963984540054, -1.0]))
    np.testing.assert_allclose(vals, [0.5, 0.975, 0.15865525393145707], atol=1e-9)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_markets": 0},
        {"min_length": 1},
        {"min_length": 50, "max_length": 40},
        {"max_length": 600},
        {"tier_weights": (0.5, 0.5, 0.5)},
    ],
)
def test_config_validation(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        MarketConfig(**kwargs)


# ------------------------------------------------------------------ book builder
def test_touch_prices_normal_and_edges() -> None:
    assert touch_prices(0.50, 2) == (48, 52)
    assert touch_prices(0.02, 3) == (1, 5)
    assert touch_prices(0.99, 3) == (96, 99)
    with pytest.raises(ValueError, match="half_spread"):
        touch_prices(0.5, 0)


def test_build_book_geometric_depth_without_noise() -> None:
    profile = DepthProfile(top_depth=100, decay=0.5, n_levels=4)
    book = build_book(0.40, 1, profile)
    assert book.depth(Side.BUY) == [(39, 100), (38, 50), (37, 25), (36, 12)]
    assert book.depth(Side.SELL) == [(41, 100), (42, 50), (43, 25), (44, 12)]


def test_build_book_truncates_at_price_bounds() -> None:
    book = build_book(0.03, 1, DepthProfile(top_depth=10, decay=1.0, n_levels=5))
    assert [p for p, _ in book.depth(Side.BUY)] == [2, 1]
    assert len(book.depth(Side.SELL)) == 5


def test_depth_dependent_slippage() -> None:
    thin = build_book(0.5, 1, DepthProfile(top_depth=20, decay=0.8))
    deep = build_book(0.5, 1, DepthProfile(top_depth=500, decay=0.9))
    thin_px = thin.submit_market(Side.BUY, 150).avg_price
    deep_px = deep.submit_market(Side.BUY, 150).avg_price
    assert thin_px is not None
    assert deep_px is not None
    assert thin_px > deep_px == pytest.approx(0.51)


def test_level_sizes_noise_is_positive() -> None:
    sizes = DepthProfile(top_depth=5, decay=0.5).level_sizes(np.random.default_rng(1))
    assert all(s >= 1 for s in sizes)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_depth": 0, "decay": 0.5},
        {"top_depth": 1, "decay": 0.0},
        {"top_depth": 1, "decay": 0.5, "n_levels": 0},
    ],
)
def test_depth_profile_validation(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        DepthProfile(**kwargs)


# ------------------------------------------------------------------ data script
def test_generate_data_script(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "data.csv"
    assert generate_data.main(["--seed", "5", "--n-markets", "2", "--out", str(out)]) == 0
    lines = out.read_text().splitlines()
    assert lines[0].startswith("# SYNTHETIC DATA")
    assert lines[1].split(",") == list(generate_data.COLUMNS)
    assert all(line.startswith("synthetic,") for line in lines[2:])
    assert "synthetic rows" in capsys.readouterr().out
    # Reproducible byte-for-byte.
    out2 = tmp_path / "data2.csv"
    generate_data.main(["--seed", "5", "--n-markets", "2", "--out", str(out2)])
    assert out.read_bytes() == out2.read_bytes()
