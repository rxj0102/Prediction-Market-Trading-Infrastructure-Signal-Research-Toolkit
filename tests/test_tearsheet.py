"""Tear sheet construction and export."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from execution.backtester import BacktestConfig, Backtester, BacktestResult, ExecutionStyle
from market_sim.markets import MarketConfig, generate_universe
from monitoring.tearsheet import TearSheet, build_tearsheet, daily_returns, sparkline
from tests.helpers import make_signal

Sheets = dict[str, tuple[BacktestResult, TearSheet]]


@pytest.fixture(scope="module")
def sheets() -> Sheets:
    u = generate_universe(
        MarketConfig(n_markets=6, seed=9, min_length=60, max_length=80, horizon=100)
    )
    sigs = {
        m.market_id: make_signal(m, np.clip(0.5 * m.true_prob + 0.5 * m.outcome, 0.02, 0.98))
        for m in u
    }
    out: Sheets = {}
    for style in ExecutionStyle:
        res = Backtester(u, sigs, BacktestConfig(style=style)).run()
        out[style.value] = (res, build_tearsheet(res, "test", 0.17, 0.18))
    return out


def test_tearsheet_numbers_match_result(sheets: Sheets) -> None:
    res, ts = sheets["aggressive"]
    assert ts.final_equity == pytest.approx(res.final_equity)
    assert ts.total_pnl == pytest.approx(res.total_pnl)
    assert ts.total_return == pytest.approx(res.total_pnl / 10_000)
    assert ts.n_steps == len(res.equity)
    assert ts.n_fills == len(res.records)
    assert ts.total_cost == pytest.approx(ts.spread_cost + ts.slippage_cost + ts.fees)
    assert 0.0 <= ts.max_drawdown < 1.0
    assert 0.0 <= ts.market_hit_rate <= 1.0
    assert ts.passive_fill_rate is None
    assert ts.brier_composite == 0.17
    assert not ts.breaker_fired
    assert sum(int(c["n_fills"]) for c in ts.cost_by_tier.values()) == ts.n_fills
    assert len(ts.equity_curve) == ts.n_steps
    passive_ts = sheets["passive"][1]
    assert passive_ts.passive_fill_rate is not None
    assert 0.0 < passive_ts.passive_fill_rate <= 1.0


def test_render_contains_every_section(sheets: Sheets) -> None:
    text = sheets["passive"][1].render()
    for needle in (
        "TEAR SHEET",
        "SYNTHETIC",
        "Sharpe (daily, ann.)",
        "max drawdown",
        "hit rate",
        "Brier composite",
        "slippage",
        "cost per contract",
        "circuit breaker trips",
        "Equity curve",
        "sparkline",
    ):
        assert needle in text


def test_write_exports_numeric_files(sheets: Sheets, tmp_path: Path) -> None:
    ts = sheets["aggressive"][1]
    txt, curve, summary = ts.write(tmp_path / "out", "ts")
    assert txt.read_text().startswith("=" * 78)
    rows = list(csv.reader(curve.open()))
    assert rows[0] == ["step", "equity", "drawdown", "gross_exposure"]
    assert len(rows) == ts.n_steps + 1
    data = json.loads(summary.read_text())
    assert data["data_source"] == "synthetic"
    assert data["total_pnl"] == pytest.approx(ts.total_pnl)
    assert "equity_curve" not in data


def test_render_handles_missing_values(sheets: Sheets) -> None:
    ts = sheets["aggressive"][1]
    ts2 = type(ts)(
        **{
            **ts.__dict__,
            "brier_composite": None,
            "sharpe_daily_annualised": math.nan,
            "equity_curve": [],
            "circuit_breaker_trips": [5],
        }
    )
    text = ts2.render()
    assert "n/a" in text
    assert "circuit breaker trips      5" in text
    assert ts2.breaker_fired


def test_sparkline_and_daily_returns() -> None:
    assert sparkline([]) == ""
    assert len(set(sparkline([1.0, 1.0, 1.0]))) == 1
    line = sparkline([1.0, 2.0, 3.0])
    assert line[0] == " "
    assert line[-1] == "@"
    assert daily_returns([]) == []
    eq = [100.0] * 24 + [110.0] * 24 + [121.0]
    np.testing.assert_allclose(daily_returns(eq, 24), [0.1, 0.1])
    np.testing.assert_allclose(daily_returns([100.0, 105.0, 110.0], 24), [0.1])
