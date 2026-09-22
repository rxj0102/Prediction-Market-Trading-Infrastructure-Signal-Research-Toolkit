"""End-to-end pipeline: market sim -> signal -> execution -> monitoring -> tear sheet."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import main
from execution.backtester import BacktestConfig, Backtester
from market_sim.markets import MarketConfig, SyntheticMarket, generate_universe
from monitoring.tearsheet import build_tearsheet
from signals.research import MarketSignal, SignalModel, evaluate
from tests.helpers import make_signal


def test_full_pipeline_runs_and_produces_valid_tearsheets(tmp_path: Path) -> None:
    lines: list[str] = []
    art = main.run(
        seed=11,
        n_markets=10,
        n_history=30,
        out_dir=tmp_path,
        log_every=48,
        sensitivity=False,
        printer=lines.append,
    )
    text = "\n".join(lines)
    # Every module reported.
    for section in (
        "MODULE 1",
        "MODULE 2",
        "MODULES 3 & 4",
        "Circuit-breaker stress test",
        "Execution style comparison",
        "Passive fill-probability model check",
    ):
        assert section in text
    assert "SYNTHETIC" in text
    # Tear sheets are complete and numerically sane for both styles.
    assert set(art.tearsheets) == {"aggressive", "passive"}
    for style, ts in art.tearsheets.items():
        res = art.results[style]
        assert ts.n_steps == len(res.equity) > 0
        assert math.isfinite(ts.final_equity)
        assert ts.final_equity == pytest.approx(10_000 + sum(res.market_pnl.values()))
        assert ts.brier_composite is not None
        assert 0.0 <= ts.brier_composite <= 1.0
        assert 0.0 <= ts.max_drawdown < 1.0
        for suffix in (".txt", "_equity.csv", "_summary.json"):
            assert (tmp_path / f"tearsheet_{style}{suffix}").exists()
    assert (tmp_path / "synthetic_markets.csv").exists()
    assert art.sensitivity == []
    assert art.report.n_test_markets == 10


def test_pipeline_components_compose_directly() -> None:
    live = generate_universe(MarketConfig(n_markets=6, seed=2))
    hist = generate_universe(MarketConfig(n_markets=20, seed=2 + main.HISTORY_SEED_OFFSET))
    model = SignalModel().fit(hist)
    report = evaluate(model, hist, live)
    signals = {m.market_id: model.compute(m) for m in live}
    res = Backtester(live, signals, BacktestConfig()).run()
    sheet = build_tearsheet(
        res, "direct", report.scores["composite"].brier, report.scores["market_mid"].brier
    )
    assert "TEAR SHEET" in sheet.render()
    assert res.final_positions == {}


def test_main_cli_and_breaker_section(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main.main(
            [
                "--seed",
                "3",
                "--n-markets",
                "6",
                "--history-markets",
                "20",
                "--output-dir",
                str(tmp_path),
                "--skip-sensitivity",
                "--log-every",
                "96",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "done." in out
    assert "CHECK: breaker" in out or "did NOT fire" in out


def test_circuit_breaker_section_reports_pass() -> None:
    live = generate_universe(MarketConfig(n_markets=12, seed=42))
    hist = generate_universe(MarketConfig(n_markets=40, seed=42 + main.HISTORY_SEED_OFFSET))
    model = SignalModel().fit(hist)
    signals = {m.market_id: model.compute(m) for m in live}
    lines: list[str] = []
    res = main.section_circuit_breaker(lines.append, live, signals)
    assert res.breaker_events
    assert any("CHECK: breaker fired and flattened all positions -> PASS" in x for x in lines)


def test_circuit_breaker_section_when_nothing_trades() -> None:
    live = generate_universe(MarketConfig(n_markets=3, seed=1))
    flat = {m.market_id: model_free_signal(m) for m in live}
    lines: list[str] = []
    res = main.section_circuit_breaker(lines.append, live, flat)
    assert not res.breaker_events
    assert any("did NOT fire" in x for x in lines)


def model_free_signal(m: SyntheticMarket) -> MarketSignal:
    return make_signal(m, m.mid)  # composite == mid -> zero edge -> no trades


def test_sensitivity_section_small() -> None:
    lines: list[str] = []
    rows = main.section_sensitivity(lines.append, seed=4, n_markets=4, n_history=12)
    assert len(rows) == 7
    assert all(np.isfinite(r["pnl"]) for r in rows)
    assert any("analyst noise" in x for x in lines)
