"""End-to-end run of the toolkit on SYNTHETIC data.

    python main.py                      # full run, prints everything, writes output/
    python main.py --seed 7 --n-markets 30

Pipeline: synthetic markets + order books (Module 1) -> signal research and
calibration (Module 2) -> event-driven backtests with aggressive and passive
execution (Module 3) -> live-style monitoring, circuit breaker and tear sheets
(Module 4). Every number printed here is computed by this run; nothing is
hard-coded, and none of it describes a real market.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from execution.backtester import (
    BacktestConfig,
    Backtester,
    BacktestResult,
    ExecutionStyle,
)
from market_sim.book_builder import build_book
from market_sim.generate_data import write_csv
from market_sim.markets import (
    DEFAULT_SEED,
    LiquidityTier,
    MarketConfig,
    SyntheticMarket,
    generate_universe,
)
from market_sim.order_book import LimitOrderBook, Side
from monitoring.tearsheet import TearSheet, build_tearsheet
from signals.research import MarketSignal, SignalModel, SignalReport, evaluate

HISTORY_SEED_OFFSET = 10_000
Printer = Callable[[str], None]


@dataclass
class RunArtifacts:
    live: list[SyntheticMarket]
    report: SignalReport
    results: dict[str, BacktestResult]
    tearsheets: dict[str, TearSheet]
    stress: BacktestResult
    sensitivity: list[dict[str, float]]


def header(p: Printer, title: str) -> None:
    p("")
    p("#" * 78)
    p(f"# {title}")
    p("#" * 78)


def fmt(x: float, spec: str = ".4f") -> str:
    return "n/a" if math.isnan(x) else format(x, spec)


# ----------------------------------------------------------------- Module 1
def section_market_sim(
    p: Printer, live: list[SyntheticMarket], cfg: MarketConfig, out_dir: Path
) -> None:
    header(p, "MODULE 1 - Synthetic prediction markets & limit order book")
    p(
        f"generator seed {cfg.seed}; {len(live)} markets; global horizon {cfg.horizon} steps "
        "(1 step = 1 synthetic hour)"
    )
    p(f"YES outcomes: {sum(m.outcome for m in live)}/{len(live)}")
    p(
        f"{'tier':<7} {'n':>3} {'avg len':>8} {'half-spread(c)':>15} {'top depth':>10} "
        f"{'vol/step':>9}"
    )
    for tier in LiquidityTier:
        ms = [m for m in live if m.tier is tier]
        if not ms:
            continue
        p(
            f"{tier.value:<7} {len(ms):>3} {np.mean([m.length for m in ms]):>8.1f} "
            f"{np.mean([m.half_spread_ticks.mean() for m in ms]):>15.2f} "
            f"{np.mean([m.top_depth.mean() for m in ms]):>10.1f} "
            f"{np.mean([m.volume.mean() for m in ms]):>9.1f}"
        )
    rows = write_csv(live, out_dir / "synthetic_markets.csv", cfg.seed)
    p(f"wrote {rows} rows of synthetic market data to {out_dir / 'synthetic_markets.csv'}")

    p("")
    p("Order book mechanics demo (price-time priority + partial fill):")
    book = LimitOrderBook()
    book.submit_limit(Side.SELL, 55, 10, owner="maker_A")
    book.submit_limit(Side.SELL, 55, 10, owner="maker_B")
    book.submit_limit(Side.SELL, 56, 5, owner="maker_C")
    res = book.submit_market(Side.BUY, 22, owner="taker")
    for f in res.fills:
        p(f"  fill {f.qty:>3} @ {f.price_dollars:.2f} vs {f.maker_owner}")
    p(
        f"  filled {res.filled_qty}/22, avg {res.avg_price:.4f}; maker_B left with "
        f"{book.level_qty(Side.SELL, 55)} at 0.55"
    )
    res2 = book.submit_market(Side.BUY, 10, owner="taker")
    p(
        f"  follow-up market buy 10: filled {res2.filled_qty}, unfilled {res2.remaining_qty} "
        "(book exhausted -> partial fill, remainder cancelled)"
    )

    p("")
    p("Depth-dependent slippage: market BUY vs mid, one representative market per tier")
    sizes = (10, 50, 100, 250, 500)
    p(f"  {'tier':<7} " + " ".join(f"{'q=' + str(q):>16}" for q in sizes))
    rng = np.random.default_rng(cfg.seed)
    for tier in LiquidityTier:
        ms = [m for m in live if m.tier is tier]
        if not ms:
            continue
        m = ms[0]
        i = m.length // 2
        cells = []
        for q in sizes:
            b = build_book(float(m.mid[i]), int(m.half_spread_ticks[i]), m.depth_profile(i), rng)
            mid = b.mid()
            r = b.submit_market(Side.BUY, q)
            assert mid is not None
            assert r.avg_price is not None
            cells.append(f"{(r.avg_price - mid) * 100:+6.2f}c {r.filled_qty:>4}/{q:<4}")
        p(f"  {tier.value:<7} " + " ".join(f"{c:>16}" for c in cells))
    p("  (cost per contract in cents vs mid; filled/requested)")


# ----------------------------------------------------------------- Module 2
def section_signals(
    p: Printer, live: list[SyntheticMarket], history: list[SyntheticMarket]
) -> tuple[SignalModel, SignalReport, dict[str, MarketSignal]]:
    header(p, "MODULE 2 - Mispricing signal research & calibration")
    model = SignalModel().fit(history)
    report = evaluate(model, history, live)
    p(
        f"training universe: {len(history)} separate synthetic resolved markets "
        f"(seed offset {HISTORY_SEED_OFFSET}); evaluation: the {len(live)} live markets "
        "(out-of-sample)"
    )
    p(f"L2 chosen by grouped 5-fold CV on training markets: {model.selected_l2}")
    for stage, coefs in report.coefficients.items():
        p(f"  {stage:<9} coefficients: " + ", ".join(f"{k}={v:+.3f}" for k, v in coefs.items()))
    p(f"evaluation rows: {report.n_obs} (every 4th step), base rate {report.base_rate:.2f}")
    p("")
    p(
        f"  {'forecast':<18} {'Brier':>7} {'logloss':>8} {'ECE':>7} {'reliab':>8} "
        f"{'resol':>8} {'uncert':>8}"
    )
    for name, s in report.scores.items():
        d = s.decomposition
        p(
            f"  {name:<18} {s.brier:7.4f} {s.log_loss:8.4f} {s.ece:7.4f} {d.reliability:8.4f} "
            f"{d.resolution:8.4f} {d.uncertainty:8.4f}"
        )
    p(
        "  (oracle_true_prob is the latent generator probability - a diagnostic ceiling that no "
        "strategy code can see)"
    )
    for name in ("composite", "market_mid"):
        p("")
        p(f"  calibration curve - {name}")
        p(f"    {'bucket':<11} {'n':>5} {'mean forecast':>14} {'observed YES':>13} {'gap':>7}")
        for b in report.scores[name].bins:
            p(
                f"    [{b.lower:.1f}, {b.upper:.1f}) {b.count:>5} {b.mean_forecast:>14.3f} "
                f"{b.observed_freq:>13.3f} {b.gap:>+7.3f}"
            )
    p("")
    p("  information coefficients (Pearson correlation):")
    for k, v in report.information_coefficient.items():
        p(f"    {k:<40} {v:+.4f}")
    p("")
    p(
        "  PnL-if-traded (1 contract toward the edge, hold to resolution, "
        f"|edge| > {model.config.edge_threshold}):"
    )
    p(
        f"    {'rule':<30} {'trades':>6} {'mkts':>5} {'hit(mid)':>9} {'pnl/ct mid':>11} "
        f"{'pnl/ct touch':>13} {'total touch':>12}"
    )
    for t in report.trade_evals:
        p(
            f"    {t.label:<30} {t.n_trades:>6} {t.n_markets:>5} {fmt(t.hit_rate, '9.3f'):>9} "
            f"{fmt(t.mean_pnl_mid, '+.4f'):>11} {fmt(t.mean_pnl_touch, '+.4f'):>13} "
            f"{t.total_pnl_touch:>+12.2f}"
        )
    signals = {m.market_id: model.compute(m) for m in live}
    return model, report, signals


# --------------------------------------------------------------- Module 3+4
def section_backtests(
    p: Printer,
    live: list[SyntheticMarket],
    signals: dict[str, MarketSignal],
    report: SignalReport,
    out_dir: Path,
    log_every: int,
) -> tuple[dict[str, BacktestResult], dict[str, TearSheet]]:
    header(p, "MODULES 3 & 4 - Event-driven backtests with live monitoring")
    base = BacktestConfig()
    lim = base.limits
    p(
        f"capital ${base.initial_capital:,.0f}; max position {lim.max_position_per_market} "
        f"contracts/market; max gross exposure ${lim.max_gross_exposure:,.0f} (worst-case loss); "
        f"Kelly x{lim.kelly_fraction}; entry/exit edge {lim.entry_edge}/{lim.exit_edge}; "
        f"drawdown throttle {lim.throttle_start:.0%}->{lim.throttle_full:.0%}; "
        f"circuit breaker at {base.monitor.max_drawdown_limit:.0%} drawdown; "
        f"fees ${base.fee_per_contract}/contract; impact {base.impact_ticks_per_depth} ticks per "
        f"top-depth taken, decay {base.impact_decay}/step"
    )
    results: dict[str, BacktestResult] = {}
    sheets: dict[str, TearSheet] = {}
    for style in ExecutionStyle:
        p("")
        p(f"--- backtest: {style.value} execution (monitor snapshot every {log_every} steps) ---")
        res = Backtester(
            live, signals, replace(base, style=style), log=p, log_every=log_every
        ).run()
        results[style.value] = res
        sheet = build_tearsheet(
            res,
            "composite signal",
            report.scores["composite"].brier,
            report.scores["market_mid"].brier,
        )
        sheets[style.value] = sheet
        p(sheet.render())
        paths = sheet.write(out_dir, f"tearsheet_{style.value}")
        p("wrote " + ", ".join(str(x) for x in paths))

    p("")
    p("Execution style comparison (same signal, same markets, same risk limits)")
    agg, pas = results["aggressive"], results["passive"]
    rows: list[tuple[str, Callable[[BacktestResult], str]]] = [
        ("fills", lambda r: f"{r.cost_summary().n_fills:d}"),
        ("contracts traded", lambda r: f"{r.cost_summary().contracts:,d}"),
        ("spread cost $", lambda r: f"{r.cost_summary().spread_cost:+,.2f}"),
        ("slippage cost $", lambda r: f"{r.cost_summary().slippage_cost:+,.2f}"),
        ("total cost $", lambda r: f"{r.cost_summary().total_cost:+,.2f}"),
        ("cost / contract (c)", lambda r: f"{r.cost_summary().cost_per_contract * 100:+.3f}"),
        ("markout / contract (c)", lambda r: f"{r.markout_per_contract() * 100:+.3f}"),
        ("net PnL $", lambda r: f"{r.total_pnl:+,.2f}"),
    ]
    p(f"  {'metric':<24} {'aggressive':>14} {'passive':>14}")
    for label, fn in rows:
        p(f"  {label:<24} {fn(agg):>14} {fn(pas):>14}")
    sharpe_agg = fmt(sheets["aggressive"].sharpe_daily_annualised, ".2f")
    sharpe_pas = fmt(sheets["passive"].sharpe_daily_annualised, ".2f")
    p(f"  {'sharpe (daily, ann.)':<24} {sharpe_agg:>14} {sharpe_pas:>14}")
    p(
        f"  {'max drawdown':<24} {sheets['aggressive'].max_drawdown:>14.2%}"
        f" {sheets['passive'].max_drawdown:>14.2%}"
    )
    ps = pas.passive_stats
    p(
        f"  passive: {ps.orders} orders, {ps.contracts_filled:,}/{ps.contracts_ordered:,} "
        f"contracts filled ({ps.fill_rate:.1%}), {ps.fully_filled} fully filled, {ps.cancelled} "
        "cancelled/partial"
    )
    diff = (agg.cost_summary().cost_per_contract - pas.cost_summary().cost_per_contract) * 100
    p(f"  realised cost difference (aggressive - passive): {diff:+.3f} cents per filled contract")

    p("")
    p("Passive fill-probability model check (ex-ante Poisson estimate vs realised full fill)")
    exp = np.array(ps.expected_fill_prob)
    real = np.array(ps.realized_full_fill)
    p(f"  {'predicted bucket':<17} {'orders':>7} {'mean predicted':>15} {'realised':>9}")
    for lo, hi in ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)):
        mask = (exp >= lo) & (exp < hi)
        if mask.any():
            p(
                f"  [{lo:.1f}, {min(hi, 1.0):.1f}){'':<7} {int(mask.sum()):>7} "
                f"{exp[mask].mean():>15.3f} {real[mask].mean():>9.3f}"
            )
    return results, sheets


def section_circuit_breaker(
    p: Printer, live: list[SyntheticMarket], signals: dict[str, MarketSignal]
) -> BacktestResult:
    header(p, "MODULE 4 - Circuit-breaker stress test")
    p(
        "Scenario: the strategy trades the INVERTED signal (a deliberately broken model) with "
        "the default risk limits, so losses accumulate until the drawdown breaker fires."
    )
    cfg = BacktestConfig(invert_signal=True)
    res = Backtester(live, signals, cfg, log=p, log_every=None).run()
    if not res.breaker_events:
        p("circuit breaker did NOT fire in this scenario")
        return res
    ev = res.breaker_events[0]
    p(
        f"breaker fired at step {ev.step}: drawdown {ev.drawdown:.2%} "
        f"(limit {cfg.monitor.max_drawdown_limit:.0%}), equity ${ev.equity:,.2f}"
    )
    p(
        f"open positions at trip: {len(ev.positions_before)} markets, "
        f"{sum(abs(v) for v in ev.positions_before.values()):,} contracts"
    )
    flat_cost = sum(r.total_cost for r in ev.flatten_records)
    p(
        f"flatten orders: {len(ev.flatten_records)} fills, cost ${flat_cost:,.2f}; "
        f"flat by step {ev.completed_step}"
    )
    trades_after = [r for r in res.records if r.step > (ev.completed_step or ev.step)]
    all_flat = ev.completed_step is not None and not ev.positions_after
    p(
        f"positions after flatten: {ev.positions_after or '{}'}; new trades after halt: "
        f"{len(trades_after)}"
    )
    p(f"final equity ${res.final_equity:,.2f}; max drawdown {res.monitor.max_drawdown:.2%}")
    p(
        f"CHECK: breaker fired and flattened all positions -> "
        f"{'PASS' if all_flat and not trades_after else 'FAIL'}"
    )
    return res


def section_sensitivity(
    p: Printer, seed: int, n_markets: int, n_history: int
) -> list[dict[str, float]]:
    header(p, "Sensitivity: results as a function of the generator's assumed inefficiency")
    p(
        "The edge exists only because the generator builds it in. Re-running the full pipeline "
        "(new markets, refit, aggressive backtest) while varying those assumptions:"
    )
    p(
        f"  {'analyst noise':>13} {'sentiment':>9} {'kappa':>5} {'impact':>6} "
        f"{'Brier comp-mkt':>14} {'IC outcome':>10} {'IC fwd':>7} {'PnL $':>10} "
        f"{'Sharpe(d)':>9} {'maxDD':>7}"
    )
    rows: list[dict[str, float]] = []
    grid = [
        (0.05, 0.12, 0.08, 2.0),
        (0.10, 0.12, 0.08, 2.0),
        (0.20, 0.12, 0.08, 2.0),
        (0.10, 0.06, 0.08, 2.0),
        (0.10, 0.03, 0.08, 2.0),
        (0.10, 0.12, 0.30, 2.0),
        (0.10, 0.03, 0.30, 2.0),
        (0.10, 0.12, 0.08, 0.0),
        (0.10, 0.12, 0.08, 6.0),
    ]
    for noise, senti, kappa, impact in grid:
        mc = MarketConfig(
            n_markets=n_markets,
            seed=seed,
            fundamental_noise_sd=noise,
            poll_noise_sd=2 * noise,
            sentiment_sigma=senti,
            market_kappa=kappa,
        )
        live = generate_universe(mc)
        hist = generate_universe(replace(mc, n_markets=n_history, seed=seed + HISTORY_SEED_OFFSET))
        model = SignalModel().fit(hist)
        rep = evaluate(model, hist, live)
        sigs = {m.market_id: model.compute(m) for m in live}
        res = Backtester(live, sigs, BacktestConfig(impact_ticks_per_depth=impact)).run()
        sheet = build_tearsheet(res, "sensitivity")
        ics = list(rep.information_coefficient.values())
        row = {
            "analyst_noise": noise,
            "sentiment_sigma": senti,
            "market_kappa": kappa,
            "impact": impact,
            "brier_gap": rep.scores["composite"].brier - rep.scores["market_mid"].brier,
            "ic_outcome": ics[1],
            "ic_forward": ics[2],
            "pnl": res.total_pnl,
            "sharpe_daily": sheet.sharpe_daily_annualised,
            "max_dd": sheet.max_drawdown,
        }
        rows.append(row)
        p(
            f"  {noise:>13.2f} {senti:>9.2f} {kappa:>5.2f} {impact:>6.1f} "
            f"{row['brier_gap']:>+14.4f} {row['ic_outcome']:>+10.3f} {row['ic_forward']:>+7.3f} "
            f"{row['pnl']:>+10.2f} {fmt(row['sharpe_daily'], '.2f'):>9} {row['max_dd']:>7.2%}"
        )
    p("  (default configuration: analyst noise 0.10, sentiment 0.12, kappa 0.08, impact 2.0;")
    p("   kappa = share of the gap to the latent probability the price closes each step)")
    return rows


def run(
    seed: int = DEFAULT_SEED,
    n_markets: int = 50,
    n_history: int = 200,
    out_dir: Path = Path("output"),
    log_every: int = 48,
    sensitivity: bool = True,
    printer: Printer = print,
) -> RunArtifacts:
    p = printer
    p("=" * 78)
    p("PREDICTION MARKET TRADING ENGINE - end-to-end run")
    p("ALL MARKET DATA BELOW IS SYNTHETIC (generated from a fixed seed). No real market data,")
    p("and no result here is a claim about real prediction markets.")
    p("=" * 78)
    cfg = MarketConfig(n_markets=n_markets, seed=seed)
    live = generate_universe(cfg)
    history = generate_universe(replace(cfg, n_markets=n_history, seed=seed + HISTORY_SEED_OFFSET))
    section_market_sim(p, live, cfg, out_dir)
    _, report, signals = section_signals(p, live, history)
    results, sheets = section_backtests(p, live, signals, report, out_dir, log_every)
    stress = section_circuit_breaker(p, live, signals)
    sens = section_sensitivity(p, seed, n_markets, n_history) if sensitivity else []
    p("")
    p("done.")
    return RunArtifacts(live, report, results, sheets, stress, sens)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the full synthetic pipeline.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-markets", type=int, default=50)
    parser.add_argument("--history-markets", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--log-every", type=int, default=48)
    parser.add_argument("--skip-sensitivity", action="store_true")
    args = parser.parse_args(argv)
    run(
        args.seed,
        args.n_markets,
        args.history_markets,
        args.output_dir,
        args.log_every,
        not args.skip_sensitivity,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
