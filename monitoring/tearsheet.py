"""End-of-run performance tear sheet: numbers first, written to text/CSV/JSON.

Sharpe ratios are reported two ways because the synthetic clock is hourly:
per-step returns annualised with sqrt(8760), and returns aggregated to 24-step
"days" annualised with sqrt(365). The whole run covers only a few hundred
synthetic hours, so any annualised figure is an extrapolation of a short
synthetic sample, not a forecast.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from execution.backtester import BacktestResult
from monitoring.metrics import drawdown_series, max_drawdown, sharpe_ratio, simple_returns

SPARK = " .:-=+*#%@"


@dataclass
class TearSheet:
    title: str
    style: str
    initial_capital: float
    final_equity: float
    total_pnl: float
    total_return: float
    sharpe_hourly_annualised: float
    sharpe_daily_annualised: float
    max_drawdown: float
    n_steps: int
    n_fills: int
    contracts_traded: int
    markets_traded: int
    market_hit_rate: float
    avg_market_pnl: float
    best_market_pnl: float
    worst_market_pnl: float
    brier_composite: float | None
    brier_market: float | None
    spread_cost: float
    slippage_cost: float
    fees: float
    total_cost: float
    cost_per_contract: float
    avg_slippage_per_fill: float
    markout_per_contract: float
    passive_fill_rate: float | None
    circuit_breaker_trips: list[int]
    max_gross_exposure: float
    cost_by_tier: dict[str, dict[str, float]] = field(default_factory=dict)
    equity_curve: list[tuple[int, float, float, float]] = field(default_factory=list)

    @property
    def breaker_fired(self) -> bool:
        return bool(self.circuit_breaker_trips)

    # ----------------------------------------------------------------- output
    def render(self, curve_rows: int = 12) -> str:
        def f(x: float | None, spec: str) -> str:
            return (
                "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)
            )

        lines = [
            "=" * 78,
            f"TEAR SHEET - {self.title}  [style: {self.style}]  (SYNTHETIC DATA)",
            "=" * 78,
            "Performance",
            f"  initial capital            ${self.initial_capital:,.2f}",
            f"  final equity               ${self.final_equity:,.2f}",
            f"  total PnL                  ${self.total_pnl:+,.2f}  ({self.total_return:+.2%})",
            f"  Sharpe (hourly, ann.)      {f(self.sharpe_hourly_annualised, '.2f')}",
            f"  Sharpe (daily, ann.)       {f(self.sharpe_daily_annualised, '.2f')}",
            f"  max drawdown               {self.max_drawdown:.2%}",
            f"  max gross exposure         ${self.max_gross_exposure:,.2f}",
            f"  steps (synthetic hours)    {self.n_steps}",
            "Trading",
            f"  fills / contracts          {self.n_fills} / {self.contracts_traded:,}",
            f"  markets traded             {self.markets_traded}",
            f"  hit rate (markets w/ PnL>0) {f(self.market_hit_rate, '.2%')}",
            f"  avg / best / worst market  ${self.avg_market_pnl:+,.2f} / "
            f"${self.best_market_pnl:+,.2f} / ${self.worst_market_pnl:+,.2f}",
            f"  passive fill rate          {f(self.passive_fill_rate, '.2%')}",
            "Forecast quality (out-of-sample markets)",
            f"  Brier composite signal     {f(self.brier_composite, '.4f')}",
            f"  Brier market mid           {f(self.brier_market, '.4f')}",
            "Transaction costs (vs. decision mid; + = cost)",
            f"  spread                     ${self.spread_cost:+,.2f}",
            f"  slippage (book walking)    ${self.slippage_cost:+,.2f}",
            f"  fees                       ${self.fees:+,.2f}",
            f"  total                      ${self.total_cost:+,.2f}",
            f"  cost per contract          {self.cost_per_contract * 100:+.3f} cents",
            f"  avg slippage per fill      ${self.avg_slippage_per_fill:+.4f}",
            f"  markout per contract       {self.markout_per_contract * 100:+.3f} cents "
            "(mid 10 steps later vs fill; - = adverse selection)",
            "  by liquidity tier:",
        ]
        for tier, c in self.cost_by_tier.items():
            lines.append(
                f"    {tier:<7} fills {int(c['n_fills']):5d}  contracts "
                f"{int(c['contracts']):8,d}  cost/contract {c['cost_per_contract'] * 100:+.3f}c"
                f"  slippage ${c['slippage_cost']:+,.2f}"
            )
        trips = ", ".join(str(s) for s in self.circuit_breaker_trips) or "none"
        lines += [
            "Risk controls",
            f"  circuit breaker trips      {trips}",
            "Equity curve (sampled)",
            f"  {'step':>5} {'equity':>12} {'drawdown':>9} {'gross exp':>10}",
        ]
        if self.equity_curve:
            idx = np.linspace(
                0, len(self.equity_curve) - 1, min(curve_rows, len(self.equity_curve))
            )
            for k in sorted({round(float(v)) for v in idx}):
                s, eq, dd, gx = self.equity_curve[k]
                lines.append(f"  {s:5d} {eq:12,.2f} {dd:9.2%} {gx:10,.0f}")
            lines.append("  sparkline: " + sparkline([e for _, e, _, _ in self.equity_curve]))
        lines.append("=" * 78)
        return "\n".join(lines)

    def write(self, out_dir: Path, stem: str) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        txt = out_dir / f"{stem}.txt"
        txt.write_text(self.render() + "\n")
        curve = out_dir / f"{stem}_equity.csv"
        with curve.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["step", "equity", "drawdown", "gross_exposure"])
            w.writerows(self.equity_curve)
        summary = out_dir / f"{stem}_summary.json"
        data = asdict(self)
        data.pop("equity_curve")
        data["data_source"] = "synthetic"
        summary.write_text(json.dumps(data, indent=2, default=_json_default) + "\n")
        return [txt, curve, summary]


def _json_default(x: object) -> object:  # pragma: no cover - defensive
    return str(x)


def sparkline(values: list[float], width: int = 60) -> str:
    if not values:
        return ""
    arr = np.asarray(values, dtype=np.float64)
    idx = np.linspace(0, len(arr) - 1, min(width, len(arr))).round().astype(int)
    v = arr[idx]
    lo, hi = float(v.min()), float(v.max())
    if hi == lo:
        return SPARK[len(SPARK) // 2] * len(v)
    return "".join(SPARK[int((x - lo) / (hi - lo) * (len(SPARK) - 1))] for x in v)


def daily_returns(equity: list[float], steps_per_day: int = 24) -> list[float]:
    eq = np.asarray(equity, dtype=np.float64)
    if len(eq) == 0:
        return []
    closes = (
        np.concatenate((eq[::steps_per_day], eq[-1:]))
        if (len(eq) - 1) % steps_per_day
        else eq[::steps_per_day]
    )
    return [float(x) for x in simple_returns(closes)]


def build_tearsheet(
    result: BacktestResult,
    title: str,
    brier_composite: float | None = None,
    brier_market: float | None = None,
) -> TearSheet:
    eq = result.equity
    dd = drawdown_series(eq)
    ppy = result.config.monitor.periods_per_year
    pnl = list(result.market_pnl.values())
    costs = result.cost_summary()
    by_tier: dict[str, dict[str, float]] = {}
    for tier in sorted(set(result.market_tier.values())):
        c = result.cost_summary(tier=tier)
        if c.n_fills:
            by_tier[tier] = {
                "n_fills": c.n_fills,
                "contracts": c.contracts,
                "cost_per_contract": c.cost_per_contract,
                "slippage_cost": c.slippage_cost,
                "total_cost": c.total_cost,
            }
    passive = result.passive_stats
    return TearSheet(
        title=title,
        style=result.config.style.value,
        initial_capital=result.config.initial_capital,
        final_equity=result.final_equity,
        total_pnl=result.total_pnl,
        total_return=result.total_pnl / result.config.initial_capital,
        sharpe_hourly_annualised=sharpe_ratio(simple_returns(eq), ppy),
        sharpe_daily_annualised=sharpe_ratio(daily_returns(eq, 24), 365.0),
        max_drawdown=max_drawdown(eq),
        n_steps=len(eq),
        n_fills=costs.n_fills,
        contracts_traded=costs.contracts,
        markets_traded=len(pnl),
        market_hit_rate=float(np.mean([p > 0 for p in pnl])) if pnl else float("nan"),
        avg_market_pnl=float(np.mean(pnl)) if pnl else 0.0,
        best_market_pnl=max(pnl, default=0.0),
        worst_market_pnl=min(pnl, default=0.0),
        brier_composite=brier_composite,
        brier_market=brier_market,
        spread_cost=costs.spread_cost,
        slippage_cost=costs.slippage_cost,
        fees=costs.fees,
        total_cost=costs.total_cost,
        cost_per_contract=costs.cost_per_contract,
        avg_slippage_per_fill=costs.avg_slippage_per_trade,
        markout_per_contract=result.markout_per_contract(),
        passive_fill_rate=passive.fill_rate if passive.orders else None,
        circuit_breaker_trips=[e.step for e in result.breaker_events],
        max_gross_exposure=max(result.gross_exposure, default=0.0),
        cost_by_tier=by_tier,
        equity_curve=[
            (s, e, float(d), g)
            for s, e, d, g in zip(result.steps, eq, dd, result.gross_exposure, strict=True)
        ],
    )
