"""Event-driven backtester.

Events are processed from a priority queue ordered by (step, kind, sequence):

* ``RESOLUTION``   - a market resolves: cancel its resting order and settle the
  position at $1 (YES) or $0 (NO).
* ``MARKET_DATA``  - a new snapshot for one market: mark to mid, advance any
  resting passive order, read the signal, size, and trade against a book
  rebuilt from the snapshot.
* ``END_OF_STEP``  - after every market has updated: mark the portfolio,
  call the risk monitor, and carry out any circuit-breaker action.

Market impact. Walking the book is the *temporary* cost of a large order. On
top of that, each aggressive fill shifts the centre of the books this
strategy trades against by ``impact_ticks_per_depth`` ticks per unit of
top-of-book depth consumed; the shift decays by ``impact_decay`` per step.
Without it the book would refill around an unchanged mid every step and the
same mispricing could be harvested repeatedly at no extra cost. Positions are
still marked at the synthetic mid (impact is treated as a cost, not a gain).

The strategy only reads information available at the current step: the
signal arrays are causal (tested in ``tests/test_signals.py``), the fill flow
used for passive orders is the flow *during* the step being processed, and
the passive executor's expected-flow estimate uses past flow only.
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum, IntEnum

import numpy as np

from execution.costs import CostSummary, ExecutionRecord
from execution.executors import AggressiveExecutor, PassiveExecutor, PassiveOrder
from execution.sizing import PositionSizer, RiskLimits, worst_case_exposure
from market_sim.book_builder import build_book
from market_sim.markets import SyntheticMarket
from market_sim.order_book import LimitOrderBook, Side
from monitoring.monitor import Action, MonitorConfig, RiskMonitor
from signals.research import MarketSignal


class ExecutionStyle(str, Enum):
    AGGRESSIVE = "aggressive"
    PASSIVE = "passive"


class EventKind(IntEnum):
    RESOLUTION = 0
    MARKET_DATA = 1
    END_OF_STEP = 2


@dataclass(frozen=True, order=True)
class Event:
    step: int
    kind: EventKind
    seq: int
    market_id: str = field(default="", compare=False)


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 10_000.0
    style: ExecutionStyle = ExecutionStyle.AGGRESSIVE
    limits: RiskLimits = field(default_factory=RiskLimits)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    fee_per_contract: float = 0.0
    passive_max_wait: int = 5
    markout_horizon: int = 10
    no_new_risk_last_steps: int = 6
    flow_lookback: int = 24
    book_seed: int = 0
    impact_ticks_per_depth: float = 2.0  # mid shift (ticks) per top-of-book depth we take
    impact_decay: float = 0.9  # per-step decay of the transient impact
    invert_signal: bool = False  # stress test: trade against the model

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.markout_horizon < 1 or self.flow_lookback < 1:
            raise ValueError("markout_horizon and flow_lookback must be >= 1")
        if self.impact_ticks_per_depth < 0 or not 0.0 <= self.impact_decay < 1.0:
            raise ValueError("need impact_ticks_per_depth >= 0 and impact_decay in [0, 1)")


class Portfolio:
    """Cash + signed contract positions; tracks per-market cash flows for attribution."""

    def __init__(self, cash: float) -> None:
        self.cash = cash
        self.positions: dict[str, int] = {}
        self.cashflow: dict[str, float] = {}
        self.market_pnl: dict[str, float] = {}

    def position(self, market_id: str) -> int:
        return self.positions.get(market_id, 0)

    def apply(self, rec: ExecutionRecord) -> None:
        flow = -rec.signed_qty * rec.avg_price - rec.fee
        self.cash += flow
        self.cashflow[rec.market_id] = self.cashflow.get(rec.market_id, 0.0) + flow
        pos = self.positions.get(rec.market_id, 0) + rec.signed_qty
        if pos == 0:
            self.positions.pop(rec.market_id, None)
        else:
            self.positions[rec.market_id] = pos

    def settle(self, market_id: str, outcome: int) -> float:
        """Pay out the position at resolution; returns the market's lifetime PnL."""
        payout = self.positions.pop(market_id, 0) * float(outcome)
        self.cash += payout
        if market_id not in self.cashflow:
            return 0.0
        pnl = self.cashflow[market_id] + payout
        self.market_pnl[market_id] = pnl
        return pnl

    def equity(self, marks: Mapping[str, float]) -> float:
        return self.cash + sum(q * marks[m] for m, q in self.positions.items())

    def exposures(self, marks: Mapping[str, float]) -> dict[str, float]:
        return {m: worst_case_exposure(q, marks[m]) for m, q in self.positions.items()}


@dataclass
class PassiveStats:
    orders: int = 0
    contracts_ordered: int = 0
    contracts_filled: int = 0
    fully_filled: int = 0
    cancelled: int = 0
    expected_fill_prob: list[float] = field(default_factory=list)
    realized_full_fill: list[int] = field(default_factory=list)

    @property
    def fill_rate(self) -> float:
        return self.contracts_filled / self.contracts_ordered if self.contracts_ordered else 0.0

    def record_final(self, order: PassiveOrder) -> None:
        self.orders += 1
        self.contracts_ordered += order.original_qty
        self.contracts_filled += order.filled
        full = int(order.remaining == 0)
        self.fully_filled += full
        self.cancelled += 1 - full
        self.expected_fill_prob.append(order.expected_fill_prob)
        self.realized_full_fill.append(full)


@dataclass
class BreakerEvent:
    step: int
    drawdown: float
    equity: float
    positions_before: dict[str, int]
    positions_after: dict[str, int] = field(default_factory=dict)
    flatten_records: list[ExecutionRecord] = field(default_factory=list)
    completed_step: int | None = None


@dataclass
class BacktestResult:
    config: BacktestConfig
    steps: list[int]
    equity: list[float]
    gross_exposure: list[float]
    records: list[ExecutionRecord]
    market_pnl: dict[str, float]
    market_tier: dict[str, str]
    monitor: RiskMonitor
    breaker_events: list[BreakerEvent]
    passive_stats: PassiveStats
    final_positions: dict[str, int]

    @property
    def final_equity(self) -> float:
        return self.equity[-1] if self.equity else self.config.initial_capital

    @property
    def total_pnl(self) -> float:
        return self.final_equity - self.config.initial_capital

    def cost_summary(self, style: str | None = None, tier: str | None = None) -> CostSummary:
        return CostSummary.from_records(
            r
            for r in self.records
            if (style is None or r.style == style) and (tier is None or r.tier == tier)
        )

    def markout_per_contract(self, style: str | None = None) -> float:
        recs = [r for r in self.records if (style is None or r.style == style)]
        contracts = sum(r.qty for r in recs)
        total = sum(r.markout_pnl or 0.0 for r in recs)
        return total / contracts if contracts else 0.0


class Backtester:
    def __init__(
        self,
        markets: Sequence[SyntheticMarket],
        signals: Mapping[str, MarketSignal],
        config: BacktestConfig | None = None,
        log: Callable[[str], None] | None = None,
        log_every: int | None = None,
    ) -> None:
        if not markets:
            raise ValueError("need at least one market")
        missing = {m.market_id for m in markets} - set(signals)
        if missing:
            raise ValueError(f"missing signals for {sorted(missing)}")
        self.markets = {m.market_id: m for m in markets}
        self.signals = signals
        self.config = config or BacktestConfig()
        self.log = log
        self.log_every = log_every
        cfg = self.config
        self.portfolio = Portfolio(cfg.initial_capital)
        self.sizer = PositionSizer(cfg.limits)
        self.aggressive = AggressiveExecutor(cfg.fee_per_contract)
        self.passive = PassiveExecutor(cfg.fee_per_contract, max_wait=cfg.passive_max_wait)
        self.monitor = RiskMonitor(cfg.monitor, cfg.initial_capital)
        self.rng = np.random.default_rng(cfg.book_seed)
        self.pending: dict[str, PassiveOrder] = {}
        self.passive_stats = PassiveStats()
        self.records: list[ExecutionRecord] = []
        self.marks: dict[str, float] = {}
        self.impact: dict[str, float] = {}
        self.breaker_events: list[BreakerEvent] = []
        self.flattening = False
        self.steps: list[int] = []
        self.equity: list[float] = []
        self.gross: list[float] = []
        self._queue: list[Event] = []
        self._seq = itertools.count()

    # ---------------------------------------------------------------- plumbing
    def _push(self, step: int, kind: EventKind, market_id: str = "") -> None:
        heapq.heappush(self._queue, Event(step, kind, next(self._seq), market_id))

    def _book(self, market: SyntheticMarket, i: int) -> LimitOrderBook:
        mid = float(np.clip(market.mid[i] + self.impact.get(market.market_id, 0.0), 0.02, 0.98))
        return build_book(mid, int(market.half_spread_ticks[i]), market.depth_profile(i), self.rng)

    def _apply_impact(self, market: SyntheticMarket, i: int, rec: ExecutionRecord | None) -> None:
        if rec is None:
            return
        shift = 0.01 * self.config.impact_ticks_per_depth * rec.qty / float(market.top_depth[i])
        self.impact[market.market_id] = self.impact.get(market.market_id, 0.0) + (
            rec.side.value * shift
        )

    def _emit(self, msg: str) -> None:
        if self.log is not None:
            self.log(msg)

    def _record(self, rec: ExecutionRecord | None) -> None:
        if rec is not None:
            self.portfolio.apply(rec)
            self.records.append(rec)

    def _finalize_passive(self, market_id: str) -> None:
        order = self.pending.pop(market_id, None)
        if order is not None:
            self.passive_stats.record_final(order)

    # -------------------------------------------------------------------- run
    def run(self) -> BacktestResult:
        start = min(m.open_step for m in self.markets.values())
        end = max(m.close_step for m in self.markets.values())
        for m in self.markets.values():
            for g in range(m.open_step, m.close_step):
                self._push(g, EventKind.MARKET_DATA, m.market_id)
            self._push(m.close_step, EventKind.RESOLUTION, m.market_id)
        for g in range(start, end + 1):
            self._push(g, EventKind.END_OF_STEP)
        while self._queue:
            ev = heapq.heappop(self._queue)
            if ev.kind is EventKind.RESOLUTION:
                self._on_resolution(ev)
            elif ev.kind is EventKind.MARKET_DATA:
                self._on_market_data(ev)
            else:
                self._on_end_of_step(ev)
        self._attach_markouts()
        return BacktestResult(
            config=self.config,
            steps=self.steps,
            equity=self.equity,
            gross_exposure=self.gross,
            records=self.records,
            market_pnl=dict(self.portfolio.market_pnl),
            market_tier={m.market_id: m.tier.value for m in self.markets.values()},
            monitor=self.monitor,
            breaker_events=self.breaker_events,
            passive_stats=self.passive_stats,
            final_positions=dict(self.portfolio.positions),
        )

    # --------------------------------------------------------------- handlers
    def _on_resolution(self, ev: Event) -> None:
        m = self.markets[ev.market_id]
        self._finalize_passive(m.market_id)
        self.portfolio.settle(m.market_id, m.outcome)
        self.marks.pop(m.market_id, None)

    def _on_market_data(self, ev: Event) -> None:
        m = self.markets[ev.market_id]
        i = m.local_index(ev.step)
        mid = float(m.mid[i])
        self.marks[m.market_id] = mid
        if m.market_id in self.impact:
            self.impact[m.market_id] *= self.config.impact_decay

        order = self.pending.get(m.market_id)
        if order is not None and order.placed_step < ev.step:
            flow = int(m.sell_flow[i] if order.side is Side.BUY else m.buy_flow[i])
            self._record(self.passive.process_step(order, self._book(m, i), ev.step, flow))
            if order.remaining == 0 or self.passive.expired(order):
                self._finalize_passive(m.market_id)

        if self.flattening or self.monitor.is_halted(ev.step):
            return
        if m.length - i <= self.config.no_new_risk_last_steps:
            return  # hold what we have into resolution; no new risk
        self._trade(m, i, ev.step, mid)

    def _trade(self, m: SyntheticMarket, i: int, step: int, mid: float) -> None:
        cfg = self.config
        prob = float(self.signals[m.market_id].composite_prob[i])
        if cfg.invert_signal:
            prob = float(np.clip(2.0 * mid - prob, 1e-3, 1.0 - 1e-3))
        pos = self.portfolio.position(m.market_id)
        pending = self.pending.get(m.market_id)
        working = pos + (pending.signed_remaining if pending else 0)
        exposures = self.portfolio.exposures(self.marks)
        other = sum(v for k, v in exposures.items() if k != m.market_id)
        equity = self.portfolio.equity(self.marks)
        target = self.sizer.target_position(
            prob, mid, equity, self.monitor.drawdown, working, other
        )
        over = other + exposures.get(m.market_id, 0.0) > cfg.limits.max_gross_exposure
        if not self.sizer.should_trade(target, working, over):
            return
        delta = target - working
        if pending is not None:
            if np.sign(delta) == pending.side.value:
                return  # let the resting order work
            self._finalize_passive(m.market_id)
            if not self.sizer.should_trade(target, pos, over):
                return
            delta = target - pos
        side = Side.BUY if delta > 0 else Side.SELL
        book = self._book(m, i)
        visible = book.total_depth(side.opposite, cfg.limits.participation_levels)
        qty = self.sizer.clip_to_participation(abs(delta), visible)
        if qty < cfg.limits.min_trade_size:
            return
        tier = m.tier.value
        if cfg.style is ExecutionStyle.AGGRESSIVE:
            rec = self.aggressive.execute(book, m.market_id, tier, step, side, qty, mid)
            self._record(rec)
            self._apply_impact(m, i, rec)
        else:
            flows = m.sell_flow if side is Side.BUY else m.buy_flow
            hist = flows[max(0, i - cfg.flow_lookback + 1) : i + 1]
            new = self.passive.place(
                book, m.market_id, tier, step, side, qty, mid, float(np.mean(hist))
            )
            if new is not None:
                self.pending[m.market_id] = new

    def _on_end_of_step(self, ev: Event) -> None:
        equity = self.portfolio.equity(self.marks)
        exposures = self.portfolio.exposures(self.marks)
        tick = self.monitor.on_tick(ev.step, equity, exposures)
        self.steps.append(ev.step)
        self.equity.append(equity)
        self.gross.append(tick.snapshot.gross_exposure)
        for alert in tick.alerts:
            self._emit(
                f"  ALERT [{alert.level.value.upper()}] step {alert.step}: "
                f"{alert.rule} - {alert.message}"
            )
        if tick.action is Action.FLATTEN_ALL:
            self.flattening = True
            self.breaker_events.append(
                BreakerEvent(
                    ev.step, tick.snapshot.drawdown, equity, dict(self.portfolio.positions)
                )
            )
        if self.flattening:
            self._flatten(ev.step)
        if self.log_every and ev.step % self.log_every == 0:
            self._emit("  " + self.monitor.format_snapshot(tick.snapshot))

    def _flatten(self, step: int) -> None:
        """Close every open position with marketable orders; continues next step if the
        visible book is too thin to absorb a position in one go."""
        event = self.breaker_events[-1]
        for market_id in list(self.pending):
            self._finalize_passive(market_id)
        for market_id, pos in list(self.portfolio.positions.items()):
            m = self.markets[market_id]
            i = m.local_index(step)
            side = Side.SELL if pos > 0 else Side.BUY
            rec = self.aggressive.execute(
                self._book(m, i),
                market_id,
                m.tier.value,
                step,
                side,
                abs(pos),
                float(m.mid[i]),
                style="flatten",
            )
            self._record(rec)
            self._apply_impact(m, i, rec)
            if rec is not None:
                event.flatten_records.append(rec)
        if not self.portfolio.positions:
            self.flattening = False
            event.completed_step = step
            event.positions_after = {}
            self._emit(
                f"  circuit breaker: all positions flat at step {step} "
                f"({len(event.flatten_records)} flatten fills)"
            )
        else:
            event.positions_after = dict(self.portfolio.positions)

    def _attach_markouts(self) -> None:
        h = self.config.markout_horizon
        out = []
        for r in self.records:
            m = self.markets[r.market_id]
            j = min(r.step - m.open_step + h, m.length - 1)
            out.append(r.with_markout(float(m.mid[j])))
        self.records = out
