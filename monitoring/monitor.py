"""Tick-by-tick risk and performance monitor with a drawdown circuit breaker.

The backtester calls :meth:`RiskMonitor.on_tick` once per global step after
marking every open position to the current mid. The monitor updates rolling
PnL / Sharpe / drawdown and exposure, raises alerts when a rule changes state,
and returns :attr:`Action.FLATTEN_ALL` when the circuit breaker trips. The
backtester is responsible for carrying the action out.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from monitoring.metrics import sharpe_ratio


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class Action(Enum):
    NONE = "none"
    FLATTEN_ALL = "flatten_all"


@dataclass(frozen=True)
class Alert:
    step: int
    level: AlertLevel
    rule: str
    message: str


@dataclass(frozen=True)
class MonitorConfig:
    window: int = 48  # rolling window in steps (48 synthetic hours)
    periods_per_year: float = 8760.0  # 1 step == 1 synthetic hour, markets trade 24/7
    max_drawdown_limit: float = 0.15  # circuit-breaker threshold (fraction of peak equity)
    breaker_cooldown: int | None = None  # None -> stay halted for the rest of the run
    drawdown_warn_fraction: float = 0.5  # warn at this fraction of the breaker threshold
    exposure_limit: float | None = None  # gross $ at risk; warn when near it
    exposure_warn_fraction: float = 0.9

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError("window must be >= 2")
        if not 0.0 < self.max_drawdown_limit < 1.0:
            raise ValueError("max_drawdown_limit must be in (0, 1)")
        if self.breaker_cooldown is not None and self.breaker_cooldown < 1:
            raise ValueError("breaker_cooldown must be >= 1 or None")


class CircuitBreaker:
    """Trips when equity falls ``max_drawdown`` below its high-water mark.

    After tripping the book is halted. With ``cooldown_steps=None`` the halt
    lasts for the rest of the run; otherwise trading resumes after the
    cooldown and the high-water mark is rebased to the equity at resumption
    (otherwise it would re-trip immediately).
    """

    def __init__(self, max_drawdown: float, cooldown_steps: int | None = None) -> None:
        if not 0.0 < max_drawdown < 1.0:
            raise ValueError("max_drawdown must be in (0, 1)")
        self.max_drawdown = max_drawdown
        self.cooldown_steps = cooldown_steps
        self.trips: list[int] = []
        self._peak: float | None = None
        self._halted = False
        self._resume_step: int | None = None
        self.drawdown = 0.0

    def is_halted(self, step: int) -> bool:
        if self._halted and self._resume_step is not None and step >= self._resume_step:
            return False
        return self._halted

    def update(self, step: int, equity: float) -> bool:
        """Feed a new equity mark. Returns True only on the step the breaker trips."""
        if self._halted:
            if self._resume_step is None or step < self._resume_step:
                return False
            self._halted = False
            self._resume_step = None
            self._peak = equity
        self._peak = equity if self._peak is None else max(self._peak, equity)
        self.drawdown = 1.0 - equity / self._peak if self._peak > 0 else 1.0
        if self.drawdown >= self.max_drawdown:
            self._halted = True
            self.trips.append(step)
            if self.cooldown_steps is not None:
                self._resume_step = step + self.cooldown_steps
            return True
        return False


@dataclass(frozen=True)
class MonitorSnapshot:
    step: int
    equity: float
    pnl: float
    rolling_pnl: float
    rolling_sharpe: float
    drawdown: float
    max_drawdown: float
    gross_exposure: float
    n_positions: int
    largest_market: str | None
    largest_exposure: float
    halted: bool


@dataclass(frozen=True)
class TickResult:
    snapshot: MonitorSnapshot
    alerts: tuple[Alert, ...]
    action: Action


@dataclass
class RiskMonitor:
    config: MonitorConfig
    initial_equity: float
    snapshots: list[MonitorSnapshot] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)
    exposure_history: list[dict[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.initial_equity <= 0:
            raise ValueError("initial_equity must be positive")
        self.breaker = CircuitBreaker(self.config.max_drawdown_limit, self.config.breaker_cooldown)
        self._peak = self.initial_equity
        self._prev_equity = self.initial_equity
        self._returns: deque[float] = deque(maxlen=self.config.window)
        self._equity_window: deque[float] = deque(
            [self.initial_equity], maxlen=self.config.window + 1
        )
        self._max_dd = 0.0
        self._dd_warned = False
        self._exp_warned = False

    @property
    def drawdown(self) -> float:
        return 1.0 - self._prev_equity / self._peak

    @property
    def max_drawdown(self) -> float:
        return self._max_dd

    def is_halted(self, step: int) -> bool:
        return self.breaker.is_halted(step)

    def on_tick(self, step: int, equity: float, exposures: Mapping[str, float]) -> TickResult:
        cfg = self.config
        self._returns.append(equity / self._prev_equity - 1.0 if self._prev_equity > 0 else 0.0)
        self._prev_equity = equity
        self._equity_window.append(equity)
        self._peak = max(self._peak, equity)
        dd = 1.0 - equity / self._peak
        self._max_dd = max(self._max_dd, dd)
        gross = float(sum(abs(v) for v in exposures.values()))
        largest = max(exposures.items(), key=lambda kv: abs(kv[1]), default=(None, 0.0))
        alerts: list[Alert] = []

        warn_level = cfg.drawdown_warn_fraction * cfg.max_drawdown_limit
        if dd >= warn_level and not self._dd_warned:
            alerts.append(
                Alert(
                    step,
                    AlertLevel.WARNING,
                    "drawdown_warning",
                    f"drawdown {dd:.2%} >= warning level {warn_level:.2%}",
                )
            )
            self._dd_warned = True
        elif dd < warn_level:
            self._dd_warned = False

        if cfg.exposure_limit is not None:
            near = gross >= cfg.exposure_warn_fraction * cfg.exposure_limit
            if near and not self._exp_warned:
                alerts.append(
                    Alert(
                        step,
                        AlertLevel.WARNING,
                        "exposure_warning",
                        f"gross exposure ${gross:,.0f} >= "
                        f"{cfg.exposure_warn_fraction:.0%} of ${cfg.exposure_limit:,.0f}",
                    )
                )
            self._exp_warned = near

        action = Action.NONE
        if self.breaker.update(step, equity):
            action = Action.FLATTEN_ALL
            alerts.append(
                Alert(
                    step,
                    AlertLevel.CRITICAL,
                    "circuit_breaker",
                    f"drawdown {self.breaker.drawdown:.2%} >= limit "
                    f"{cfg.max_drawdown_limit:.2%}: flatten all positions and halt",
                )
            )

        rets = np.fromiter(self._returns, dtype=np.float64)
        snap = MonitorSnapshot(
            step=step,
            equity=equity,
            pnl=equity - self.initial_equity,
            rolling_pnl=equity - self._equity_window[0],
            rolling_sharpe=sharpe_ratio(rets, cfg.periods_per_year),
            drawdown=dd,
            max_drawdown=self._max_dd,
            gross_exposure=gross,
            n_positions=sum(1 for v in exposures.values() if v != 0),
            largest_market=largest[0],
            largest_exposure=float(abs(largest[1])),
            halted=self.breaker.is_halted(step),
        )
        self.snapshots.append(snap)
        self.exposure_history.append(dict(exposures))
        self.alerts.extend(alerts)
        return TickResult(snap, tuple(alerts), action)

    def format_snapshot(self, snap: MonitorSnapshot) -> str:
        sharpe = "n/a" if math.isnan(snap.rolling_sharpe) else f"{snap.rolling_sharpe:6.2f}"
        top = f"{snap.largest_market} ${snap.largest_exposure:,.0f}" if snap.largest_market else "-"
        return (
            f"step {snap.step:4d} | equity ${snap.equity:10,.2f} | pnl {snap.pnl:+9.2f} | "
            f"roll{self.config.window} pnl {snap.rolling_pnl:+8.2f} | roll sharpe {sharpe} | "
            f"dd {snap.drawdown:6.2%} (max {snap.max_drawdown:6.2%}) | "
            f"gross ${snap.gross_exposure:8,.0f} | pos {snap.n_positions:2d} | top {top}"
            f"{' | HALTED' if snap.halted else ''}"
        )
