"""Module 4: tick-by-tick risk/performance monitoring, circuit breaker, tear sheet.

``monitoring.tearsheet`` depends on the backtester and is imported directly
(``from monitoring.tearsheet import build_tearsheet``) to avoid an import cycle.
"""

from monitoring.metrics import drawdown_series, max_drawdown, sharpe_ratio, simple_returns
from monitoring.monitor import (
    Action,
    Alert,
    AlertLevel,
    CircuitBreaker,
    MonitorConfig,
    MonitorSnapshot,
    RiskMonitor,
)

__all__ = [
    "Action",
    "Alert",
    "AlertLevel",
    "CircuitBreaker",
    "MonitorConfig",
    "MonitorSnapshot",
    "RiskMonitor",
    "drawdown_series",
    "max_drawdown",
    "sharpe_ratio",
    "simple_returns",
]
