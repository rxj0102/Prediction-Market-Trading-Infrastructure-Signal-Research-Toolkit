"""Transaction-cost accounting.

Every execution is measured against the mid at decision time (implementation
shortfall), split into:

* spread cost   - distance from the mid to the touch (half the quoted spread for
  an aggressive order; negative for a passive order resting inside/at the
  touch, i.e. a captured spread);
* slippage      - distance from the touch to the achieved average price, i.e.
  the cost of walking the book beyond the best level;
* fees          - an explicit per-contract fee (0 by default; configurable).

All three are signed so that positive = cost to the strategy.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from market_sim.order_book import Side


@dataclass(frozen=True)
class ExecutionRecord:
    step: int
    market_id: str
    tier: str
    style: str  # "aggressive" | "passive" | "flatten"
    side: Side
    qty: int
    avg_price: float
    decision_mid: float
    touch_price: float  # best opposite quote (aggressive) or our limit price (passive)
    fee: float = 0.0
    markout_mid: float | None = None  # mid ``markout_horizon`` steps later, filled in later

    @property
    def signed_qty(self) -> int:
        return self.side.value * self.qty

    @property
    def spread_cost(self) -> float:
        return self.side.value * (self.touch_price - self.decision_mid) * self.qty

    @property
    def slippage_cost(self) -> float:
        return self.side.value * (self.avg_price - self.touch_price) * self.qty

    @property
    def total_cost(self) -> float:
        return self.spread_cost + self.slippage_cost + self.fee

    @property
    def cost_per_contract(self) -> float:
        return self.total_cost / self.qty

    @property
    def markout_pnl(self) -> float | None:
        """PnL of the fill marked to the later mid (negative = adverse selection)."""
        if self.markout_mid is None:
            return None
        return self.side.value * (self.markout_mid - self.avg_price) * self.qty

    def with_markout(self, mid: float) -> ExecutionRecord:
        return ExecutionRecord(
            self.step,
            self.market_id,
            self.tier,
            self.style,
            self.side,
            self.qty,
            self.avg_price,
            self.decision_mid,
            self.touch_price,
            self.fee,
            mid,
        )


@dataclass(frozen=True)
class CostSummary:
    n_fills: int
    contracts: int
    notional: float
    spread_cost: float
    slippage_cost: float
    fees: float

    @property
    def total_cost(self) -> float:
        return self.spread_cost + self.slippage_cost + self.fees

    @property
    def cost_per_contract(self) -> float:
        return self.total_cost / self.contracts if self.contracts else 0.0

    @property
    def avg_slippage_per_trade(self) -> float:
        return self.slippage_cost / self.n_fills if self.n_fills else 0.0

    @property
    def cost_bps_of_notional(self) -> float:
        return 1e4 * self.total_cost / self.notional if self.notional else 0.0

    @classmethod
    def from_records(cls, records: Iterable[ExecutionRecord]) -> CostSummary:
        recs = list(records)
        return cls(
            n_fills=len(recs),
            contracts=sum(r.qty for r in recs),
            notional=sum(r.qty * r.avg_price for r in recs),
            spread_cost=sum(r.spread_cost for r in recs),
            slippage_cost=sum(r.slippage_cost for r in recs),
            fees=sum(r.fee for r in recs),
        )
