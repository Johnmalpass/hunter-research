"""Risk subsystem. Hard caps enforced in code before any order goes to a broker.

`limits.RiskLimits` is the single source of truth for position size, drawdown,
and daily-loss caps. The TRADER agent proposes; CONSCIENCE checks against
RiskLimits; the broker only sees orders that passed both.
"""
from quant.risk.limits import RiskLimits, RiskCheck, RiskHalted

__all__ = ["RiskLimits", "RiskCheck", "RiskHalted"]
