"""CONSCIENCE — pre-trade risk veto, the agent that says "no".

Every proposed order from TRADER passes through CONSCIENCE before any
broker call. CONSCIENCE has read-only access to the proposed order, the
current portfolio, the regime state, and the RiskLimits configuration. It
returns a `ConscienceVerdict` of one of:

    APPROVE          fully allowed; pass straight to execution
    REDUCE_SIZE      allowed at smaller size; specifies the reduced size
    VETO             not allowed; explain why
    INQUIRY          edge case; open an inquiry for the operator

The point of CONSCIENCE being a distinct module (not a method on TRADER):
the same set of rules can be invoked manually, exposed in the doctor
output, or replaced with an Opus-backed reviewer in a future version
without touching TRADER. Separation of concerns: TRADER proposes,
CONSCIENCE judges, broker executes.

Hard rules (always enforced):
  - position size vs RiskLimits.max_position_pct
  - portfolio drawdown vs RiskLimits.max_drawdown_pct (kill switch)
  - daily loss vs RiskLimits.max_daily_loss_pct
  - silo concentration vs RiskLimits.max_concentration_per_silo_pct
  - gross leverage vs RiskLimits.max_gross_leverage

Soft rules (open inquiries instead of hard veto):
  - regime mismatch (mechanism's allowed regimes don't include current)
  - cold-start mechanism with size > 1% (genuinely large bet on untested logic)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from quant.risk.limits import RiskLimits, RiskState


class Verdict(Enum):
    APPROVE = "approve"
    REDUCE_SIZE = "reduce_size"
    VETO = "veto"
    INQUIRY = "inquiry"


@dataclass
class OpenPosition:
    """Minimum portfolio bookkeeping for risk checks."""

    asset: str
    direction: str
    size_dollars: float
    entry_date: datetime
    silo: Optional[str] = None  # tag if known (e.g. "cmbs", "pharma")


@dataclass
class ProposedOrder:
    """The trade TRADER wants to make."""

    asset: str
    direction: str
    size_pct_of_nav: float
    size_dollars: float
    rationale: str
    holding_period_days: int
    confidence: float
    contributing_mechanisms: list[str] = field(default_factory=list)
    silo: Optional[str] = None
    cold_start: bool = False  # any contributing mechanism has < 5 closed trades


@dataclass
class ConscienceVerdict:
    """Verdict on one ProposedOrder."""

    verdict: Verdict
    reason: str
    adjusted_size_pct: Optional[float] = None
    open_inquiry: Optional[dict] = None  # {type, body, urgency, options} if INQUIRY


def review_order(
    order: ProposedOrder,
    *,
    nav: float,
    open_positions: list[OpenPosition],
    risk_state: RiskState,
    risk_limits: Optional[RiskLimits] = None,
    regime_probability_in_allowed: Optional[float] = None,
) -> ConscienceVerdict:
    """Apply hard + soft rules. Return verdict."""
    risk_limits = risk_limits or RiskLimits()

    # ── Hard rules ────────────────────────────────────────────────────

    # 1. Halted from prior catastrophic loss
    if risk_state.is_halted():
        return ConscienceVerdict(
            verdict=Verdict.VETO,
            reason=f"trading halted until {risk_state.halted_until}",
        )

    # 2. Drawdown kill switch
    dd = risk_limits.check_drawdown(risk_state.nav, risk_state.peak_nav)
    if not dd.ok:
        return ConscienceVerdict(verdict=Verdict.VETO, reason=dd.reason)

    # 3. Daily loss limit
    if risk_state.nav_yesterday is not None:
        daily = risk_limits.check_daily_loss(risk_state.nav, risk_state.nav_yesterday)
        if not daily.ok:
            return ConscienceVerdict(verdict=Verdict.VETO, reason=daily.reason)

    # 4. Position size cap
    pos = risk_limits.check_position_size(order.size_dollars, nav)
    if not pos.ok:
        capped_pct = risk_limits.max_position_pct
        return ConscienceVerdict(
            verdict=Verdict.REDUCE_SIZE,
            reason=pos.reason,
            adjusted_size_pct=capped_pct,
        )

    # 5. Silo concentration (if we know the silo)
    if order.silo:
        same_silo_dollars = sum(
            abs(p.size_dollars) for p in open_positions if p.silo == order.silo
        ) + abs(order.size_dollars)
        silo = risk_limits.check_silo_concentration(
            same_silo_dollars, nav, order.silo
        )
        if not silo.ok:
            return ConscienceVerdict(verdict=Verdict.VETO, reason=silo.reason)

    # 6. Gross leverage
    new_gross = sum(abs(p.size_dollars) for p in open_positions) + abs(
        order.size_dollars
    )
    lev = risk_limits.check_gross_leverage(new_gross, nav)
    if not lev.ok:
        return ConscienceVerdict(verdict=Verdict.VETO, reason=lev.reason)

    # ── Soft rules: open inquiry instead of hard veto ─────────────────

    # 7. Regime mismatch on a sizable bet
    if (
        regime_probability_in_allowed is not None
        and regime_probability_in_allowed < 0.30
        and order.size_pct_of_nav > 0.01
    ):
        return ConscienceVerdict(
            verdict=Verdict.INQUIRY,
            reason=(
                f"P(allowed regime) = {regime_probability_in_allowed:.0%} < 30% "
                f"but size = {order.size_pct_of_nav:.2%} of NAV"
            ),
            open_inquiry={
                "type": "decision",
                "urgency": "high",
                "body": (
                    f"Mechanism wants to {order.direction} {order.asset} at "
                    f"{order.size_pct_of_nav:.2%} but regime support is only "
                    f"{regime_probability_in_allowed:.0%}. Approve, reduce, or veto?"
                ),
                "options": ["approve", "reduce_to_0.5pct", "veto"],
            },
        )

    # 8. Cold-start mechanism with materially large bet
    if order.cold_start and order.size_pct_of_nav > 0.015:
        return ConscienceVerdict(
            verdict=Verdict.INQUIRY,
            reason=(
                f"cold-start mechanism (no track record) bet "
                f"{order.size_pct_of_nav:.2%} of NAV"
            ),
            open_inquiry={
                "type": "decision",
                "urgency": "medium",
                "body": (
                    f"New mechanism {order.contributing_mechanisms} has no track "
                    f"record but wants {order.size_pct_of_nav:.2%}. OK to proceed?"
                ),
                "options": ["approve", "reduce_to_0.5pct", "wait_for_track_record"],
            },
        )

    return ConscienceVerdict(verdict=Verdict.APPROVE, reason="all checks passed")
