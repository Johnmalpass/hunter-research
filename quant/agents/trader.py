"""TRADER — the orchestrator that ties every piece together.

One `run_cycle(...)` call:

  1. Reads MarketState (the unified data store)
  2. Detects current regime
  3. Evaluates every registered mechanism against the state
  4. Aggregates signals via Coalition (synergy-weighted voting)
  5. Sizes each coalition vote via Kelly + regime + track-record
  6. Passes each proposed order through CONSCIENCE
  7. Approved orders -> log to mechanism_signals ledger
  8. Vetoed/inquiry orders -> open inquiries for the operator
  9. Returns a TradingCycleResult report (json-serialisable)

This is pure Python. No LLM calls, no broker calls. The TRADER produces
*decisions*, not executions. Phase 6 wires this to IBKR for live trading.

Why this matters:
  Before this module, every piece (mechanisms, coalition, sizing, regime,
  ledger, risk limits, conscience) lived in isolation. After this module,
  one CLI command runs the full system. Adding a new mechanism takes one
  file. Adding a new data adapter takes one file. Everything else is
  already wired.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from quant.agents.conscience import (
    ConscienceVerdict,
    OpenPosition,
    ProposedOrder,
    Verdict,
    review_order,
)
from quant.agents.inquiry import open_inquiry
from quant.data.base import MarketState
from quant.research.coalition import CoalitionVote, aggregate_signals
from quant.research.ledger import log_signal, track_record
from quant.research.mechanism import (
    Mechanism,
    Signal,
    _import_all_mechanisms,
    list_mechanisms,
    get_mechanism,
)
from quant.research.regime import RegimeDetector, RegimeState
from quant.risk.limits import RiskLimits, RiskState
from quant.risk.sizing import SizingDecision, size_position


@dataclass
class CycleStep:
    """One step's record — useful for the audit log of a cycle."""

    mechanism_id: str
    n_signals: int
    elapsed_ms: int
    error: Optional[str] = None


@dataclass
class OrderRecord:
    """Trader's record of a proposed order + its conscience verdict."""

    order: ProposedOrder
    verdict: Verdict
    verdict_reason: str
    sizing: dict  # SizingDecision serialised
    inquiry_id: Optional[int] = None
    logged_signal_ids: list[int] = field(default_factory=list)


@dataclass
class TradingCycleResult:
    """Top-level report from one `run_cycle` call."""

    asof: datetime
    nav: float
    regime: Optional[str]
    regime_probabilities: dict[str, float]
    n_mechanisms_evaluated: int
    n_signals_emitted: int
    n_coalition_votes: int
    n_orders_proposed: int
    n_approved: int
    n_vetoed: int
    n_inquiries_opened: int
    steps: list[CycleStep]
    orders: list[OrderRecord]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "asof": self.asof.isoformat(),
            "nav": self.nav,
            "regime": self.regime,
            "regime_probabilities": self.regime_probabilities,
            "n_mechanisms_evaluated": self.n_mechanisms_evaluated,
            "n_signals_emitted": self.n_signals_emitted,
            "n_coalition_votes": self.n_coalition_votes,
            "n_orders_proposed": self.n_orders_proposed,
            "n_approved": self.n_approved,
            "n_vetoed": self.n_vetoed,
            "n_inquiries_opened": self.n_inquiries_opened,
            "steps": [s.__dict__ for s in self.steps],
            "orders": [
                {
                    "asset": o.order.asset,
                    "direction": o.order.direction,
                    "size_pct_of_nav": o.order.size_pct_of_nav,
                    "size_dollars": o.order.size_dollars,
                    "confidence": o.order.confidence,
                    "rationale": o.order.rationale,
                    "verdict": o.verdict.value,
                    "verdict_reason": o.verdict_reason,
                    "inquiry_id": o.inquiry_id,
                    "logged_signal_ids": o.logged_signal_ids,
                    "sizing": o.sizing,
                }
                for o in self.orders
            ],
            "rationale": self.rationale,
        }


# ============================================================
# Helpers
# ============================================================

def _evaluate_all_mechanisms(
    state: MarketState,
    asof: datetime,
) -> tuple[dict[str, list[Signal]], list[CycleStep]]:
    """Run every registered mechanism's evaluate(). Returns signals + step log."""
    import time

    _import_all_mechanisms()
    signals_by_mech: dict[str, list[Signal]] = {}
    steps: list[CycleStep] = []
    for mech_id in list_mechanisms():
        cls = get_mechanism(mech_id)
        try:
            mech = cls()
        except Exception as e:
            steps.append(CycleStep(mech_id, 0, 0, f"instantiate failed: {e}"))
            continue
        t0 = time.monotonic()
        try:
            signals = mech.evaluate(state, asof)
        except Exception as e:
            elapsed = int((time.monotonic() - t0) * 1000)
            steps.append(CycleStep(mech_id, 0, elapsed, f"evaluate failed: {e}"))
            continue
        elapsed = int((time.monotonic() - t0) * 1000)
        steps.append(CycleStep(mech_id, len(signals), elapsed))
        if signals:
            signals_by_mech[mech_id] = signals
    return signals_by_mech, steps


def _vote_to_proposed_order(
    vote: CoalitionVote,
    nav: float,
    regime: Optional[RegimeState],
    db_path: Optional[Path | str],
) -> tuple[ProposedOrder, SizingDecision]:
    """Convert a CoalitionVote + sizing into a ProposedOrder."""
    if regime is None:
        regime_p_in_allowed = 0.5  # no regime info; mid value
    else:
        # Use the dominant regime for the contributing mechanisms; the prudent
        # default is to size against the mechanism's expected operating regime,
        # which we don't have per-mech here. Use the current dominant regime
        # probability as a conservative bound.
        regime_p_in_allowed = regime.probabilities.get(regime.regime, 1.0)

    # Aggregate per-contributor track record (simple mean for v1)
    win_rates: list[float] = []
    n_completes: list[int] = []
    cold_start = False
    for mech_id in vote.contributing_mechanism_ids:
        try:
            tr = track_record(mech_id, lookback_days=180, db_path=db_path)
        except Exception:
            continue
        if tr.n_completed >= 5:
            win_rates.append(tr.win_rate or 0.0)
            n_completes.append(tr.n_completed)
        else:
            cold_start = True
    avg_win = sum(win_rates) / len(win_rates) if win_rates else None
    total_n = sum(n_completes)

    sd = size_position(
        signal_confidence=vote.net_confidence,
        expected_gain_pct=0.05,
        expected_loss_pct=0.04,
        regime_probability_in_allowed=regime_p_in_allowed,
        track_record_win_rate=avg_win,
        track_record_n=total_n,
    )

    direction = vote.net_direction if vote.net_direction != "flat" else "long"
    rationale = (
        f"coalition({len(vote.contributing_mechanism_ids)} mechs, "
        f"net={vote.net_direction}, conf={vote.net_confidence:.2f}); "
        f"{sd.reason}"
    )

    order = ProposedOrder(
        asset=vote.asset,
        direction=direction,
        size_pct_of_nav=sd.final_size_pct,
        size_dollars=sd.final_size_pct * nav,
        rationale=rationale,
        holding_period_days=60,  # default; future: pull from contributing mechs
        confidence=vote.net_confidence,
        contributing_mechanisms=vote.contributing_mechanism_ids,
        cold_start=cold_start,
    )
    return order, sd


# ============================================================
# Main entry
# ============================================================

def run_cycle(
    *,
    nav: float,
    open_positions: Optional[list[OpenPosition]] = None,
    asof: Optional[datetime] = None,
    risk_limits: Optional[RiskLimits] = None,
    risk_state: Optional[RiskState] = None,
    db_path: Optional[Path | str] = None,
    dry_run: bool = True,
) -> TradingCycleResult:
    """Run one full TRADER cycle.

    `dry_run=True` (default): proposes orders, runs CONSCIENCE, but does NOT
    log signals to the ledger or open inquiries. Use this for testing or
    daily what-if reports.

    `dry_run=False`: approved orders are logged to mechanism_signals;
    inquiry-flagged orders open real inquiries you can answer.

    Returns a TradingCycleResult with the full audit trail.
    """
    asof = asof or datetime.now(timezone.utc)
    open_positions = open_positions or []
    risk_limits = risk_limits or RiskLimits()
    risk_state = risk_state or RiskState(nav=nav, peak_nav=nav)

    state = MarketState(db_path)
    try:
        regime = RegimeDetector(state).detect(asof)
        signals_by_mech, steps = _evaluate_all_mechanisms(state, asof)
    finally:
        state.close()

    n_signals_total = sum(len(v) for v in signals_by_mech.values())

    # Aggregate to coalition votes
    votes = aggregate_signals(signals_by_mech, asof=asof, db_path=db_path)

    # Convert each vote to a proposed order + size + conscience verdict
    order_records: list[OrderRecord] = []
    n_approved = n_vetoed = n_inquiries = 0
    for vote in votes:
        order, sd = _vote_to_proposed_order(vote, nav, regime, db_path)
        if order.size_dollars <= 0:
            continue
        regime_p = (
            None if regime is None
            else regime.probabilities.get(regime.regime, 1.0)
        )
        verdict = review_order(
            order,
            nav=nav,
            open_positions=open_positions,
            risk_state=risk_state,
            risk_limits=risk_limits,
            regime_probability_in_allowed=regime_p,
        )

        rec = OrderRecord(
            order=order,
            verdict=verdict.verdict,
            verdict_reason=verdict.reason,
            sizing={
                "raw_kelly": sd.raw_kelly,
                "fractional_kelly_used": sd.fractional_kelly_used,
                "regime_multiplier": sd.regime_multiplier,
                "track_record_multiplier": sd.track_record_multiplier,
                "final_size_pct": sd.final_size_pct,
                "capped_by": sd.capped_by,
                "win_prob": sd.win_prob,
            },
        )

        if verdict.verdict == Verdict.APPROVE:
            n_approved += 1
            if not dry_run:
                # Log signal for each contributing mechanism
                for mech_id in order.contributing_mechanisms:
                    sid = log_signal(
                        mechanism_id=mech_id,
                        asof=asof,
                        asset=order.asset,
                        direction=order.direction,
                        raw_size_pct=order.size_pct_of_nav,
                        confidence=order.confidence,
                        rationale=order.rationale,
                        regime_at_signal=(
                            None if regime is None else regime.to_dict()
                        ),
                        signal_metadata={
                            "verdict": verdict.verdict.value,
                            "coalition_size": len(
                                order.contributing_mechanisms
                            ),
                        },
                        db_path=db_path,
                    )
                    rec.logged_signal_ids.append(sid)

        elif verdict.verdict == Verdict.VETO:
            n_vetoed += 1

        elif verdict.verdict == Verdict.REDUCE_SIZE:
            # Resize and accept. Treat as approved at the new cap.
            n_approved += 1
            if verdict.adjusted_size_pct is not None:
                order.size_pct_of_nav = verdict.adjusted_size_pct
                order.size_dollars = verdict.adjusted_size_pct * nav
            if not dry_run:
                for mech_id in order.contributing_mechanisms:
                    sid = log_signal(
                        mechanism_id=mech_id,
                        asof=asof,
                        asset=order.asset,
                        direction=order.direction,
                        raw_size_pct=order.size_pct_of_nav,
                        confidence=order.confidence,
                        rationale=(
                            order.rationale
                            + f"; reduced by CONSCIENCE: {verdict.reason}"
                        ),
                        regime_at_signal=(
                            None if regime is None else regime.to_dict()
                        ),
                        signal_metadata={
                            "verdict": verdict.verdict.value,
                            "reduced_to_pct": verdict.adjusted_size_pct,
                        },
                        db_path=db_path,
                    )
                    rec.logged_signal_ids.append(sid)

        elif verdict.verdict == Verdict.INQUIRY:
            n_inquiries += 1
            if not dry_run and verdict.open_inquiry:
                rec.inquiry_id = open_inquiry(
                    inquiry_type=verdict.open_inquiry["type"],
                    body=verdict.open_inquiry["body"],
                    urgency=verdict.open_inquiry["urgency"],
                    options=verdict.open_inquiry.get("options"),
                    related_files=", ".join(order.contributing_mechanisms),
                    db_path=db_path,
                )

        order_records.append(rec)

    return TradingCycleResult(
        asof=asof,
        nav=nav,
        regime=regime.regime if regime else None,
        regime_probabilities=(
            regime.probabilities if regime else {}
        ),
        n_mechanisms_evaluated=len(steps),
        n_signals_emitted=n_signals_total,
        n_coalition_votes=len(votes),
        n_orders_proposed=len(order_records),
        n_approved=n_approved,
        n_vetoed=n_vetoed,
        n_inquiries_opened=n_inquiries,
        steps=steps,
        orders=order_records,
        rationale=(
            f"Cycle at {asof.isoformat()[:19]}: regime="
            f"{regime.regime if regime else 'unknown'}, "
            f"{len(steps)} mechs evaluated, {n_signals_total} signals, "
            f"{len(votes)} coalition votes, {len(order_records)} proposed orders, "
            f"{n_approved} approved, {n_vetoed} vetoed, "
            f"{n_inquiries} inquiries"
            + (" [DRY-RUN]" if dry_run else " [LIVE]")
        ),
    )
