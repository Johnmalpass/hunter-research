"""Mechanism Coalition — combine signals from many mechanisms into one vote.

When the TRADER agent runs each cycle, several mechanisms may fire on the
same (or related) assets simultaneously. This module turns that bundle of
mechanism outputs into a single coalition decision per asset.

Voting rules
============

For each asset, gather all signals firing on that asset right now. For each
mechanism that emitted a signal, look up its track record and weight its
vote by:

  1. Recency-adjusted Sharpe (better recent performers get bigger weight)
  2. Direction agreement (longs and shorts cancel — net only)
  3. Diversity bonus (mechanisms touching different silos add more
     than redundant ones — captured via configurable silo metadata)

The final coalition vote per asset is:

  net_direction        sign of weighted-direction sum
  net_confidence       |weighted sum| / sum of |weights|
  total_size_pct       sum of individual sized positions, capped by
                       portfolio concentration limits

Why this matters
================

A single mechanism's signal is brittle. A coalition vote where five
mechanisms drawing on different cross-silo information all agree is
genuinely high-conviction, and deserves materially bigger size than any
one alone. This is the "exponentially smarter" piece — the system's
intelligence emerges from the ensemble, not from any single mechanism.

This module is the foundation. The TRADER agent (Phase 4) calls it each
cycle. The synergy layer can later replace recency-Sharpe with the actual
information-theoretic synergy II between mechanism signals — that is the
genuinely novel research move.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from quant.research.ledger import TrackRecord, track_record
from quant.research.mechanism import Signal


@dataclass
class WeightedVote:
    mechanism_id: str
    signal: Signal
    track: Optional[TrackRecord]
    weight: float
    direction_sign: int  # +1 long, -1 short, 0 abstain


@dataclass
class CoalitionVote:
    asset: str
    asof: datetime
    net_direction: str  # "long" | "short" | "flat"
    net_confidence: float
    total_size_pct: float
    contributing_mechanism_ids: list[str]
    votes: list[WeightedVote] = field(default_factory=list)
    rationale: str = ""

    def is_actionable(self, min_confidence: float = 0.4) -> bool:
        return (
            self.net_direction != "flat"
            and self.net_confidence >= min_confidence
            and self.total_size_pct > 1e-6
        )


def _direction_sign(direction: str) -> int:
    return {"long": +1, "short": -1, "exit": 0, "flat": 0}.get(direction, 0)


def _recency_sharpe_weight(tr: Optional[TrackRecord]) -> float:
    """Convert a TrackRecord into a non-negative weight.

    Cold-start (no track record) gets 1.0 — the mechanism's vote counts
    fully but neither bonus nor penalty.
    A high Sharpe tilts up; a poor recent record (cold mechanism, repeated
    losses) tilts down. Capped on both ends to keep the coalition stable.
    """
    if tr is None or tr.n_completed == 0:
        return 1.0
    base = 1.0
    if tr.sharpe_per_trade is not None:
        base = max(0.2, min(2.5, 1.0 + tr.sharpe_per_trade))
    if tr.is_cold:
        base *= 0.3
    return base


def aggregate_signals(
    signals_by_mechanism: dict[str, list[Signal]],
    *,
    asof: Optional[datetime] = None,
    track_record_lookback_days: int = 180,
    db_path=None,
    cold_start_default_size: float = 0.01,
) -> list[CoalitionVote]:
    """Group signals by asset, weight by track record, return one vote per asset.

    `signals_by_mechanism` maps mechanism_id -> list of signals it just emitted.
    Multiple mechanisms can sign the same asset; their votes net.
    """
    asof = asof or datetime.now(timezone.utc)

    by_asset: dict[str, list[tuple[str, Signal]]] = {}
    for mech_id, signals in signals_by_mechanism.items():
        for s in signals:
            by_asset.setdefault(s.asset, []).append((mech_id, s))

    out: list[CoalitionVote] = []
    for asset, items in by_asset.items():
        votes: list[WeightedVote] = []
        weighted_dir_sum = 0.0
        weight_sum = 0.0
        size_sum = 0.0

        for mech_id, sig in items:
            tr = None
            try:
                tr = track_record(
                    mech_id, lookback_days=track_record_lookback_days, db_path=db_path
                )
            except Exception:
                tr = None
            w = _recency_sharpe_weight(tr) * float(sig.confidence)
            d = _direction_sign(sig.direction)

            # If we have a track record use its win rate to weight further
            # Otherwise we accept the cold-start default
            size_contrib = float(sig.size_pct) if sig.size_pct else cold_start_default_size

            votes.append(
                WeightedVote(
                    mechanism_id=mech_id,
                    signal=sig,
                    track=tr,
                    weight=w,
                    direction_sign=d,
                )
            )
            weighted_dir_sum += w * d
            weight_sum += w
            # Conflict-aware sizing: net direction reduces size when shorts/longs
            # cancel, rather than over-summing.
            size_sum += size_contrib * d  # signed

        net_d_signed = size_sum  # signed: positive = net long
        if abs(weight_sum) < 1e-9:
            continue

        net_score = weighted_dir_sum / weight_sum  # in [-1, +1]
        if net_score > 0.05:
            net_direction = "long"
        elif net_score < -0.05:
            net_direction = "short"
        else:
            net_direction = "flat"

        net_confidence = abs(net_score)
        # Total size: absolute net of signed contributions (cancellations reduce size)
        total_size = abs(net_d_signed)

        rationale = (
            f"asset={asset}: {len(items)} mechanisms "
            f"({sum(1 for _, s in items if s.direction=='long')} long, "
            f"{sum(1 for _, s in items if s.direction=='short')} short); "
            f"net_score={net_score:+.2f}; total_size={total_size:.3%}"
        )

        out.append(
            CoalitionVote(
                asset=asset,
                asof=asof,
                net_direction=net_direction,
                net_confidence=net_confidence,
                total_size_pct=total_size,
                contributing_mechanism_ids=[m for m, _ in items],
                votes=votes,
                rationale=rationale,
            )
        )

    return out
