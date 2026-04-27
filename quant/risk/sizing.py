"""Position sizing — Kelly-derived, regime-conditioned, track-record-aware.

Inputs (all explicit, all auditable):

  signal           the mechanism's emitted Signal (asset, direction, confidence)
  regime           current RegimeState from RegimeDetector (may be None)
  allowed_regimes  the regimes this mechanism is calibrated for
  track_record     mechanism's historical hit rate / Sharpe (may be None)
  expected_gain_pct, expected_loss_pct  per-trade scenario magnitudes
  fractional_kelly fraction of full Kelly to use (default 0.25 for safety)
  risk_limits      RiskLimits cap (the hard ceiling)
  nav              account NAV (for absolute sizing)

Output: SizingDecision with the final fraction, all the multipliers exposed,
and a human-readable reason string.

Why Kelly: it provably maximises long-run geometric growth IF inputs are
correct. Inputs are never correct, so we use 25-50% of Kelly. This avoids
the volatility that full Kelly produces while still capturing the asymmetry:
*high-conviction, high-edge bets get materially bigger sizes than
low-conviction ones.* This is exactly the asymmetry you wanted.

Why regime conditioning: a mechanism calibrated in risk-on can be wrong in
risk-off. Multiplying size by P(regime in allowed_set) collapses size to
ZERO when the regime detector is confident the mechanism shouldn't fire,
and full size when the regime fully aligns. Smooth, not binary.

Why track-record conditioning: mechanisms that have lost money recently
should bet smaller. Mechanisms with strong recent performance should bet
bigger. Posterior win-probability shrinkage toward the prior, weighted by
sample size.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from quant.risk.limits import RiskLimits


@dataclass
class SizingDecision:
    raw_kelly: float
    fractional_kelly_used: float
    edge_pct: float
    win_prob: float
    regime_multiplier: float
    track_record_multiplier: float
    final_size_pct: float
    capped_by: Optional[str]
    reason: str

    def is_zero(self) -> bool:
        return self.final_size_pct <= 1e-9


def _full_kelly_fraction(
    win_prob: float,
    expected_gain_pct: float,
    expected_loss_pct: float,
) -> float:
    """Standard Kelly for binary win/loss outcomes.

    f* = (p * b - q) / b
    where b = gain / loss (odds), p = win prob, q = 1-p.

    Returns 0 if Kelly is negative (don't bet) or inputs are invalid.
    """
    if expected_loss_pct <= 0 or expected_gain_pct <= 0:
        return 0.0
    if not (0.0 < win_prob < 1.0):
        return 0.0
    b = expected_gain_pct / expected_loss_pct
    q = 1.0 - win_prob
    f = (win_prob * b - q) / b
    return max(0.0, f)


def _shrink_win_prob(
    raw_confidence: float,
    track_record_win_rate: Optional[float],
    track_record_n: int = 0,
    prior_win_prob: float = 0.55,
) -> float:
    """Posterior win probability from confidence + history.

    With no track record we trust the mechanism's own confidence (clamped to
    [0.50, 0.85] — neither too humble to bet nor too cocky). With a track
    record of N completed trades and observed win rate w, we shrink toward
    the prior with weight ~ 10/(10+N): more data = trust the data.
    """
    confidence_clamped = max(0.50, min(0.85, raw_confidence))
    if track_record_win_rate is None or track_record_n <= 0:
        return confidence_clamped
    shrinkage = 10.0 / (10.0 + track_record_n)
    return shrinkage * prior_win_prob + (1.0 - shrinkage) * track_record_win_rate


def size_position(
    *,
    signal_confidence: float,
    expected_gain_pct: float = 0.04,
    expected_loss_pct: float = 0.04,
    regime_probability_in_allowed: float = 1.0,
    track_record_win_rate: Optional[float] = None,
    track_record_n: int = 0,
    fractional_kelly: float = 0.25,
    risk_limits: Optional[RiskLimits] = None,
    liquidity_cap_pct: Optional[float] = None,
) -> SizingDecision:
    """Compose all the multipliers into a single position-size fraction.

    Returns a SizingDecision exposing every step so the trader's Conscience
    agent can audit what was decided and why.
    """
    risk_limits = risk_limits or RiskLimits()

    win_prob = _shrink_win_prob(
        signal_confidence, track_record_win_rate, track_record_n
    )
    full_k = _full_kelly_fraction(win_prob, expected_gain_pct, expected_loss_pct)
    k_used = full_k * fractional_kelly

    edge = win_prob * expected_gain_pct - (1.0 - win_prob) * expected_loss_pct
    regime_mult = max(0.0, min(1.0, regime_probability_in_allowed))

    # Track-record multiplier: scale by Sharpe-like signal. Conservative shrinkage.
    if track_record_win_rate is None or track_record_n < 5:
        tr_mult = 0.7  # cold start: smaller until we know the mechanism works
    else:
        # Smooth: 0.4 at win=0.4, 1.0 at win=0.55, 1.4 at win=0.70 (capped 1.5)
        tr_mult = max(0.2, min(1.5, 1.0 + 4.0 * (track_record_win_rate - 0.55)))

    raw_size = k_used * regime_mult * tr_mult

    # Cap by risk limits
    capped_by: Optional[str] = None
    final = raw_size
    if final > risk_limits.max_position_pct:
        final = risk_limits.max_position_pct
        capped_by = "max_position_pct"
    if liquidity_cap_pct is not None and final > liquidity_cap_pct:
        final = liquidity_cap_pct
        capped_by = "liquidity"

    reason = (
        f"win_prob={win_prob:.2f} (raw conf={signal_confidence:.2f}, "
        f"tr_n={track_record_n}, tr_win={track_record_win_rate}); "
        f"full Kelly={full_k:.3f}, frac×{fractional_kelly}={k_used:.3f}; "
        f"regime×{regime_mult:.2f}; track×{tr_mult:.2f}; "
        f"raw={raw_size:.3%}, final={final:.3%}"
        + (f" capped by {capped_by}" if capped_by else "")
    )

    return SizingDecision(
        raw_kelly=full_k,
        fractional_kelly_used=k_used,
        edge_pct=edge,
        win_prob=win_prob,
        regime_multiplier=regime_mult,
        track_record_multiplier=tr_mult,
        final_size_pct=final,
        capped_by=capped_by,
        reason=reason,
    )
