"""Maxwell-Demon Index for compositional alpha.

The thermodynamic analogy
=========================

In physics, a Maxwell demon extracts work from a thermal system by *measuring*
microstates and selectively letting molecules through a door. The work the
demon can extract is bounded by the information content of its measurement:

    W_extractable <= k_B T * I(measurement; system_state) * ln(2)   [bits to nats]

(Bennett 1982, Sagawa-Ueda 2010, Parrondo et al. 2015 for modern formulations.)

The market analogue
===================

A real market is *not* perfectly informationally efficient because no single
participant reads every silo. Information is stratified — patent lawyers read
patents, actuaries read reserves, etc. The market's aggregate price embeds
each specialist's local view but never the joint view across silos.

HUNTER reads across silos and observes the joint state. It is the demon at
the door. The "work" it can extract = compositional alpha.

The bound is the **synergistic information** of the cross-silo composition:

    alpha_extractable_bits <= S(future_return; silo_A, silo_B, ..., silo_N)

where S is the synergy term in partial information decomposition. We already
have a synergy estimator (`quant.research.synergy.SynergyEstimator`), so we
can plug in directly.

What this module does
=====================

Per mechanism:
  1. Compute the synergy bits S of the mechanism's signal vs. realised alpha
     (using the ledger as observation history).
  2. Compute the realised alpha per signal.
  3. The DEMON INDEX = realised_alpha_per_bit. Mechanisms that extract more
     alpha per bit of synergy are *more thermodynamically efficient*.
  4. The DEMON CEILING = S * C, where C is a calibration constant from the
     historical (alpha, synergy) regression. Tells you how much MORE alpha
     this mechanism could potentially extract.

Read the result like this:
  - High demon index + headroom remaining: scale up size; mechanism has
    untapped extraction capacity.
  - High demon index + at ceiling: mechanism is near its thermodynamic limit;
    don't expect much more.
  - Low demon index + high synergy: information IS there but the mechanism
    is failing to exploit it. Refine the mechanism or the signal.

This is, as far as we have searched, a novel application of information
thermodynamics (Landauer 1961; Bennett 1982; Parrondo, Horowitz, Sagawa
2015) to compositional financial inference. The framework is coherent on
its own and can be cited even before empirical validation; the empirical
demon-index calibration becomes a deliverable of the summer 2026 study.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from quant.research.ledger import track_record
from quant.research.synergy import SynergyEstimator


@dataclass
class DemonIndex:
    """Per-mechanism thermodynamic alpha-extraction summary."""

    mechanism_id: str
    n_completed_trades: int
    realised_alpha_per_trade: float  # mean realised return per trade
    synergy_bits: Optional[float]  # S(realised_alpha; signal_features)
    alpha_per_bit: Optional[float]  # realised_alpha / synergy_bits
    ceiling_alpha: Optional[float]  # S * C with C from cross-mechanism calibration
    headroom_pct: Optional[float]  # (ceiling - realised) / ceiling, in [0, 1]
    interpretation: str

    def to_dict(self) -> dict:
        return {
            "mechanism_id": self.mechanism_id,
            "n_completed_trades": self.n_completed_trades,
            "realised_alpha_per_trade": (
                None if self.realised_alpha_per_trade is None
                else round(self.realised_alpha_per_trade, 4)
            ),
            "synergy_bits": (
                None if self.synergy_bits is None
                else round(self.synergy_bits, 4)
            ),
            "alpha_per_bit": (
                None if self.alpha_per_bit is None
                else round(self.alpha_per_bit, 4)
            ),
            "ceiling_alpha": (
                None if self.ceiling_alpha is None
                else round(self.ceiling_alpha, 4)
            ),
            "headroom_pct": (
                None if self.headroom_pct is None
                else round(self.headroom_pct, 4)
            ),
            "interpretation": self.interpretation,
        }


def compute_demon_index(
    mechanism_id: str,
    *,
    db_path: Optional[Path | str] = None,
    lookback_days: int = 365,
    calibration_alpha_per_bit: float = 0.02,
) -> DemonIndex:
    """Demon Index for a single mechanism.

    `calibration_alpha_per_bit` is the empirical conversion factor between
    a bit of synergistic information and a unit of realised alpha. It is
    the "C" in `alpha <= C * S` and must be calibrated cross-mechanism in
    the summer study. Default 0.02 = 2% expected alpha per bit; this is the
    placeholder until we have real data to fit it.
    """
    import sqlite3
    from quant.data.base import DEFAULT_DB

    path = Path(db_path) if db_path else DEFAULT_DB

    tr = track_record(mechanism_id, lookback_days=lookback_days, db_path=path)
    n = tr.n_completed
    if n < 5:
        return DemonIndex(
            mechanism_id=mechanism_id,
            n_completed_trades=n,
            realised_alpha_per_trade=tr.mean_return_pct or 0.0,
            synergy_bits=None,
            alpha_per_bit=None,
            ceiling_alpha=None,
            headroom_pct=None,
            interpretation=(
                "insufficient closed trades (need >= 5); demon index unavailable"
            ),
        )

    # Pull the (signal_metadata, realised_return) sequence for this mechanism
    conn = sqlite3.connect(str(path))
    try:
        rows = conn.execute(
            "SELECT s.signal_metadata, s.confidence, o.realised_return_pct "
            "FROM mechanism_signals s "
            "JOIN mechanism_outcomes o ON o.signal_id = s.id "
            "WHERE s.mechanism_id = ? AND o.realised_return_pct IS NOT NULL "
            "ORDER BY s.asof",
            (mechanism_id,),
        ).fetchall()
    finally:
        conn.close()

    if len(rows) < 5:
        return DemonIndex(
            mechanism_id=mechanism_id,
            n_completed_trades=len(rows),
            realised_alpha_per_trade=tr.mean_return_pct or 0.0,
            synergy_bits=None,
            alpha_per_bit=None,
            ceiling_alpha=None,
            headroom_pct=None,
            interpretation="insufficient observations to estimate synergy",
        )

    # Extract two features per signal: confidence + a scalar from metadata
    # (e.g. z_score for ZScorePredicate-derived signals).
    import json as _json

    confidences = []
    secondary = []
    returns = []
    for meta_json, conf, ret in rows:
        try:
            meta = _json.loads(meta_json) if meta_json else {}
        except _json.JSONDecodeError:
            meta = {}
        # Try common metadata keys; default 0
        sec = (
            meta.get("z_score")
            or meta.get("spread")
            or meta.get("magnitude")
            or 0.0
        )
        try:
            sec = float(sec)
        except (ValueError, TypeError):
            sec = 0.0
        confidences.append(float(conf or 0.0))
        secondary.append(sec)
        returns.append(float(ret))

    x = np.array(returns, dtype=float)
    a = np.array(confidences, dtype=float)
    b = np.array(secondary, dtype=float)

    # Use KSG (continuous) when we have enough data; fall back to discrete
    method = "ksg" if len(x) >= 30 else "discrete"
    estimator = SynergyEstimator(method=method, k=min(4, len(x) // 4))
    try:
        result = estimator.measure(x, a, b)
        synergy_bits = max(0.0, result.synergy_bits)
    except Exception:
        synergy_bits = 0.0

    realised = float(np.mean(x))
    if synergy_bits > 1e-9:
        alpha_per_bit = realised / synergy_bits
    else:
        alpha_per_bit = None

    ceiling = synergy_bits * calibration_alpha_per_bit
    if ceiling > 1e-9 and realised > 0:
        headroom = max(0.0, (ceiling - realised) / ceiling)
    else:
        headroom = None

    if synergy_bits < 0.1:
        interp = (
            "near-zero synergy: this mechanism's signal carries little "
            "compositional information; check if it's effectively single-silo"
        )
    elif alpha_per_bit and alpha_per_bit > calibration_alpha_per_bit * 0.8:
        interp = (
            "high efficiency: extracting most of the available compositional "
            "alpha; size up cautiously, near thermodynamic ceiling"
        )
    elif alpha_per_bit and alpha_per_bit < calibration_alpha_per_bit * 0.2:
        interp = (
            "low efficiency: synergy bits are present but mechanism isn't "
            "exploiting them; refine signal or scoring"
        )
    else:
        interp = (
            "moderate efficiency: extracting some of available alpha; "
            "headroom remains, scale-up plausible"
        )

    return DemonIndex(
        mechanism_id=mechanism_id,
        n_completed_trades=len(rows),
        realised_alpha_per_trade=realised,
        synergy_bits=synergy_bits,
        alpha_per_bit=alpha_per_bit,
        ceiling_alpha=ceiling if ceiling > 0 else None,
        headroom_pct=headroom,
        interpretation=interp,
    )


def compute_demon_index_all(
    db_path: Optional[Path | str] = None,
    lookback_days: int = 365,
    calibration_alpha_per_bit: float = 0.02,
) -> list[DemonIndex]:
    """Demon index for every mechanism in the ledger."""
    import sqlite3
    from quant.data.base import DEFAULT_DB

    path = Path(db_path) if db_path else DEFAULT_DB
    if not path.exists():
        return []
    conn = sqlite3.connect(str(path))
    try:
        try:
            rows = conn.execute(
                "SELECT DISTINCT mechanism_id FROM mechanism_signals "
                "ORDER BY mechanism_id"
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    finally:
        conn.close()

    return [
        compute_demon_index(
            r[0],
            db_path=path,
            lookback_days=lookback_days,
            calibration_alpha_per_bit=calibration_alpha_per_bit,
        )
        for r in rows
    ]
