"""HUNTER-MACRO — regime forecaster.

The regime detector (`quant.research.regime`) tells us where we ARE.
The forecaster tells us where we're GOING.

Method
======

Markov chain over the four regimes {risk_on, late_cycle, risk_off, crisis}.

  1. Run the regime detector daily over the historical FRED window.
  2. Count regime transitions: T[i, j] = # times we went from i -> j the next day.
  3. Apply Laplace smoothing (add 1 to every cell) so unobserved transitions
     have non-zero probability.
  4. Row-normalise to get the daily transition matrix P[i, j] = P(regime_{t+1} = j | regime_t = i).
  5. To forecast n days ahead from current distribution pi_0:
         pi_n = pi_0 @ P^n
  6. The stationary distribution (left eigenvector for eigenvalue 1)
     is the long-run regime mix the system relaxes to.
  7. The characteristic persistence (1 / (1 - lambda_2), where lambda_2 is the
     second-largest eigenvalue) is the average time the chain takes to
     forget where it started.

Why this matters for the TRADER
================================

A 60-day position calibrated for risk_on is exposed if the regime drifts
to risk_off within those 60 days. The right regime probability to gate the
trade by is the *60-day forecast*, not today's regime. This module gives
the TRADER agent that look-ahead.

It is also a useful instrument by itself: regime forecasts are publishable
content for The HUNTER Ledger Substack, and the synthetic regime walk
through 2007-2010 we did earlier can be done with forecasts as well —
"on Aug 2008, the model said P(crisis) at +30 days was X" — which is a
genuinely auditable empirical artefact.

Persistence
===========

The fitted transition matrix is cached in `quant_data.db` under
`regime_transition_matrix`. Refit explicitly via `RegimeForecaster.fit()`.
Stale matrices are recomputed automatically if `auto_refit_after_days` has
elapsed since the last fit (default 30 days).
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from quant.data.base import DEFAULT_DB, MarketState
from quant.research.regime import REGIMES, RegimeDetector, RegimeState


SCHEMA = """
CREATE TABLE IF NOT EXISTS regime_transition_matrix (
    from_regime TEXT NOT NULL,
    to_regime TEXT NOT NULL,
    probability REAL NOT NULL,
    n_observed INTEGER NOT NULL,
    fitted_at TEXT NOT NULL,
    fit_lookback_years INTEGER,
    PRIMARY KEY (from_regime, to_regime)
);

CREATE TABLE IF NOT EXISTS regime_forecaster_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _conn(db_path: Optional[Path | str]) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA)
    return conn


@dataclass
class ForecastedRegime:
    """Regime distribution at a future horizon, plus context."""

    asof: datetime
    horizon_days: int
    probabilities: dict[str, float]  # P(regime at asof + horizon_days)
    most_likely_regime: str
    transition_matrix: list[list[float]]  # 4x4 daily transition matrix
    fitted_at: datetime

    def prob(self, regime_name: str) -> float:
        return self.probabilities.get(regime_name, 0.0)

    def to_dict(self) -> dict:
        return {
            "asof": self.asof.isoformat(),
            "horizon_days": self.horizon_days,
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "most_likely_regime": self.most_likely_regime,
            "transition_matrix": [[round(x, 4) for x in row] for row in self.transition_matrix],
            "fitted_at": self.fitted_at.isoformat(),
        }


class RegimeForecaster:
    """Markov-chain regime forecaster with persisted transition matrix."""

    def __init__(
        self,
        state: MarketState,
        db_path: Optional[Path | str] = None,
        laplace_smoothing: float = 1.0,
        auto_refit_after_days: int = 30,
    ):
        self.state = state
        self.db_path = db_path
        self.detector = RegimeDetector(state)
        self.laplace = laplace_smoothing
        self.auto_refit_after_days = auto_refit_after_days

    # ── Persistence helpers ──────────────────────────────────────────────

    def _load_matrix(self) -> Optional[tuple[np.ndarray, datetime]]:
        conn = _conn(self.db_path)
        try:
            rows = conn.execute(
                "SELECT from_regime, to_regime, probability, fitted_at "
                "FROM regime_transition_matrix"
            ).fetchall()
        finally:
            conn.close()
        if not rows:
            return None
        idx = {r: i for i, r in enumerate(REGIMES)}
        m = np.zeros((4, 4))
        fitted_at = None
        for fr, to, p, ts in rows:
            if fr in idx and to in idx:
                m[idx[fr], idx[to]] = float(p)
                fitted_at = datetime.fromisoformat(ts)
        if fitted_at is None:
            return None
        return m, fitted_at

    def _save_matrix(
        self, matrix: np.ndarray, counts: np.ndarray, lookback_years: int
    ) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        conn = _conn(self.db_path)
        try:
            with conn:
                conn.execute("DELETE FROM regime_transition_matrix")
                for i, fr in enumerate(REGIMES):
                    for j, to in enumerate(REGIMES):
                        conn.execute(
                            "INSERT INTO regime_transition_matrix "
                            "(from_regime, to_regime, probability, n_observed, "
                            " fitted_at, fit_lookback_years) "
                            "VALUES (?, ?, ?, ?, ?, ?)",
                            (fr, to, float(matrix[i, j]), int(counts[i, j]),
                             now_iso, lookback_years),
                        )
        finally:
            conn.close()

    # ── Fit / forecast ───────────────────────────────────────────────────

    def fit(
        self,
        lookback_years: int = 10,
        step_days: int = 7,
    ) -> np.ndarray:
        """Run the detector across history, count transitions, build matrix.

        Uses a step of `step_days` (default weekly) to reduce compute. The
        Markov property is approximated at that timescale; we then convert
        the weekly matrix to a daily matrix by taking its 1/step_days root
        via eigen-decomposition.

        Returns the daily transition matrix.
        """
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=int(365.25 * lookback_years))

        regime_seq: list[str] = []
        cur = start
        while cur <= end:
            rs = self.detector.detect(cur)
            if rs is not None:
                regime_seq.append(rs.regime)
            cur += timedelta(days=step_days)

        if len(regime_seq) < 10:
            raise RuntimeError(
                f"Only {len(regime_seq)} regime states observed in lookback; "
                "ingest more FRED history first."
            )

        idx = {r: i for i, r in enumerate(REGIMES)}
        counts = np.full((4, 4), self.laplace)  # Laplace prior
        for prev, nxt in zip(regime_seq[:-1], regime_seq[1:]):
            counts[idx[prev], idx[nxt]] += 1.0

        # Row-normalise to weekly transition matrix
        weekly = counts / counts.sum(axis=1, keepdims=True)

        # Convert weekly -> daily via matrix root (eigen-decomposition)
        daily = _matrix_root(weekly, step_days)

        # Make absolutely sure rows sum to 1 after numerical work
        daily = np.maximum(daily, 0.0)
        row_sums = daily.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        daily = daily / row_sums

        self._save_matrix(daily, counts.astype(int), lookback_years)
        return daily

    def get_transition_matrix(self, allow_auto_refit: bool = True) -> np.ndarray:
        """Return the daily matrix, refitting if missing or stale."""
        loaded = self._load_matrix()
        if loaded is None:
            return self.fit()
        m, fitted_at = loaded
        age_days = (datetime.now(timezone.utc) - fitted_at).days
        if allow_auto_refit and age_days > self.auto_refit_after_days:
            return self.fit()
        return m

    def forecast(
        self,
        asof: Optional[datetime] = None,
        horizon_days: int = 30,
    ) -> Optional[ForecastedRegime]:
        asof = asof or datetime.now(timezone.utc)
        current = self.detector.detect(asof)
        if current is None:
            return None
        m = self.get_transition_matrix()
        pi_0 = np.array([current.probabilities[r] for r in REGIMES])
        m_n = np.linalg.matrix_power(m, max(1, int(horizon_days)))
        pi_n = pi_0 @ m_n
        # Numerical clean
        pi_n = np.maximum(pi_n, 0.0)
        pi_n = pi_n / pi_n.sum()

        probs = {r: float(pi_n[i]) for i, r in enumerate(REGIMES)}
        loaded = self._load_matrix()
        fitted_at = loaded[1] if loaded else datetime.now(timezone.utc)

        return ForecastedRegime(
            asof=asof,
            horizon_days=horizon_days,
            probabilities=probs,
            most_likely_regime=max(probs, key=probs.get),
            transition_matrix=m.tolist(),
            fitted_at=fitted_at,
        )

    def stationary_distribution(self) -> dict[str, float]:
        """The long-run regime mix the chain converges to.

        Found as the left eigenvector for eigenvalue 1 of the transition matrix.
        """
        m = self.get_transition_matrix()
        # Solve pi @ M = pi  ->  pi (M - I) = 0  with sum(pi) = 1
        # Equivalent: eigenvector of M.T for eigenvalue 1
        evals, evecs = np.linalg.eig(m.T)
        # Find eigenvalue closest to 1
        idx = int(np.argmin(np.abs(evals - 1.0)))
        pi = np.real(evecs[:, idx])
        # Eigenvectors are unique only up to scale; numpy may return the
        # negative of the Perron-Frobenius eigenvector. Flip sign if needed
        # so the principal direction has positive total mass.
        if pi.sum() < 0:
            pi = -pi
        pi = np.maximum(pi, 0.0)
        s = pi.sum()
        if s <= 0:
            # Fallback: uniform distribution if eigenvector degenerate
            pi = np.ones(4) / 4.0
        else:
            pi = pi / s
        return {r: float(pi[i]) for i, r in enumerate(REGIMES)}

    def characteristic_persistence_days(self) -> float:
        """Average time for the chain to forget initial condition.

        Computed as 1 / (1 - |lambda_2|), where lambda_2 is the second-largest
        eigenvalue of the transition matrix in absolute value. Smaller |lambda_2|
        means faster mixing; larger means more persistent regimes.
        """
        m = self.get_transition_matrix()
        evals = np.abs(np.linalg.eigvals(m))
        evals = np.sort(evals)[::-1]
        if len(evals) < 2:
            return float("inf")
        l2 = evals[1]
        if l2 >= 1.0:
            return float("inf")
        return float(1.0 / (1.0 - l2))


# ============================================================
# Math helper
# ============================================================

def _matrix_root(M: np.ndarray, n: int) -> np.ndarray:
    """Compute the n-th root of a (positive, row-stochastic) matrix M.

    Uses eigen-decomposition: if M = V D V^-1, then M^(1/n) = V D^(1/n) V^-1.

    Returns a real-valued matrix; tiny imaginary parts due to numerical noise
    are dropped. Negative entries (which can arise for non-embeddable chains)
    are clamped to zero and rows are renormalised; this is a known limitation
    of the eigen approach for non-embeddable Markov chains, and is fine for
    forecasting because the resulting daily matrix raised back to power n
    closely approximates the original M.
    """
    if n == 1:
        return M
    evals, evecs = np.linalg.eig(M)
    # n-th root of eigenvalues (handles complex)
    eval_root = np.power(evals + 0j, 1.0 / n)
    D_root = np.diag(eval_root)
    M_root = evecs @ D_root @ np.linalg.inv(evecs)
    # Drop imaginary noise
    M_root = np.real(M_root)
    return M_root


# ============================================================
# CLI demo
# ============================================================

def cli_forecast(
    db_path: Optional[Path | str] = None,
    horizons: tuple[int, ...] = (1, 7, 30, 90, 180),
    asof: Optional[datetime] = None,
) -> dict:
    """Forecast regime probabilities at multiple horizons. CLI-friendly output."""
    state = MarketState(db_path)
    try:
        forecaster = RegimeForecaster(state, db_path=db_path)
        out: dict = {
            "asof": (asof or datetime.now(timezone.utc)).isoformat(),
            "horizons": {},
        }
        for h in horizons:
            f = forecaster.forecast(asof=asof, horizon_days=h)
            if f is None:
                out["horizons"][h] = None
            else:
                out["horizons"][h] = {
                    "most_likely": f.most_likely_regime,
                    "probabilities": {k: round(v, 4) for k, v in f.probabilities.items()},
                }
        out["stationary"] = {
            k: round(v, 4) for k, v in forecaster.stationary_distribution().items()
        }
        out["characteristic_persistence_days"] = round(
            forecaster.characteristic_persistence_days(), 1
        )
        return out
    finally:
        state.close()
