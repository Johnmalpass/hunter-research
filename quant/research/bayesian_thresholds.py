"""Online Bayesian threshold update for predicates.

Each predicate (ZScorePredicate, SpreadPredicate, ThresholdPredicate) has a
fixed threshold. In production, the optimal threshold for a given predicate
in a given regime drifts. This module learns it online via Beta-Bernoulli
conjugate updates plus Thompson sampling.

Approach
========

Discretise the threshold space into K candidate values (e.g., z in
{1.0, 1.5, 2.0, 2.5, 3.0}). For each candidate threshold t_k maintain a
Beta(α_k, β_k) posterior over the win rate of trades that fired at t_k.

After every closed trade that was triggered at threshold t_k with outcome
o ∈ {win, loss}:

    α_k += 1[win]
    β_k += 1[loss]

Thompson sampling at each cycle: sample p_k ~ Beta(α_k, β_k) and pick the
threshold with the highest sample. Argmax-posterior-mean is the more
exploitative choice; Thompson balances explore/exploit.

The state is persisted in SQLite so it survives across runs. Each predicate
identifier (free-form string, typically "<mechanism_id>:<predicate_name>") has
its own posterior over its own discretised threshold grid.

Why the Beta-Bernoulli conjugate
================================

  - Closed-form posterior update (one addition per trade)
  - Natural credible intervals (qbeta inverse CDF)
  - Robust to small samples (the prior keeps things sane)
  - Easy to inspect: posterior mean = α/(α+β); after N trades the variance
    of the posterior shrinks like 1/N, and the operator can see when the
    posterior is sharp enough to trust.

The Beta prior defaults to Beta(1, 1) (uniform). Pass an informative prior
if you have one — e.g. a Beta(α=5, β=5) says "I expect 50/50 with some
prior weight."

This is the literal mathematical implementation of the system getting
exponentially smarter over time without retraining.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from quant.data.base import DEFAULT_DB


SCHEMA = """
CREATE TABLE IF NOT EXISTS bayesian_thresholds (
    predicate_id TEXT NOT NULL,
    threshold_value REAL NOT NULL,
    alpha REAL NOT NULL DEFAULT 1.0,
    beta REAL NOT NULL DEFAULT 1.0,
    n_observed INTEGER NOT NULL DEFAULT 0,
    last_updated TEXT,
    PRIMARY KEY (predicate_id, threshold_value)
);
"""


def _conn(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA)
    return conn


@dataclass
class ThresholdPosterior:
    threshold: float
    alpha: float
    beta: float
    n_observed: int

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        s = a + b
        return (a * b) / (s * s * (s + 1.0))

    def credible_interval(self, level: float = 0.90) -> tuple[float, float]:
        from scipy.stats import beta as _beta

        return _beta.interval(level, self.alpha, self.beta)


class BayesianThresholdGrid:
    """Online posterior over a discretised threshold grid for one predicate.

    Persisted to SQLite under `predicate_id`. Initialise once; every
    `update(threshold, won)` call commits to disk.
    """

    def __init__(
        self,
        predicate_id: str,
        candidate_thresholds: Sequence[float],
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        db_path: Path | str | None = None,
    ):
        if not candidate_thresholds:
            raise ValueError("need at least one candidate threshold")
        self.predicate_id = predicate_id
        self.candidates = sorted(float(t) for t in candidate_thresholds)
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)
        self.db_path = db_path
        self._ensure_seeded()

    def _ensure_seeded(self) -> None:
        conn = _conn(self.db_path)
        try:
            with conn:
                for t in self.candidates:
                    conn.execute(
                        "INSERT OR IGNORE INTO bayesian_thresholds "
                        "(predicate_id, threshold_value, alpha, beta, n_observed) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (self.predicate_id, t, self.prior_alpha, self.prior_beta, 0),
                    )
        finally:
            conn.close()

    def _nearest_candidate(self, t: float) -> float:
        return min(self.candidates, key=lambda x: abs(x - t))

    def update(self, threshold: float, won: bool) -> None:
        t = self._nearest_candidate(threshold)
        delta_a = 1.0 if won else 0.0
        delta_b = 0.0 if won else 1.0
        conn = _conn(self.db_path)
        try:
            with conn:
                conn.execute(
                    "UPDATE bayesian_thresholds "
                    "SET alpha = alpha + ?, beta = beta + ?, "
                    "    n_observed = n_observed + 1, last_updated = ? "
                    "WHERE predicate_id = ? AND threshold_value = ?",
                    (
                        delta_a, delta_b,
                        datetime.now(timezone.utc).isoformat(),
                        self.predicate_id, t,
                    ),
                )
        finally:
            conn.close()

    def posterior(self, threshold: float) -> Optional[ThresholdPosterior]:
        t = self._nearest_candidate(threshold)
        conn = _conn(self.db_path)
        try:
            row = conn.execute(
                "SELECT alpha, beta, n_observed FROM bayesian_thresholds "
                "WHERE predicate_id = ? AND threshold_value = ?",
                (self.predicate_id, t),
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        return ThresholdPosterior(threshold=t, alpha=row[0], beta=row[1], n_observed=row[2])

    def all_posteriors(self) -> list[ThresholdPosterior]:
        conn = _conn(self.db_path)
        try:
            rows = conn.execute(
                "SELECT threshold_value, alpha, beta, n_observed "
                "FROM bayesian_thresholds WHERE predicate_id = ? "
                "ORDER BY threshold_value",
                (self.predicate_id,),
            ).fetchall()
        finally:
            conn.close()
        return [
            ThresholdPosterior(threshold=r[0], alpha=r[1], beta=r[2], n_observed=r[3])
            for r in rows
        ]

    def thompson_sample(
        self,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[float, float]:
        """Sample a (threshold, sampled_win_probability). Use the threshold."""
        rng = rng or np.random.default_rng()
        posts = self.all_posteriors()
        if not posts:
            return self.candidates[0], self.prior_alpha / (self.prior_alpha + self.prior_beta)
        samples = [(p.threshold, float(rng.beta(p.alpha, p.beta))) for p in posts]
        best = max(samples, key=lambda x: x[1])
        return best

    def best_by_posterior_mean(self) -> ThresholdPosterior:
        return max(self.all_posteriors(), key=lambda p: p.mean)

    def report(self) -> list[dict]:
        out = []
        for p in self.all_posteriors():
            ci_lo, ci_hi = p.credible_interval(0.90)
            out.append(
                {
                    "threshold": p.threshold,
                    "n_observed": p.n_observed,
                    "posterior_mean": round(p.mean, 4),
                    "ci_90": (round(ci_lo, 4), round(ci_hi, 4)),
                    "alpha": round(p.alpha, 2),
                    "beta": round(p.beta, 2),
                }
            )
        return out


def reset_predicate(predicate_id: str, db_path: Path | str | None = None) -> int:
    """Wipe all posteriors for a predicate. Returns rows deleted."""
    conn = _conn(db_path)
    try:
        with conn:
            cur = conn.execute(
                "DELETE FROM bayesian_thresholds WHERE predicate_id = ?",
                (predicate_id,),
            )
            return cur.rowcount
    finally:
        conn.close()
