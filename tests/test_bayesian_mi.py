"""Tests for online Bayesian thresholds + Mechanism MI Network."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from quant.data.base import DataPoint, write_points
from quant.research.bayesian_thresholds import (
    BayesianThresholdGrid,
    ThresholdPosterior,
    reset_predicate,
)
from quant.research.ledger import log_signal
from quant.research.mi_network import compute_mi_network


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test_bm.db"


# ============================================================
# Bayesian thresholds
# ============================================================

def test_bayesian_grid_seeds_priors(tmp_db: Path):
    grid = BayesianThresholdGrid(
        predicate_id="test_pred",
        candidate_thresholds=[1.0, 1.5, 2.0, 2.5, 3.0],
        prior_alpha=1.0, prior_beta=1.0,
        db_path=tmp_db,
    )
    posts = grid.all_posteriors()
    assert len(posts) == 5
    for p in posts:
        assert p.alpha == 1.0
        assert p.beta == 1.0
        assert p.n_observed == 0
        assert p.mean == pytest.approx(0.5)


def test_bayesian_update_shifts_posterior(tmp_db: Path):
    grid = BayesianThresholdGrid(
        predicate_id="z_humira",
        candidate_thresholds=[1.5, 2.0, 2.5, 3.0],
        db_path=tmp_db,
    )
    # Win 7 times at z=2.0, lose 3 times
    for _ in range(7):
        grid.update(2.0, won=True)
    for _ in range(3):
        grid.update(2.0, won=False)

    p = grid.posterior(2.0)
    assert p is not None
    assert p.n_observed == 10
    # Posterior mean = (1+7)/(1+7+1+3) = 8/12 = 0.667
    assert p.mean == pytest.approx(8.0 / 12.0)
    # Other thresholds untouched
    assert grid.posterior(2.5).n_observed == 0


def test_bayesian_thompson_sampling_picks_likely_winner(tmp_db: Path):
    grid = BayesianThresholdGrid(
        predicate_id="z_proven",
        candidate_thresholds=[1.0, 2.0, 3.0],
        db_path=tmp_db,
    )
    # Make threshold 2.0 dramatically better than the others
    for _ in range(50):
        grid.update(2.0, won=True)
    for _ in range(2):
        grid.update(2.0, won=False)
    for _ in range(50):
        grid.update(1.0, won=False)
    for _ in range(50):
        grid.update(3.0, won=False)

    rng = np.random.default_rng(42)
    picks = [grid.thompson_sample(rng)[0] for _ in range(200)]
    # threshold 2.0 should dominate
    pct_2 = sum(1 for p in picks if p == 2.0) / len(picks)
    assert pct_2 > 0.80, f"expected ~all picks to be 2.0; got {pct_2:.0%}"


def test_bayesian_persistence_across_instances(tmp_db: Path):
    grid_a = BayesianThresholdGrid(
        predicate_id="persist_test", candidate_thresholds=[1.0, 2.0],
        db_path=tmp_db,
    )
    grid_a.update(2.0, won=True)
    grid_a.update(2.0, won=True)

    # New instance reads same db
    grid_b = BayesianThresholdGrid(
        predicate_id="persist_test", candidate_thresholds=[1.0, 2.0],
        db_path=tmp_db,
    )
    p = grid_b.posterior(2.0)
    assert p.n_observed == 2
    assert p.alpha == 3.0  # 1 prior + 2 wins


def test_credible_interval_narrows_with_data(tmp_db: Path):
    grid = BayesianThresholdGrid(
        predicate_id="ci_test", candidate_thresholds=[2.0],
        db_path=tmp_db,
    )
    p_initial = grid.posterior(2.0)
    lo0, hi0 = p_initial.credible_interval(0.90)
    initial_width = hi0 - lo0

    for _ in range(100):
        grid.update(2.0, won=True)
    p_final = grid.posterior(2.0)
    lo1, hi1 = p_final.credible_interval(0.90)
    final_width = hi1 - lo1
    assert final_width < initial_width / 5  # CI should shrink dramatically


def test_reset_predicate(tmp_db: Path):
    grid = BayesianThresholdGrid(
        predicate_id="reset_test",
        candidate_thresholds=[1.0, 2.0, 3.0],
        db_path=tmp_db,
    )
    grid.update(2.0, True)
    n_deleted = reset_predicate("reset_test", db_path=tmp_db)
    assert n_deleted == 3
    # New grid starts fresh
    grid2 = BayesianThresholdGrid(
        predicate_id="reset_test",
        candidate_thresholds=[1.0, 2.0, 3.0],
        db_path=tmp_db,
    )
    p = grid2.posterior(2.0)
    assert p.n_observed == 0


# ============================================================
# MI Network
# ============================================================

def test_mi_network_empty_ledger(tmp_db: Path):
    result = compute_mi_network(db_path=tmp_db)
    assert result.mechanisms == []
    assert result.diversity_score == 1.0


def test_mi_network_single_mechanism(tmp_db: Path):
    asof = datetime.now(timezone.utc) - timedelta(days=180)
    for d in range(40):
        log_signal(
            mechanism_id="solo",
            asof=asof + timedelta(days=d),
            asset="X", direction="long", raw_size_pct=0.01, confidence=0.5,
            rationale="test", db_path=tmp_db,
        )
    result = compute_mi_network(db_path=tmp_db, lookback_days=365)
    assert result.mechanisms == ["solo"]
    assert result.diversity_score == 1.0


def test_mi_network_correlated_mechanisms_clustered(tmp_db: Path):
    """Two mechanisms firing on identical days should be in the same cluster."""
    asof = datetime.now(timezone.utc) - timedelta(days=300)
    n_days = 200  # gives ~40 firing dates per mech, above min_overlap_days
    for d in range(n_days):
        if d % 5 == 0:
            log_signal(
                mechanism_id="mech_A",
                asof=asof + timedelta(days=d),
                asset="X", direction="long", raw_size_pct=0.01, confidence=0.5,
                rationale="t", db_path=tmp_db,
            )
            log_signal(
                mechanism_id="mech_B",
                asof=asof + timedelta(days=d),
                asset="Y", direction="long", raw_size_pct=0.01, confidence=0.5,
                rationale="t", db_path=tmp_db,
            )

    result = compute_mi_network(
        db_path=tmp_db, lookback_days=400, mi_edge_threshold_bits=0.01
    )
    assert "mech_A" in result.mechanisms and "mech_B" in result.mechanisms
    assert any(
        (e.src == "mech_A" and e.dst == "mech_B")
        or (e.src == "mech_B" and e.dst == "mech_A")
        for e in result.edges
    )
    assert result.clusters["mech_A"] == result.clusters["mech_B"]


def test_mi_network_perfectly_disjoint_mechanisms(tmp_db: Path):
    """Mechanisms that fire on strictly disjoint days should have low MI.

    Designed to avoid the (0,0)-bin dominance of the plug-in MI estimator
    that inflates apparent dependence between independent sparse Bernoulli
    streams. With strict disjoint firing, the estimator correctly sees no
    co-firing -> MI below threshold -> no edge -> different clusters.
    """
    asof = datetime.now(timezone.utc) - timedelta(days=400)
    n_days = 300
    for d in range(n_days):
        if d % 7 == 0:  # mech A fires every 7th day
            log_signal(
                mechanism_id="disj_A", asof=asof + timedelta(days=d),
                asset="X", direction="long", raw_size_pct=0.01, confidence=0.5,
                rationale="t", db_path=tmp_db,
            )
        elif d % 11 == 3:  # mech B fires on different residue mod 11
            log_signal(
                mechanism_id="disj_B", asof=asof + timedelta(days=d),
                asset="Y", direction="long", raw_size_pct=0.01, confidence=0.5,
                rationale="t", db_path=tmp_db,
            )

    result = compute_mi_network(
        db_path=tmp_db, lookback_days=500, mi_edge_threshold_bits=0.05
    )
    assert "disj_A" in result.mechanisms and "disj_B" in result.mechanisms
    # Strictly disjoint streams -> different clusters (or no edges => singletons)
    assert result.clusters["disj_A"] != result.clusters["disj_B"] or not result.edges
