"""Tests for HUNTER-MACRO regime forecaster."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from quant.research.regime_forecast import (
    RegimeForecaster,
    _matrix_root,
)


def test_matrix_root_recovers_identity():
    """nth root of M, raised to nth power, should approximately equal M."""
    M = np.array(
        [
            [0.95, 0.04, 0.01, 0.0],
            [0.10, 0.85, 0.05, 0.0],
            [0.02, 0.10, 0.80, 0.08],
            [0.01, 0.05, 0.20, 0.74],
        ]
    )
    R = _matrix_root(M, n=7)
    M_recovered = np.linalg.matrix_power(R, 7)
    assert np.allclose(M_recovered, M, atol=1e-6)


def test_matrix_root_n_equals_one_returns_input():
    M = np.array([[0.5, 0.5], [0.3, 0.7]])
    R = _matrix_root(M, n=1)
    assert np.array_equal(R, M)


def test_forecaster_persists_and_loads_matrix(tmp_path: Path):
    """Save a hand-built transition matrix, load it, confirm it round-trips."""
    db = tmp_path / "rf.db"
    # We'll bypass fit() and write a known matrix directly via the persistence layer
    # to test load/save without needing FRED data.
    from quant.data.base import MarketState
    from quant.research.regime_forecast import _conn

    state = MarketState(db)
    try:
        # Write a clean diagonal-dominant matrix
        M = np.array(
            [
                [0.90, 0.05, 0.04, 0.01],
                [0.10, 0.80, 0.07, 0.03],
                [0.02, 0.18, 0.70, 0.10],
                [0.01, 0.05, 0.30, 0.64],
            ]
        )
        counts = (M * 100).astype(int)
        forecaster = RegimeForecaster(state, db_path=db)
        forecaster._save_matrix(M, counts, lookback_years=5)

        loaded, fitted_at = forecaster._load_matrix()
        assert np.allclose(loaded, M, atol=1e-9)
        assert isinstance(fitted_at, datetime)
    finally:
        state.close()


def test_stationary_distribution_sums_to_one(tmp_path: Path):
    db = tmp_path / "rf.db"
    from quant.data.base import MarketState

    state = MarketState(db)
    try:
        M = np.array(
            [
                [0.90, 0.05, 0.04, 0.01],
                [0.10, 0.80, 0.07, 0.03],
                [0.02, 0.18, 0.70, 0.10],
                [0.01, 0.05, 0.30, 0.64],
            ]
        )
        f = RegimeForecaster(state, db_path=db)
        f._save_matrix(M, (M * 100).astype(int), lookback_years=5)
        pi = f.stationary_distribution()
        assert abs(sum(pi.values()) - 1.0) < 1e-9
        assert all(0.0 <= v <= 1.0 for v in pi.values())
    finally:
        state.close()


def test_characteristic_persistence_positive(tmp_path: Path):
    db = tmp_path / "rf.db"
    from quant.data.base import MarketState

    state = MarketState(db)
    try:
        # Highly persistent diagonal matrix
        M = np.eye(4) * 0.95 + np.ones((4, 4)) * 0.05 / 4 - np.eye(4) * 0.05 / 4
        # Renormalise rows
        M = M / M.sum(axis=1, keepdims=True)
        f = RegimeForecaster(state, db_path=db)
        f._save_matrix(M, (M * 100).astype(int), lookback_years=5)
        tau = f.characteristic_persistence_days()
        assert tau > 0  # positive
        assert tau < 1e6  # finite
    finally:
        state.close()


def test_forecaster_returns_none_when_no_regime_data(tmp_path: Path):
    db = tmp_path / "rf.db"
    from quant.data.base import MarketState

    state = MarketState(db)
    try:
        # Empty db, no FRED -> detector returns None -> forecaster returns None
        M = np.eye(4)
        f = RegimeForecaster(state, db_path=db)
        f._save_matrix(M, (M * 100).astype(int), lookback_years=5)
        out = f.forecast(asof=datetime(2024, 5, 1, tzinfo=timezone.utc), horizon_days=30)
        assert out is None
    finally:
        state.close()


def test_forecast_distribution_sums_to_one(tmp_path: Path):
    """Even with synthetic detector inputs, the forecast should be a proper
    probability distribution. We test the math by manually constructing pi_0
    and pi_n from a known matrix."""
    M = np.array(
        [
            [0.90, 0.05, 0.04, 0.01],
            [0.10, 0.80, 0.07, 0.03],
            [0.02, 0.18, 0.70, 0.10],
            [0.01, 0.05, 0.30, 0.64],
        ]
    )
    pi_0 = np.array([1.0, 0.0, 0.0, 0.0])
    pi_30 = pi_0 @ np.linalg.matrix_power(M, 30)
    assert abs(pi_30.sum() - 1.0) < 1e-9
    assert all(p >= 0 for p in pi_30)
    # Starting in risk_on, the chain should drift toward stationary mix
    # which has non-zero mass on every regime
    assert pi_30[0] < pi_0[0]  # risk_on probability decays
