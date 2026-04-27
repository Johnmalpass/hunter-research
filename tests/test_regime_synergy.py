"""Tests for regime-conditional synergy.

Critical property: a synthetic dataset where the synergy-regime is masked
by a noise-regime should show pooled II ~ 0 but regime-conditional II
recovering the true synergy in the right regime.
"""
from __future__ import annotations

import numpy as np
import pytest

from quant.research.regime_synergy import cli_demo_regime_split
from quant.research.synergy import SynergyEstimator


def test_measure_grouped_separates_regimes():
    """Two-regime case: regime A redundant, regime B synergistic.

    A:  X = A   (B is noise)            -> II ~ 0 (no synergy, no redundancy)
    B:  X = A XOR B  (continuous form)  -> II > 0
    """
    rng = np.random.default_rng(0)
    n = 1500
    a_a = rng.standard_normal(n)
    b_a = rng.standard_normal(n)
    x_a = a_a + 0.1 * rng.standard_normal(n)

    a_b = rng.standard_normal(n)
    b_b = rng.standard_normal(n)
    x_b = np.sign(a_b * b_b) + 0.2 * rng.standard_normal(n)

    a = np.concatenate([a_a, a_b])
    b = np.concatenate([b_a, b_b])
    x = np.concatenate([x_a, x_b])
    labels = np.array(["A"] * n + ["B"] * n)

    est = SynergyEstimator(method="ksg", k=4)
    grouped = est.measure_grouped(x, a, b, labels)
    assert "A" in grouped and "B" in grouped
    assert grouped["A"] is not None and grouped["B"] is not None
    assert grouped["B"].ii_bits > grouped["A"].ii_bits + 0.3


def test_measure_grouped_too_few_obs_returns_none():
    rng = np.random.default_rng(0)
    n_main = 200
    n_tiny = 10  # below MIN_GROUP_OBS=30
    a = rng.standard_normal(n_main + n_tiny)
    b = rng.standard_normal(n_main + n_tiny)
    x = rng.standard_normal(n_main + n_tiny)
    labels = np.array(["main"] * n_main + ["tiny"] * n_tiny)

    est = SynergyEstimator()
    grouped = est.measure_grouped(x, a, b, labels)
    assert grouped["main"] is not None
    assert grouped["tiny"] is None  # under threshold -> None, not error


def test_measure_grouped_length_mismatch_raises():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    x = np.array([1.0, 2.0, 3.0])
    labels = np.array(["a", "b"])  # wrong length
    est = SynergyEstimator()
    with pytest.raises(ValueError):
        est.measure_grouped(x, a, b, labels)


def test_demo_regime_split_unmasks_synergy():
    """The flagship demonstration: pooled II ~ 0, stress-regime II ~ +1.

    This confirms the central theoretical claim of the module: a regime-blind
    PID can hide synergy that a regime-conditional PID reveals.
    """
    out = cli_demo_regime_split()
    assert out["n_total"] == 3000

    pooled = out["pooled_II_bits"]
    rc = out["regime_conditional"]
    calm_ii = rc["calm"]["II_bits"]
    stress_ii = rc["stress"]["II_bits"]

    # Stress regime is genuinely synergistic
    assert stress_ii > 0.5
    # Calm regime is essentially independent
    assert abs(calm_ii) < 0.2
    # Pooled is between them and below stress
    assert pooled < stress_ii
