"""Tests for the synergistic information estimator.

Validates against three textbook cases:
  XOR        II ~ +1 bit (perfect synergy)
  Mirror     II ~ -1 bit (perfect redundancy)
  Independent II ~ 0 bit (A is sufficient, B is noise)

Also verifies KSG estimator on a continuous nonlinear interaction.
"""
from __future__ import annotations

import numpy as np
import pytest

from quant.research.synergy import (
    SynergyEstimator,
    discrete_mi,
    interaction_information,
    ksg_mi,
)


def test_discrete_mi_independent_zero():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 4, size=10000)
    b = rng.integers(0, 4, size=10000)
    mi = discrete_mi(a, b)
    assert mi == pytest.approx(0.0, abs=0.05), f"got {mi}"


def test_discrete_mi_perfect_dependence_one_bit():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, size=10000)
    b = a.copy()
    mi = discrete_mi(a, b)
    assert mi == pytest.approx(1.0, abs=0.02), f"got {mi}"


def test_xor_is_synergistic():
    """X = A XOR B. Each marginal carries 0 bits; jointly they carry 1 bit."""
    rng = np.random.default_rng(0)
    n = 20000
    a = rng.integers(0, 2, size=n)
    b = rng.integers(0, 2, size=n)
    x = (a ^ b).astype(int)

    ii = interaction_information(x, a, b)
    assert ii > 0.9, f"expected ~+1 bit synergy, got II = {ii:.3f}"
    assert ii < 1.1, f"II suspiciously large: {ii:.3f}"


def test_mirror_is_redundant():
    """B = A; X = A. Both A and B carry the same 1 bit. II = -1."""
    rng = np.random.default_rng(0)
    n = 20000
    a = rng.integers(0, 2, size=n)
    b = a.copy()
    x = a.copy()

    ii = interaction_information(x, a, b)
    assert ii < -0.9, f"expected ~-1 bit redundancy, got II = {ii:.3f}"


def test_independent_redundant_is_zero():
    """X depends only on A; B is independent noise. II ~ 0."""
    rng = np.random.default_rng(0)
    n = 20000
    a = rng.integers(0, 2, size=n)
    b = rng.integers(0, 2, size=n)
    x = a.copy()

    ii = interaction_information(x, a, b)
    assert abs(ii) < 0.05, f"expected ~0 bit, got II = {ii:.3f}"


def test_estimator_class_dispatches():
    rng = np.random.default_rng(0)
    n = 5000
    a = rng.integers(0, 2, size=n)
    b = rng.integers(0, 2, size=n)
    x = (a ^ b).astype(int)

    est = SynergyEstimator(method="discrete")
    r = est.measure(x, a, b)
    assert r.synergistic
    assert r.synergy_bits == pytest.approx(r.ii_bits, abs=1e-9)
    assert r.redundancy_bits == 0.0
    assert r.n == n


def test_ksg_independent_continuous_near_zero():
    rng = np.random.default_rng(0)
    n = 2000
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    mi = ksg_mi(a, b, k=4)
    assert mi < 0.05, f"expected near-zero MI for independent gaussians, got {mi:.3f}"


def test_ksg_correlated_continuous_positive():
    rng = np.random.default_rng(0)
    n = 5000
    a = rng.standard_normal(n)
    b = a + 0.3 * rng.standard_normal(n)
    mi = ksg_mi(a, b, k=4)
    assert mi > 0.5, f"expected substantial MI for correlated gaussians, got {mi:.3f}"
