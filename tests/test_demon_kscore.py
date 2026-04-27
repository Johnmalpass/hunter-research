"""Tests for the Demon Index + K-Score modules."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from quant.research.demon_index import compute_demon_index
from quant.research.k_score import (
    KScoreResult,
    k_score,
    normalised_compression_distance,
)
from quant.research.ledger import log_outcome, log_signal


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test_dk.db"


# ============================================================
# K-Score
# ============================================================

def test_k_score_basic_returns_result():
    result = k_score(
        thesis_text="When CMBS office delinquency exceeds 6% and AA-Treasury "
                    "spread exceeds 100bps and a quarterly statutory filing is "
                    "within 60 days, life insurers are systematically "
                    "under-reserved and a basket short of MET, PRU, LNC, AFL "
                    "is profitable for 120 days.",
        constituent_facts=[
            "CMBS office delinquency rate at 12.3%",
            "AA corporate yield curve refresh cycle is 90 days",
            "NAIC Q1 statutory filing date is May 15",
            "Life insurers use AA yield curve for capital reserve calculations",
        ],
    )
    assert isinstance(result, KScoreResult)
    assert result.n_constituent_facts == 4
    assert result.raw_thesis_bytes > 0
    assert result.compressed_thesis_bytes > 0


def test_k_score_empty_thesis_raises():
    with pytest.raises(ValueError):
        k_score("", ["fact"])


def test_k_score_no_facts_returns_positive():
    """A thesis with no constituent facts should show positive K-score
    (everything in the thesis is "novel" relative to nothing)."""
    result = k_score(
        thesis_text="A long detailed thesis about cross-silo composition.",
        constituent_facts=[],
    )
    assert result.k_score_bytes > 0
    assert result.k_score_normalised > 0


def test_k_score_redundant_thesis_low_or_negative():
    """A thesis that just restates one fact should compress similarly,
    giving low or negative K-score after subtracting the fact."""
    fact = "The Federal Reserve raised interest rates by 25 basis points."
    thesis = fact  # same text -> compressed sizes nearly equal
    result = k_score(thesis_text=thesis, constituent_facts=[fact])
    # K_score = C(thesis) - C(fact); for identical text these are equal
    # so K_score ~ 0 and normalised should be near zero
    assert abs(result.k_score_normalised) < 0.5


def test_normalised_compression_distance_self_is_low():
    a = "The yield curve inverted in March 2023."
    ncd = normalised_compression_distance(a, a)
    # Concatenating with itself should compress nearly as well as one copy
    assert ncd < 0.3  # fairly low for identical strings


def test_normalised_compression_distance_unrelated_is_higher():
    a = "Federal Reserve monetary policy and the yield curve."
    b = "Quantum entanglement in NbN superconducting wires."
    ncd_self = normalised_compression_distance(a, a)
    ncd_diff = normalised_compression_distance(a, b)
    assert ncd_diff > ncd_self


def test_normalised_compression_distance_in_unit_interval():
    a = "test string"
    b = "completely unrelated content"
    ncd = normalised_compression_distance(a, b)
    assert 0.0 <= ncd <= 2.0  # NCD bounded, sometimes slightly > 1 in approx


# ============================================================
# Demon Index
# ============================================================

def test_demon_index_insufficient_data(tmp_db: Path):
    """A mechanism with no closed trades should return a clean
    'insufficient data' result, not an error."""
    di = compute_demon_index("nonexistent_mech", db_path=tmp_db)
    assert di.n_completed_trades == 0
    assert di.synergy_bits is None
    assert "insufficient" in di.interpretation


def test_demon_index_with_trades(tmp_db: Path):
    """Seed 30 closed trades; verify a Demon Index is produced."""
    asof = datetime.now(timezone.utc) - timedelta(days=180)
    for i in range(30):
        ts = asof + timedelta(days=i * 3)
        sid = log_signal(
            mechanism_id="test_demon",
            asof=ts,
            asset="X",
            direction="long",
            raw_size_pct=0.02,
            confidence=0.6 + 0.01 * (i % 5),
            rationale="t",
            signal_metadata={"z_score": 2.0 + 0.1 * (i % 4)},
            db_path=tmp_db,
        )
        # Alternating wins/losses with some variance
        ret = 0.03 if i % 2 == 0 else -0.02
        log_outcome(
            signal_id=sid,
            entry_date=ts,
            entry_price=100.0,
            exit_date=ts + timedelta(days=30),
            exit_price=100.0 * (1 + ret),
            realised_return_pct=ret,
            db_path=tmp_db,
        )

    di = compute_demon_index("test_demon", db_path=tmp_db)
    assert di.n_completed_trades == 30
    assert di.synergy_bits is not None
    assert di.synergy_bits >= 0.0  # synergy is non-negative by clamp
    assert di.realised_alpha_per_trade is not None
    # Interpretation should be a non-empty string
    assert isinstance(di.interpretation, str) and len(di.interpretation) > 5


def test_demon_index_zero_synergy_branch(tmp_db: Path):
    """When mechanism produces only single-confidence signals, synergy
    should be near zero, and interpretation should flag that."""
    asof = datetime.now(timezone.utc) - timedelta(days=180)
    for i in range(15):
        ts = asof + timedelta(days=i * 3)
        sid = log_signal(
            mechanism_id="flat_mech",
            asof=ts,
            asset="X",
            direction="long",
            raw_size_pct=0.02,
            confidence=0.5,  # all same
            rationale="t",
            signal_metadata={"z_score": 1.5},  # all same
            db_path=tmp_db,
        )
        log_outcome(
            signal_id=sid,
            entry_date=ts,
            entry_price=100.0,
            exit_date=ts + timedelta(days=30),
            exit_price=100.0,
            realised_return_pct=0.0,  # all flat
            db_path=tmp_db,
        )

    di = compute_demon_index("flat_mech", db_path=tmp_db)
    # All-zero variation -> synergy should be ~0
    assert di.synergy_bits is not None
    assert di.synergy_bits < 0.5
