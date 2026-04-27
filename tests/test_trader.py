"""Tests for the TRADER agent + CONSCIENCE.

Exercises the integration: mechanisms (registered via the existing
decorators) -> coalition -> sizing -> conscience -> ledger.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from quant.agents.conscience import (
    OpenPosition,
    ProposedOrder,
    Verdict,
    review_order,
)
from quant.agents.trader import run_cycle
from quant.risk.limits import RiskLimits, RiskState


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test_trader.db"


# ============================================================
# CONSCIENCE
# ============================================================

def test_conscience_approves_clean_order():
    order = ProposedOrder(
        asset="ABBV", direction="short", size_pct_of_nav=0.02,
        size_dollars=200.0, rationale="t", holding_period_days=30,
        confidence=0.7, contributing_mechanisms=["mech_a"],
    )
    verdict = review_order(
        order, nav=10000.0, open_positions=[],
        risk_state=RiskState(nav=10000, peak_nav=10000),
        regime_probability_in_allowed=0.8,
    )
    assert verdict.verdict == Verdict.APPROVE


def test_conscience_caps_oversized():
    order = ProposedOrder(
        asset="MET", direction="short", size_pct_of_nav=0.10,
        size_dollars=1000.0, rationale="t", holding_period_days=30,
        confidence=0.9, contributing_mechanisms=["m"],
    )
    verdict = review_order(
        order, nav=10000.0, open_positions=[],
        risk_state=RiskState(nav=10000, peak_nav=10000),
        regime_probability_in_allowed=1.0,
    )
    assert verdict.verdict == Verdict.REDUCE_SIZE
    assert verdict.adjusted_size_pct == pytest.approx(0.05)


def test_conscience_vetoes_in_drawdown():
    order = ProposedOrder(
        asset="X", direction="long", size_pct_of_nav=0.01, size_dollars=100,
        rationale="t", holding_period_days=30, confidence=0.7,
        contributing_mechanisms=["m"],
    )
    # 90% nav vs 100% peak = 10% drawdown -> blown
    state = RiskState(nav=9000, peak_nav=10000)
    verdict = review_order(
        order, nav=9000.0, open_positions=[], risk_state=state,
    )
    assert verdict.verdict == Verdict.VETO
    assert "drawdown" in verdict.reason


def test_conscience_inquiry_on_regime_mismatch():
    order = ProposedOrder(
        asset="X", direction="long", size_pct_of_nav=0.03, size_dollars=300,
        rationale="t", holding_period_days=30, confidence=0.7,
        contributing_mechanisms=["m"],
    )
    verdict = review_order(
        order, nav=10000.0, open_positions=[],
        risk_state=RiskState(nav=10000, peak_nav=10000),
        regime_probability_in_allowed=0.10,  # very low support
    )
    assert verdict.verdict == Verdict.INQUIRY
    assert verdict.open_inquiry is not None
    assert verdict.open_inquiry["type"] == "decision"


def test_conscience_silo_concentration_veto():
    order = ProposedOrder(
        asset="ABBV", direction="short", size_pct_of_nav=0.05, size_dollars=500,
        rationale="t", holding_period_days=30, confidence=0.7,
        contributing_mechanisms=["m"], silo="pharma",
    )
    # Already 18% of nav in pharma; this would push to 23% > 20% cap
    open_pos = [
        OpenPosition(asset="MRNA", direction="short", size_dollars=900,
                     entry_date=datetime.now(timezone.utc), silo="pharma"),
        OpenPosition(asset="GILD", direction="short", size_dollars=900,
                     entry_date=datetime.now(timezone.utc), silo="pharma"),
    ]
    verdict = review_order(
        order, nav=10000.0, open_positions=open_pos,
        risk_state=RiskState(nav=10000, peak_nav=10000),
        regime_probability_in_allowed=1.0,
    )
    assert verdict.verdict == Verdict.VETO
    assert "silo" in verdict.reason.lower()


# ============================================================
# TRADER end-to-end
# ============================================================

def test_run_cycle_dry_run_no_signals_logged(tmp_db: Path):
    """In dry-run mode, run_cycle should not log signals."""
    asof = datetime(2024, 5, 1, tzinfo=timezone.utc)
    result = run_cycle(
        nav=10000.0, asof=asof, db_path=tmp_db, dry_run=True,
    )
    assert result.nav == 10000.0
    # No mechanism's data is in tmp_db so signals should be empty
    assert result.n_signals_emitted == 0
    # No signals were logged
    import sqlite3
    conn = sqlite3.connect(str(tmp_db))
    try:
        try:
            n = conn.execute("SELECT COUNT(*) FROM mechanism_signals").fetchone()[0]
        except sqlite3.OperationalError:
            n = 0
    finally:
        conn.close()
    assert n == 0


def test_run_cycle_returns_full_audit_trail(tmp_db: Path):
    asof = datetime(2024, 5, 1, tzinfo=timezone.utc)
    result = run_cycle(
        nav=10000.0, asof=asof, db_path=tmp_db, dry_run=True,
    )
    # All counters consistent
    assert result.n_orders_proposed == (
        result.n_approved + result.n_vetoed + result.n_inquiries_opened
    )
    assert result.n_mechanisms_evaluated >= 2  # we have at least 2 registered
    # Steps record one entry per mechanism
    assert len(result.steps) == result.n_mechanisms_evaluated
    # to_dict round-trip is JSON-serialisable
    import json
    d = result.to_dict()
    s = json.dumps(d)
    assert "asof" in s


def test_run_cycle_reflects_regime(tmp_db: Path):
    asof = datetime.now(timezone.utc)
    result = run_cycle(
        nav=10000.0, asof=asof, db_path=tmp_db, dry_run=True,
    )
    # If FRED data isn't ingested in this tmp db, regime is None — both fine
    if result.regime is not None:
        # Probabilities sum to 1
        total = sum(result.regime_probabilities.values())
        assert abs(total - 1.0) < 1e-6
