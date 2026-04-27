"""Tests for sizing, ledger, and coalition modules."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from quant.research.ledger import (
    log_outcome,
    log_signal,
    signals_for_asset,
    track_record,
)
from quant.research.coalition import aggregate_signals
from quant.research.mechanism import Signal
from quant.risk.limits import RiskLimits
from quant.risk.sizing import SizingDecision, _full_kelly_fraction, size_position


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test_slc.db"


# ============================================================
# Sizing
# ============================================================

def test_kelly_fraction_basic():
    # 60% win, 1:1 odds: f = (0.6*1 - 0.4)/1 = 0.2
    f = _full_kelly_fraction(0.6, 0.05, 0.05)
    assert f == pytest.approx(0.2, abs=1e-3)


def test_kelly_fraction_negative_zeros():
    # 50% win, 1:1 odds: edge = 0, Kelly = 0
    assert _full_kelly_fraction(0.5, 0.05, 0.05) == pytest.approx(0.0, abs=1e-3)
    # 40% win, 1:1 odds: negative edge -> 0
    assert _full_kelly_fraction(0.4, 0.05, 0.05) == 0.0


def test_size_position_high_conviction_pinned_at_cap():
    """Strong signal + aligned regime + good track record + full Kelly
    naturally exceeds the 5% RiskLimits cap and gets pinned there."""
    d = size_position(
        signal_confidence=0.85,
        expected_gain_pct=0.06,
        expected_loss_pct=0.04,
        regime_probability_in_allowed=1.0,
        track_record_win_rate=0.65,
        track_record_n=20,
        fractional_kelly=0.50,
    )
    assert d.final_size_pct == pytest.approx(0.05, abs=1e-9)
    assert d.capped_by == "max_position_pct"


def test_size_position_regime_misaligned_shrinks_size():
    """When P(allowed regime) is tiny, size shrinks proportionally.

    Compare RAW Kelly (pre-cap) so the RiskLimits ceiling doesn't hide the
    ratio. RAW raw_kelly is unaffected by regime_probability; what we test
    is final_size_pct at very low regime probability is itself small.
    """
    misaligned = size_position(
        signal_confidence=0.85,
        expected_gain_pct=0.06,
        expected_loss_pct=0.04,
        regime_probability_in_allowed=0.05,
        track_record_win_rate=0.65,
        track_record_n=20,
    )
    # With 5% regime prob, size should be < 1% (well under the 5% cap)
    assert misaligned.final_size_pct < 0.01
    # Regime multiplier is exactly the input
    assert misaligned.regime_multiplier == pytest.approx(0.05)


def test_size_position_no_track_record_smaller():
    """Cold-start mechanism gets smaller size than one with proven record."""
    cold = size_position(
        signal_confidence=0.7,
        expected_gain_pct=0.05,
        expected_loss_pct=0.05,
        regime_probability_in_allowed=1.0,
    )
    proven = size_position(
        signal_confidence=0.7,
        expected_gain_pct=0.05,
        expected_loss_pct=0.05,
        regime_probability_in_allowed=1.0,
        track_record_win_rate=0.65,
        track_record_n=30,
    )
    assert proven.final_size_pct >= cold.final_size_pct


def test_size_position_capped_by_risk_limits():
    """Even with absurd Kelly, final size cannot exceed RiskLimits cap."""
    limits = RiskLimits(max_position_pct=0.02)
    d = size_position(
        signal_confidence=0.95,
        expected_gain_pct=0.20,
        expected_loss_pct=0.02,
        regime_probability_in_allowed=1.0,
        track_record_win_rate=0.80,
        track_record_n=100,
        fractional_kelly=1.0,
        risk_limits=limits,
    )
    assert d.final_size_pct == pytest.approx(0.02, abs=1e-9)
    assert d.capped_by == "max_position_pct"


# ============================================================
# Ledger
# ============================================================

def test_log_signal_and_track_record(tmp_db: Path):
    asof = datetime(2024, 5, 1, tzinfo=timezone.utc)
    sid = log_signal(
        mechanism_id="test_mech",
        asof=asof,
        asset="MET",
        direction="short",
        raw_size_pct=0.03,
        confidence=0.7,
        rationale="test",
        regime_at_signal={"regime": "risk_off"},
        signal_metadata={"z_score": 2.5},
        db_path=tmp_db,
    )
    assert sid > 0

    log_outcome(
        signal_id=sid,
        entry_date=asof + timedelta(days=1),
        entry_price=80.0,
        exit_date=asof + timedelta(days=31),
        exit_price=75.0,
        realised_return_pct=0.0625,  # short profit
        db_path=tmp_db,
    )

    tr = track_record("test_mech", db_path=tmp_db)
    assert tr.n_signals == 1
    assert tr.n_completed == 1
    assert tr.win_rate == pytest.approx(1.0)
    assert tr.mean_return_pct == pytest.approx(0.0625)
    assert tr.best_regime == "risk_off"


def test_consecutive_loss_detection(tmp_db: Path):
    """3 wins, then 5 losses, then 1 win — max streak should be 5 -> is_cold."""
    asof = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for i in range(9):
        sid = log_signal(
            mechanism_id="losing_mech",
            asof=asof + timedelta(days=i * 10),
            asset="X",
            direction="long",
            raw_size_pct=0.01,
            confidence=0.6,
            rationale=f"trade {i}",
            db_path=tmp_db,
        )
        is_win = i < 3 or i == 8  # wins on indices 0,1,2,8; losses on 3-7
        ret = 0.02 if is_win else -0.03
        log_outcome(
            signal_id=sid,
            entry_date=asof + timedelta(days=i * 10 + 1),
            entry_price=100.0,
            exit_date=asof + timedelta(days=i * 10 + 31),
            exit_price=100.0 * (1 + ret),
            realised_return_pct=ret,
            db_path=tmp_db,
        )

    tr = track_record("losing_mech", db_path=tmp_db)
    assert tr.max_consecutive_losses == 5
    assert tr.is_cold  # >= 5 in a row triggers cold flag


def test_signals_for_asset_recent_only(tmp_db: Path):
    now = datetime(2024, 6, 1, tzinfo=timezone.utc)
    log_signal(
        mechanism_id="m1",
        asof=now - timedelta(days=10),
        asset="AAPL",
        direction="long",
        raw_size_pct=0.02,
        confidence=0.7,
        rationale="recent",
        db_path=tmp_db,
    )
    log_signal(
        mechanism_id="m2",
        asof=now - timedelta(days=100),
        asset="AAPL",
        direction="short",
        raw_size_pct=0.02,
        confidence=0.7,
        rationale="old",
        db_path=tmp_db,
    )
    recent = signals_for_asset(
        "AAPL", asof=now, lookback_days=30, db_path=tmp_db
    )
    assert len(recent) == 1
    assert recent[0]["mechanism_id"] == "m1"


# ============================================================
# Coalition
# ============================================================

def _signal(mech: str, asset: str, direction: str, conf: float, size: float) -> Signal:
    return Signal(
        asset=asset,
        direction=direction,
        size_pct=size,
        confidence=conf,
        holding_period_days=30,
        rationale=f"{mech} signal",
        asof=datetime(2024, 5, 1, tzinfo=timezone.utc),
    )


def test_coalition_single_signal(tmp_db: Path):
    sigs = {"m1": [_signal("m1", "MET", "short", 0.7, 0.02)]}
    votes = aggregate_signals(sigs, db_path=tmp_db)
    assert len(votes) == 1
    v = votes[0]
    assert v.asset == "MET"
    assert v.net_direction == "short"
    assert v.is_actionable()


def test_coalition_canceling_signals(tmp_db: Path):
    """Equal-weight long + short on same asset cancels to flat."""
    sigs = {
        "m1": [_signal("m1", "MET", "long", 0.7, 0.02)],
        "m2": [_signal("m2", "MET", "short", 0.7, 0.02)],
    }
    votes = aggregate_signals(sigs, db_path=tmp_db)
    assert len(votes) == 1
    v = votes[0]
    # Net should be near zero
    assert abs(v.total_size_pct) < 1e-6
    assert v.net_direction == "flat"


def test_coalition_diversified_long(tmp_db: Path):
    """Three mechanisms all long the same asset -> bigger size."""
    sigs = {
        "m1": [_signal("m1", "ABBV", "short", 0.6, 0.02)],
        "m2": [_signal("m2", "ABBV", "short", 0.7, 0.02)],
        "m3": [_signal("m3", "ABBV", "short", 0.8, 0.02)],
    }
    votes = aggregate_signals(sigs, db_path=tmp_db)
    assert len(votes) == 1
    v = votes[0]
    assert v.net_direction == "short"
    assert v.total_size_pct == pytest.approx(0.06, abs=1e-9)
    assert len(v.contributing_mechanism_ids) == 3
    assert v.is_actionable()


def test_coalition_separate_assets(tmp_db: Path):
    sigs = {
        "m1": [
            _signal("m1", "AAPL", "long", 0.7, 0.02),
            _signal("m1", "MET", "short", 0.7, 0.02),
        ],
    }
    votes = aggregate_signals(sigs, db_path=tmp_db)
    assert len(votes) == 2
    assets = {v.asset for v in votes}
    assert assets == {"AAPL", "MET"}
