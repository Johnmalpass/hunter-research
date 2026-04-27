"""Tests for regime detector and predicate library."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from quant.data.base import DataPoint, MarketState, write_points
from quant.research.predicates import (
    And,
    Not,
    Or,
    PredicateResult,
    RegimePredicate,
    SpreadPredicate,
    ThresholdPredicate,
    WithinDaysOfPredicate,
    ZScorePredicate,
)
from quant.research.regime import REGIMES, RegimeDetector, RegimeState


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test_rp.db"


def _ts(days_ago: int, asof: datetime) -> datetime:
    return asof - timedelta(days=days_ago)


# ============================================================
# Threshold + Spread + Z-score
# ============================================================

def test_threshold_predicate_fires(tmp_db: Path):
    asof = datetime(2024, 5, 1, tzinfo=timezone.utc)
    write_points(
        [DataPoint(asof - timedelta(days=1), "DGS10", "value", 4.5, "fred", {})],
        tmp_db,
    )
    p = ThresholdPredicate("DGS10", "value", 4.0, ">=")
    with MarketState(tmp_db) as state:
        r = p(state, asof)
    assert r is not None
    assert r.fired
    assert r.magnitude == pytest.approx(0.5)


def test_threshold_predicate_misses(tmp_db: Path):
    asof = datetime(2024, 5, 1, tzinfo=timezone.utc)
    write_points(
        [DataPoint(asof - timedelta(days=1), "DGS10", "value", 3.5, "fred", {})],
        tmp_db,
    )
    p = ThresholdPredicate("DGS10", "value", 4.0, ">=")
    with MarketState(tmp_db) as state:
        r = p(state, asof)
    assert r is not None
    assert not r.fired


def test_spread_predicate_basis_points(tmp_db: Path):
    asof = datetime(2024, 5, 1, tzinfo=timezone.utc)
    write_points(
        [
            DataPoint(asof, "BAMLC0A2CAA", "value", 5.6, "fred", {}),
            DataPoint(asof, "DGS10", "value", 4.2, "fred", {}),
        ],
        tmp_db,
    )
    # 1.4 percentage points = 140 bps; threshold 100 bps
    p = SpreadPredicate(
        "BAMLC0A2CAA", "value", "DGS10", "value",
        threshold=100.0, scale=100.0, op=">=",
    )
    with MarketState(tmp_db) as state:
        r = p(state, asof)
    assert r is not None
    assert r.fired
    assert r.magnitude == pytest.approx(140.0, abs=1e-6)


def test_zscore_fires_on_spike(tmp_db: Path):
    asof = datetime(2024, 6, 1, tzinfo=timezone.utc)
    points = []
    for d in range(120):
        ts = asof - timedelta(days=120 - d)
        v = 10.0 if d < 113 else 50.0
        points.append(DataPoint(ts, "TEST", "value", v, "test", {}))
    write_points(points, tmp_db)

    p = ZScorePredicate("TEST", "value", threshold=2.0, window_days=90)
    with MarketState(tmp_db) as state:
        r = p(state, asof)
    assert r is not None
    assert r.fired
    assert r.magnitude > 2.0


def test_zscore_no_data(tmp_db: Path):
    p = ZScorePredicate("MISSING", "value", threshold=2.0)
    with MarketState(tmp_db) as state:
        r = p(state, datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert r is None  # missing data is None, not False


# ============================================================
# Combinators
# ============================================================

def test_and_short_circuits_on_missing(tmp_db: Path):
    """And returns None if any child has missing data."""
    asof = datetime(2024, 5, 1, tzinfo=timezone.utc)
    write_points(
        [DataPoint(asof, "DGS10", "value", 4.5, "fred", {})], tmp_db,
    )
    p = And(
        ThresholdPredicate("DGS10", "value", 4.0, ">="),
        ThresholdPredicate("MISSING", "value", 1.0, ">="),
    )
    with MarketState(tmp_db) as state:
        r = p(state, asof)
    assert r is None


def test_and_fires_when_all_fire(tmp_db: Path):
    asof = datetime(2024, 5, 1, tzinfo=timezone.utc)
    write_points(
        [
            DataPoint(asof, "A", "value", 5.0, "test", {}),
            DataPoint(asof, "B", "value", 3.0, "test", {}),
        ],
        tmp_db,
    )
    p = And(
        ThresholdPredicate("A", "value", 4.0, ">="),
        ThresholdPredicate("B", "value", 2.0, ">="),
    )
    with MarketState(tmp_db) as state:
        r = p(state, asof)
    assert r is not None
    assert r.fired


def test_or_fires_if_any(tmp_db: Path):
    asof = datetime(2024, 5, 1, tzinfo=timezone.utc)
    write_points(
        [
            DataPoint(asof, "A", "value", 5.0, "test", {}),
            DataPoint(asof, "B", "value", 0.5, "test", {}),
        ],
        tmp_db,
    )
    p = Or(
        ThresholdPredicate("A", "value", 4.0, ">="),
        ThresholdPredicate("B", "value", 1.0, ">="),
    )
    with MarketState(tmp_db) as state:
        r = p(state, asof)
    assert r is not None
    assert r.fired


def test_not_inverts(tmp_db: Path):
    asof = datetime(2024, 5, 1, tzinfo=timezone.utc)
    write_points(
        [DataPoint(asof, "A", "value", 5.0, "test", {})],
        tmp_db,
    )
    p = Not(ThresholdPredicate("A", "value", 4.0, ">="))
    with MarketState(tmp_db) as state:
        r = p(state, asof)
    assert r is not None
    assert not r.fired


# ============================================================
# Calendar
# ============================================================

def test_within_days_of_filing():
    asof = datetime(2024, 5, 10, tzinfo=timezone.utc)
    p = WithinDaysOfPredicate(calendar_days=[(5, 15)], max_days=30)
    r = p(state=None, asof=asof)
    assert r is not None
    assert r.fired
    assert r.evidence["days_to_next"] == 5


def test_within_days_of_far_future_misses():
    asof = datetime(2024, 1, 10, tzinfo=timezone.utc)
    p = WithinDaysOfPredicate(calendar_days=[(11, 14)], max_days=30)
    r = p(state=None, asof=asof)
    assert r is not None
    assert not r.fired
    assert r.evidence["days_to_next"] > 30


# ============================================================
# Regime detector
# ============================================================

def _build_macro_history(tmp_db: Path, asof: datetime, scenario: str):
    """Synthesise enough FRED-shaped history for the regime detector."""
    points: list[DataPoint] = []
    base = asof - timedelta(days=400)
    for d in range(400):
        ts = base + timedelta(days=d)
        # Vary inputs so percentile ranks are well-defined
        if scenario == "risk_on":
            ten = 4.5 + 0.0001 * d
            two = 3.0
            hy = 3.0 + 0.001 * d
        elif scenario == "crisis":
            # Today is much higher HY spread than history; today's curve inverted
            ten = 4.0 - 0.001 * d
            two = 3.5
            hy = 3.5 + 0.001 * d
        else:
            ten = 4.0
            two = 3.5
            hy = 4.5
        points.append(DataPoint(ts, "DGS10", "value", ten, "fred", {}))
        points.append(DataPoint(ts, "DGS2", "value", two, "fred", {}))
        points.append(DataPoint(ts, "BAA10Y", "value", hy, "fred", {}))

    if scenario == "crisis":
        # Override today: yield curve inverted hard, HY spread blown out
        points.append(DataPoint(asof, "DGS10", "value", 3.5, "fred", {}))
        points.append(DataPoint(asof, "DGS2", "value", 4.5, "fred", {}))
        points.append(DataPoint(asof, "BAA10Y", "value", 12.0, "fred", {}))

    write_points(points, tmp_db)


def test_regime_detect_risk_on(tmp_db: Path):
    asof = datetime(2024, 5, 1, tzinfo=timezone.utc)
    _build_macro_history(tmp_db, asof, "risk_on")
    with MarketState(tmp_db) as state:
        det = RegimeDetector(state)
        rs = det.detect(asof)
    assert rs is not None
    # Probabilities sum to 1
    assert sum(rs.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    # All four regimes present
    assert set(rs.probabilities) == set(REGIMES)


def test_regime_detect_crisis(tmp_db: Path):
    asof = datetime(2024, 5, 1, tzinfo=timezone.utc)
    _build_macro_history(tmp_db, asof, "crisis")
    with MarketState(tmp_db) as state:
        det = RegimeDetector(state)
        rs = det.detect(asof)
    assert rs is not None
    # Crisis-shaped state should land in risk_off or crisis, not risk_on
    assert rs.regime in ("risk_off", "crisis")
    assert rs.is_risk_off_or_crisis


def test_regime_detect_no_data(tmp_db: Path):
    asof = datetime(2024, 5, 1, tzinfo=timezone.utc)
    with MarketState(tmp_db) as state:
        det = RegimeDetector(state)
        rs = det.detect(asof)
    assert rs is None  # graceful, not crash


def test_regime_predicate_uses_detector(tmp_db: Path):
    asof = datetime(2024, 5, 1, tzinfo=timezone.utc)
    _build_macro_history(tmp_db, asof, "crisis")
    p = RegimePredicate(allowed_regimes=["risk_off", "crisis"], min_probability=0.4)
    with MarketState(tmp_db) as state:
        r = p(state, asof)
    assert r is not None
    assert r.fired
    assert r.evidence["regime"] in ("risk_off", "crisis")
