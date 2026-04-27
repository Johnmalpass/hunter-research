"""Tests for the Mechanism Compiler framework."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from quant.data.base import DataPoint, MarketState, write_points
from quant.research.mechanism import (
    Mechanism,
    MechanismRequirement,
    Signal,
    _import_all_mechanisms,
    get_mechanism,
    list_mechanisms,
    register,
)


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test_mech.db"


def test_imports_register_mechanisms():
    _import_all_mechanisms()
    names = list_mechanisms()
    assert "thesis_328" in names
    assert "pharma_adverse_spike" in names


def test_thesis_328_metadata():
    _import_all_mechanisms()
    cls = get_mechanism("thesis_328")
    m = cls()
    assert m.thesis_id == "thesis_328"
    assert "MET" in m.universe
    assert m.direction == "short"
    # All FRED requirements should declare the fred adapter
    fred_reqs = [r for r in m.requirements if "DGS10" in r.asset_id or "BAML" in r.asset_id or "DRCRE" in r.asset_id]
    assert len(fred_reqs) >= 3
    assert all(r.suggested_adapter == "fred" for r in fred_reqs)


def test_thesis_328_returns_empty_when_data_missing(tmp_db: Path):
    """Mechanism must not crash when state is empty; should return []."""
    _import_all_mechanisms()
    m = get_mechanism("thesis_328")()
    with MarketState(tmp_db) as state:
        signals = m.evaluate(state, datetime(2024, 5, 1, tzinfo=timezone.utc))
    assert signals == []


def test_thesis_328_triggers_when_thresholds_cross(tmp_db: Path):
    """Populate FRED-shaped data above thresholds and confirm a basket signal fires."""
    _import_all_mechanisms()
    m = get_mechanism("thesis_328")()
    asof = datetime(2024, 5, 10, tzinfo=timezone.utc)  # within filing window of May 15
    points = [
        DataPoint(asof - timedelta(days=10), "DRCRELACBS", "value", 8.5, "fred", {}),
        DataPoint(asof - timedelta(days=2), "BAMLC0A2CAA", "value", 5.6, "fred", {}),
        DataPoint(asof - timedelta(days=2), "DGS10", "value", 4.2, "fred", {}),
    ]
    write_points(points, tmp_db)
    with MarketState(tmp_db) as state:
        signals = m.evaluate(state, asof)
    assert len(signals) == 4  # MET, PRU, LNC, AFL
    assert all(s.direction == "short" for s in signals)
    assert {s.asset for s in signals} == {"MET", "PRU", "LNC", "AFL"}


def test_thesis_328_no_signal_when_spread_too_tight(tmp_db: Path):
    _import_all_mechanisms()
    m = get_mechanism("thesis_328")()
    asof = datetime(2024, 5, 10, tzinfo=timezone.utc)
    points = [
        DataPoint(asof - timedelta(days=10), "DRCRELACBS", "value", 8.5, "fred", {}),
        DataPoint(asof - timedelta(days=2), "BAMLC0A2CAA", "value", 4.5, "fred", {}),
        DataPoint(asof - timedelta(days=2), "DGS10", "value", 4.2, "fred", {}),  # spread = 30bps < 100bps
    ]
    write_points(points, tmp_db)
    with MarketState(tmp_db) as state:
        signals = m.evaluate(state, asof)
    assert signals == []


def test_pharma_spike_triggers_on_z_score(tmp_db: Path):
    _import_all_mechanisms()
    cls = get_mechanism("pharma_adverse_spike")
    m = cls(drug="TESTDRUG", manufacturer_ticker="TEST")
    asof = datetime(2024, 6, 1, tzinfo=timezone.utc)

    # 113 days at baseline (count=10), then 7 days elevated (count=50).
    # The 90-day rolling baseline therefore stays mostly at 10, today's z is
    # large, and the 14-day lookback contains 7 elevated days (>= min 5).
    points = []
    base_date = asof - timedelta(days=120)
    for d in range(120):
        ts = base_date + timedelta(days=d)
        count = 10 if d < 113 else 50
        points.append(
            DataPoint(ts, "TESTDRUG", "faers_reports_count_1d", count, "faers", {})
        )
    write_points(points, tmp_db)

    with MarketState(tmp_db) as state:
        signals = m.evaluate(state, asof)
    assert len(signals) == 1
    assert signals[0].direction == "short"
    assert signals[0].asset == "TEST"
    assert signals[0].metadata["z_score"] > 2.0


def test_pharma_spike_no_signal_when_flat(tmp_db: Path):
    _import_all_mechanisms()
    cls = get_mechanism("pharma_adverse_spike")
    m = cls(drug="FLATDRUG", manufacturer_ticker="FLAT")
    asof = datetime(2024, 6, 1, tzinfo=timezone.utc)

    points = []
    base_date = asof - timedelta(days=120)
    for d in range(120):
        ts = base_date + timedelta(days=d)
        points.append(
            DataPoint(ts, "FLATDRUG", "faers_reports_count_1d", 10, "faers", {})
        )
    write_points(points, tmp_db)

    with MarketState(tmp_db) as state:
        signals = m.evaluate(state, asof)
    assert signals == []


def test_data_check_reports_counts(tmp_db: Path):
    _import_all_mechanisms()
    m = get_mechanism("thesis_328")()
    asof = datetime(2024, 5, 10, tzinfo=timezone.utc)
    points = [
        DataPoint(asof, "DRCRELACBS", "value", 8.5, "fred", {}),
        DataPoint(asof, "DGS10", "value", 4.2, "fred", {}),
    ]
    write_points(points, tmp_db)
    with MarketState(tmp_db) as state:
        check = m.check_data(state)
    assert check["DRCRELACBS/value"] == 1
    assert check["DGS10/value"] == 1
    assert check["BAMLC0A2CAA/value"] == 0


def test_signal_dataclass_serialises():
    s = Signal(
        asset="MET",
        direction="short",
        size_pct=0.02,
        confidence=0.7,
        holding_period_days=120,
        rationale="test",
        asof=datetime(2024, 5, 1, tzinfo=timezone.utc),
    )
    assert s.asset == "MET"
    assert s.size_pct == 0.02


def test_register_decorator_idempotent():
    """Calling _import_all_mechanisms twice should not duplicate registry."""
    _import_all_mechanisms()
    n1 = len(list_mechanisms())
    _import_all_mechanisms()
    n2 = len(list_mechanisms())
    assert n1 == n2
