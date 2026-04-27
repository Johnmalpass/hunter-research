"""Tests for quant.data — schema, write/read round-trip, MarketState queries."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from quant.data.base import DataPoint, MarketState, get_connection, write_points


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test_quant.db"


def _ts(day: int) -> datetime:
    return datetime(2026, 4, 1, tzinfo=timezone.utc) + timedelta(days=day)


def test_schema_creates(tmp_db: Path) -> None:
    conn = get_connection(tmp_db)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        names = {r[0] for r in rows}
        assert "data_points" in names
        assert "adapter_runs" in names
        assert "asset_aliases" in names
    finally:
        conn.close()


def test_round_trip(tmp_db: Path) -> None:
    points = [
        DataPoint(_ts(0), "AAPL", "price_close", 175.0, "polygon", {"adj": True}),
        DataPoint(_ts(1), "AAPL", "price_close", 176.5, "polygon", {"adj": True}),
        DataPoint(_ts(0), "DGS10", "value", 4.25, "fred", {"label": "10y"}),
    ]
    n = write_points(points, tmp_db)
    assert n == 3

    with MarketState(tmp_db) as state:
        latest = state.latest("AAPL", "price_close")
        assert latest is not None
        assert latest.value == pytest.approx(176.5)
        assert latest.source == "polygon"

        hist = state.history("AAPL", "price_close", _ts(-1), _ts(2))
        assert [p.value for p in hist] == pytest.approx([175.0, 176.5])

        assert set(state.assets()) == {"AAPL", "DGS10"}
        assert set(state.fields("AAPL")) == {"price_close"}


def test_idempotent_overwrite(tmp_db: Path) -> None:
    p1 = DataPoint(_ts(0), "MET", "price_close", 80.0, "polygon", {})
    p2 = DataPoint(_ts(0), "MET", "price_close", 81.5, "polygon", {})  # same key, new value
    write_points([p1], tmp_db)
    write_points([p2], tmp_db)
    with MarketState(tmp_db) as state:
        latest = state.latest("MET", "price_close")
        assert latest.value == pytest.approx(81.5)


def test_two_sources_no_collision(tmp_db: Path) -> None:
    """Same asset+field+ts from two sources should both persist (PK includes source)."""
    p_polygon = DataPoint(_ts(0), "AAPL", "price_close", 175.0, "polygon", {})
    p_yahoo = DataPoint(_ts(0), "AAPL", "price_close", 175.05, "yahoo", {})
    write_points([p_polygon, p_yahoo], tmp_db)
    with MarketState(tmp_db) as state:
        hist = state.history("AAPL", "price_close", _ts(-1), _ts(1))
        assert len(hist) == 2
        sources = {p.source for p in hist}
        assert sources == {"polygon", "yahoo"}
