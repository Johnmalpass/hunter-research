"""Tests for seam_network + strange_loop modules."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from quant.data.base import DataPoint, write_points
from quant.research.seam_network import (
    add_seam,
    export_atlas_dump,
    find_seams,
    get_seam,
    graph_stats,
    log_seam_use,
)
from quant.research.strange_loop import (
    assess_strange_loop,
    estimate_articulation_impact_from_history,
)


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test_ssl.db"


# ============================================================
# Seam network
# ============================================================

def test_add_seam_creates_record(tmp_db: Path):
    sid = add_seam(
        source_silo="patents",
        source_phrase="bismuth substitute for silver in PV cells",
        target_silo="commodities",
        target_phrase="silver inventory at COMEX vaults",
        underlying_reality_label="solar manufacturing silver demand",
        db_path=tmp_db,
    )
    assert sid > 0
    seam = get_seam(sid, db_path=tmp_db)
    assert seam is not None
    assert seam.source_silo == "patents"
    assert seam.target_silo == "commodities"
    assert seam.n_uses == 1


def test_add_seam_increments_existing(tmp_db: Path):
    """Adding the same seam twice increments n_uses, not creates a duplicate."""
    sid1 = add_seam(
        source_silo="cmbs", source_phrase="office delinquency 12%",
        target_silo="insurance", target_phrase="AA reserve assumption stale",
        db_path=tmp_db,
    )
    sid2 = add_seam(
        source_silo="cmbs", source_phrase="office delinquency 12%",
        target_silo="insurance", target_phrase="AA reserve assumption stale",
        db_path=tmp_db,
    )
    assert sid1 == sid2  # same id
    seam = get_seam(sid1, db_path=tmp_db)
    assert seam.n_uses == 2


def test_log_seam_use_increments_alpha(tmp_db: Path):
    sid = add_seam(
        source_silo="a", source_phrase="x",
        target_silo="b", target_phrase="y",
        db_path=tmp_db,
    )
    log_seam_use(sid, mechanism_id="m1", realised_alpha=0.05, db_path=tmp_db)
    log_seam_use(sid, mechanism_id="m1", realised_alpha=0.03, db_path=tmp_db)
    log_seam_use(sid, mechanism_id="m1", realised_alpha=-0.01, db_path=tmp_db)
    seam = get_seam(sid, db_path=tmp_db)
    assert seam.alpha_generated_total == pytest.approx(0.07, abs=1e-9)


def test_find_seams_filters(tmp_db: Path):
    add_seam(source_silo="a", source_phrase="1", target_silo="b", target_phrase="2", db_path=tmp_db)
    add_seam(source_silo="a", source_phrase="3", target_silo="c", target_phrase="4", db_path=tmp_db)
    add_seam(source_silo="b", source_phrase="5", target_silo="c", target_phrase="6", db_path=tmp_db)

    a_b = find_seams(source_silo="a", target_silo="b", db_path=tmp_db)
    assert len(a_b) == 1
    assert a_b[0].target_silo == "b"

    from_a = find_seams(source_silo="a", db_path=tmp_db)
    assert len(from_a) == 2


def test_graph_stats(tmp_db: Path):
    for src, tgt in [("a", "b"), ("b", "c"), ("c", "a"), ("a", "b")]:
        sid = add_seam(
            source_silo=src, source_phrase=f"{src}_text",
            target_silo=tgt, target_phrase=f"{tgt}_text",
            db_path=tmp_db,
        )
        log_seam_use(sid, realised_alpha=0.01, db_path=tmp_db)

    stats = graph_stats(db_path=tmp_db)
    # 3 unique seams (the second a->b just increments)
    assert stats["n_seams"] == 3
    assert stats["n_silos_touched"] == 3
    assert stats["total_alpha_via_seams"] >= 0.04 - 1e-9
    assert len(stats["top_seams"]) == 3


def test_export_atlas_dump(tmp_db: Path, tmp_path: Path):
    add_seam(
        source_silo="a", source_phrase="x", target_silo="b", target_phrase="y",
        underlying_reality_label="test reality",
        regime_conditioning={"risk_on": True},
        db_path=tmp_db,
    )
    out_path = tmp_path / "atlas.json"
    result = export_atlas_dump(out_path, db_path=tmp_db)
    assert result["n_seams_exported"] == 1
    assert out_path.exists()
    import json
    payload = json.loads(out_path.read_text())
    assert payload["schema_version"] == "atlas.v1"
    assert len(payload["seams"]) == 1
    assert payload["seams"][0]["underlying_reality"] == "test reality"


# ============================================================
# Strange loop
# ============================================================

def test_articulation_impact_returns_zero_with_no_data(tmp_db: Path):
    impact, conf = estimate_articulation_impact_from_history(
        topic_asset_id="nonexistent_topic",
        publish_date=datetime(2024, 5, 1, tzinfo=timezone.utc),
        db_path=tmp_db,
    )
    assert impact == 0.0
    assert conf == 0.0


def test_articulation_impact_detects_post_publication_spike(tmp_db: Path):
    asof = datetime(2024, 5, 1, tzinfo=timezone.utc)
    points = []
    # 30 days of low baseline before publication
    for d in range(30):
        ts = asof - timedelta(days=30 - d)
        points.append(DataPoint(
            ts, "test_topic", "articles_count_1d", 5,
            "gdelt", {"query": "test topic"},
        ))
    # 30 days of elevated articles after publication
    for d in range(30):
        ts = asof + timedelta(days=d)
        points.append(DataPoint(
            ts, "test_topic", "articles_count_1d", 25,
            "gdelt", {"query": "test topic"},
        ))
    write_points(points, tmp_db)

    impact, conf = estimate_articulation_impact_from_history(
        topic_asset_id="test_topic",
        publish_date=asof,
        db_path=tmp_db,
    )
    assert impact > 0  # post is higher than pre
    assert conf > 0.1  # we have meaningful signal


def test_strange_loop_assessment_recommends_action(tmp_db: Path):
    """Assess a candidate thesis and check we get a sensible recommendation."""
    # Seed some GDELT data on the topic
    asof = datetime.now(timezone.utc) - timedelta(days=10)
    points = []
    for d in range(60):
        ts = asof - timedelta(days=60 - d)
        n = 5 + (1 if d % 7 == 0 else 0)  # mostly flat baseline
        points.append(DataPoint(
            ts, "humira_safety", "articles_count_1d", n,
            "gdelt", {"query": "humira safety"},
        ))
    write_points(points, tmp_db)

    result = assess_strange_loop(
        thesis_id="humira_short",
        candidate_topic_asset_id="humira_safety",
        baseline_alpha_estimate=0.05,
        db_path=tmp_db,
    )
    assert result.thesis_id == "humira_short"
    assert result.recommendation in {
        "publish_and_trade", "trade_first_then_publish",
        "publish_only", "trade_only", "hold",
    }
    assert result.net_alpha_estimate <= result.baseline_alpha_estimate  # never increased
    assert 0.0 <= result.confidence <= 1.0


def test_strange_loop_zero_baseline_recommends_hold(tmp_db: Path):
    result = assess_strange_loop(
        thesis_id="weak_thesis",
        candidate_topic_asset_id="any_topic",
        baseline_alpha_estimate=0.001,
        db_path=tmp_db,
    )
    assert result.recommendation in {"hold", "publish_only"}
