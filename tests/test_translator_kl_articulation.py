"""Tests for the audience translator + dialect KL + articulation lead modules."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from quant.data.base import DataPoint, write_points
from quant.research.articulation_lead import compute_articulation_lead
from quant.research.audience_translator import (
    DEFAULT_PROFILES,
    build_translation_prompt,
    translate_for_all_audiences,
    translate_for_audience,
)
from quant.research.dialect_kl import (
    compute_silo_kl_matrix,
    fit_dialect_distribution,
    jensen_shannon,
    kl_gaussian,
)


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test_tka.db"


# ============================================================
# Audience translator
# ============================================================

def test_default_profiles_loaded():
    assert "substack" in DEFAULT_PROFILES
    assert "ssrn" in DEFAULT_PROFILES
    assert "sell_side" in DEFAULT_PROFILES
    assert "treasury" in DEFAULT_PROFILES
    assert "twitter" in DEFAULT_PROFILES


def test_each_profile_has_required_fields():
    for name, p in DEFAULT_PROFILES.items():
        assert p.name == name
        assert p.target_length_words > 0
        assert p.voice
        assert p.structural_template
        assert isinstance(p.expected_terms, list)
        assert isinstance(p.forbidden_terms, list)


def test_build_prompt_includes_thesis_text():
    p = DEFAULT_PROFILES["substack"]
    prompt = build_translation_prompt(
        thesis_text="When CMBS office delinq > 6% and AA spread > 100bps, "
                    "life insurers are under-reserved.",
        profile=p,
    )
    assert "narrative-first-person" in prompt
    assert "1100 words" in prompt
    assert "CMBS office delinq" in prompt


def test_dry_run_returns_prompt_no_api_call():
    result = translate_for_audience(
        thesis_text="Test thesis text.",
        audience="twitter",
        thesis_id="test_x",
        dry_run=True,
    )
    assert result.dry_run
    assert result.cost_usd == 0.0
    assert result.confidence == 0.0
    assert "DRY-RUN" in result.translated_text
    assert "twitter" in result.translated_text  # the prompt mentions the audience


def test_translate_for_all_audiences_dry_run():
    out = translate_for_all_audiences(
        thesis_text="Test thesis.",
        thesis_id="test_x",
        dry_run=True,
    )
    assert set(out) == set(DEFAULT_PROFILES)
    for aud, t in out.items():
        assert t.dry_run
        assert t.audience == aud


def test_unknown_audience_raises():
    with pytest.raises(ValueError):
        translate_for_audience(
            thesis_text="t", audience="invalid_audience_name", dry_run=True,
        )


# ============================================================
# Dialect KL
# ============================================================

def test_kl_gaussian_self_is_zero():
    rng = np.random.default_rng(0)
    mu = rng.standard_normal(8)
    cov = np.eye(8) + 0.1 * rng.standard_normal((8, 8))
    cov = cov @ cov.T  # ensure positive-definite
    kl = kl_gaussian(mu, cov, mu, cov)
    # KL of a distribution from itself should be ~0
    assert abs(kl) < 1e-6


def test_kl_gaussian_positive_for_different():
    rng = np.random.default_rng(0)
    mu_a = np.zeros(8)
    mu_b = np.ones(8) * 2.0
    cov = np.eye(8)
    kl = kl_gaussian(mu_a, cov, mu_b, cov)
    assert kl > 0


def test_jensen_shannon_symmetric_and_zero_for_identical():
    a_statements = ["the cat sat on the mat", "felines on textiles"]
    b_statements = ["the cat sat on the mat", "felines on textiles"]
    a = fit_dialect_distribution("a", a_statements + a_statements)
    b = fit_dialect_distribution("b", b_statements + b_statements)
    js_ab = jensen_shannon(a, b)
    js_ba = jensen_shannon(b, a)
    assert abs(js_ab - js_ba) < 1e-9
    assert js_ab < 0.1  # near-zero for nearly-identical distributions


def test_compute_silo_kl_matrix_shape():
    statements_by_silo = {
        "patents": [
            "method for fabricating semiconductors using novel photoresist",
            "claim 1: a system comprising memory and processor",
            "claim 2: said memory is volatile",
            "method of producing improved chemical yield",
            "an apparatus for solar cell fabrication",
            "process for depositing thin films on substrates",
        ],
        "insurance": [
            "RBC capital ratio computation",
            "actuarial assumption for mortality",
            "policy reserve under SAP",
            "asset-liability matching for life insurer",
            "NAIC Schedule D classification",
            "statutory accounting principle 5R",
        ],
        "regulation": [
            "Federal Register notice on rule-making",
            "comment period for proposed regulation",
            "promulgation under section 553 APA",
            "executive order on federal procurement",
            "OMB circular A-130",
            "Office of Management and Budget review",
        ],
    }
    matrix = compute_silo_kl_matrix(statements_by_silo)
    assert len(matrix.silos) == 3
    assert matrix.asymmetric_kl.shape == (3, 3)
    assert matrix.jensen_shannon.shape == (3, 3)
    # Diagonals are zero
    assert all(matrix.asymmetric_kl[i, i] == 0.0 for i in range(3))
    assert all(matrix.jensen_shannon[i, i] == 0.0 for i in range(3))
    # Symmetry of JS
    for i in range(3):
        for j in range(3):
            assert abs(matrix.jensen_shannon[i, j] - matrix.jensen_shannon[j, i]) < 1e-9


def test_top_pairs_returns_sorted():
    statements_by_silo = {
        "a": ["banana fruit yellow", "apple fruit red", "grape fruit purple"] * 3,
        "b": ["banana fruit yellow", "apple fruit red", "grape fruit purple"] * 3,
        "c": ["matrix factorisation", "eigenvalue decomposition",
              "differential equation"] * 3,
    }
    matrix = compute_silo_kl_matrix(statements_by_silo)
    top = matrix.top_pairs(k=3)
    assert len(top) == 3
    # All triples (a,b,c) -> expect a-c and b-c to have higher JS than a-b
    js_dict = {tuple(sorted([s1, s2])): v for s1, s2, v in top}
    assert js_dict[("a", "c")] > js_dict[("a", "b")]
    assert js_dict[("b", "c")] > js_dict[("a", "b")]


# ============================================================
# Articulation lead
# ============================================================

def test_articulation_returns_no_data_when_db_empty(tmp_db: Path):
    record = compute_articulation_lead(
        "thesis_test",
        thesis_keywords=["cmbs office delinquency"],
        hunter_articulated_at=datetime(2024, 5, 1, tzinfo=timezone.utc),
        db_path=tmp_db,
    )
    # tmp_db doesn't exist yet, so this hits the no-db branch
    assert record.market_articulated_at is None
    assert record.lead_time_days is None


def test_articulation_no_market_articulation_returns_lead_none(tmp_db: Path):
    """Baseline data exists but no spike afterwards -> market hasn't articulated."""
    asof = datetime(2024, 5, 1, tzinfo=timezone.utc)
    points = []
    # 90 days of baseline at count=5
    for d in range(90):
        ts = asof - timedelta(days=90 - d)
        points.append(DataPoint(
            ts, "cmbs_office_delinquency", "articles_count_1d", 5,
            "gdelt", {"query": "cmbs office delinquency"},
        ))
    # 60 days post-HUNTER, still at count=5 (no spike)
    for d in range(60):
        ts = asof + timedelta(days=d)
        points.append(DataPoint(
            ts, "cmbs_office_delinquency", "articles_count_1d", 5,
            "gdelt", {"query": "cmbs office delinquency"},
        ))
    write_points(points, tmp_db)

    record = compute_articulation_lead(
        "thesis_test",
        thesis_keywords=["cmbs office delinquency"],
        hunter_articulated_at=asof,
        db_path=tmp_db,
    )
    assert record.market_articulated_at is None
    assert record.lead_time_days is None
    assert "not yet articulated" in record.interpretation


def test_articulation_spike_detected(tmp_db: Path):
    """Baseline at 5, then a spike to 50 after HUNTER articulates."""
    asof = datetime(2024, 5, 1, tzinfo=timezone.utc)
    points = []
    for d in range(90):
        ts = asof - timedelta(days=90 - d)
        points.append(DataPoint(
            ts, "cmbs_office_delinquency", "articles_count_1d", 5,
            "gdelt", {"query": "cmbs office delinquency"},
        ))
    for d in range(60):
        ts = asof + timedelta(days=d)
        # Day 30 onwards: huge spike
        n = 50 if d >= 30 else 5
        points.append(DataPoint(
            ts, "cmbs_office_delinquency", "articles_count_1d", n,
            "gdelt", {"query": "cmbs office delinquency"},
        ))
    write_points(points, tmp_db)

    record = compute_articulation_lead(
        "thesis_test",
        thesis_keywords=["cmbs office delinquency"],
        hunter_articulated_at=asof,
        db_path=tmp_db,
    )
    assert record.market_articulated_at is not None
    assert record.lead_time_days is not None
    # Lead time should be ~30 days (the spike happens day 30 post-HUNTER)
    assert 25 <= record.lead_time_days <= 35


def test_articulation_insufficient_baseline_returns_clean_message(tmp_db: Path):
    asof = datetime(2024, 5, 1, tzinfo=timezone.utc)
    # Only 3 days of baseline -> too few
    for d in range(3):
        ts = asof - timedelta(days=3 - d)
        write_points([DataPoint(
            ts, "rare_keyword", "articles_count_1d", 5,
            "gdelt", {"query": "rare keyword"},
        )], tmp_db)
    record = compute_articulation_lead(
        "thesis_test",
        thesis_keywords=["rare keyword"],
        hunter_articulated_at=asof,
        db_path=tmp_db,
    )
    assert record.market_articulated_at is None
    assert "insufficient baseline" in record.interpretation
