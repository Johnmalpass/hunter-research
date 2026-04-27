"""Articulation Lead Time — does HUNTER articulate compositions before the market?

The hypothesis being tested
===========================

If HUNTER's edge is *articulation* (Theory B from the conversation), the time
between when HUNTER posts a thesis and when the market itself articulates it
publicly should correlate with realised alpha. Longer lead time = more alpha.

If HUNTER's edge is *finding* (Theory A), the synergy bits per thesis should
correlate but lead time should not — because the market would never
independently articulate the same composition.

This module measures lead time and lets us test the hypothesis.

How "market articulation" is detected
=====================================

We use GDELT article-count time series for keywords from the thesis. The
market is "articulated" when:

    daily_article_count(thesis_keywords) >= mean_article_count + Z * stdev

with Z=2 by default. Once that fires, we record the date and compute the
lead time as (articulation_date - thesis_posted_date).

If the keywords never spike, we cap the lead time at the observation horizon
and flag the thesis as "not yet articulated" (which is excellent news for HUNTER —
the market still hasn't caught up).

Inputs and integration
======================

This module reads from the existing GDELT adapter in `quant.data.adapters.gdelt`
and from the `mechanism_signals` ledger. To use:

  1. Ingest GDELT for thesis keywords:
     `python -m quant ingest gdelt --queries "cmbs office delinquency,life insurer reserves"`
  2. Run articulation analysis:
     `compute_articulation_lead("thesis_328", thesis_keywords=["cmbs office delinquency"])`

The result is an ArticulationRecord that we can then correlate against
realised alpha when summer fills the ledger.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from quant.data.base import DEFAULT_DB


def _query_to_asset_id(query: str) -> str:
    """Mirror of GdeltAdapter's sanitiser so we look up the same asset_id."""
    return re.sub(r"[^a-z0-9]+", "_", query.lower()).strip("_") or "unknown"


@dataclass
class ArticulationRecord:
    thesis_id: str
    keywords_searched: list[str]
    hunter_articulated_at: datetime
    market_articulated_at: Optional[datetime]
    lead_time_days: Optional[int]
    n_articles_at_articulation: Optional[int]
    baseline_mean: float
    baseline_std: float
    z_threshold: float
    interpretation: str

    def to_dict(self) -> dict:
        return {
            "thesis_id": self.thesis_id,
            "keywords": self.keywords_searched,
            "hunter_articulated_at": self.hunter_articulated_at.isoformat(),
            "market_articulated_at": (
                None if self.market_articulated_at is None
                else self.market_articulated_at.isoformat()
            ),
            "lead_time_days": self.lead_time_days,
            "n_articles_at_articulation": self.n_articles_at_articulation,
            "baseline_mean": round(self.baseline_mean, 2),
            "baseline_std": round(self.baseline_std, 2),
            "z_threshold": self.z_threshold,
            "interpretation": self.interpretation,
        }


def compute_articulation_lead(
    thesis_id: str,
    *,
    thesis_keywords: list[str],
    hunter_articulated_at: Optional[datetime] = None,
    z_threshold: float = 2.0,
    baseline_window_days: int = 90,
    detection_window_days: int = 365,
    db_path: Optional[Path | str] = None,
) -> ArticulationRecord:
    """For one thesis, compute when (if ever) the market articulated it.

    `thesis_keywords` should be the set of GDELT queries you used during
    ingestion. The asset_ids in the data store are the sanitised forms of
    those queries.
    """
    if not thesis_keywords:
        raise ValueError("thesis_keywords must be non-empty")

    if hunter_articulated_at is None:
        hunter_articulated_at = datetime.now(timezone.utc)
    elif hunter_articulated_at.tzinfo is None:
        hunter_articulated_at = hunter_articulated_at.replace(tzinfo=timezone.utc)

    asset_ids = [_query_to_asset_id(k) for k in thesis_keywords]
    floor = hunter_articulated_at - timedelta(days=baseline_window_days)
    end = hunter_articulated_at + timedelta(days=detection_window_days)

    path = Path(db_path) if db_path else DEFAULT_DB
    if not path.exists():
        return ArticulationRecord(
            thesis_id=thesis_id,
            keywords_searched=thesis_keywords,
            hunter_articulated_at=hunter_articulated_at,
            market_articulated_at=None,
            lead_time_days=None,
            n_articles_at_articulation=None,
            baseline_mean=0.0,
            baseline_std=0.0,
            z_threshold=z_threshold,
            interpretation="no quant_data.db; ingest GDELT first",
        )

    conn = sqlite3.connect(str(path))
    try:
        # Daily aggregated article counts across all keywords' asset_ids
        rows = conn.execute(
            f"""
            SELECT date(ts) AS day, SUM(value_numeric) AS n
            FROM data_points
            WHERE asset_id IN ({','.join('?' * len(asset_ids))})
              AND field = 'articles_count_1d'
              AND ts >= ? AND ts <= ?
            GROUP BY day
            ORDER BY day
            """,
            asset_ids + [floor.isoformat(), end.isoformat()],
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return ArticulationRecord(
            thesis_id=thesis_id,
            keywords_searched=thesis_keywords,
            hunter_articulated_at=hunter_articulated_at,
            market_articulated_at=None,
            lead_time_days=None,
            n_articles_at_articulation=None,
            baseline_mean=0.0,
            baseline_std=0.0,
            z_threshold=z_threshold,
            interpretation="no GDELT data for these keywords; ingest first",
        )

    # Split into baseline (pre-HUNTER) and detection (post-HUNTER) windows
    baseline = [n for d, n in rows if d <= hunter_articulated_at.date().isoformat()]
    detection = [(d, n) for d, n in rows if d > hunter_articulated_at.date().isoformat()]

    if len(baseline) < 10:
        return ArticulationRecord(
            thesis_id=thesis_id,
            keywords_searched=thesis_keywords,
            hunter_articulated_at=hunter_articulated_at,
            market_articulated_at=None,
            lead_time_days=None,
            n_articles_at_articulation=None,
            baseline_mean=0.0,
            baseline_std=0.0,
            z_threshold=z_threshold,
            interpretation=(
                f"insufficient baseline ({len(baseline)} days); need >= 10 "
                "before hunter_articulated_at"
            ),
        )

    base_arr = np.asarray(baseline, dtype=float)
    base_mean = float(base_arr.mean())
    base_std = float(base_arr.std()) or 1.0
    threshold = base_mean + z_threshold * base_std

    market_articulated_at: Optional[datetime] = None
    n_at_artic: Optional[int] = None
    for d_str, n in detection:
        if float(n) >= threshold:
            market_articulated_at = datetime.fromisoformat(d_str + "T00:00:00").replace(
                tzinfo=timezone.utc
            )
            n_at_artic = int(n)
            break

    if market_articulated_at is None:
        lead_days = None
        interp = (
            f"market has not yet articulated (over {len(detection)} days post-HUNTER); "
            "alpha potentially intact"
        )
    else:
        lead_days = (market_articulated_at - hunter_articulated_at).days
        if lead_days >= 30:
            interp = (
                f"strong articulation lead: HUNTER articulated {lead_days} days "
                "before market; expect substantial alpha if Theory B holds"
            )
        elif lead_days >= 7:
            interp = (
                f"moderate lead ({lead_days} days); meaningful alpha plausible"
            )
        elif lead_days >= 0:
            interp = (
                f"short lead ({lead_days} days); near-simultaneous articulation"
            )
        else:
            interp = (
                f"market articulated FIRST ({-lead_days} days before HUNTER); "
                "Theory B suggests no alpha here"
            )

    return ArticulationRecord(
        thesis_id=thesis_id,
        keywords_searched=thesis_keywords,
        hunter_articulated_at=hunter_articulated_at,
        market_articulated_at=market_articulated_at,
        lead_time_days=lead_days,
        n_articles_at_articulation=n_at_artic,
        baseline_mean=base_mean,
        baseline_std=base_std,
        z_threshold=z_threshold,
        interpretation=interp,
    )
