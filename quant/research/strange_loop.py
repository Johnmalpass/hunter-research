"""Strange Loop — HUNTER explicitly models its own influence on the consensus.

The deeper philosophical claim (Observer-Generated Reality):

  HUNTER's articulations do not just *predict* price changes; they
  partially *cause* them. The market price is co-constituted by HUNTER's
  observations. HUNTER is not a passive predictor but an active observer-
  actor in the market's self-construction. This is hyperstition (Nick Land
  1996, Reza Negarestani) operationalised for finance — Hofstadter's strange
  loop applied to markets.

What this module computes
=========================

Given a candidate thesis T that HUNTER might publish, compute:

    baseline_alpha(T)         what alpha would be IF the market never
                              learned about HUNTER's view (counterfactual)
    articulation_impact(T)    estimated movement in consensus caused by
                              HUNTER's own publication (the strange-loop
                              feedback term)
    net_alpha(T)              baseline_alpha - articulation_impact
                              (what HUNTER actually captures after its own
                              publication accelerates absorption)

Trader implications
===================

1. **Trade FIRST, publish later.** When net_alpha < baseline_alpha, the
   alpha is partially destroyed by your own publication. Build the position
   before publishing; let the publication move the consensus AFTER you're in.

2. **Publish without trading.** When you have an inquiry-flagged or
   already-vetoed signal, publishing for ARTICULATION VALUE (audience
   reach, brand, pre-registration timestamp) without trading captures the
   downstream consensus-impact for the brand without the trade decay.

3. **Optimise for hyperstitional impact.** When choosing which of N
   candidate theses to publish, pick the one that MOST changes consensus
   if your goal is influence. Pick the one that LEAST changes consensus
   if your goal is alpha. The trader picks per-thesis depending on
   business objective.

4. **Time the publication.** A thesis published into a high-attention
   regime has more articulation impact than one published into noise. The
   regime detector + GDELT theme density determines publication timing.

Estimating articulation impact in practice
==========================================

We use historical HUNTER articulations as training signal:

  1. Pull HUNTER's past prediction-board posts with timestamps.
  2. For each, query GDELT for the topic in the 30 days BEFORE and AFTER
     publication.
  3. Compute the delta in article counts attributable to HUNTER's publication
     (above what would be expected from prior trend).
  4. Train a small regression: articulation_impact = f(thesis_features).
  5. Apply to new theses.

Until we have summer data, this module operates with conservative defaults
and dry-run output. The summer study generates the calibration data.

This is, as far as we have searched, the first explicit operationalisation
of hyperstitional feedback in financial markets. The conceptual seed is in
the CCRU writing of the late 1990s; nobody has built the engine.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class StrangeLoopAssessment:
    """Strange-loop self-assessment of one candidate thesis."""

    thesis_id: str
    asof: datetime
    baseline_alpha_estimate: float
    articulation_impact_estimate: float
    net_alpha_estimate: float
    confidence: float  # how much we trust the estimate (0-1)
    n_similar_past_articulations: int
    recommendation: str  # "publish_and_trade" | "trade_first_then_publish" | "publish_only" | "trade_only" | "hold"
    rationale: str

    def to_dict(self) -> dict:
        return {
            "thesis_id": self.thesis_id,
            "asof": self.asof.isoformat(),
            "baseline_alpha_estimate": round(self.baseline_alpha_estimate, 4),
            "articulation_impact_estimate": round(self.articulation_impact_estimate, 4),
            "net_alpha_estimate": round(self.net_alpha_estimate, 4),
            "confidence": round(self.confidence, 3),
            "n_similar_past_articulations": self.n_similar_past_articulations,
            "recommendation": self.recommendation,
            "rationale": self.rationale,
        }


def _query_gdelt_articles(
    asset_id: str,
    start: datetime,
    end: datetime,
    db_path: Optional[Path | str],
) -> list[tuple[datetime, int]]:
    """Pull GDELT article-count rows for a topic. Returns list of (ts, count)."""
    from quant.data.base import DEFAULT_DB

    path = Path(db_path) if db_path else DEFAULT_DB
    if not path.exists():
        return []
    conn = sqlite3.connect(str(path))
    try:
        try:
            rows = conn.execute(
                "SELECT ts, value_numeric FROM data_points "
                "WHERE asset_id = ? AND field = 'articles_count_1d' "
                "  AND ts >= ? AND ts <= ? "
                "ORDER BY ts",
                (asset_id, start.isoformat(), end.isoformat()),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    finally:
        conn.close()
    out: list[tuple[datetime, int]] = []
    for ts_str, n in rows:
        try:
            out.append(
                (datetime.fromisoformat(ts_str.replace("Z", "+00:00")), int(n or 0))
            )
        except (ValueError, TypeError):
            continue
    return out


def estimate_articulation_impact_from_history(
    topic_asset_id: str,
    publish_date: datetime,
    *,
    pre_window_days: int = 30,
    post_window_days: int = 30,
    db_path: Optional[Path | str] = None,
) -> tuple[float, float]:
    """For an already-published topic, estimate the consensus drift attributable
    to publication.

    Returns (estimated_impact, confidence).

    Method: compare mean article count in the [-pre_window, 0] window to mean
    in the [0, +post_window] window. The ratio is the proxy for "how much did
    consensus shift after publication." This is a noisy proxy but bounded and
    interpretable.
    """
    if publish_date.tzinfo is None:
        publish_date = publish_date.replace(tzinfo=timezone.utc)
    pre_start = publish_date - timedelta(days=pre_window_days)
    post_end = publish_date + timedelta(days=post_window_days)
    rows = _query_gdelt_articles(topic_asset_id, pre_start, post_end, db_path)
    if len(rows) < 10:
        return 0.0, 0.0  # not enough data

    pre = [n for ts, n in rows if ts <= publish_date]
    post = [n for ts, n in rows if ts > publish_date]
    if len(pre) < 5 or len(post) < 5:
        return 0.0, 0.0

    mean_pre = float(np.mean(pre))
    mean_post = float(np.mean(post))
    if mean_pre <= 0:
        return 0.0, 0.0

    # Relative impact: how much higher is post than pre?
    impact = (mean_post - mean_pre) / max(mean_pre, 1.0)

    # Confidence based on sample sizes and signal-to-noise
    pre_std = float(np.std(pre)) or 1.0
    z = (mean_post - mean_pre) / pre_std
    confidence = float(max(0.0, min(1.0, abs(z) / 3.0)))

    return impact, confidence


def assess_strange_loop(
    *,
    thesis_id: str,
    candidate_topic_asset_id: str,
    baseline_alpha_estimate: float,
    asof: Optional[datetime] = None,
    publication_objective: str = "alpha",  # "alpha" | "influence" | "both"
    db_path: Optional[Path | str] = None,
) -> StrangeLoopAssessment:
    """Estimate net alpha after self-publication impact + recommend action.

    `baseline_alpha_estimate` is the expected alpha assuming HUNTER doesn't
    publish. We estimate the publication impact from the historical
    distribution of similar past articulations (via GDELT spike analysis).
    """
    asof = asof or datetime.now(timezone.utc)

    # Use the existing GDELT data on this topic to estimate how much a
    # publication-style spike adds to consensus drift.
    rows = _query_gdelt_articles(
        candidate_topic_asset_id,
        asof - timedelta(days=180),
        asof,
        db_path,
    )
    if len(rows) < 30:
        # Not enough history; conservative default
        articulation_impact = 0.10 * baseline_alpha_estimate
        confidence = 0.2
        n_similar = 0
    else:
        # Find historical local maxima as proxies for "publication-driven spikes"
        counts = np.array([n for _, n in rows])
        baseline_count = float(np.median(counts))
        threshold = baseline_count + 2.0 * float(np.std(counts))
        n_spikes = int((counts >= threshold).sum())
        n_similar = n_spikes
        # Map spike frequency to articulation impact: more spikes -> the
        # topic responds to articulation more strongly -> larger impact
        spike_density = n_spikes / max(1, len(counts))
        articulation_impact = (
            baseline_alpha_estimate * float(min(0.6, 0.05 + spike_density * 5.0))
        )
        confidence = float(min(0.9, 0.3 + n_spikes / 30.0))

    net_alpha = baseline_alpha_estimate - articulation_impact

    # Recommendation logic
    if articulation_impact / max(abs(baseline_alpha_estimate), 1e-9) > 0.4:
        # Publication will eat 40%+ of the alpha; trade first
        if publication_objective == "alpha":
            rec = "trade_first_then_publish"
            rationale = (
                f"articulation impact {articulation_impact:.4f} is a large "
                f"fraction of baseline alpha {baseline_alpha_estimate:.4f}; "
                "build position before publishing"
            )
        elif publication_objective == "influence":
            rec = "publish_only"
            rationale = (
                "high consensus-impact thesis; publish for influence even "
                "though alpha is largely consumed by the publication"
            )
        else:
            rec = "trade_first_then_publish"
            rationale = "default: capture alpha before publication absorbs it"
    elif net_alpha > 0.005:
        rec = "publish_and_trade"
        rationale = (
            f"net alpha {net_alpha:.4f} after publication impact "
            f"{articulation_impact:.4f}; safe to publish and trade together"
        )
    elif baseline_alpha_estimate > 0.005:
        rec = "publish_and_trade"
        rationale = "modest baseline alpha; publication impact is small"
    else:
        rec = "hold"
        rationale = "baseline alpha too small to act on"

    return StrangeLoopAssessment(
        thesis_id=thesis_id,
        asof=asof,
        baseline_alpha_estimate=baseline_alpha_estimate,
        articulation_impact_estimate=articulation_impact,
        net_alpha_estimate=net_alpha,
        confidence=confidence,
        n_similar_past_articulations=n_similar,
        recommendation=rec,
        rationale=rationale,
    )
