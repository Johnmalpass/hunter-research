"""Portfolio → Scoring feedback loop.

Problem: the scorer and the portfolio are currently disconnected. The
scorer doesn't know that the last 5 CRE-contamination theses all lost
money, so it keeps scoring the 6th similarly high. This module closes
that loop.

Mechanism: every time a position closes, we record the thesis embedding
+ the realised P&L. Every time a new hypothesis is formed, we look up
the most similar recently-closed theses and apply a calibration delta
to the score (penalty for loser-like theses, small boost for winner-like).

This is a simple, interpretable, auditable form of online learning.
No ML model to train. Just cosine similarity against a rolling window
of closed positions.

Framing: this is "regime-aware scoring" — the system
learns which kinds of theses have been rewarded by the market *recently*
and adjusts confidence accordingly. Not a black box.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from database import get_connection

logger = logging.getLogger("hunter.portfolio_feedback")

# Score adjustment bounds — keep modest so a few bad trades don't
# permanently cap a new regime's upside.
MAX_PENALTY = -12.0
MAX_BOOST = +6.0

# Similarity threshold to count as "related"
RELATED_THRESHOLD = 0.55

# Look-back window for closed positions
LOOKBACK_DAYS = 180

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _embed(text: str):
    if not text or not text.strip():
        return None
    model = _get_model()
    return model.encode(text[:2000], normalize_embeddings=True)


def _ensure_feedback_table():
    conn = get_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scoring_feedback_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hypothesis_id INTEGER,
                raw_score REAL,
                adjustment REAL,
                adjusted_score REAL,
                num_related INTEGER,
                top_related_ticker TEXT,
                top_related_pnl_pct REAL,
                top_related_similarity REAL,
                reason TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.commit()
    finally:
        conn.close()


def _recent_closed_positions(lookback_days: int = LOOKBACK_DAYS):
    """Pull closed positions within lookback window, with their thesis text."""
    conn = get_connection()
    try:
        cutoff = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        rows = conn.execute("""
            SELECT id, ticker, direction, hypothesis_text, pnl_pct,
                   close_date, close_reason, diamond_score
            FROM portfolio_positions
            WHERE status = 'closed'
              AND hypothesis_text IS NOT NULL
              AND (close_date >= ? OR close_date IS NULL)
        """, (cutoff,)).fetchall()
        out = []
        for r in rows:
            pid, ticker, direction, hyp, pnl, close_date, close_reason, ds = r
            out.append({
                "position_id": pid,
                "ticker": ticker,
                "direction": direction,
                "hypothesis_text": hyp,
                "pnl_pct": pnl or 0.0,
                "close_date": close_date,
                "close_reason": close_reason,
                "diamond_score": ds,
            })
        return out
    finally:
        conn.close()


def compute_score_adjustment(
    hypothesis_text: str,
    raw_score: float,
    lookback_days: int = LOOKBACK_DAYS,
) -> dict:
    """Compute a score adjustment based on similarity to recent closed positions.

    Returns dict with:
        adjustment: float, clipped to [MAX_PENALTY, MAX_BOOST]
        adjusted_score: raw_score + adjustment, floored at 1
        num_related: count of past positions with sim >= RELATED_THRESHOLD
        top_related: dict with nearest closed position info
        reason: human-readable explanation
    """
    result = {
        "adjustment": 0.0,
        "adjusted_score": raw_score,
        "num_related": 0,
        "top_related": None,
        "reason": "no related closed positions in lookback window",
    }

    if not hypothesis_text or raw_score is None:
        return result

    try:
        new_emb = _embed(hypothesis_text)
    except Exception as e:
        logger.warning(f"Embedding failed: {e}")
        return result
    if new_emb is None:
        return result

    closed = _recent_closed_positions(lookback_days)
    if not closed:
        return result

    related = []
    for p in closed:
        try:
            p_emb = _embed(p["hypothesis_text"])
        except Exception:
            continue
        if p_emb is None:
            continue
        sim = float(np.dot(new_emb, p_emb))
        if sim >= RELATED_THRESHOLD:
            related.append({
                **p,
                "similarity": sim,
            })

    if not related:
        return result

    related.sort(key=lambda r: -r["similarity"])
    top = related[0]

    # Weighted average P&L of related positions, weighted by similarity
    total_w = sum(r["similarity"] for r in related)
    weighted_pnl = sum(r["pnl_pct"] * r["similarity"] for r in related) / max(0.001, total_w)

    # Map weighted P&L to adjustment. -5% P&L → ~-6 pts penalty.
    # +5% P&L → ~+3 pts boost. Asymmetric: penalise losers more than we reward winners.
    if weighted_pnl < 0:
        adj = max(MAX_PENALTY, weighted_pnl * 1.2)  # e.g. -10% → -12 pts
    else:
        adj = min(MAX_BOOST, weighted_pnl * 0.6)    # e.g. +10% → +6 pts

    # Soften if few related positions (low statistical support)
    if len(related) == 1:
        adj *= 0.5
    elif len(related) == 2:
        adj *= 0.75

    adj = round(adj, 2)
    adjusted = max(1.0, raw_score + adj)

    reason_parts = [
        f"{len(related)} related closed position{'s' if len(related) != 1 else ''} "
        f"(weighted P&L {weighted_pnl:+.2f}%)",
        f"closest: {top['ticker']} {top['direction']} "
        f"P&L {top['pnl_pct']:+.2f}% sim={top['similarity']:.2f}",
    ]

    result.update({
        "adjustment": adj,
        "adjusted_score": round(adjusted, 2),
        "num_related": len(related),
        "top_related": {
            "ticker": top["ticker"],
            "direction": top["direction"],
            "pnl_pct": top["pnl_pct"],
            "similarity": round(top["similarity"], 4),
            "close_date": top["close_date"],
        },
        "reason": " | ".join(reason_parts),
    })

    return result


def log_adjustment(hypothesis_id: Optional[int], raw_score: float,
                   adjustment_result: dict):
    """Persist a feedback adjustment row for later analysis."""
    _ensure_feedback_table()
    conn = get_connection()
    try:
        top = adjustment_result.get("top_related") or {}
        conn.execute("""
            INSERT INTO scoring_feedback_log
            (hypothesis_id, raw_score, adjustment, adjusted_score,
             num_related, top_related_ticker, top_related_pnl_pct,
             top_related_similarity, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            hypothesis_id,
            raw_score,
            adjustment_result.get("adjustment", 0.0),
            adjustment_result.get("adjusted_score", raw_score),
            adjustment_result.get("num_related", 0),
            top.get("ticker"),
            top.get("pnl_pct"),
            top.get("similarity"),
            adjustment_result.get("reason", ""),
        ))
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "simulate":
        # Dry-run: apply feedback to every surviving hypothesis we already
        # have, and show what adjustments WOULD be applied going forward.
        conn = get_connection()
        rows = conn.execute("""
            SELECT id, hypothesis_text, diamond_score
            FROM hypotheses
            WHERE survived_kill = 1
              AND hypothesis_text IS NOT NULL
            ORDER BY diamond_score DESC
            LIMIT 20
        """).fetchall()
        conn.close()

        print(f"Simulating feedback on top 20 surviving hypotheses...\n")
        print(f"{'ID':>4} {'raw':>4} {'adj':>6} {'new':>5} {'n':>3} {'closest':>20}")
        print("-" * 70)
        for hid, hyp, ds in rows:
            if ds is None:
                continue
            r = compute_score_adjustment(hyp, float(ds))
            tr = r.get("top_related") or {}
            closest = f"{tr.get('ticker','--')} {tr.get('pnl_pct', 0):+.1f}%" if tr else "--"
            print(f"{hid:>4} {ds:>4} {r['adjustment']:>+6.2f} "
                  f"{r['adjusted_score']:>5.1f} {r['num_related']:>3} {closest:>20}")
        sys.exit(0)

    # Default: score adjustment from stdin
    text = sys.stdin.read().strip()
    if not text:
        print("Usage: python portfolio_feedback.py simulate")
        print("   or: echo 'hypothesis' | python portfolio_feedback.py")
        sys.exit(1)
    r = compute_score_adjustment(text, 70.0)
    print(json.dumps(r, indent=2))
