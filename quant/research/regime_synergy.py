"""Regime-conditional synergistic information.

For a target X (realised alpha or diamond score) and two information
sources A, B (e.g. number of silos, domain distance), the interaction
information

    II(X; A; B) = I(X; A,B) - I(X; A) - I(X; B)

is a first-order proxy for synergy: II > 0 means the joint distribution
of (A, B) reveals more about X than either marginal alone; II < 0 means
the two sources carry overlapping information.

This module computes II *separately within each macro regime*:

    II(X; A; B | regime = R)   for R in {risk_on, late_cycle, risk_off, crisis}

The observation is operational: the same composition can be synergistic
in one regime and redundant in another. A mechanism whose
II(risk_off) > 0 but II(risk_on) ≈ 0 should fire only in risk_off.

As far as we have searched, partial information decomposition (Williams &
Beer 2010) has not previously been conditioned on macro regime in
finance. The conditioned form is the rigorous mathematical operationalisation
of "this composition produces alpha *only when the macro environment
allows the cross-silo channel to function*."

Inputs:
    corpus_db: HUNTER corpus SQLite (live or Zenodo)
    quant_db:  the quant data store containing FRED data for regime detection
                (defaults to quant.data.base.DEFAULT_DB)

Output: a dict
    {
        "n_total": int,
        "n_per_regime": {regime: int, ...},
        "results": {regime: SynergyResult or None, ...},
        "global": SynergyResult,                 # un-conditioned baseline
        "delta_vs_global": {regime: float, ...}, # II(R) - II(global), in bits
    }
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from quant.data.base import MarketState
from quant.research.regime import RegimeDetector
from quant.research.synergy import SynergyEstimator


def _load_collisions_with_timestamps(corpus_db: str | Path) -> list[dict]:
    """Pull (collision_id, num_domains, domain_distance, score, timestamp).

    Tries the same join as hunter_bridge but pulls a usable timestamp from
    the collisions row so we can attach a regime to each observation.
    """
    conn = sqlite3.connect(str(corpus_db))
    try:
        table_names = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        # Detect which timestamp column collisions has — schemas vary slightly
        ts_col_candidates = ["created_at", "discovered_at", "ts", "timestamp"]
        if "collisions" in table_names:
            cols = {
                r[1]
                for r in conn.execute("PRAGMA table_info(collisions)").fetchall()
            }
        else:
            cols = set()
        ts_col = next((c for c in ts_col_candidates if c in cols), None)

        rows: list[dict] = []
        for table in ("hypotheses", "hypotheses_archive"):
            if table not in table_names:
                continue
            try:
                ts_select = f"col.{ts_col}" if ts_col else "h.created_at"
                cur = conn.execute(
                    f"SELECT col.id, col.num_domains, col.domain_distance, "
                    f"       h.diamond_score, {ts_select} "
                    f"FROM {table} h JOIN collisions col ON col.id = h.collision_id "
                    f"WHERE h.diamond_score IS NOT NULL "
                    f"  AND col.num_domains IS NOT NULL"
                )
                for cid, nd, dd, score, ts_str in cur:
                    if ts_str is None:
                        continue
                    try:
                        ts = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                    except (ValueError, TypeError):
                        continue
                    rows.append(
                        {
                            "collision_id": cid,
                            "num_domains": int(nd),
                            "domain_distance": (
                                float(dd) if dd is not None else None
                            ),
                            "diamond_score": float(score),
                            "timestamp": ts,
                            "source_table": table,
                        }
                    )
            except sqlite3.OperationalError:
                continue
    finally:
        conn.close()
    return rows


def _attach_regimes(
    rows: list[dict],
    quant_db: Optional[Path | str] = None,
) -> tuple[list[dict], list[str]]:
    """For each row, query the regime detector at row['timestamp'].

    Returns the original rows (filtered to those where regime detection
    succeeded) plus the list of regime labels in the same order.
    """
    state = MarketState(quant_db)
    detector = RegimeDetector(state)

    kept: list[dict] = []
    labels: list[str] = []
    try:
        for row in rows:
            rs = detector.detect(row["timestamp"])
            if rs is None:
                continue
            kept.append(row)
            labels.append(rs.regime)
    finally:
        state.close()
    return kept, labels


def compute_regime_conditional_synergy(
    corpus_db: str | Path,
    *,
    quant_db: Optional[Path | str] = None,
    method: str = "discrete",
) -> dict[str, Any]:
    """End-to-end: load collisions, attach regimes, compute II per regime.

    Returns a clean dict, never raises on empty/under-powered corpora.
    """
    rows = _load_collisions_with_timestamps(corpus_db)
    if not rows:
        return {
            "status": "empty",
            "message": (
                "No scored collisions with timestamps found. The HUNTER db "
                "must have rows in `hypotheses` or `hypotheses_archive` joined "
                "to `collisions` with both num_domains and a usable timestamp."
            ),
        }

    rows, labels = _attach_regimes(rows, quant_db=quant_db)
    if not rows:
        return {
            "status": "no_regime_data",
            "n_collisions": 0,
            "message": (
                "Found scored collisions but the regime detector returned "
                "None for all of them. Ingest FRED data first: "
                "`python -m quant ingest fred --series DGS10,DGS2,BAA10Y`."
            ),
        }

    score = np.array([r["diamond_score"] for r in rows], dtype=float)
    n_dom = np.array([r["num_domains"] for r in rows], dtype=float)
    dist = np.array(
        [r["domain_distance"] if r["domain_distance"] is not None else np.nan
         for r in rows],
        dtype=float,
    )
    # Drop rows with missing distance
    mask = ~np.isnan(dist)
    score = score[mask]
    n_dom = n_dom[mask]
    dist = dist[mask]
    labels_arr = np.array(labels)[mask]

    estimator = SynergyEstimator(method=method)

    global_result = estimator.measure(score, n_dom, dist)
    grouped = estimator.measure_grouped(score, n_dom, dist, labels_arr)

    n_per_regime: dict[str, int] = {}
    for label in np.unique(labels_arr):
        n_per_regime[str(label)] = int((labels_arr == label).sum())

    delta = {}
    for label, res in grouped.items():
        delta[label] = (
            None
            if res is None
            else round(res.ii_bits - global_result.ii_bits, 4)
        )

    return {
        "status": "ok",
        "method": method,
        "n_total": len(score),
        "n_per_regime": n_per_regime,
        "global_II_bits": round(global_result.ii_bits, 4),
        "global": {
            "I(X;A)": round(global_result.i_xa_bits, 4),
            "I(X;B)": round(global_result.i_xb_bits, 4),
            "I(X;A,B)": round(global_result.i_xab_bits, 4),
            "II_bits": round(global_result.ii_bits, 4),
        },
        "per_regime": {
            label: (
                None
                if res is None
                else {
                    "n": n_per_regime.get(label, 0),
                    "I(X;A)": round(res.i_xa_bits, 4),
                    "I(X;B)": round(res.i_xb_bits, 4),
                    "I(X;A,B)": round(res.i_xab_bits, 4),
                    "II_bits": round(res.ii_bits, 4),
                }
            )
            for label, res in grouped.items()
        },
        "delta_vs_global_bits": delta,
    }


def cli_demo_regime_split() -> dict[str, Any]:
    """Synthetic demo where one regime is synergistic and the other is not.

    Construct two regimes' worth of data:
      In regime "calm":   X is independent of (A, B) — pure noise. II ~= 0.
      In regime "stress": X = sign(A) XOR sign(B) — pure synergy. II ~= +1 bit.

    A pooled (regime-blind) II computation will be heavily diluted toward zero.
    Conditioning on regime recovers the truth. This is exactly the value the
    method adds: it sees signal that the unconditional version cannot.
    """
    rng = np.random.default_rng(42)
    n_per = 1500

    # Regime "calm": pure noise, II ~ 0
    a_calm = rng.standard_normal(n_per)
    b_calm = rng.standard_normal(n_per)
    x_calm = rng.standard_normal(n_per)

    # Regime "stress": X = sign(A * B) -> synergistic, marginals ~ 0
    a_stress = rng.standard_normal(n_per)
    b_stress = rng.standard_normal(n_per)
    x_stress = np.sign(a_stress * b_stress) + 0.2 * rng.standard_normal(n_per)

    # Pool
    a = np.concatenate([a_calm, a_stress])
    b = np.concatenate([b_calm, b_stress])
    x = np.concatenate([x_calm, x_stress])
    labels = np.array(["calm"] * n_per + ["stress"] * n_per)

    est = SynergyEstimator(method="ksg", k=4)
    pooled = est.measure(x, a, b)
    grouped = est.measure_grouped(x, a, b, labels)

    return {
        "n_total": int(a.shape[0]),
        "pooled_II_bits": round(pooled.ii_bits, 4),
        "pooled_I(X;A,B)": round(pooled.i_xab_bits, 4),
        "regime_conditional": {
            label: (
                None
                if r is None
                else {
                    "II_bits": round(r.ii_bits, 4),
                    "I(X;A)": round(r.i_xa_bits, 4),
                    "I(X;B)": round(r.i_xb_bits, 4),
                    "I(X;A,B)": round(r.i_xab_bits, 4),
                }
            )
            for label, r in grouped.items()
        },
        "interpretation": (
            "Pooled II is heavily diluted toward zero because the synergistic "
            "regime is averaged with the noisy regime. Conditioning recovers "
            "the truth: ~0 in calm, ~+1 bit in stress. A mechanism gated to "
            "stress regime captures the full synergistic alpha."
        ),
    }


if __name__ == "__main__":
    import json

    print("Regime-conditional synergy: synthetic demo")
    print("=" * 60)
    print(json.dumps(cli_demo_regime_split(), indent=2, default=str))
