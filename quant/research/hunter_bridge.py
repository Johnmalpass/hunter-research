"""Bridge from a HUNTER research corpus (live or Zenodo) to the synergy estimator.

Loads collisions and linked diamond scores from a HUNTER SQLite database and
computes the synergistic information II(X; A; B), where:

  X   = adversarial diamond score (or realised alpha when summer fills)
  A   = num_domains   (silo count of the collision)
  B   = domain_distance  (cross-silo information-theoretic distance)

Reading: how many bits about score do (silos x distance) carry SYNERGISTICALLY
beyond each marginal alone? Positive II means the framework's central claim is
operationally true in the data. Negative II means the two features carry
redundant information about score (no compositional gain).

When summer fills the live db with realised alpha, swap X for realised_alpha
and the same function answers the headline empirical question of the summer
paper.

Schema-tolerant: looks in both `hypotheses` and `hypotheses_archive`. Joins
through `collisions` on `collision_id`.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from quant.research.synergy import SynergyEstimator


def _load_collisions_with_scores(db_path: str | Path) -> list[dict]:
    """Return rows of (collision_id, num_domains, domain_distance, diamond_score)."""
    conn = sqlite3.connect(str(db_path))
    rows: list[dict] = []
    try:
        table_names = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        for table in ("hypotheses", "hypotheses_archive"):
            if table not in table_names:
                continue
            try:
                cur = conn.execute(
                    f"SELECT col.id, col.num_domains, col.domain_distance, h.diamond_score "
                    f"FROM {table} h JOIN collisions col ON col.id = h.collision_id "
                    f"WHERE h.diamond_score IS NOT NULL "
                    f"  AND col.num_domains IS NOT NULL"
                )
                for cid, nd, dd, score in cur:
                    rows.append(
                        {
                            "collision_id": cid,
                            "num_domains": int(nd) if nd is not None else None,
                            "domain_distance": (
                                float(dd) if dd is not None else None
                            ),
                            "diamond_score": float(score),
                            "source_table": table,
                        }
                    )
            except sqlite3.OperationalError:
                continue
    finally:
        conn.close()
    return rows


def compute_collision_synergy(db_path: str | Path) -> dict[str, Any]:
    """II(score; num_domains; domain_distance) over scored collisions.

    Returns a clean dict — never raises on empty/under-powered corpora.
    """
    rows = _load_collisions_with_scores(db_path)

    if not rows:
        return {
            "status": "empty",
            "n": 0,
            "message": (
                "No scored collisions found. Either the live db hasn't been "
                "populated yet (run the HUNTER pipeline) or the Zenodo corpus "
                "needs to be downloaded first."
            ),
        }

    usable = [
        r for r in rows
        if r["num_domains"] is not None and r["domain_distance"] is not None
    ]
    if len(usable) < 50:
        return {
            "status": "underpowered",
            "n": len(usable),
            "n_total": len(rows),
            "message": (
                f"Only {len(usable)} collisions have all three of "
                "(num_domains, domain_distance, diamond_score). Need >= 50 "
                "for a meaningful synergy estimate."
            ),
        }

    score = np.array([r["diamond_score"] for r in usable], dtype=float)
    n_dom = np.array([r["num_domains"] for r in usable], dtype=float)
    dist = np.array([r["domain_distance"] for r in usable], dtype=float)

    est_d = SynergyEstimator(method="discrete", n_bins=8)
    rd = est_d.measure(score, n_dom, dist)

    est_c = SynergyEstimator(method="ksg", k=4)
    rc = est_c.measure(score, n_dom, dist)

    return {
        "status": "ok",
        "n": len(usable),
        "n_total": len(rows),
        "summary": {
            "mean_score": float(score.mean()),
            "mean_num_domains": float(n_dom.mean()),
            "mean_domain_distance": float(dist.mean()),
        },
        "discrete_II": {
            "I(score; num_domains)": rd.i_xa_bits,
            "I(score; domain_distance)": rd.i_xb_bits,
            "I(score; (num_domains, domain_distance))": rd.i_xab_bits,
            "II_bits": rd.ii_bits,
            "synergistic": rd.synergistic,
            "synergy_bits": rd.synergy_bits,
            "redundancy_bits": rd.redundancy_bits,
        },
        "ksg_II": {
            "I(score; num_domains)": rc.i_xa_bits,
            "I(score; domain_distance)": rc.i_xb_bits,
            "I(score; (num_domains, domain_distance))": rc.i_xab_bits,
            "II_bits": rc.ii_bits,
        },
    }


def cli_demo_on_synthetic() -> dict[str, Any]:
    """Demonstrate what the bridge will report under three regimes.

    Useful before the corpus fills: shows what synergistic, redundant, and
    independent look like on HUNTER-shaped data. KSG handles continuous data.

    Recipes (each constructed so the sign of II is unambiguous):

      synergistic_xor    X = sign(num_dom > median) * sign(dist > median) + noise
                         Each marginal carries ~0 info (X is symmetric in the
                         other variable). Joint determines X. II > 0.

      redundant_proxy    A second variable B that is essentially a noisy copy
                         of num_dom; X depends only on num_dom. Both A and B
                         carry the same info -> II < 0.

      independent        B is pure noise; X depends only on num_dom -> II ~ 0.
    """
    rng = np.random.default_rng(0)
    n = 2000

    n_dom = rng.integers(1, 7, size=n).astype(float)
    dist = rng.uniform(0.0, 1.0, size=n)

    # 1. Synergistic (XOR-like): each marginal is symmetric in the other
    sign_n = np.where(n_dom > 3.5, 1.0, -1.0)
    sign_d = np.where(dist > 0.5, 1.0, -1.0)
    synergistic_x = sign_n * sign_d + rng.standard_normal(n) * 0.3

    # 2. Redundant: B is a noisy copy of A; X depends on A only
    proxy_of_n_dom = (n_dom - 3.5) / 3.0 + rng.standard_normal(n) * 0.05
    redundant_x = (n_dom - 3.5) + rng.standard_normal(n) * 0.4

    # 3. Independent: B is pure noise; X depends on A only
    independent_x = (n_dom - 3.5) * 2 + rng.standard_normal(n) * 0.5

    out: dict[str, Any] = {}
    est = SynergyEstimator(method="ksg", k=4)

    for label, x_vec, a_vec, b_vec, expected_sign in [
        ("synergistic_xor",      synergistic_x, n_dom, dist,            "II > 0"),
        ("redundant_proxy",      redundant_x,   n_dom, proxy_of_n_dom,  "II < 0"),
        ("independent_b_noise",  independent_x, n_dom, dist,            "II ~ 0"),
    ]:
        r = est.measure(x_vec, a_vec, b_vec)
        out[label] = {
            "expected": expected_sign,
            "II_bits": round(r.ii_bits, 4),
            "I(X;A)": round(r.i_xa_bits, 4),
            "I(X;B)": round(r.i_xb_bits, 4),
            "I(X;A,B)": round(r.i_xab_bits, 4),
        }
    return out


if __name__ == "__main__":
    import json

    print("Synthetic HUNTER-shape demo (KSG):")
    print(json.dumps(cli_demo_on_synthetic(), indent=2, default=str))
