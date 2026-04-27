"""Seam Network — the persistent translation graph that becomes HUNTER's moat.

The deeper insight: every successful cross-silo translation HUNTER performs
is a *learned seam* — a documented connection between concept_in_silo_A and
concept_in_silo_B, with an associated underlying-reality anchor and the
realised alpha that flowed through it.

Over time these seams accumulate into a graph. The graph has structure that
matters:

  - Direct seams              A <-> B
  - Transitive seams          A -> B + B -> C implies A -> C (after enough uses)
  - Cycle-closing seams       A -> B -> C -> A (most stable; all routes consistent)
  - Bridge seams              connect previously-disconnected components

THE SEAM NETWORK IS THE MOAT. It compounds with use. It cannot be replicated
by reading code or by cloning the corpus. It is emergent from running. It
forms the canonical reference structure for cross-silo translation in
finance — the kind of asset that becomes citable infrastructure (like CRSP
for stock prices, or COMPUSTAT for fundamentals) for the field that does
not yet have such a reference.

Schema (in quant_data.db):

    seams
        id, source_silo, source_phrase, target_silo, target_phrase,
        underlying_reality_label, first_observed, n_uses, last_used,
        alpha_generated_total, half_life_days_estimate,
        regime_conditioning_json, is_validated, validation_source

    seam_uses
        seam_id, used_at, mechanism_id, signal_id, realised_alpha
        (one row per time the seam contributed to a generated thesis)

Public-asset path: when summer fills the network, export it as a versioned
Zenodo dataset under your name. Every paper using cross-silo methods cites
it. Citation count compounds. The seam-network DOI becomes the canonical
reference. **This is the academic-priority moat.**
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from quant.data.base import DEFAULT_DB


SCHEMA = """
CREATE TABLE IF NOT EXISTS seams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_silo TEXT NOT NULL,
    source_phrase TEXT NOT NULL,
    target_silo TEXT NOT NULL,
    target_phrase TEXT NOT NULL,
    underlying_reality_label TEXT,
    first_observed TEXT NOT NULL,
    last_used TEXT,
    n_uses INTEGER NOT NULL DEFAULT 1,
    alpha_generated_total REAL NOT NULL DEFAULT 0.0,
    half_life_days_estimate REAL,
    regime_conditioning_json TEXT,
    is_validated INTEGER NOT NULL DEFAULT 0,
    validation_source TEXT
);

CREATE INDEX IF NOT EXISTS ix_seams_source ON seams (source_silo, source_phrase);
CREATE INDEX IF NOT EXISTS ix_seams_target ON seams (target_silo, target_phrase);
CREATE INDEX IF NOT EXISTS ix_seams_silos ON seams (source_silo, target_silo);

CREATE TABLE IF NOT EXISTS seam_uses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    seam_id INTEGER NOT NULL,
    used_at TEXT NOT NULL,
    mechanism_id TEXT,
    signal_id INTEGER,
    realised_alpha REAL,
    FOREIGN KEY (seam_id) REFERENCES seams (id)
);

CREATE INDEX IF NOT EXISTS ix_seam_uses_seam ON seam_uses (seam_id);
CREATE INDEX IF NOT EXISTS ix_seam_uses_signal ON seam_uses (signal_id);
"""


def _conn(db_path: Optional[Path | str]) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA)
    return conn


@dataclass
class Seam:
    id: int
    source_silo: str
    source_phrase: str
    target_silo: str
    target_phrase: str
    underlying_reality_label: Optional[str]
    first_observed: datetime
    last_used: Optional[datetime]
    n_uses: int
    alpha_generated_total: float
    half_life_days_estimate: Optional[float]
    regime_conditioning: dict
    is_validated: bool
    validation_source: Optional[str]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_silo": self.source_silo,
            "source_phrase": self.source_phrase,
            "target_silo": self.target_silo,
            "target_phrase": self.target_phrase,
            "underlying_reality": self.underlying_reality_label,
            "first_observed": self.first_observed.isoformat(),
            "last_used": (
                None if self.last_used is None else self.last_used.isoformat()
            ),
            "n_uses": self.n_uses,
            "alpha_generated_total": round(self.alpha_generated_total, 4),
            "half_life_days_estimate": self.half_life_days_estimate,
            "regime_conditioning": self.regime_conditioning,
            "is_validated": self.is_validated,
            "validation_source": self.validation_source,
        }


def _row_to_seam(row) -> Seam:
    return Seam(
        id=row[0],
        source_silo=row[1],
        source_phrase=row[2],
        target_silo=row[3],
        target_phrase=row[4],
        underlying_reality_label=row[5],
        first_observed=datetime.fromisoformat(row[6]),
        last_used=(
            datetime.fromisoformat(row[7]) if row[7] else None
        ),
        n_uses=row[8],
        alpha_generated_total=row[9],
        half_life_days_estimate=row[10],
        regime_conditioning=json.loads(row[11]) if row[11] else {},
        is_validated=bool(row[12]),
        validation_source=row[13],
    )


def add_seam(
    *,
    source_silo: str,
    source_phrase: str,
    target_silo: str,
    target_phrase: str,
    underlying_reality_label: Optional[str] = None,
    regime_conditioning: Optional[dict] = None,
    db_path: Optional[Path | str] = None,
) -> int:
    """Log a new translation seam (or increment its use-count if it exists).

    Seam identity = (source_silo, source_phrase, target_silo, target_phrase).
    Repeated observations of the same seam increment n_uses and update
    last_used. Returns the seam_id (existing or new).
    """
    now = datetime.now(timezone.utc).isoformat()
    conn = _conn(db_path)
    try:
        existing = conn.execute(
            "SELECT id FROM seams "
            "WHERE source_silo = ? AND source_phrase = ? "
            "  AND target_silo = ? AND target_phrase = ?",
            (source_silo, source_phrase, target_silo, target_phrase),
        ).fetchone()
        if existing:
            sid = int(existing[0])
            with conn:
                conn.execute(
                    "UPDATE seams SET n_uses = n_uses + 1, last_used = ? WHERE id = ?",
                    (now, sid),
                )
            return sid
        with conn:
            cur = conn.execute(
                "INSERT INTO seams "
                "(source_silo, source_phrase, target_silo, target_phrase, "
                " underlying_reality_label, first_observed, last_used, n_uses, "
                " alpha_generated_total, regime_conditioning_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 1, 0.0, ?)",
                (
                    source_silo, source_phrase, target_silo, target_phrase,
                    underlying_reality_label,
                    now, now,
                    json.dumps(regime_conditioning or {}, default=str),
                ),
            )
            return int(cur.lastrowid)
    finally:
        conn.close()


def log_seam_use(
    seam_id: int,
    *,
    mechanism_id: Optional[str] = None,
    signal_id: Optional[int] = None,
    realised_alpha: Optional[float] = None,
    db_path: Optional[Path | str] = None,
) -> int:
    """Log that a seam was used in producing a thesis or signal."""
    now = datetime.now(timezone.utc).isoformat()
    conn = _conn(db_path)
    try:
        with conn:
            cur = conn.execute(
                "INSERT INTO seam_uses "
                "(seam_id, used_at, mechanism_id, signal_id, realised_alpha) "
                "VALUES (?, ?, ?, ?, ?)",
                (seam_id, now, mechanism_id, signal_id, realised_alpha),
            )
            if realised_alpha is not None:
                conn.execute(
                    "UPDATE seams SET alpha_generated_total = alpha_generated_total + ? "
                    "WHERE id = ?",
                    (float(realised_alpha), seam_id),
                )
            return int(cur.lastrowid)
    finally:
        conn.close()


def get_seam(seam_id: int, db_path: Optional[Path | str] = None) -> Optional[Seam]:
    conn = _conn(db_path)
    try:
        row = conn.execute(
            "SELECT id, source_silo, source_phrase, target_silo, target_phrase, "
            "       underlying_reality_label, first_observed, last_used, n_uses, "
            "       alpha_generated_total, half_life_days_estimate, "
            "       regime_conditioning_json, is_validated, validation_source "
            "FROM seams WHERE id = ?",
            (seam_id,),
        ).fetchone()
    finally:
        conn.close()
    return _row_to_seam(row) if row else None


def find_seams(
    *,
    source_silo: Optional[str] = None,
    target_silo: Optional[str] = None,
    min_n_uses: int = 1,
    db_path: Optional[Path | str] = None,
) -> list[Seam]:
    """Query seams matching the given filters."""
    conn = _conn(db_path)
    try:
        clauses = ["n_uses >= ?"]
        params: list = [min_n_uses]
        if source_silo is not None:
            clauses.append("source_silo = ?")
            params.append(source_silo)
        if target_silo is not None:
            clauses.append("target_silo = ?")
            params.append(target_silo)
        sql = (
            "SELECT id, source_silo, source_phrase, target_silo, target_phrase, "
            "       underlying_reality_label, first_observed, last_used, n_uses, "
            "       alpha_generated_total, half_life_days_estimate, "
            "       regime_conditioning_json, is_validated, validation_source "
            "FROM seams WHERE " + " AND ".join(clauses) + " ORDER BY n_uses DESC"
        )
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return [_row_to_seam(r) for r in rows]


def graph_stats(db_path: Optional[Path | str] = None) -> dict:
    """Return summary statistics on the seam network."""
    conn = _conn(db_path)
    try:
        n_seams = conn.execute("SELECT COUNT(*) FROM seams").fetchone()[0]
        n_uses = conn.execute("SELECT COALESCE(SUM(n_uses), 0) FROM seams").fetchone()[0]
        total_alpha = conn.execute(
            "SELECT COALESCE(SUM(alpha_generated_total), 0) FROM seams"
        ).fetchone()[0]

        silo_pairs = conn.execute(
            "SELECT source_silo, target_silo, COUNT(*), SUM(n_uses) "
            "FROM seams GROUP BY source_silo, target_silo"
        ).fetchall()

        all_silos: set[str] = set()
        for ss, ts, _, _ in silo_pairs:
            all_silos.add(ss)
            all_silos.add(ts)

        validated = conn.execute(
            "SELECT COUNT(*) FROM seams WHERE is_validated = 1"
        ).fetchone()[0]

        top_seams_rows = conn.execute(
            "SELECT id, source_silo, target_silo, n_uses, alpha_generated_total "
            "FROM seams ORDER BY n_uses DESC LIMIT 5"
        ).fetchall()
        top_seams = [
            {
                "seam_id": r[0],
                "source_silo": r[1],
                "target_silo": r[2],
                "n_uses": r[3],
                "alpha": round(r[4], 4),
            }
            for r in top_seams_rows
        ]
    finally:
        conn.close()

    return {
        "n_seams": n_seams,
        "n_silos_touched": len(all_silos),
        "n_uses_total": int(n_uses or 0),
        "total_alpha_via_seams": round(float(total_alpha or 0.0), 4),
        "n_silo_pairs_with_seams": len(silo_pairs),
        "n_validated_seams": validated,
        "top_seams": top_seams,
    }


def export_atlas_dump(
    output_path: Path | str,
    db_path: Optional[Path | str] = None,
) -> dict:
    """Export the seam network as a JSON dump suitable for Zenodo deposit.

    Output is a single JSON file with metadata + every seam. This is the
    citable artefact that becomes the Open Compositional Atlas. Versioned
    deposits accumulate into a permanent reference resource.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seams = find_seams(min_n_uses=1, db_path=db_path)
    stats = graph_stats(db_path=db_path)
    payload = {
        "schema_version": "atlas.v1",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "stats": stats,
        "seams": [s.to_dict() for s in seams],
    }
    output_path.write_text(json.dumps(payload, indent=2, default=str))
    return {
        "output_path": str(output_path),
        "n_seams_exported": len(seams),
        "n_silos_touched": stats["n_silos_touched"],
    }
