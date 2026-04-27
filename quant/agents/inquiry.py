"""System inquiry queue — the system asks; the operator answers.

The autonomous agents (TRADER, CONSCIENCE, AUDITOR, mechanism compiler, regime
detector) write questions when they are uncertain or need a human decision.
The operator reviews + answers daily.

This is the human-in-the-loop layer that gives the operator real-time
insight into what the system is thinking AND lets the system get smarter
from human judgment without auto-pilot mistakes.

Inquiry types:
    decision    needs operator's call (e.g. "should we exit position X early?")
    data        operator may have access to data we don't (e.g. "do you know
                whether ABBV's Q2 earnings call is May 1 or May 8?")
    validation  system found something surprising; sanity check (e.g. "this
                mechanism just emitted 12 signals in a day; normal?")
    review      end-of-day "tell me what surprised you" prompt

CLI:
    python -m quant inquiries list
    python -m quant inquiries answer <id> "<text>"
    python -m quant inquiries dismiss <id>

Schema in quant_data.db:
    inquiries
        id, created_at, inquiry_type, urgency, context, body,
        options_json, related_files, answered_at, answer, dismissed
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from quant.data.base import DEFAULT_DB


INQUIRY_SCHEMA = """
CREATE TABLE IF NOT EXISTS inquiries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    inquiry_type TEXT NOT NULL,
    urgency TEXT NOT NULL DEFAULT 'medium',
    context TEXT,
    body TEXT NOT NULL,
    options_json TEXT,
    related_files TEXT,
    answered_at TEXT,
    answer TEXT,
    dismissed INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS ix_inq_status ON inquiries (answered_at, dismissed);
CREATE INDEX IF NOT EXISTS ix_inq_urgency ON inquiries (urgency);
"""


URGENCY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}


@dataclass
class Inquiry:
    id: int
    created_at: datetime
    inquiry_type: str
    urgency: str
    context: Optional[str]
    body: str
    options: list[str]
    related_files: Optional[str]
    answered_at: Optional[datetime]
    answer: Optional[str]
    dismissed: bool

    @property
    def is_open(self) -> bool:
        return self.answered_at is None and not self.dismissed


def _conn(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.executescript(INQUIRY_SCHEMA)
    return conn


def open_inquiry(
    *,
    inquiry_type: str,
    body: str,
    urgency: str = "medium",
    context: Optional[str] = None,
    options: Optional[list[str]] = None,
    related_files: Optional[str] = None,
    db_path: Path | str | None = None,
) -> int:
    """Create a new inquiry, return its id."""
    if inquiry_type not in {"decision", "data", "validation", "review"}:
        raise ValueError(f"unknown inquiry_type: {inquiry_type}")
    if urgency not in URGENCY_RANK:
        raise ValueError(f"unknown urgency: {urgency}")
    conn = _conn(db_path)
    try:
        with conn:
            cur = conn.execute(
                "INSERT INTO inquiries "
                "(created_at, inquiry_type, urgency, context, body, "
                " options_json, related_files) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    datetime.now(timezone.utc).isoformat(),
                    inquiry_type,
                    urgency,
                    context,
                    body,
                    json.dumps(options or []),
                    related_files,
                ),
            )
            return int(cur.lastrowid)
    finally:
        conn.close()


def list_open_inquiries(
    db_path: Path | str | None = None,
    limit: int = 50,
) -> list[Inquiry]:
    """Open + un-dismissed, ordered by urgency then created_at."""
    conn = _conn(db_path)
    try:
        rows = conn.execute(
            "SELECT id, created_at, inquiry_type, urgency, context, body, "
            "       options_json, related_files, answered_at, answer, dismissed "
            "FROM inquiries WHERE answered_at IS NULL AND dismissed = 0 "
            "ORDER BY urgency, created_at LIMIT ?",
            (limit,),
        ).fetchall()
        out: list[Inquiry] = []
        for r in rows:
            try:
                opts = json.loads(r[6]) if r[6] else []
            except json.JSONDecodeError:
                opts = []
            out.append(
                Inquiry(
                    id=r[0],
                    created_at=datetime.fromisoformat(r[1]),
                    inquiry_type=r[2],
                    urgency=r[3],
                    context=r[4],
                    body=r[5],
                    options=opts,
                    related_files=r[7],
                    answered_at=datetime.fromisoformat(r[8]) if r[8] else None,
                    answer=r[9],
                    dismissed=bool(r[10]),
                )
            )
        # Re-sort by urgency rank since SQLite text-sort puts "critical"
        # alphabetically after "high", etc.
        out.sort(key=lambda i: (URGENCY_RANK.get(i.urgency, 99), i.created_at))
        return out
    finally:
        conn.close()


def answer_inquiry(
    inquiry_id: int,
    answer: str,
    db_path: Path | str | None = None,
) -> None:
    conn = _conn(db_path)
    try:
        with conn:
            conn.execute(
                "UPDATE inquiries SET answered_at = ?, answer = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), answer, inquiry_id),
            )
    finally:
        conn.close()


def dismiss_inquiry(
    inquiry_id: int,
    db_path: Path | str | None = None,
) -> None:
    conn = _conn(db_path)
    try:
        with conn:
            conn.execute(
                "UPDATE inquiries SET dismissed = 1 WHERE id = ?",
                (inquiry_id,),
            )
    finally:
        conn.close()


def get_inquiry(
    inquiry_id: int,
    db_path: Path | str | None = None,
) -> Optional[Inquiry]:
    conn = _conn(db_path)
    try:
        r = conn.execute(
            "SELECT id, created_at, inquiry_type, urgency, context, body, "
            "       options_json, related_files, answered_at, answer, dismissed "
            "FROM inquiries WHERE id = ?",
            (inquiry_id,),
        ).fetchone()
        if not r:
            return None
        try:
            opts = json.loads(r[6]) if r[6] else []
        except json.JSONDecodeError:
            opts = []
        return Inquiry(
            id=r[0],
            created_at=datetime.fromisoformat(r[1]),
            inquiry_type=r[2],
            urgency=r[3],
            context=r[4],
            body=r[5],
            options=opts,
            related_files=r[7],
            answered_at=datetime.fromisoformat(r[8]) if r[8] else None,
            answer=r[9],
            dismissed=bool(r[10]),
        )
    finally:
        conn.close()
