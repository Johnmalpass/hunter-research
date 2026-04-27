"""Base classes for the HUNTER market data layer.

Every external source writes through `BaseAdapter` into one SQLite store with a
single normalised schema. Agents read via `MarketState` with no knowledge of
which adapter produced a given fact, so swapping FRED for Refinitiv or Polygon
for IEX is a one-file change.

Schema (data_points):
    ts             ISO-8601 timestamp of the EVENT (not ingest time)
    asset_id       stable string id (ticker, CIK, FRED series, vessel IMO, ...)
    field          short snake_case field name (price, gdp_growth, filing_10k, ...)
    value          stringified payload (JSON for structs)
    value_numeric  float copy of value when numeric, else NULL — for fast SQL math
    source         adapter name
    metadata       JSON blob (URL, units, label, retrieved_at, ...)
    ingested_at    when this row hit the store

The composite primary key (ts, asset_id, field, source) makes re-ingesting
the same observation idempotent. Two sources can record the same field for
the same asset and timestamp without colliding.
"""

from __future__ import annotations

import json
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

DEFAULT_DB = Path(__file__).resolve().parent.parent.parent / "quant_data.db"


@dataclass
class DataPoint:
    """A single observation from any source."""

    timestamp: datetime
    asset_id: str
    field: str
    value: Any
    source: str
    metadata: dict = dc_field(default_factory=dict)


SCHEMA = """
CREATE TABLE IF NOT EXISTS data_points (
    ts TEXT NOT NULL,
    asset_id TEXT NOT NULL,
    field TEXT NOT NULL,
    value TEXT NOT NULL,
    value_numeric REAL,
    source TEXT NOT NULL,
    metadata TEXT,
    ingested_at TEXT NOT NULL,
    PRIMARY KEY (ts, asset_id, field, source)
);

CREATE INDEX IF NOT EXISTS ix_dp_asset_field ON data_points (asset_id, field);
CREATE INDEX IF NOT EXISTS ix_dp_field_ts ON data_points (field, ts);
CREATE INDEX IF NOT EXISTS ix_dp_source ON data_points (source);

CREATE TABLE IF NOT EXISTS adapter_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    adapter TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    rows_written INTEGER DEFAULT 0,
    error TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS asset_aliases (
    asset_id TEXT PRIMARY KEY,
    name TEXT,
    asset_type TEXT,
    primary_source TEXT,
    metadata TEXT
);
"""


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA)
    return conn


def _serialise_value(v: Any) -> tuple[str, Optional[float]]:
    """Return (text_repr, numeric_repr_or_None)."""
    if v is None:
        return "", None
    if isinstance(v, bool):
        return str(v), None
    if isinstance(v, (int, float)):
        return str(v), float(v)
    if isinstance(v, str):
        return v, None
    return json.dumps(v, default=str), None


def write_points(
    points: Iterable[DataPoint],
    db_path: Path | str | None = None,
) -> int:
    """Persist DataPoints. Idempotent: re-ingesting a point overwrites the row."""
    conn = get_connection(db_path)
    now = datetime.now(timezone.utc).isoformat()
    rows = 0
    try:
        with conn:
            for p in points:
                v_text, v_num = _serialise_value(p.value)
                ts = p.timestamp
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                conn.execute(
                    "INSERT OR REPLACE INTO data_points "
                    "(ts, asset_id, field, value, value_numeric, source, metadata, ingested_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        ts.isoformat(),
                        p.asset_id,
                        p.field,
                        v_text,
                        v_num,
                        p.source,
                        json.dumps(p.metadata or {}, default=str),
                        now,
                    ),
                )
                rows += 1
    finally:
        conn.close()
    return rows


class BaseAdapter(ABC):
    """One subclass per data source.

    Subclass contract:
      - set `name` to a stable string ("fred", "edgar", "polygon", ...)
      - set `rate_limit_per_minute` to keep the adapter under the source's limit
      - implement `fetch(**kwargs)` returning an iterable of DataPoints

    The harness calls `run()` which wraps `fetch()` with throttling, persistence,
    and run-tracking. Subclasses do not touch the database directly.
    """

    name: str = "base"
    rate_limit_per_minute: int = 60

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = db_path
        self._last_call: float = 0.0

    def _throttle(self) -> None:
        gap = 60.0 / max(1, self.rate_limit_per_minute)
        elapsed = time.monotonic() - self._last_call
        if elapsed < gap:
            time.sleep(gap - elapsed)
        self._last_call = time.monotonic()

    @abstractmethod
    def fetch(self, **kwargs) -> Iterable[DataPoint]:
        ...

    def run(self, **kwargs) -> int:
        """Fetch, persist, log. Returns row count."""
        conn = get_connection(self.db_path)
        started = datetime.now(timezone.utc).isoformat()
        try:
            with conn:
                cur = conn.execute(
                    "INSERT INTO adapter_runs (adapter, started_at) VALUES (?, ?)",
                    (self.name, started),
                )
                run_id = cur.lastrowid
        finally:
            conn.close()

        try:
            n = write_points(self.fetch(**kwargs), self.db_path)
        except Exception as e:
            conn = get_connection(self.db_path)
            try:
                with conn:
                    conn.execute(
                        "UPDATE adapter_runs SET finished_at = ?, error = ? WHERE id = ?",
                        (datetime.now(timezone.utc).isoformat(), str(e), run_id),
                    )
            finally:
                conn.close()
            raise

        conn = get_connection(self.db_path)
        try:
            with conn:
                conn.execute(
                    "UPDATE adapter_runs SET finished_at = ?, rows_written = ? WHERE id = ?",
                    (datetime.now(timezone.utc).isoformat(), n, run_id),
                )
        finally:
            conn.close()
        return n


class MarketState:
    """Read-only unified view over every adapter's output.

    Construct one per decision cycle. Agents query through this object and
    never see the underlying adapters or schema.
    """

    def __init__(self, db_path: Path | str | None = None):
        self._db_path = db_path
        self.conn = get_connection(db_path)

    def latest(self, asset_id: str, field: str) -> Optional[DataPoint]:
        row = self.conn.execute(
            "SELECT ts, asset_id, field, value, value_numeric, source, metadata "
            "FROM data_points WHERE asset_id = ? AND field = ? "
            "ORDER BY ts DESC LIMIT 1",
            (asset_id, field),
        ).fetchone()
        return self._row_to_point(row)

    def latest_as_of(
        self,
        asset_id: str,
        field: str,
        asof: datetime,
    ) -> Optional[DataPoint]:
        """Point-in-time read: latest value at or before `asof`. No look-ahead."""
        row = self.conn.execute(
            "SELECT ts, asset_id, field, value, value_numeric, source, metadata "
            "FROM data_points WHERE asset_id = ? AND field = ? AND ts <= ? "
            "ORDER BY ts DESC LIMIT 1",
            (asset_id, field, asof.isoformat()),
        ).fetchone()
        return self._row_to_point(row)

    def history(
        self,
        asset_id: str,
        field: str,
        start: datetime,
        end: Optional[datetime] = None,
    ) -> list[DataPoint]:
        end = end or datetime.now(timezone.utc)
        rows = self.conn.execute(
            "SELECT ts, asset_id, field, value, value_numeric, source, metadata "
            "FROM data_points WHERE asset_id = ? AND field = ? "
            "AND ts >= ? AND ts <= ? ORDER BY ts",
            (asset_id, field, start.isoformat(), end.isoformat()),
        ).fetchall()
        return [self._row_to_point(r) for r in rows if r is not None]

    def fields(self, asset_id: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT field FROM data_points WHERE asset_id = ?",
            (asset_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def assets(self, field: Optional[str] = None) -> list[str]:
        if field is not None:
            rows = self.conn.execute(
                "SELECT DISTINCT asset_id FROM data_points WHERE field = ?",
                (field,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT DISTINCT asset_id FROM data_points"
            ).fetchall()
        return [r[0] for r in rows]

    def adapter_runs(self, adapter: Optional[str] = None, limit: int = 20) -> list[dict]:
        if adapter:
            rows = self.conn.execute(
                "SELECT adapter, started_at, finished_at, rows_written, error "
                "FROM adapter_runs WHERE adapter = ? ORDER BY id DESC LIMIT ?",
                (adapter, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT adapter, started_at, finished_at, rows_written, error "
                "FROM adapter_runs ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "adapter": r[0],
                "started_at": r[1],
                "finished_at": r[2],
                "rows_written": r[3],
                "error": r[4],
            }
            for r in rows
        ]

    @staticmethod
    def _row_to_point(row) -> Optional[DataPoint]:
        if not row:
            return None
        ts, asset_id, field, value, value_numeric, source, metadata = row
        try:
            md = json.loads(metadata) if metadata else {}
        except json.JSONDecodeError:
            md = {}
        v = value_numeric if value_numeric is not None else value
        return DataPoint(
            timestamp=datetime.fromisoformat(ts),
            asset_id=asset_id,
            field=field,
            value=v,
            source=source,
            metadata=md,
        )

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
