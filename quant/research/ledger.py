"""Mechanism Ledger — every signal a mechanism emits + its realised outcome.

This is the substrate for self-aware mechanisms. Without it, every mechanism
is forever stateless: same inputs -> same output, never updated by experience.
With it, the system can answer questions like:

  - what's mechanism X's hit rate over the last 90 days?
  - in which regime does mechanism X work best?
  - has the mechanism gone cold (last 5 signals all losses)?
  - what's the cross-correlation of signals between mechanism X and Y?

Tables (created in the same SQLite store as quant.data, keyed off
`quant_data.db` by default):

  mechanism_signals
    id, mechanism_id, asof, asset, direction, raw_size_pct, confidence,
    rationale, regime_at_signal, signal_metadata, created_at

  mechanism_outcomes
    signal_id, entry_date, entry_price, exit_date, exit_price,
    realised_return_pct, holding_days, status

The outcome table is filled by either the live trader (when positions
close) or by the backtest harness (when simulating closures).
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

from quant.data.base import DEFAULT_DB


LEDGER_SCHEMA = """
CREATE TABLE IF NOT EXISTS mechanism_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mechanism_id TEXT NOT NULL,
    asof TEXT NOT NULL,
    asset TEXT NOT NULL,
    direction TEXT NOT NULL,
    raw_size_pct REAL,
    confidence REAL,
    rationale TEXT,
    regime_at_signal TEXT,
    signal_metadata TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_ms_mech_asof ON mechanism_signals (mechanism_id, asof);
CREATE INDEX IF NOT EXISTS ix_ms_asset_asof ON mechanism_signals (asset, asof);

CREATE TABLE IF NOT EXISTS mechanism_outcomes (
    signal_id INTEGER PRIMARY KEY,
    entry_date TEXT,
    entry_price REAL,
    exit_date TEXT,
    exit_price REAL,
    realised_return_pct REAL,
    holding_days INTEGER,
    status TEXT,
    closed_at TEXT,
    FOREIGN KEY (signal_id) REFERENCES mechanism_signals (id)
);

CREATE INDEX IF NOT EXISTS ix_mo_status ON mechanism_outcomes (status);
"""


def _conn(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.executescript(LEDGER_SCHEMA)
    return conn


@dataclass
class TrackRecord:
    """Aggregated stats for a mechanism's history."""

    mechanism_id: str
    n_signals: int
    n_completed: int
    n_open: int
    win_rate: Optional[float]
    mean_return_pct: Optional[float]
    median_return_pct: Optional[float]
    sharpe_per_trade: Optional[float]
    max_loss_pct: Optional[float]
    max_consecutive_losses: int
    last_signal_at: Optional[datetime]
    best_regime: Optional[str]
    worst_regime: Optional[str]

    @property
    def is_cold(self) -> bool:
        """5+ consecutive losses suggests the mechanism is broken right now."""
        return self.max_consecutive_losses >= 5


def log_signal(
    *,
    mechanism_id: str,
    asof: datetime,
    asset: str,
    direction: str,
    raw_size_pct: float,
    confidence: float,
    rationale: str,
    regime_at_signal: Optional[dict] = None,
    signal_metadata: Optional[dict] = None,
    db_path: Path | str | None = None,
) -> int:
    """Log a freshly-emitted signal. Returns its row id."""
    if asof.tzinfo is None:
        asof = asof.replace(tzinfo=timezone.utc)
    conn = _conn(db_path)
    try:
        with conn:
            cur = conn.execute(
                "INSERT INTO mechanism_signals "
                "(mechanism_id, asof, asset, direction, raw_size_pct, "
                " confidence, rationale, regime_at_signal, signal_metadata, "
                " created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    mechanism_id,
                    asof.isoformat(),
                    asset,
                    direction,
                    raw_size_pct,
                    confidence,
                    rationale,
                    json.dumps(regime_at_signal or {}, default=str),
                    json.dumps(signal_metadata or {}, default=str),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            return int(cur.lastrowid)
    finally:
        conn.close()


def log_outcome(
    *,
    signal_id: int,
    entry_date: datetime,
    entry_price: float,
    exit_date: datetime,
    exit_price: float,
    realised_return_pct: float,
    status: str = "closed",
    db_path: Path | str | None = None,
) -> None:
    """Persist the realised outcome for a previously-logged signal."""
    if entry_date.tzinfo is None:
        entry_date = entry_date.replace(tzinfo=timezone.utc)
    if exit_date.tzinfo is None:
        exit_date = exit_date.replace(tzinfo=timezone.utc)
    holding_days = max(0, (exit_date - entry_date).days)
    conn = _conn(db_path)
    try:
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO mechanism_outcomes "
                "(signal_id, entry_date, entry_price, exit_date, exit_price, "
                " realised_return_pct, holding_days, status, closed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    signal_id,
                    entry_date.isoformat(),
                    entry_price,
                    exit_date.isoformat(),
                    exit_price,
                    realised_return_pct,
                    holding_days,
                    status,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
    finally:
        conn.close()


def track_record(
    mechanism_id: str,
    *,
    lookback_days: Optional[int] = None,
    db_path: Path | str | None = None,
) -> TrackRecord:
    """Compute aggregate stats for a mechanism's history."""
    conn = _conn(db_path)
    try:
        params: list[Any] = [mechanism_id]
        ts_filter = ""
        if lookback_days is not None:
            cutoff = (
                datetime.now(timezone.utc) - timedelta(days=int(lookback_days))
            ).isoformat()
            ts_filter = "AND s.asof >= ?"
            params.append(cutoff)

        rows = conn.execute(
            f"""
            SELECT s.id, s.asof, s.regime_at_signal,
                   o.realised_return_pct, o.status
            FROM mechanism_signals s
            LEFT JOIN mechanism_outcomes o ON o.signal_id = s.id
            WHERE s.mechanism_id = ? {ts_filter}
            ORDER BY s.asof
            """,
            params,
        ).fetchall()

        n_signals = len(rows)
        completed = [r for r in rows if r[3] is not None]
        n_completed = len(completed)
        n_open = n_signals - n_completed

        win_rate: Optional[float] = None
        mean_ret: Optional[float] = None
        median_ret: Optional[float] = None
        sharpe: Optional[float] = None
        max_loss: Optional[float] = None
        max_consec_losses = 0
        last_signal_at: Optional[datetime] = None

        if n_completed:
            rets = sorted(r[3] for r in completed)
            wins = sum(1 for r in rets if r > 0)
            win_rate = wins / n_completed
            mean_ret = sum(rets) / n_completed
            median_ret = (
                rets[n_completed // 2]
                if n_completed % 2
                else 0.5 * (rets[n_completed // 2 - 1] + rets[n_completed // 2])
            )
            if n_completed >= 2:
                var = sum((r - mean_ret) ** 2 for r in rets) / (n_completed - 1)
                std = var ** 0.5
                sharpe = (mean_ret / std) if std > 0 else 0.0
            max_loss = min(rets)

            # Consecutive loss streak
            cur_streak = 0
            best_streak = 0
            for r in [c[3] for c in completed]:
                if r < 0:
                    cur_streak += 1
                    best_streak = max(best_streak, cur_streak)
                else:
                    cur_streak = 0
            max_consec_losses = best_streak

        if n_signals:
            last_signal_at = datetime.fromisoformat(rows[-1][1])

        # Best / worst regime by mean return per regime
        best_regime: Optional[str] = None
        worst_regime: Optional[str] = None
        regime_to_rets: dict[str, list[float]] = {}
        for sid, asof, regime_json, ret_pct, status in completed:
            try:
                regime_obj = json.loads(regime_json) if regime_json else {}
            except json.JSONDecodeError:
                regime_obj = {}
            regime_label = regime_obj.get("regime") or "unknown"
            regime_to_rets.setdefault(regime_label, []).append(ret_pct)
        if regime_to_rets:
            regime_means = {
                k: sum(v) / len(v) for k, v in regime_to_rets.items() if v
            }
            if regime_means:
                best_regime = max(regime_means, key=regime_means.get)
                worst_regime = min(regime_means, key=regime_means.get)

        return TrackRecord(
            mechanism_id=mechanism_id,
            n_signals=n_signals,
            n_completed=n_completed,
            n_open=n_open,
            win_rate=win_rate,
            mean_return_pct=mean_ret,
            median_return_pct=median_ret,
            sharpe_per_trade=sharpe,
            max_loss_pct=max_loss,
            max_consecutive_losses=max_consec_losses,
            last_signal_at=last_signal_at,
            best_regime=best_regime,
            worst_regime=worst_regime,
        )
    finally:
        conn.close()


def signals_for_asset(
    asset: str,
    *,
    asof: Optional[datetime] = None,
    lookback_days: int = 30,
    db_path: Path | str | None = None,
) -> list[dict]:
    """Recent signals from any mechanism on a given asset."""
    asof = asof or datetime.now(timezone.utc)
    cutoff = (asof - timedelta(days=lookback_days)).isoformat()
    conn = _conn(db_path)
    try:
        rows = conn.execute(
            "SELECT id, mechanism_id, asof, asset, direction, raw_size_pct, "
            "       confidence, rationale, regime_at_signal "
            "FROM mechanism_signals "
            "WHERE asset = ? AND asof >= ? AND asof <= ? "
            "ORDER BY asof DESC",
            (asset, cutoff, asof.isoformat()),
        ).fetchall()
        return [
            {
                "id": r[0],
                "mechanism_id": r[1],
                "asof": r[2],
                "asset": r[3],
                "direction": r[4],
                "raw_size_pct": r[5],
                "confidence": r[6],
                "rationale": r[7],
                "regime_at_signal": json.loads(r[8]) if r[8] else {},
            }
            for r in rows
        ]
    finally:
        conn.close()
