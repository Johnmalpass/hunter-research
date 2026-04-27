"""Mechanism Compiler — convert HUNTER theses into executable Python.

A `Mechanism` is the operational form of a HUNTER thesis. It is a pure,
deterministic function that, given point-in-time `MarketState`, returns
`Signal`s the trader can act on.

  HUNTER thesis (English) -> Mechanism (this file's subclass) -> Backtest -> Live

This separation matters because:

  - The LLM-written thesis is suggestive but not executable. The compiled
    mechanism is executable but not LLM-dependent at runtime. We compile
    once, run a million times.
  - A mechanism is testable. Same MarketState in -> same Signal out. We
    can backtest 30 years of history without invoking any LLM call.
  - The same mechanism object runs in both backtest mode (historical
    state) and live mode (today's state). One implementation, two contexts.

Subclasses set `thesis_id`, `name`, `universe` (tickers/series the trade
touches), `required_fields` (data the mechanism reads from MarketState),
`holding_period_days`, and `params` (tunable thresholds). They override
`evaluate(state, asof) -> list[Signal]`.

A `MechanismRequirement` describes what data the mechanism needs and where
to get it. The CLI's `quant backtest --thesis X` reads this and prints
clean instructions if the data isn't ingested yet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from quant.data.base import MarketState


@dataclass
class Signal:
    """A single trade decision emitted by a Mechanism."""

    asset: str
    direction: str  # "long" | "short" | "exit"
    size_pct: float  # of NAV — final size set by sizing layer
    confidence: float  # 0.0 to 1.0
    holding_period_days: int
    rationale: str
    asof: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MechanismRequirement:
    """Declares one piece of data a mechanism needs from MarketState."""

    asset_id: str
    field: str
    suggested_adapter: str  # name of the quant.data adapter that supplies this
    note: str = ""


@dataclass(kw_only=True)
class Mechanism:
    """Base class. Each HUNTER thesis becomes one subclass.

    Override `evaluate(state, asof)` to return zero or more Signals.
    Override class attributes to declare metadata and data requirements.
    """

    thesis_id: str
    name: str
    universe: list[str]
    requirements: list[MechanismRequirement]
    holding_period_days: int = 90
    direction: str = "both"  # "long" | "short" | "both"
    params: dict[str, float] = field(default_factory=dict)
    description: str = ""

    def check_data(self, state: MarketState) -> dict[str, int]:
        """Return {requirement_key: row_count} for each declared requirement.

        A row_count of 0 means the mechanism cannot run; the CLI surfaces
        the suggested_adapter so the user can ingest the missing data.
        """
        from datetime import datetime as _dt, timezone as _tz

        out: dict[str, int] = {}
        # historical floor; we only care about presence
        floor = _dt(1990, 1, 1, tzinfo=_tz.utc)
        for req in self.requirements:
            history = state.history(req.asset_id, req.field, floor)
            out[f"{req.asset_id}/{req.field}"] = len(history)
        return out

    def evaluate(self, state: MarketState, asof: datetime) -> list[Signal]:
        """Override in subclass. Return list of signals (possibly empty)."""
        raise NotImplementedError


# ============================================================
# Registry of compiled mechanisms
# ============================================================

_MECHANISMS: dict[str, type[Mechanism]] = {}


def register(thesis_id: str):
    """Decorator: register a Mechanism subclass under its thesis_id."""

    def wrap(cls: type[Mechanism]) -> type[Mechanism]:
        _MECHANISMS[thesis_id] = cls
        return cls

    return wrap


def get_mechanism(thesis_id: str) -> type[Mechanism]:
    if thesis_id not in _MECHANISMS:
        raise KeyError(
            f"No mechanism registered for thesis_id={thesis_id}. "
            f"Available: {sorted(_MECHANISMS)}"
        )
    return _MECHANISMS[thesis_id]


def list_mechanisms() -> list[str]:
    return sorted(_MECHANISMS)


def _import_all_mechanisms():
    """Import every module in quant.research.mechanisms so registration runs."""
    import importlib
    import pkgutil

    from quant.research import mechanisms as pkg

    for _, modname, _ in pkgutil.iter_modules(pkg.__path__):
        importlib.import_module(f"{pkg.__name__}.{modname}")
