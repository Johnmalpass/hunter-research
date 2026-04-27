"""Predicates: Level-1 building blocks for compiled mechanisms.

Each predicate is a callable

    pred(state: MarketState, asof: datetime) -> Optional[PredicateResult]

returning None when required data is missing, or a `PredicateResult` with:

    fired       bool   did the predicate condition hold
    magnitude   float  by how much (z-score, spread, days, probability)
    evidence    dict   raw values that fed the decision

Predicates compose via `And`, `Or`, `Not`, `Within`. A mechanism that
expresses its logic entirely in predicates + combinators is "DSL-compiled":
auditable, fast, easy for an LLM to generate without writing free-form
Python.

The L1 + L2 layers cover the majority of HUNTER thesis logic (z-scores,
threshold crosses, spread widenings, regime gating, calendar windows). When
a thesis genuinely needs custom logic, free-form Python (Level 3) is the
escape hatch — and free-form mechanisms are encouraged to call predicates
as building blocks anyway.

Available predicates (Level 1):
    ThresholdPredicate        latest value of (asset, field) vs constant
    ZScorePredicate           z(today) over rolling window vs threshold
    SpreadPredicate           (a - b) * scale vs threshold
    RegimePredicate           current regime in allowed set with min prob
    WithinDaysOfPredicate     within N days of an annual calendar event

Combinators (Level 2):
    And, Or, Not              logical composition; missing data short-circuits
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional, Sequence

from quant.data.base import MarketState


@dataclass
class PredicateResult:
    fired: bool
    magnitude: float = 0.0
    evidence: dict = field(default_factory=dict)


Predicate = Callable[[MarketState, datetime], Optional[PredicateResult]]


# ============================================================
# Level 1: leaf predicates
# ============================================================

class ThresholdPredicate:
    """Latest value of (asset, field) compared to a fixed threshold."""

    SUPPORTED_OPS = (">=", ">", "<=", "<", "==")

    def __init__(self, asset_id: str, field: str, threshold: float, op: str = ">="):
        if op not in self.SUPPORTED_OPS:
            raise ValueError(f"unsupported op: {op}")
        self.asset_id = asset_id
        self.field = field
        self.threshold = float(threshold)
        self.op = op

    def __call__(self, state: MarketState, asof: datetime) -> Optional[PredicateResult]:
        pt = state.latest_as_of(self.asset_id, self.field, asof)
        if pt is None or pt.value is None:
            return None
        try:
            v = float(pt.value)
        except (TypeError, ValueError):
            return None
        cmp = {
            ">=": v >= self.threshold,
            ">":  v >  self.threshold,
            "<=": v <= self.threshold,
            "<":  v <  self.threshold,
            "==": v == self.threshold,
        }
        return PredicateResult(
            fired=cmp[self.op],
            magnitude=v - self.threshold,
            evidence={
                "asset": self.asset_id, "field": self.field,
                "value": v, "threshold": self.threshold, "op": self.op,
            },
        )


class ZScorePredicate:
    """Today's value z-scores above threshold over a rolling window."""

    def __init__(
        self,
        asset_id: str,
        field: str,
        threshold: float,
        window_days: int = 90,
        min_obs: int = 30,
    ):
        self.asset_id = asset_id
        self.field = field
        self.threshold = float(threshold)
        self.window_days = int(window_days)
        self.min_obs = int(min_obs)

    def __call__(self, state: MarketState, asof: datetime) -> Optional[PredicateResult]:
        floor = asof - timedelta(days=self.window_days)
        history = state.history(self.asset_id, self.field, floor, asof)
        if len(history) < self.min_obs:
            return None
        try:
            values = [float(p.value) for p in history]
        except (TypeError, ValueError):
            return None
        today = values[-1]
        prior = values[:-1]
        if len(prior) < 2:
            return None
        n = len(prior)
        mean = sum(prior) / n
        var = sum((v - mean) ** 2 for v in prior) / n
        std = (var ** 0.5) or 1e-9
        z = (today - mean) / std
        return PredicateResult(
            fired=z >= self.threshold,
            magnitude=z,
            evidence={
                "asset": self.asset_id, "field": self.field,
                "today": today, "mean": mean, "std": std, "z": z,
                "window_days": self.window_days,
            },
        )


class SpreadPredicate:
    """(value(a) - value(b)) * scale compared to a threshold."""

    def __init__(
        self,
        a_asset: str, a_field: str,
        b_asset: str, b_field: str,
        threshold: float,
        scale: float = 1.0,
        op: str = ">=",
    ):
        if op not in (">=", ">", "<=", "<"):
            raise ValueError(f"unsupported op: {op}")
        self.a = (a_asset, a_field)
        self.b = (b_asset, b_field)
        self.threshold = float(threshold)
        self.scale = float(scale)
        self.op = op

    def __call__(self, state: MarketState, asof: datetime) -> Optional[PredicateResult]:
        pa = state.latest_as_of(self.a[0], self.a[1], asof)
        pb = state.latest_as_of(self.b[0], self.b[1], asof)
        if not (pa and pb):
            return None
        try:
            spread = (float(pa.value) - float(pb.value)) * self.scale
        except (TypeError, ValueError):
            return None
        cmp = {
            ">=": spread >= self.threshold,
            ">":  spread >  self.threshold,
            "<=": spread <= self.threshold,
            "<":  spread <  self.threshold,
        }
        return PredicateResult(
            fired=cmp[self.op],
            magnitude=spread,
            evidence={
                "a": pa.value, "b": pb.value,
                "spread": spread, "scale": self.scale,
                "threshold": self.threshold,
            },
        )


class RegimePredicate:
    """Current regime is one of `allowed_regimes` with prob >= min_probability."""

    def __init__(self, allowed_regimes: Sequence[str], min_probability: float = 0.5):
        self.allowed = set(allowed_regimes)
        self.min_probability = float(min_probability)

    def __call__(self, state: MarketState, asof: datetime) -> Optional[PredicateResult]:
        # Lazy import to avoid cycle with regime.py
        from quant.research.regime import RegimeDetector

        det = RegimeDetector(state)
        rs = det.detect(asof)
        if rs is None:
            return None
        prob_in_allowed = sum(rs.prob(r) for r in self.allowed)
        return PredicateResult(
            fired=prob_in_allowed >= self.min_probability,
            magnitude=prob_in_allowed,
            evidence={
                "regime": rs.regime,
                "probabilities": rs.probabilities,
                "allowed": sorted(self.allowed),
                "prob_in_allowed": prob_in_allowed,
            },
        )


class WithinDaysOfPredicate:
    """`asof` is within `max_days` of an annual calendar event (month, day)."""

    def __init__(self, calendar_days: Sequence[tuple[int, int]], max_days: int):
        self.calendar = list(calendar_days)
        self.max_days = int(max_days)

    def __call__(self, state: MarketState, asof: datetime) -> Optional[PredicateResult]:
        candidates: list[int] = []
        for month, day in self.calendar:
            for offset in (0, 1):
                try:
                    d = datetime(
                        asof.year + offset, month, day, tzinfo=asof.tzinfo
                    )
                except ValueError:
                    continue
                delta = (d - asof).days
                if delta >= 0:
                    candidates.append(delta)
        if not candidates:
            return PredicateResult(
                fired=False,
                magnitude=0.0,
                evidence={"days_to_next": None},
            )
        days = min(candidates)
        return PredicateResult(
            fired=days <= self.max_days,
            magnitude=float(days),
            evidence={"days_to_next": days, "max_days": self.max_days},
        )


# ============================================================
# Level 2: combinators
# ============================================================

class And:
    def __init__(self, *children: Predicate):
        self.children = list(children)

    def __call__(self, state: MarketState, asof: datetime) -> Optional[PredicateResult]:
        evid: dict = {}
        results: list[PredicateResult] = []
        for i, c in enumerate(self.children):
            r = c(state, asof)
            if r is None:
                return None  # short-circuit on missing data
            results.append(r)
            evid[f"child_{i}"] = {"fired": r.fired, "magnitude": r.magnitude}
        fired = all(r.fired for r in results)
        mags = [r.magnitude for r in results if r.magnitude is not None]
        return PredicateResult(
            fired=fired,
            magnitude=min(mags) if mags else 0.0,
            evidence=evid,
        )


class Or:
    def __init__(self, *children: Predicate):
        self.children = list(children)

    def __call__(self, state: MarketState, asof: datetime) -> Optional[PredicateResult]:
        results: list[PredicateResult] = []
        for c in self.children:
            r = c(state, asof)
            if r is not None:
                results.append(r)
        if not results:
            return None
        fired = any(r.fired for r in results)
        mags = [r.magnitude for r in results if r.magnitude is not None]
        return PredicateResult(
            fired=fired,
            magnitude=max(mags) if mags else 0.0,
            evidence={f"child_{i}": {"fired": r.fired} for i, r in enumerate(results)},
        )


class Not:
    def __init__(self, child: Predicate):
        self.child = child

    def __call__(self, state: MarketState, asof: datetime) -> Optional[PredicateResult]:
        r = self.child(state, asof)
        if r is None:
            return None
        return PredicateResult(
            fired=not r.fired,
            magnitude=-r.magnitude,
            evidence={"inner_fired": r.fired},
        )
