"""Regime detector: classify the macro environment into 4 states.

Inputs (from FRED, ingested via quant.data.adapters.fred):
    DGS10, DGS2     -> yield-curve slope (10y minus 2y)
    BAA10Y          -> Moody's Baa corporate yield minus 10y Treasury
                       (credit-stress proxy with deep history back to 1986;
                       used in preference to BAA10Y because FRED's
                       BAA10Y series was re-indexed and now only
                       returns post-2023 data via the API)

Output:
    RegimeState with one of {risk_on, late_cycle, risk_off, crisis}
    + soft probabilities over all four
    + the raw inputs and their 5-year percentile ranks

Method
======

For each of the two macro variables we compute its 5-year rolling percentile
rank (where does today sit in the historical distribution?). We then
soft-classify by Gaussian similarity to four prototype centres in
(yield_curve_pct, hy_spread_pct) space:

    risk_on      (0.70, 0.25)   steep curve, tight spreads
    late_cycle   (0.20, 0.40)   flat/inverted curve, modest spreads
    risk_off     (0.20, 0.75)   flat/inverted curve, wide spreads
    crisis       (0.10, 0.95)   inverted curve, panic spreads

The softmax is sharp enough that the highest-probability regime usually has
> 0.6 mass; transitions show two regimes near 0.4 each, a useful signal.

Why this matters
================

Mechanisms compiled from HUNTER theses cannot know that they were written
in a different macro regime than the one they're now firing in. The regime
detector is the macro overlay every mechanism reads. Mechanisms can:

  - declare which regimes they are active in (RegimePredicate)
  - scale position size by regime probability (sizing layer)
  - turn themselves off when the regime they were calibrated in disappears

The detector is a pure function of MarketState. Same data -> same regime.
No LLM. No internet calls at evaluation time.

Future extension: regime-conditional synergy. Compute the synergistic-info
score II(score; A; B | regime=R) separately within each regime. Some
compositions are synergistic in risk-on but not in risk-off; the framework
should recognise this and gate accordingly. That's the next-turn build.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from quant.data.base import MarketState


REGIMES = ("risk_on", "late_cycle", "risk_off", "crisis")

# Prototype centres: (yield_curve_5y_percentile, high_yield_spread_5y_percentile)
PROTOTYPES: dict[str, tuple[float, float]] = {
    "risk_on":    (0.70, 0.25),
    "late_cycle": (0.20, 0.40),
    "risk_off":   (0.20, 0.75),
    "crisis":     (0.10, 0.95),
}


@dataclass
class RegimeState:
    asof: datetime
    regime: str
    probabilities: dict[str, float]
    inputs: dict[str, float]

    def prob(self, regime_name: str) -> float:
        return self.probabilities.get(regime_name, 0.0)

    @property
    def is_risk_off_or_crisis(self) -> bool:
        return self.regime in ("risk_off", "crisis")

    def to_dict(self) -> dict:
        return {
            "asof": self.asof.isoformat(),
            "regime": self.regime,
            "probabilities": self.probabilities,
            "inputs": self.inputs,
        }


class RegimeDetector:
    """Detect the macro regime from FRED data already in MarketState."""

    def __init__(
        self,
        state: MarketState,
        rolling_window_years: int = 5,
        softmax_sharpness: float = 6.0,
    ):
        self.state = state
        self.rolling_window_years = rolling_window_years
        self.softmax_sharpness = softmax_sharpness

    # ── helpers ──────────────────────────────────────────────────────────

    def _yield_curve_slope_history(
        self, asof: datetime, years: int
    ) -> list[tuple[datetime, float]]:
        """Pair DGS10 and DGS2 by date and compute the slope at each."""
        floor = asof - timedelta(days=int(365.25 * years))
        ten = self.state.history("DGS10", "value", floor, asof)
        two = self.state.history("DGS2", "value", floor, asof)
        ten_by_date = {p.timestamp.date(): float(p.value) for p in ten}
        two_by_date = {p.timestamp.date(): float(p.value) for p in two}
        common = sorted(set(ten_by_date) & set(two_by_date))
        return [
            (datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc),
             ten_by_date[d] - two_by_date[d])
            for d in common
        ]

    def _hy_spread_history(self, asof: datetime, years: int) -> list[float]:
        floor = asof - timedelta(days=int(365.25 * years))
        history = self.state.history("BAA10Y", "value", floor, asof)
        return [float(p.value) for p in history]

    @staticmethod
    def _percentile_rank(value: float, sample: list[float]) -> float:
        if not sample:
            return 0.5
        rank = sum(1 for v in sample if v <= value)
        return rank / len(sample)

    # ── main entrypoint ──────────────────────────────────────────────────

    def detect(self, asof: Optional[datetime] = None) -> Optional[RegimeState]:
        asof = asof or datetime.now(timezone.utc)

        yc_history = self._yield_curve_slope_history(asof, self.rolling_window_years)
        hy_history = self._hy_spread_history(asof, self.rolling_window_years)

        if not yc_history or not hy_history:
            return None

        yc_today = yc_history[-1][1]
        hy_today = hy_history[-1]
        yc_pct = self._percentile_rank(yc_today, [v for _, v in yc_history])
        hy_pct = self._percentile_rank(hy_today, hy_history)

        # Softmax over negative squared distance to each prototype
        scores: dict[str, float] = {}
        for name, (cy, ch) in PROTOTYPES.items():
            d2 = (yc_pct - cy) ** 2 + (hy_pct - ch) ** 2
            scores[name] = -d2

        max_s = max(scores.values())
        exps = {
            k: math.exp((v - max_s) * self.softmax_sharpness)
            for k, v in scores.items()
        }
        total = sum(exps.values())
        probs = {k: v / total for k, v in exps.items()}
        regime = max(probs, key=probs.get)

        return RegimeState(
            asof=asof,
            regime=regime,
            probabilities=probs,
            inputs={
                "yield_curve_slope": yc_today,
                "yield_curve_pct_5y": yc_pct,
                "high_yield_spread": hy_today,
                "high_yield_spread_pct_5y": hy_pct,
            },
        )

    # ── batch utilities ──────────────────────────────────────────────────

    def detect_history(
        self,
        start: datetime,
        end: Optional[datetime] = None,
        step_days: int = 7,
    ) -> list[RegimeState]:
        """Build a regime-time-series by re-running detect() at each step."""
        end = end or datetime.now(timezone.utc)
        out: list[RegimeState] = []
        cur = start
        while cur <= end:
            rs = self.detect(cur)
            if rs is not None:
                out.append(rs)
            cur += timedelta(days=step_days)
        return out
