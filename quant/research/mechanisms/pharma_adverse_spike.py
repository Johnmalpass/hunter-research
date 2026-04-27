"""Mechanism: pharma adverse-event spike → manufacturer short.

Cross-silo composition: FDA FAERS adverse-event reports + equity prices.

Signal logic
============

For a target drug-manufacturer pair (e.g. atorvastatin -> a generics maker
or a single-product biotech), we compute a rolling z-score of daily
adverse-event report counts:

    z(t) = (count(t) - mean_90d(count)) / std_90d(count)

When z(t) >= z_threshold and the recent count has been elevated for at
least `min_elevated_days` of the last `lookback_days`, we emit a SHORT
signal on the manufacturer's stock with a `holding_period_days` window.

Cross-silo edge
===============

FAERS is read by clinical pharmacologists and FDA reviewers; almost never
by the equity analyst covering the manufacturer's stock or the healthcare
REIT that owns the operator's facilities. A spike that looks routine from
either silo alone is jointly informative when both are read.

Tunable params
==============

  z_threshold              z-score that triggers a short
  lookback_days            window over which we measure elevation
  min_elevated_days        days of z >= 1 needed within lookback
  size_pct                 % of NAV per name
  rolling_window_days      moving-window size for mean/std (default 90)

This file is the template. The CLI registers it under thesis_id
"pharma_adverse_spike" and the backtest harness can run it on any
(drug, ticker) pair you've ingested.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from quant.data.base import MarketState
from quant.research.mechanism import (
    Mechanism,
    MechanismRequirement,
    Signal,
    register,
)


def _z_score_of_today(
    state: MarketState,
    drug: str,
    asof: datetime,
    rolling_window_days: int,
) -> Optional[tuple[float, float, float]]:
    """Return (today_count, rolling_mean, rolling_std). None if insufficient data."""
    floor = asof - timedelta(days=rolling_window_days)
    history = state.history(drug, "faers_reports_count_1d", floor, asof)
    if len(history) < 30:
        return None
    counts = [float(p.value) for p in history]
    today_count = counts[-1]
    prior = counts[:-1]
    if len(prior) < 30:
        return None
    mu = statistics.fmean(prior)
    sigma = statistics.pstdev(prior) or 1e-9
    return today_count, mu, sigma


@register("pharma_adverse_spike")
@dataclass(kw_only=True)
class PharmaAdverseSpikeMechanism(Mechanism):
    """Concentration-flag spike on FAERS -> short the manufacturer.

    Construct with `drug` and `manufacturer_ticker`. e.g.

        m = PharmaAdverseSpikeMechanism(
            drug="HUMIRA",
            manufacturer_ticker="ABBV",
        )

    Other params have safe defaults but should be tuned per drug.
    """

    # Class-level defaults for the inherited Mechanism fields, so the dataclass
    # init does not require them as positional kwargs. __post_init__ then
    # overrides them based on the chosen drug/manufacturer.
    thesis_id: str = "pharma_adverse_spike"
    name: str = "FAERS adverse-event spike -> manufacturer short"
    universe: list[str] | None = None
    requirements: list[MechanismRequirement] | None = None
    holding_period_days: int = 30
    direction: str = "short"
    description: str = (
        "Adverse-event count z-score above threshold + sustained elevation "
        "-> short the manufacturer for holding_period_days."
    )

    drug: str = "HUMIRA"
    manufacturer_ticker: str = "ABBV"

    def __post_init__(self) -> None:
        self.thesis_id = (
            f"pharma_adverse_spike_{self.drug.lower()}_"
            f"{self.manufacturer_ticker.lower()}"
        )
        self.name = (
            f"FAERS adverse-event spike on {self.drug} -> short "
            f"{self.manufacturer_ticker}"
        )
        self.universe = [self.manufacturer_ticker]
        self.requirements = [
            MechanismRequirement(
                asset_id=self.drug,
                field="faers_reports_count_1d",
                suggested_adapter="faers",
                note=f"python -m quant ingest faers --drugs {self.drug}",
            ),
        ]
        self.params.setdefault("z_threshold", 2.0)
        self.params.setdefault("lookback_days", 14)
        self.params.setdefault("min_elevated_days", 5)
        self.params.setdefault("rolling_window_days", 90)
        self.params.setdefault("size_pct", 0.03)

    def evaluate(self, state: MarketState, asof: datetime) -> list[Signal]:
        z_info = _z_score_of_today(
            state, self.drug, asof, int(self.params["rolling_window_days"])
        )
        if z_info is None:
            return []
        today, mu, sigma = z_info
        z = (today - mu) / sigma if sigma > 0 else 0.0
        if z < self.params["z_threshold"]:
            return []

        # Sustained elevation check
        floor = asof - timedelta(days=int(self.params["lookback_days"]))
        recent = state.history(
            self.drug, "faers_reports_count_1d", floor, asof
        )
        if len(recent) < int(self.params["min_elevated_days"]):
            return []
        elevated_count = sum(
            1 for p in recent if (float(p.value) - mu) / sigma >= 1.0
        )
        if elevated_count < int(self.params["min_elevated_days"]):
            return []

        return [
            Signal(
                asset=self.manufacturer_ticker,
                direction="short",
                size_pct=float(self.params["size_pct"]),
                confidence=min(1.0, z / 4.0),
                holding_period_days=int(self.holding_period_days),
                rationale=(
                    f"FAERS {self.drug} z={z:.2f} (today={today:.0f}, "
                    f"mu_90d={mu:.1f}, sigma={sigma:.1f}); "
                    f"elevated for {elevated_count}/"
                    f"{int(self.params['lookback_days'])} days"
                ),
                asof=asof,
                metadata={
                    "drug": self.drug,
                    "z_score": z,
                    "rolling_mean": mu,
                    "rolling_std": sigma,
                },
            )
        ]
