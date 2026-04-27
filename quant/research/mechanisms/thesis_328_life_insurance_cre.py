"""Mechanism: thesis #328 — life insurance CRE credit-risk underpricing.

From docs/diamond_theses.md (diamond score 97):

  Life insurers reserve against CRE credit risk at ~0.43% default experience
  while CMBS office delinquency is running 12.34% in the same loan
  population. Reserve refresh cycles lag the market; the underpricing
  compounds across carriers because every carrier uses the same AA
  corporate yield curve construction input.

Cross-silo composition
======================

  CMBS servicer reports (silo A: structured finance / CRE credit)
  AA corporate yield curve (silo B: fixed-income index construction)
  NAIC schedule D + statutory filings (silo C: insurance regulation)
  Life insurer reserve adequacy (silo D: actuarial)

Trigger
=======

When CMBS office delinquency is high AND the AA yield curve has widened
materially relative to the Treasury 10Y AND a Q1 statutory filing date is
within `filing_window_days`, life insurers are systematically under-reserved.
Short a basket of life insurers for `holding_period_days`.

Required data
=============

  DRCRELACBS  FRED  delinquency rate on commercial real estate loans
  BAMLC0A2CAA FRED  ICE BofA AA US corporate index effective yield
  DGS10       FRED  10-year Treasury constant-maturity yield
  Manufacturer prices (MET, PRU, LNC, AFL): yfinance via the backtest harness

If FRED data is not in the store, the mechanism's `check_data()` will
report missing rows and the CLI will print:

    python -m quant ingest fred --series DRCRELACBS,BAMLC0A2CAA,DGS10

This is the template for the Mechanism Compiler: the same shape applies to
every diamond thesis. Each one becomes a file like this, with declared
requirements, parametric thresholds, and an `evaluate(state, asof)` method.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from quant.data.base import MarketState
from quant.research.mechanism import (
    Mechanism,
    MechanismRequirement,
    Signal,
    register,
)

# Q1 / Q2 / Q3 / Q4 statutory filing dates (US life insurance)
QUARTERLY_FILING_DATES = [
    (5, 15),    # Q1 due May 15
    (8, 14),    # Q2 due August 14
    (11, 14),   # Q3 due November 14
    (3, 1),     # Q4 / annual due March 1
]


def _days_to_next_filing(asof: datetime) -> int:
    candidates = []
    for month, day in QUARTERLY_FILING_DATES:
        for year_offset in (0, 1):
            try:
                d = datetime(asof.year + year_offset, month, day, tzinfo=asof.tzinfo)
            except ValueError:
                continue
            delta = (d - asof).days
            if delta >= 0:
                candidates.append(delta)
    return min(candidates) if candidates else 365


@register("thesis_328")
@dataclass(kw_only=True)
class Thesis328LifeInsuranceCre(Mechanism):
    """Short basket of life insurers when CMBS distress + AA-Treasury spread
    + filing window all align.
    """

    thesis_id: str = "thesis_328"
    name: str = "Life insurance CRE credit-risk underpricing"
    universe: list[str] | None = None
    requirements: list[MechanismRequirement] | None = None
    holding_period_days: int = 120
    direction: str = "short"
    description: str = "Diamond #328: short MET/PRU/LNC/AFL when CMBS delinq high, AA spread widened, filing date within window."

    def __post_init__(self) -> None:
        self.universe = list(self.universe or ["MET", "PRU", "LNC", "AFL"])
        self.requirements = list(
            self.requirements
            or [
                MechanismRequirement(
                    asset_id="DRCRELACBS",
                    field="value",
                    suggested_adapter="fred",
                    note="python -m quant ingest fred --series DRCRELACBS",
                ),
                MechanismRequirement(
                    asset_id="BAMLC0A2CAA",
                    field="value",
                    suggested_adapter="fred",
                    note="python -m quant ingest fred --series BAMLC0A2CAA",
                ),
                MechanismRequirement(
                    asset_id="DGS10",
                    field="value",
                    suggested_adapter="fred",
                    note="python -m quant ingest fred --series DGS10",
                ),
            ]
        )
        # Default params — published values from the thesis
        self.params.setdefault("delinq_threshold_pct", 6.0)
        self.params.setdefault("aa_treasury_spread_bps_min", 100.0)
        self.params.setdefault("filing_window_days", 60)
        self.params.setdefault("size_per_name_pct", 0.02)

    def evaluate(self, state: MarketState, asof: datetime) -> list[Signal]:
        delinq_pt = state.latest_as_of("DRCRELACBS", "value", asof)
        aa_pt = state.latest_as_of("BAMLC0A2CAA", "value", asof)
        t10_pt = state.latest_as_of("DGS10", "value", asof)
        if not (delinq_pt and aa_pt and t10_pt):
            return []

        delinq = float(delinq_pt.value)
        if delinq < self.params["delinq_threshold_pct"]:
            return []

        aa = float(aa_pt.value)
        t10 = float(t10_pt.value)
        spread_bps = (aa - t10) * 100.0
        if spread_bps < self.params["aa_treasury_spread_bps_min"]:
            return []

        days_to_filing = _days_to_next_filing(asof)
        if days_to_filing > int(self.params["filing_window_days"]):
            return []

        rationale = (
            f"CMBS office delinq {delinq:.2f}% (>= {self.params['delinq_threshold_pct']}); "
            f"AA-T10 spread {spread_bps:.0f}bps (>= {self.params['aa_treasury_spread_bps_min']}); "
            f"days to next statutory filing {days_to_filing} (<= "
            f"{self.params['filing_window_days']})"
        )

        return [
            Signal(
                asset=ticker,
                direction="short",
                size_pct=float(self.params["size_per_name_pct"]),
                confidence=0.7,
                holding_period_days=self.holding_period_days,
                rationale=rationale,
                asof=asof,
                metadata={
                    "cmbs_delinq_pct": delinq,
                    "aa_treasury_spread_bps": spread_bps,
                    "days_to_filing": days_to_filing,
                },
            )
            for ticker in self.universe
        ]
