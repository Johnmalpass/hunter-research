"""OpenFDA FAERS adapter — FDA Adverse Event Reporting System.

NO API KEY REQUIRED for low-rate access. Optional free key (just an email,
240 req/min, 120k req/day) at https://open.fda.gov/apis/authentication/.
Add to .env if you have one:
    OPENFDA_API_KEY=your_key

Why this is in the system: FAERS is one of the cleanest 'no finance person
reads this' silos. Adverse-event spikes for a given drug or manufacturer
are read by clinical-pharmacology researchers and FDA staff; almost never by
the equity analyst covering the manufacturer's stock or the healthcare REIT
that owns the operator's facilities. Cross-silo collisions through FAERS
should produce diamond-tier compositions.

Schema:
    asset_id = drug brand_name or generic_name (uppercased)
    field    = faers_reports_count_per_day
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Iterable

from quant.data.base import BaseAdapter, DataPoint


class FaersAdapter(BaseAdapter):
    name = "faers"
    rate_limit_per_minute = 60  # generous; openFDA free is 240/min unauthenticated

    BASE_URL = "https://api.fda.gov/drug/event.json"

    def __init__(self, api_key: str | None = None, db_path=None):
        super().__init__(db_path)
        self.api_key = api_key or os.environ.get("OPENFDA_API_KEY")  # optional

    def fetch_drug_event_counts(
        self,
        drug: str,
        date_from: str = "2020-01-01",
        date_to: str | None = None,
    ) -> Iterable[DataPoint]:
        """Daily count of adverse-event reports for one drug.

        Uses openFDA's `count` aggregation. Returns one DataPoint per day with
        non-zero reports.
        """
        import requests

        date_to = date_to or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        date_from_ymd = date_from.replace("-", "")
        date_to_ymd = date_to.replace("-", "")

        # Search either generic_name or brand_name (case-insensitive on field)
        search = (
            f'(patient.drug.openfda.generic_name:"{drug}" OR '
            f'patient.drug.openfda.brand_name:"{drug}") AND '
            f"receivedate:[{date_from_ymd} TO {date_to_ymd}]"
        )
        params = {"search": search, "count": "receivedate"}
        if self.api_key:
            params["api_key"] = self.api_key

        self._throttle()
        r = requests.get(self.BASE_URL, params=params, timeout=30)

        # FAERS returns 404 when the search has zero results — treat as empty
        if r.status_code == 404:
            return
        r.raise_for_status()
        results = r.json().get("results") or []

        asset_id = drug.upper()
        for entry in results:
            time_str = entry.get("time")
            count = entry.get("count")
            if not time_str or count is None:
                continue
            try:
                ts = datetime.strptime(time_str, "%Y%m%d").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            yield DataPoint(
                ts,
                asset_id,
                "faers_reports_count_1d",
                int(count),
                self.name,
                {"query_drug": drug},
            )

    def fetch(
        self,
        drugs: list[str] | None = None,
        date_from: str = "2020-01-01",
        date_to: str | None = None,
    ) -> Iterable[DataPoint]:
        for d in drugs or []:
            yield from self.fetch_drug_event_counts(d, date_from, date_to)
