"""FRED (Federal Reserve Economic Data) adapter.

Free API. Sign up for an API key (free, instant) at:
    https://fred.stlouisfed.org/docs/api/api_key.html

Add to .env:
    FRED_API_KEY=your_key_here

Used to ingest macro time series — yield curve, credit spreads, CPI, payrolls,
Fed balance sheet, mortgage rates, the broad dollar. These are the canonical
inputs the TRADER and AUDITOR agents need to track for any HUNTER mechanism
that touches macro context.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Iterable

from quant.data.base import BaseAdapter, DataPoint

# Curated baseline. Add more as new HUNTER mechanisms reference them.
DEFAULT_SERIES: dict[str, str] = {
    # Yield curve
    "DGS1MO": "1-month Treasury",
    "DGS3MO": "3-month Treasury",
    "DGS6MO": "6-month Treasury",
    "DGS1": "1-year Treasury",
    "DGS2": "2-year Treasury",
    "DGS5": "5-year Treasury",
    "DGS10": "10-year Treasury",
    "DGS30": "30-year Treasury",
    # Credit spreads
    "BAA10Y": "Moody's Baa Corporate Yield - 10y Treasury (credit-stress, since 1986)",
    "BAMLH0A0HYM2": "ICE BofA US High Yield Index OAS (post-2023 only)",
    "BAMLC0A0CM": "ICE BofA US Corporate Index OAS",
    # Inflation
    "CPIAUCSL": "CPI All Urban (SA)",
    "CPILFESL": "CPI Core (SA)",
    # Activity
    "INDPRO": "Industrial Production Index",
    "UNRATE": "Unemployment Rate",
    "PAYEMS": "Nonfarm Payrolls",
    # Liquidity / Fed
    "WALCL": "Fed Total Assets",
    "RRPONTSYD": "Reverse Repo facility usage",
    "M2SL": "M2 Money Supply",
    # Real estate
    "MORTGAGE30US": "30-yr fixed mortgage rate",
    "CSUSHPISA": "Case-Shiller Home Price Index",
    # FX
    "DTWEXBGS": "Dollar Index, Broad",
}


class FredAdapter(BaseAdapter):
    name = "fred"
    rate_limit_per_minute = 110  # FRED soft limit ~120/min; stay under

    BASE_URL = "https://api.stlouisfed.org/fred"

    def __init__(self, api_key: str | None = None, db_path=None):
        super().__init__(db_path)
        self.api_key = api_key or os.environ.get("FRED_API_KEY")
        if not self.api_key:
            raise ValueError(
                "FRED_API_KEY not set. Free key at "
                "https://fred.stlouisfed.org/docs/api/api_key.html, "
                "then add FRED_API_KEY=... to .env"
            )

    def _fetch_one_series(
        self,
        series_id: str,
        label: str,
        observation_start: str,
        max_retries: int = 2,
    ):
        """Fetch one series with retries on transient 5xx. Yields DataPoints
        (zero or more) and never raises — failed series log a warning and skip.
        """
        import logging
        import time as _time

        import requests

        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": observation_start,
        }
        logger = logging.getLogger("hunter.quant.fred")

        last_err = None
        for attempt in range(max_retries + 1):
            self._throttle()
            try:
                r = requests.get(
                    f"{self.BASE_URL}/series/observations",
                    params=params,
                    timeout=30,
                )
                if r.status_code >= 500 and attempt < max_retries:
                    _time.sleep(1.5 * (attempt + 1))
                    continue
                r.raise_for_status()
                payload = r.json()
                units = payload.get("units", "")
                obs_list = payload.get("observations", [])
                yielded = 0
                for obs in obs_list:
                    date = obs.get("date")
                    value = obs.get("value")
                    if not date or value in (".", "", None):
                        continue
                    try:
                        val = float(value)
                    except (ValueError, TypeError):
                        continue
                    yielded += 1
                    yield DataPoint(
                        timestamp=datetime.fromisoformat(date).replace(
                            tzinfo=timezone.utc
                        ),
                        asset_id=series_id,
                        field="value",
                        value=val,
                        source=self.name,
                        metadata={"label": label, "units": units},
                    )
                return
            except (requests.HTTPError, requests.ConnectionError, ValueError) as e:
                last_err = e
                if attempt < max_retries:
                    _time.sleep(1.5 * (attempt + 1))
                    continue
                logger.warning(
                    "FRED series %s failed after %d attempts: %s",
                    series_id, max_retries + 1, e,
                )
                return

    def fetch(
        self,
        series: dict[str, str] | list[str] | None = None,
        observation_start: str = "2010-01-01",
    ) -> Iterable[DataPoint]:
        if series is None:
            series = DEFAULT_SERIES
        if isinstance(series, list):
            series = {s: s for s in series}

        for series_id, label in series.items():
            yield from self._fetch_one_series(
                series_id, label, observation_start
            )
