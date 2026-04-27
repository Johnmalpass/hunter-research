"""Polygon.io adapter (real-time + historical equities, options, crypto).

Sign up at https://polygon.io. Starter tier is $29/mo and gives 5y of minute
bars + 50k API calls/day, which is enough for the AI Quant Worker's daily
research-and-trade cycle.

Add to .env:
    POLYGON_API_KEY=your_key

Endpoints used here:
    /v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}
    /v3/reference/tickers/{ticker}
    /v2/reference/news

Schema written into the unified store:
    asset_id = ticker (e.g. "AAPL")
    field    = price_close, price_open, price_high, price_low, volume, vwap
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Iterable

from quant.data.base import BaseAdapter, DataPoint


class PolygonAdapter(BaseAdapter):
    name = "polygon"
    rate_limit_per_minute = 100  # Starter tier: 5/sec = 300/min, stay well under

    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: str | None = None, db_path=None):
        super().__init__(db_path)
        self.api_key = api_key or os.environ.get("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError(
                "POLYGON_API_KEY not set. Sign up at https://polygon.io, "
                "then add POLYGON_API_KEY=... to .env"
            )

    def fetch_daily_bars(
        self,
        ticker: str,
        from_date: str,
        to_date: str | None = None,
    ) -> Iterable[DataPoint]:
        """Daily OHLCV bars between from_date and to_date (YYYY-MM-DD)."""
        import requests

        to_date = to_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        url = (
            f"{self.BASE_URL}/v2/aggs/ticker/{ticker.upper()}"
            f"/range/1/day/{from_date}/{to_date}"
        )
        params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": self.api_key}

        self._throttle()
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        for bar in r.json().get("results") or []:
            ts = datetime.fromtimestamp(bar["t"] / 1000, tz=timezone.utc)
            asset_id = ticker.upper()
            common_meta = {"timeframe": "1d"}
            yield DataPoint(ts, asset_id, "price_open", bar["o"], self.name, common_meta)
            yield DataPoint(ts, asset_id, "price_high", bar["h"], self.name, common_meta)
            yield DataPoint(ts, asset_id, "price_low", bar["l"], self.name, common_meta)
            yield DataPoint(ts, asset_id, "price_close", bar["c"], self.name, common_meta)
            yield DataPoint(ts, asset_id, "volume", bar["v"], self.name, common_meta)
            if "vw" in bar:
                yield DataPoint(ts, asset_id, "vwap", bar["vw"], self.name, common_meta)

    def fetch(
        self,
        tickers: list[str] | None = None,
        from_date: str = "2020-01-01",
        to_date: str | None = None,
    ) -> Iterable[DataPoint]:
        for t in tickers or []:
            yield from self.fetch_daily_bars(t, from_date, to_date)
