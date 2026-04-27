"""Tiingo adapter (news + EOD prices + fundamentals).

Sign up at https://tiingo.com. Starter is $10/mo and gives ticker-tagged
news with sentiment, plus EOD prices on every US listing. Crucial for the
reflexivity-phase detector (news count + sentiment delta is the primary
phase-2/phase-3 transition signal).

Add to .env:
    TIINGO_API_KEY=your_key

Endpoints:
    /tiingo/news
    /tiingo/daily/{ticker}/prices

Schema:
    asset_id = ticker (e.g. "MET")
    field    = news_count_24h, news_sentiment_avg_24h, price_close (alt to Polygon)
"""
from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Iterable

from quant.data.base import BaseAdapter, DataPoint


class TiingoAdapter(BaseAdapter):
    name = "tiingo"
    rate_limit_per_minute = 100

    BASE_URL = "https://api.tiingo.com"

    def __init__(self, api_key: str | None = None, db_path=None):
        super().__init__(db_path)
        self.api_key = api_key or os.environ.get("TIINGO_API_KEY")
        if not self.api_key:
            raise ValueError(
                "TIINGO_API_KEY not set. Sign up at https://tiingo.com, "
                "then add TIINGO_API_KEY=... to .env"
            )

    def _headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Token {self.api_key}",
        }

    def fetch_news(
        self,
        tickers: list[str],
        days_back: int = 7,
        limit: int = 1000,
    ) -> Iterable[DataPoint]:
        """Pull news, aggregate to per-ticker per-day count + mean sentiment."""
        import requests

        start = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
        params = {
            "tickers": ",".join(t.upper() for t in tickers),
            "startDate": start,
            "limit": limit,
            "sortBy": "publishedDate",
        }

        self._throttle()
        r = requests.get(
            f"{self.BASE_URL}/tiingo/news",
            headers=self._headers(),
            params=params,
            timeout=30,
        )
        r.raise_for_status()
        articles = r.json()

        # Aggregate by (ticker, date): count + mean sentiment when present
        bucket: dict[tuple[str, str], list[float]] = defaultdict(list)
        counts: dict[tuple[str, str], int] = defaultdict(int)
        for a in articles:
            pub = a.get("publishedDate", "")[:10]
            if not pub:
                continue
            sentiment = a.get("sentiment")
            for t in a.get("tickers", []):
                key = (t.upper(), pub)
                counts[key] += 1
                if sentiment is not None:
                    bucket[key].append(float(sentiment))

        for (ticker, date), n in counts.items():
            ts = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
            yield DataPoint(ts, ticker, "news_count_1d", n, self.name, {})
            if bucket[(ticker, date)]:
                avg = sum(bucket[(ticker, date)]) / len(bucket[(ticker, date)])
                yield DataPoint(ts, ticker, "news_sentiment_avg_1d", avg, self.name, {})

    def fetch(
        self,
        tickers: list[str] | None = None,
        days_back: int = 7,
    ) -> Iterable[DataPoint]:
        if tickers:
            yield from self.fetch_news(tickers, days_back=days_back)
