"""GDELT v2 Doc adapter — global news events.

NO API KEY required. Free, public, dated. Covers every theme in every country.
Where Tiingo gives ticker-tagged finance news, GDELT gives the broadest news
input we can plug into HUNTER. Useful for early warning (regulatory shifts in
emerging markets that no Bloomberg terminal has tagged) and as input to the
reflexivity detector (article counts on a HUNTER thesis topic over time).

Reference: https://api.gdeltproject.org/api/v2/doc/doc?format=html

Schema:
    asset_id = sanitised query (e.g. "cmbs_delinquency")
    field    = articles_count_1d, mean_tone_1d
    metadata = original query string
"""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Iterable

from quant.data.base import BaseAdapter, DataPoint


def _query_to_asset_id(q: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", q.lower()).strip("_") or "unknown"


class GdeltAdapter(BaseAdapter):
    name = "gdelt"
    rate_limit_per_minute = 60

    BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    def fetch_topic_news(
        self,
        query: str,
        days_back: int = 7,
        max_records: int = 250,
    ) -> Iterable[DataPoint]:
        import requests

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days_back)
        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": max_records,
            "startdatetime": start.strftime("%Y%m%d%H%M%S"),
            "enddatetime": end.strftime("%Y%m%d%H%M%S"),
        }
        self._throttle()
        r = requests.get(self.BASE_URL, params=params, timeout=30)

        # GDELT sometimes returns HTML on bad queries; soft-fail
        try:
            r.raise_for_status()
            data = r.json()
        except (requests.HTTPError, ValueError):
            return

        articles = data.get("articles") or []
        bucket_count: dict[str, int] = defaultdict(int)
        bucket_tone: dict[str, list[float]] = defaultdict(list)

        for a in articles:
            seen = a.get("seendate", "")
            if len(seen) < 8:
                continue
            day = seen[:8]
            bucket_count[day] += 1
            tone = a.get("tone")
            if tone is None:
                continue
            try:
                bucket_tone[day].append(float(tone))
            except (ValueError, TypeError):
                pass

        asset_id = _query_to_asset_id(query)
        for day, count in bucket_count.items():
            ts = datetime.strptime(day, "%Y%m%d").replace(tzinfo=timezone.utc)
            yield DataPoint(
                ts, asset_id, "articles_count_1d", count, self.name, {"query": query}
            )
            if bucket_tone[day]:
                mean_tone = sum(bucket_tone[day]) / len(bucket_tone[day])
                yield DataPoint(
                    ts, asset_id, "mean_tone_1d", mean_tone, self.name, {"query": query}
                )

    def fetch(
        self,
        queries: list[str] | None = None,
        days_back: int = 7,
    ) -> Iterable[DataPoint]:
        for q in queries or []:
            yield from self.fetch_topic_news(q, days_back=days_back)
