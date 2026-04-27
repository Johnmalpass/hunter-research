"""SEC EDGAR adapter.

No API key. SEC requires a User-Agent header identifying the requester. Set
EDGAR_USER_AGENT in .env (or accept the default in the constructor).

Free, dated, machine-readable. The single richest free financial data source.

Used to ingest:
  - Filing index: every 10-K, 10-Q, 8-K, S-1, Form 4, 13F
  - Filing URLs (for downstream HUNTER pipeline ingestion of filing text)
  - Company metadata (SIC, name, ticker history)

Reference:
    https://www.sec.gov/os/accessing-edgar-data
    https://www.sec.gov/edgar/sec-api-documentation
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Iterable

from quant.data.base import BaseAdapter, DataPoint


class EdgarAdapter(BaseAdapter):
    name = "edgar"
    rate_limit_per_minute = 600  # SEC limit is 10/sec = 600/min

    BASE_URL = "https://www.sec.gov"
    DATA_URL = "https://data.sec.gov"

    def __init__(self, user_agent: str | None = None, db_path=None):
        super().__init__(db_path)
        self.user_agent = user_agent or os.environ.get(
            "EDGAR_USER_AGENT",
            "HUNTER Research johnjosephmalpass@gmail.com",
        )
        self._ticker_cache: dict[str, str] | None = None

    def _headers(self) -> dict[str, str]:
        return {"User-Agent": self.user_agent, "Accept": "application/json"}

    def _load_ticker_map(self) -> dict[str, str]:
        if self._ticker_cache is not None:
            return self._ticker_cache
        import requests

        self._throttle()
        r = requests.get(
            f"{self.BASE_URL}/files/company_tickers.json",
            headers=self._headers(),
            timeout=30,
        )
        r.raise_for_status()
        m: dict[str, str] = {}
        for entry in r.json().values():
            t = entry.get("ticker")
            cik = entry.get("cik_str")
            if t and cik is not None:
                m[t.upper()] = str(cik).zfill(10)
        self._ticker_cache = m
        return m

    def cik_for_ticker(self, ticker: str) -> str | None:
        return self._load_ticker_map().get(ticker.upper())

    def fetch_recent_filings(
        self,
        cik: str,
        count: int = 40,
    ) -> Iterable[DataPoint]:
        """Fetch the company's recent filings index. One DataPoint per filing."""
        import requests

        cik_padded = cik.zfill(10)
        self._throttle()
        r = requests.get(
            f"{self.DATA_URL}/submissions/CIK{cik_padded}.json",
            headers=self._headers(),
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        recent = data.get("filings", {}).get("recent", {})
        accession_numbers = recent.get("accessionNumber", [])[:count]
        forms = recent.get("form", [])[:count]
        filing_dates = recent.get("filingDate", [])[:count]
        primary_docs = recent.get("primaryDocument", [])[:count]
        primary_doc_descs = recent.get("primaryDocDescription", [])[:count]

        company_name = data.get("name")
        tickers = data.get("tickers", [])
        sic = data.get("sic")

        for accession, form, date, doc, desc in zip(
            accession_numbers,
            forms,
            filing_dates,
            primary_docs,
            primary_doc_descs,
        ):
            if not date:
                continue
            field = "filing_" + form.lower().replace("/", "_").replace("-", "_")
            url = (
                f"{self.BASE_URL}/Archives/edgar/data/"
                f"{int(cik_padded)}/{accession.replace('-', '')}/{doc}"
            )
            yield DataPoint(
                timestamp=datetime.fromisoformat(date).replace(tzinfo=timezone.utc),
                asset_id=cik_padded,
                field=field,
                value=accession,
                source=self.name,
                metadata={
                    "form": form,
                    "company_name": company_name,
                    "tickers": tickers,
                    "sic": sic,
                    "primary_doc": doc,
                    "description": desc,
                    "url": url,
                },
            )

    def fetch(
        self,
        tickers: list[str] | None = None,
        ciks: list[str] | None = None,
        count: int = 40,
    ) -> Iterable[DataPoint]:
        if tickers:
            for t in tickers:
                cik = self.cik_for_ticker(t)
                if cik:
                    yield from self.fetch_recent_filings(cik, count=count)
        if ciks:
            for c in ciks:
                yield from self.fetch_recent_filings(c, count=count)
