"""Smoke test for the quant data layer.

Run from the repo root:
    python -m quant.smoke

Verifies the round-trip from adapter -> SQLite -> MarketState query.

EDGAR runs without an API key. FRED is skipped if FRED_API_KEY is missing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make `quant` importable when running this file directly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=True)

from quant.data.base import MarketState  # noqa: E402
from quant.data.adapters.edgar import EdgarAdapter  # noqa: E402
from quant.data.adapters.fred import FredAdapter  # noqa: E402


def banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def smoke_edgar() -> None:
    banner("EDGAR adapter (no API key required)")
    adapter = EdgarAdapter()
    n = adapter.run(tickers=["AAPL", "MET"], count=5)
    print(f"  ingested {n} filing rows for AAPL + MET")


def smoke_fred() -> None:
    banner("FRED adapter")
    if not os.environ.get("FRED_API_KEY"):
        print("  FRED_API_KEY not set - skipping")
        print("  free key at https://fred.stlouisfed.org/docs/api/api_key.html")
        print("  then add to .env: FRED_API_KEY=your_key")
        return
    adapter = FredAdapter()
    n = adapter.run(
        series=["DGS10", "DGS2", "MORTGAGE30US"],
        observation_start="2024-01-01",
    )
    print(f"  ingested {n} observations for DGS10, DGS2, MORTGAGE30US")


def smoke_query() -> None:
    banner("MarketState query")
    with MarketState() as state:
        assets = state.assets()
        print(f"  {len(assets)} distinct assets in store")
        for asset_id in sorted(assets)[:6]:
            fields = state.fields(asset_id)
            print(f"    {asset_id}: {len(fields)} fields")
            for fld in fields[:2]:
                latest = state.latest(asset_id, fld)
                if latest:
                    val = latest.value
                    if isinstance(val, str) and len(val) > 40:
                        val = val[:37] + "..."
                    print(
                        f"      latest {fld} @ {latest.timestamp.isoformat()[:19]}"
                        f"  src={latest.source}  val={val}"
                    )

        runs = state.adapter_runs(limit=5)
        print(f"\n  recent adapter runs:")
        for r in runs:
            status = (
                f"OK rows={r['rows_written']}"
                if not r["error"]
                else f"ERR {r['error'][:60]}"
            )
            print(f"    {r['adapter']}  {r['started_at'][:19]}  {status}")


if __name__ == "__main__":
    smoke_edgar()
    smoke_fred()
    smoke_query()
    print("\nDone. Database at: quant_data.db (in repo root)")
