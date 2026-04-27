"""Unified market data store. One SQLite file, one schema, many adapters.

Every external source (FRED, EDGAR, Polygon, Tiingo, MarineTraffic, FAERS,
GDELT, ...) writes through `BaseAdapter` into a single `data_points` table.
Agents read via `MarketState` and never know which adapter produced a value.
"""

from quant.data.base import (
    BaseAdapter,
    DataPoint,
    MarketState,
    get_connection,
    write_points,
)

__all__ = [
    "BaseAdapter",
    "DataPoint",
    "MarketState",
    "get_connection",
    "write_points",
]
