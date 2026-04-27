"""HUNTER quant — the AI quant shop layered on top of the HUNTER research engine.

Modules:
    quant.data        unified market data store + adapters (FRED, EDGAR, Polygon, ...)
    quant.research    synergy estimator, mechanism compiler, backtester  (next)
    quant.agents      TRADER, CONSCIENCE, AUDITOR, MEMORY                 (next)
    quant.execution   IBKR adapter, paper-trading mode, portfolio book    (next)
    quant.risk        hard limits, reflexivity phase detector             (next)

This package is intentionally separate from the HUNTER research code in the
repository root. HUNTER discovers cross-silo theses; quant trades them.
"""

__version__ = "0.0.1"
