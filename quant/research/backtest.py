"""Backtest harness for compiled Mechanisms.

Walks forward through a date range, evaluates the mechanism on a cadence,
opens simulated positions on emitted Signals, closes them after
holding_period_days, computes per-trade returns and aggregate stats vs SPY.

No look-ahead by construction: at every `asof` the mechanism only sees
data with timestamp <= asof (`MarketState.latest_as_of`), and entry prices
are taken on the next trading day after the signal.

Prices come from yfinance. Costs are modelled as a flat round-trip in bps,
plus a half-spread bps proxy for slippage. These are conservative for liquid
US equities and aggressive for illiquid names — tune per universe.

Output is a `BacktestResult` dataclass with summary stats and the full
trade list. JSON-serialisable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from quant.data.base import MarketState
from quant.research.mechanism import Mechanism, Signal


@dataclass
class Trade:
    asset: str
    direction: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    raw_return_pct: float
    net_return_pct: float
    size_pct: float
    holding_days: int
    rationale: str


@dataclass
class BacktestResult:
    thesis_id: str
    period_start: datetime
    period_end: datetime
    n_signals: int
    n_trades_completed: int
    win_rate: float
    avg_net_return_pct: float
    total_compound_return_pct: float
    sharpe_annualised: float
    max_drawdown_pct: float
    avg_holding_days: float
    spy_total_return_pct: float
    alpha_vs_spy_pct: float
    data_check: dict[str, int]
    trades: list[Trade] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["period_start"] = self.period_start.isoformat()
        d["period_end"] = self.period_end.isoformat()
        d["trades"] = [
            {
                **t.__dict__,
                "entry_date": t.entry_date.isoformat(),
                "exit_date": t.exit_date.isoformat(),
            }
            for t in self.trades
        ]
        return d


# ============================================================
# Price fetching (yfinance)
# ============================================================

def _fetch_prices(
    tickers: list[str],
    start: datetime,
    end: datetime,
) -> dict[str, Any]:
    """Pull adjusted-close pandas Series for each ticker. Returns {ticker: Series}."""
    import yfinance as yf

    out: dict[str, Any] = {}
    for ticker in tickers:
        try:
            hist = yf.Ticker(ticker).history(
                start=(start - timedelta(days=10)).strftime("%Y-%m-%d"),
                end=(end + timedelta(days=10)).strftime("%Y-%m-%d"),
                auto_adjust=True,
            )
            if hist.empty:
                out[ticker] = None
                continue
            out[ticker] = hist["Close"]
        except Exception:
            out[ticker] = None
    return out


def _next_price_after(prices, target_date: datetime) -> Optional[tuple[datetime, float]]:
    """First trading day's close strictly after `target_date`."""
    if prices is None:
        return None
    # pandas DatetimeIndex; convert target to tz-naive day for comparison
    target_naive = target_date.replace(tzinfo=None)
    idx = prices.index
    # tz-strip pandas index for comparison if needed
    try:
        idx_naive = idx.tz_localize(None)
    except (AttributeError, TypeError):
        idx_naive = idx
    matches = idx_naive > target_naive
    if not matches.any():
        return None
    pos = int(matches.argmax())
    return idx[pos].to_pydatetime(), float(prices.iloc[pos])


# ============================================================
# Walk-forward
# ============================================================

def backtest_mechanism(
    mechanism: Mechanism,
    state: MarketState,
    start_date: datetime,
    end_date: datetime,
    eval_freq_days: int = 7,
    cost_bps: float = 5.0,
    slippage_bps: float = 5.0,
) -> BacktestResult:
    """Run the mechanism every eval_freq_days. Return a BacktestResult."""

    # Sanity check: tz handling. We work in UTC throughout.
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    # Data pre-flight
    data_check = mechanism.check_data(state)

    # Fetch all needed prices once (universe + SPY)
    universe = list(mechanism.universe)
    benchmark = "SPY"
    all_tickers = list(set(universe + [benchmark]))
    prices = _fetch_prices(all_tickers, start_date, end_date)

    # SPY benchmark return over the full period
    spy_series = prices.get(benchmark)
    spy_total_return_pct = 0.0
    if spy_series is not None and len(spy_series) > 1:
        spy_total_return_pct = float(
            (spy_series.iloc[-1] / spy_series.iloc[0] - 1.0) * 100.0
        )

    # Walk forward
    trades: list[Trade] = []
    n_signals = 0
    cur = start_date
    one_step = timedelta(days=eval_freq_days)
    cost_drag = (cost_bps + slippage_bps) / 10000.0  # one-way; doubled below

    while cur <= end_date:
        signals = mechanism.evaluate(state, cur)
        for sig in signals:
            n_signals += 1
            entry = _next_price_after(prices.get(sig.asset), cur)
            if entry is None:
                continue
            entry_date, entry_price = entry
            exit_target = entry_date + timedelta(days=sig.holding_period_days)
            ex = _next_price_after(prices.get(sig.asset), exit_target)
            if ex is None:
                continue
            exit_date, exit_price = ex

            if sig.direction == "long":
                raw = (exit_price - entry_price) / entry_price
            elif sig.direction == "short":
                raw = (entry_price - exit_price) / entry_price
            else:
                continue

            net = raw - 2.0 * cost_drag  # round-trip costs
            trades.append(
                Trade(
                    asset=sig.asset,
                    direction=sig.direction,
                    entry_date=entry_date,
                    exit_date=exit_date,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    raw_return_pct=raw * 100.0,
                    net_return_pct=net * 100.0,
                    size_pct=sig.size_pct,
                    holding_days=(exit_date - entry_date).days,
                    rationale=sig.rationale,
                )
            )
        cur += one_step

    # Aggregate
    if not trades:
        return BacktestResult(
            thesis_id=mechanism.thesis_id,
            period_start=start_date,
            period_end=end_date,
            n_signals=n_signals,
            n_trades_completed=0,
            win_rate=0.0,
            avg_net_return_pct=0.0,
            total_compound_return_pct=0.0,
            sharpe_annualised=0.0,
            max_drawdown_pct=0.0,
            avg_holding_days=0.0,
            spy_total_return_pct=spy_total_return_pct,
            alpha_vs_spy_pct=-spy_total_return_pct,
            data_check=data_check,
            trades=[],
        )

    rets = [t.net_return_pct / 100.0 for t in trades]
    n_wins = sum(1 for r in rets if r > 0)
    win_rate = n_wins / len(rets)
    avg_net = sum(rets) / len(rets)

    compound = 1.0
    for r in rets:
        compound *= (1.0 + r)
    total_compound = (compound - 1.0) * 100.0

    if len(rets) > 1:
        mu = sum(rets) / len(rets)
        var = sum((r - mu) ** 2 for r in rets) / (len(rets) - 1)
        sigma = math.sqrt(max(var, 1e-12))
        avg_holding = sum(t.holding_days for t in trades) / len(trades)
        ann_factor = math.sqrt(365.25 / max(avg_holding, 1.0))
        sharpe = (mu / sigma) * ann_factor if sigma > 0 else 0.0
    else:
        sharpe = 0.0
        avg_holding = float(trades[0].holding_days)

    cum = []
    c = 1.0
    for r in rets:
        c *= (1.0 + r)
        cum.append(c)
    peak = -1e9
    max_dd = 0.0
    for x in cum:
        if x > peak:
            peak = x
        dd = (x - peak) / peak
        if dd < max_dd:
            max_dd = dd

    return BacktestResult(
        thesis_id=mechanism.thesis_id,
        period_start=start_date,
        period_end=end_date,
        n_signals=n_signals,
        n_trades_completed=len(trades),
        win_rate=win_rate,
        avg_net_return_pct=avg_net * 100.0,
        total_compound_return_pct=total_compound,
        sharpe_annualised=sharpe,
        max_drawdown_pct=max_dd * 100.0,
        avg_holding_days=avg_holding,
        spy_total_return_pct=spy_total_return_pct,
        alpha_vs_spy_pct=total_compound - spy_total_return_pct,
        data_check=data_check,
        trades=trades,
    )


def format_result(result: BacktestResult, max_trades_shown: int = 10) -> str:
    """Human-readable summary."""
    lines: list[str] = []
    lines.append(f"Thesis:        {result.thesis_id}")
    lines.append(
        f"Period:        {result.period_start.date()} to {result.period_end.date()}"
    )
    lines.append("")
    lines.append("Data check:")
    if not result.data_check:
        lines.append("  (no requirements)")
    for key, count in result.data_check.items():
        marker = "ok " if count > 0 else "MISSING"
        lines.append(f"  {marker}  {key}  ({count} rows)")
    lines.append("")
    lines.append("Performance:")
    lines.append(f"  Signals emitted:        {result.n_signals}")
    lines.append(f"  Trades completed:       {result.n_trades_completed}")
    lines.append(f"  Win rate:               {result.win_rate:.1%}")
    lines.append(f"  Avg net return / trade: {result.avg_net_return_pct:+.2f}%")
    lines.append(f"  Total compound return:  {result.total_compound_return_pct:+.2f}%")
    lines.append(f"  Sharpe (annualised):    {result.sharpe_annualised:+.2f}")
    lines.append(f"  Max drawdown:           {result.max_drawdown_pct:+.2f}%")
    lines.append(f"  Avg holding days:       {result.avg_holding_days:.0f}")
    lines.append(f"  SPY total return:       {result.spy_total_return_pct:+.2f}%")
    lines.append(f"  Alpha vs SPY:           {result.alpha_vs_spy_pct:+.2f}%")

    if result.trades:
        lines.append("")
        lines.append(f"Trades (first {min(max_trades_shown, len(result.trades))}):")
        for t in result.trades[:max_trades_shown]:
            lines.append(
                f"  {t.entry_date.date()} {t.direction:5s} {t.asset:6s} "
                f"@ {t.entry_price:.2f}  ->  {t.exit_date.date()} "
                f"@ {t.exit_price:.2f}  net {t.net_return_pct:+.2f}%"
            )
        if len(result.trades) > max_trades_shown:
            lines.append(f"  ...and {len(result.trades) - max_trades_shown} more")

    return "\n".join(lines)
