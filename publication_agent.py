#!/usr/bin/env python3
"""HUNTER Publication Agent.

Takes diamond-tier findings (score >= 65) from the HUNTER corpus and produces
Substack and Twitter/X ready publication cards that combine:

  1. The HUNTER cross-silo thesis (what the engine found across professional silos)
  2. Standard market consensus (yfinance: price, analyst target, sector PE, vol)
  3. A HUNTER-adjusted fair value (Opus reasons about the mechanical chain from
     cross-silo fact to price impact, produces a specific number)

The result is a publication-ready card showing the spread between street consensus
and the HUNTER view, with the mechanical chain, catalyst, and key assumption exposed.

Schema-tolerant: reads findings from either `hunter.db` (findings table, uses
title/score/full_report) or the frozen Zenodo corpus (hypotheses +
hypotheses_archive tables, uses hypothesis_text/diamond_score/full_report).

Usage:
    # One finding, dry run
    python publication_agent.py --finding-id 328 --dry-run

    # One finding, writes to publications/
    python publication_agent.py --finding-id 328

    # All diamonds in Zenodo corpus
    python publication_agent.py --all-diamonds

    # All diamonds above 85, capped at 5
    python publication_agent.py --all-diamonds --threshold 85 --limit 5

    # Live hunter.db instead of Zenodo
    python publication_agent.py --all-diamonds --db hunter.db

Cost per finding: ~$0.15-0.25 (1 Sonnet call for ticker extraction +
1-2 Opus calls for fair-value reasoning). yfinance is free.
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import anthropic
import yfinance as yf
from dotenv import load_dotenv

try:
    from config import MODEL, MODEL_DEEP, MODEL_FAST
except ImportError:
    MODEL = "claude-sonnet-4-5"
    MODEL_DEEP = "claude-opus-4-7"
    MODEL_FAST = "claude-haiku-4-5"

load_dotenv(override=True)

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(__file__).parent
PUBLICATIONS_DIR = BASE_DIR / "publications"
PUBLICATIONS_DIR.mkdir(exist_ok=True)

ZENODO_DB = "/tmp/hunter_zenodo/hunter_corpus_v1/hunter_corpus_v1.sqlite"
LIVE_DB = str(BASE_DIR / "hunter.db")


def _client():
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# ============================================================
# Schema-tolerant finding accessors
# ============================================================

def _get_title(row):
    return row.get("title") or row.get("hypothesis_text") or row.get("statement") or ""


def _get_score(row):
    for k in ("adjusted_score", "diamond_score", "score"):
        v = row.get(k)
        if v is not None:
            return v
    return 0


def _get_report(row):
    return row.get("full_report") or row.get("mechanism") or ""


def _get_summary(row):
    return row.get("summary") or row.get("hypothesis_text") or ""


def _get_pathway(row):
    # Zenodo corpus stores the cross-silo chain in fact_chain
    return row.get("fact_chain") or row.get("named_pathway") or row.get("mechanism") or ""


def _get_kill_notes(row):
    return row.get("kill_attempts") or row.get("kill_notes") or row.get("kill_log") or ""


def _get_actions(row):
    return row.get("action_steps") or ""


def _get_confidence(row):
    return row.get("confidence") or "Medium"


def _get_domain(row):
    return row.get("domain") or row.get("category") or row.get("stratum") or ""


# ============================================================
# Corpus fetching
# ============================================================

def _open(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _tables(conn):
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {r["name"] for r in rows}


def fetch_finding(db_path, finding_id):
    """Return (dict, source_table) for one finding, or (None, None)."""
    conn = _open(db_path)
    tables = _tables(conn)
    for table in ("findings", "hypotheses", "hypotheses_archive"):
        if table not in tables:
            continue
        try:
            row = conn.execute(f"SELECT * FROM {table} WHERE id = ?", (finding_id,)).fetchone()
            if row:
                return dict(row), table
        except sqlite3.OperationalError:
            continue
    return None, None


def fetch_all_diamonds(db_path, threshold=65, prefer_hypotheses=True):
    """Return list of (dict, source_table) for all findings >= threshold.

    Scans all candidate tables. If `prefer_hypotheses` is True (default),
    the newer v3 `hypotheses` + `hypotheses_archive` tables (with diamond_score
    and full mechanism/kill-phase fields) are returned first, and the older
    `findings` table is included only if it contains rows not present in
    either hypotheses table.
    """
    conn = _open(db_path)
    tables = _tables(conn)

    hypo_rows = []
    legacy_rows = []

    # Newer v3 hypotheses tables (richer schema)
    for table in ("hypotheses", "hypotheses_archive"):
        if table not in tables:
            continue
        try:
            rows = conn.execute(
                f"SELECT * FROM {table} WHERE diamond_score >= ? ORDER BY diamond_score DESC",
                (threshold,)
            ).fetchall()
            for r in rows:
                hypo_rows.append((dict(r), table))
        except sqlite3.OperationalError:
            continue

    # Older v1/v2 findings table (simpler schema, legacy scoring)
    if "findings" in tables:
        for score_col in ("adjusted_score", "score"):
            try:
                rows = conn.execute(
                    f"SELECT * FROM findings WHERE {score_col} >= ? ORDER BY {score_col} DESC",
                    (threshold,)
                ).fetchall()
                for r in rows:
                    legacy_rows.append((dict(r), "findings"))
                break
            except sqlite3.OperationalError:
                continue

    out = hypo_rows + (legacy_rows if not prefer_hypotheses or not hypo_rows else legacy_rows)
    # Dedupe by (table, id)
    seen = set()
    unique = []
    for d, t in out:
        key = (t, d.get("id"))
        if key in seen:
            continue
        seen.add(key)
        unique.append((d, t))
    return unique


# ============================================================
# Ticker extraction (Sonnet)
# ============================================================

TICKER_SYSTEM = (
    "You are a portfolio construction specialist. Given a cross-silo investment "
    "thesis, identify the tradeable vehicles whose prices most directly reflect "
    "the thesis mechanism. Return ONLY valid JSON."
)


def extract_tickers(row):
    client = _client()
    title = _get_title(row)[:500]
    summary = _get_summary(row)[:800]
    report = _get_report(row)[:1500]
    action = _get_actions(row)[:400]

    body = (
        f"TITLE: {title}\n\n"
        f"SUMMARY: {summary}\n\n"
        f"REPORT: {report}\n\n"
        f"ACTION STEPS: {action}"
    )

    prompt = f"""Read this HUNTER cross-silo thesis. Extract the tradeable positions it implies.

{body}

Rules:
- Extract up to 2 positions. Each maps to a specific ticker whose price MOST DIRECTLY reflects the mechanism.
- DO NOT use broad sector ETFs (XLF, IYR, XLE, XLV) unless the thesis is genuinely sector wide.
- For pharma theses, pick the company with the HIGHEST single-drug revenue concentration, not Pfizer or Merck.
- For CMBS or CRE theses, pick named CMBS-exposed REITs or regional banks with high CRE concentration.
- For insurance theses, pick the named life insurer or specialty insurer.
- direction: "long" if the thesis implies undervalued, "short" if overvalued.
- If no clean tradeable vehicle exists (e.g. private-market thesis), return positions: [].

Return JSON only:
{{
  "positions": [
    {{"ticker": "XYZ", "direction": "short", "reasoning": "one sentence", "weight": 0.7}}
  ],
  "tradeable": true,
  "vehicle_type": "equity|etf|option|bond|none"
}}"""

    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            temperature=0.1,
            system=TICKER_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()
        data = json.loads(text)
        return data
    except Exception as e:
        print(f"  [WARN] Ticker extraction failed: {e}")
        return {"positions": [], "tradeable": False, "vehicle_type": "none"}


# ============================================================
# Market data (yfinance)
# ============================================================

def fetch_market_snapshot(ticker):
    if not ticker or ticker.upper() in ("MANUAL", "NONE", ""):
        return None
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        hist = t.history(period="90d")

        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if not price and not hist.empty:
            price = float(hist["Close"].iloc[-1])

        vol = None
        if not hist.empty and len(hist) > 10:
            rets = hist["Close"].pct_change().dropna()
            if len(rets) > 0:
                vol = float(rets.std() * (252 ** 0.5))

        return {
            "ticker": ticker.upper(),
            "current_price": round(float(price), 2) if price else None,
            "target_mean": info.get("targetMeanPrice"),
            "target_median": info.get("targetMedianPrice"),
            "target_high": info.get("targetHighPrice"),
            "target_low": info.get("targetLowPrice"),
            "num_analysts": info.get("numberOfAnalystOpinions"),
            "recommendation": info.get("recommendationKey"),
            "market_cap": info.get("marketCap"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "beta": info.get("beta"),
            "pe_forward": info.get("forwardPE"),
            "pe_trailing": info.get("trailingPE"),
            "vol_90d_ann": round(vol, 4) if vol else None,
            "wk52_high": info.get("fiftyTwoWeekHigh"),
            "wk52_low": info.get("fiftyTwoWeekLow"),
            "short_pct": info.get("shortPercentOfFloat"),
            "div_yield": info.get("dividendYield"),
        }
    except Exception as e:
        print(f"  [WARN] Market snapshot failed for {ticker}: {e}")
        return {"ticker": ticker.upper(), "error": str(e)}


# ============================================================
# Confidence-weighted adjustment (shrinkage toward street)
# ============================================================
# Rationale: a low-confidence HUNTER finding should move the price less than
# a high-confidence one, otherwise weak theses drag portfolio accuracy.
# Weights below shrink the raw Opus adjustment toward zero (i.e. toward
# street baseline) when HUNTER's own confidence is medium or low.
CONFIDENCE_WEIGHTS = {
    "high": 1.00,    # full conviction -> full adjustment
    "medium": 0.65,  # partial conviction -> ~2/3 adjustment
    "low": 0.35,     # weak conviction -> ~1/3 adjustment, mostly deferring to street
}


def apply_confidence_weighting(valuation, market):
    """Shrink the HUNTER adjustment by confidence weight.

    Preserves the raw Opus output (under raw_* keys) and overwrites the
    primary adjustment/predicted-price fields with the confidence-weighted
    versions. These are the numbers that get published.
    """
    if not valuation:
        return valuation

    raw_adj = valuation.get("hunter_adjustment_dollars")
    if raw_adj is None:
        return valuation

    conf = (valuation.get("confidence") or "medium").lower().strip()
    weight = CONFIDENCE_WEIGHTS.get(conf, 0.65)

    street = valuation.get("street_baseline_price") or market.get("target_mean")
    current = market.get("current_price")
    if street is None:
        return valuation

    # Preserve raw outputs
    valuation["raw_adjustment_dollars"] = raw_adj
    valuation["raw_adjustment_pct"] = valuation.get("hunter_adjustment_pct")
    valuation["raw_hunter_predicted_price"] = valuation.get("hunter_predicted_price")
    valuation["confidence_weight"] = weight

    # Apply shrinkage
    weighted_adj = round(raw_adj * weight, 2)
    weighted_pct = round(weighted_adj / float(street) * 100, 2) if street else None
    new_hunter_price = round(float(street) + weighted_adj, 2)
    new_vs_current = round((new_hunter_price - float(current)) / float(current) * 100, 2) if current else None
    new_vs_street = round((new_hunter_price - float(street)) / float(street) * 100, 2) if street else None

    valuation["hunter_adjustment_dollars"] = weighted_adj
    valuation["hunter_adjustment_pct"] = weighted_pct
    valuation["hunter_predicted_price"] = new_hunter_price
    valuation["hunter_vs_current_pct"] = new_vs_current
    valuation["hunter_vs_street_pct"] = new_vs_street

    return valuation


# ============================================================
# HUNTER-adjusted fair value (Opus)
# ============================================================

FAIR_VALUE_SYSTEM = """You are a quantitative equity analyst at an institutional-grade research shop. You specialise in reconciling bottom-up company valuation with cross-silo structural information that standard sell-side models do not ingest.

Your output format is a three-step decomposition:

  1. STREET BASELINE. Start from the sell-side consensus price target. That is the street's 12-month prediction. Do not discard it, do not argue with the consensus model, just use it as the starting point.

  2. HUNTER CROSS-SILO ADJUSTMENT. Compute the specific dollar adjustment the HUNTER mechanism implies, on top of the street baseline. This is the cross-silo edge: information the street model does not ingest. Show the mechanical chain from the cross-silo fact to the financial metric (earnings, book value, cap rate, reserve ratio, FFO multiple, loss severity) to the dollar impact.

  3. HUNTER PREDICTED PRICE = street baseline + adjustment. This is the final 12-month (or stated horizon) price prediction HUNTER commits to.

Be conservative. If the HUNTER edge is 3-8 percent, that is normal cross-silo alpha. Don't inflate to 40 percent unless the thesis supports it (multi-link chain, binary regulatory event, reserve cliff). If you cannot bridge the mechanism to a specific dollar adjustment, set hunter_adjustment_dollars to null and explain why. Output JSON only."""


def _fmt_money(v):
    if v is None:
        return "n/a"
    if isinstance(v, (int, float)):
        if v > 1e9:
            return f"${v/1e9:.1f}B"
        if v > 1e6:
            return f"${v/1e6:.1f}M"
        return f"${v:,.2f}"
    return str(v)


def compute_hunter_fair_value(row, ticker, market, direction):
    if not market or not market.get("current_price"):
        return None

    client = _client()
    title = _get_title(row)
    summary = _get_summary(row)[:600]
    report = _get_report(row)[:2200]
    pathway = _get_pathway(row)[:600]
    kill_notes = _get_kill_notes(row)[:500]

    thesis = (
        f"TITLE: {title}\n\n"
        f"SUMMARY: {summary}\n\n"
        f"FULL MECHANISM: {report}\n\n"
        f"CROSS-SILO PATHWAY: {pathway}\n\n"
        f"KILL-PHASE NOTES: {kill_notes}"
    )

    mc = _fmt_money(market.get("market_cap"))
    snap = (
        f"TICKER: {ticker}\n"
        f"Proposed direction: {direction}\n"
        f"Current price: ${market.get('current_price')}\n"
        f"Street mean target (12mo): ${market.get('target_mean') or 'n/a'}\n"
        f"Street median target: ${market.get('target_median') or 'n/a'}\n"
        f"Target range: ${market.get('target_low') or 'n/a'} to ${market.get('target_high') or 'n/a'}\n"
        f"Analyst count: {market.get('num_analysts') or 'n/a'}\n"
        f"Street recommendation: {market.get('recommendation') or 'n/a'}\n"
        f"Sector / Industry: {market.get('sector')} / {market.get('industry')}\n"
        f"Market cap: {mc}\n"
        f"Forward PE: {market.get('pe_forward')}\n"
        f"Trailing PE: {market.get('pe_trailing')}\n"
        f"Beta: {market.get('beta')}\n"
        f"90d realized vol (annualised): {market.get('vol_90d_ann')}\n"
        f"52w range: ${market.get('wk52_low')} to ${market.get('wk52_high')}\n"
        f"Short interest % of float: {market.get('short_pct')}\n"
        f"Dividend yield: {market.get('div_yield')}"
    )

    prompt = f"""HUNTER THESIS:
{thesis}

MARKET SNAPSHOT:
{snap}

Produce a three-step HUNTER prediction for {ticker} with a {direction} bias.

STEP 1. STREET BASELINE (12 months). Take the street consensus 12-month price target as given. If a target mean is shown in the market snapshot, use it. If no target is available, derive one from forward PE times consensus EPS or sector cap-rate times FFO. State it as the street_baseline_price. One sentence on what street models are expecting.

STEP 2. HUNTER CROSS-SILO ADJUSTMENT. Identify the specific financial metric the HUNTER mechanism impacts (earnings, book value, cap rate, reserve ratio, FFO multiple, loss severity). Quantify the magnitude in basis points or percent. Translate to a dollar adjustment on the street baseline. The mechanism_chain must trace: cross-silo fact -> financial metric impact -> dollar adjustment.

STEP 3. HUNTER PREDICTED PRICE = street_baseline_price + hunter_adjustment_dollars. This is the final 12-month prediction you commit to.

Calibration:
  - Normal cross-silo alpha is 3 to 8 percent adjustment. Do not inflate to 40 percent unless the thesis supports it (multi-link chain, binary regulatory event, reserve cliff).
  - If the adjustment is genuinely 20 percent+, show why in mechanism_chain.
  - If you cannot bridge the mechanism to a dollar adjustment, set hunter_adjustment_dollars to null and explain.

Return JSON only:
{{
  "street_baseline_price": 70.05,
  "street_view_summary": "1 sentence on what street currently believes",
  "hunter_adjustment_dollars": -25.55,
  "hunter_adjustment_pct": -36.5,
  "mechanism_chain": "2-3 sentence chain: cross-silo fact -> financial metric impact -> dollar adjustment",
  "key_assumption": "the one quantitative assumption that drives the adjustment magnitude",
  "hunter_predicted_price": 44.50,
  "hunter_vs_current_pct": -23.9,
  "hunter_vs_street_pct": -36.5,
  "confidence": "low|medium|high",
  "catalyst": "specific upcoming event, filing, or date that triggers repricing",
  "horizon_months": 12
}}"""

    try:
        resp = client.messages.create(
            model=MODEL_DEEP,
            max_tokens=2048,
            temperature=0.2,
            system=FAIR_VALUE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()
        return json.loads(text)
    except Exception as e:
        print(f"  [WARN] Fair-value computation failed: {e}")
        return None


# ============================================================
# Output formatting
# ============================================================

def format_substack_card(row, position, market, valuation, source_table):
    title = _get_title(row)
    score = _get_score(row)
    confidence = _get_confidence(row)
    summary = _get_summary(row)

    ticker = position.get("ticker")
    direction = position.get("direction", "long")
    dir_word = "Short" if direction == "short" else "Long"

    current = market.get("current_price")
    # Three-step decomposition
    street_baseline = (valuation or {}).get("street_baseline_price") or market.get("target_mean")
    adj_dollars = (valuation or {}).get("hunter_adjustment_dollars")
    adj_pct = (valuation or {}).get("hunter_adjustment_pct")
    hunter_price = (valuation or {}).get("hunter_predicted_price")
    vs_current = (valuation or {}).get("hunter_vs_current_pct")
    vs_street = (valuation or {}).get("hunter_vs_street_pct")

    # Confidence weighting transparency (valuation-level, separate from thesis confidence)
    raw_adj = (valuation or {}).get("raw_adjustment_dollars")
    raw_adj_pct = (valuation or {}).get("raw_adjustment_pct")
    raw_hp = (valuation or {}).get("raw_hunter_predicted_price")
    conf_weight = (valuation or {}).get("confidence_weight", 1.0)
    valuation_conf = ((valuation or {}).get("confidence") or "medium").lower()

    mech = (valuation or {}).get("mechanism_chain", "")
    catalyst = (valuation or {}).get("catalyst", "")
    street_view = (valuation or {}).get("street_view_summary", "")
    key_assumption = (valuation or {}).get("key_assumption", "")
    horizon = (valuation or {}).get("horizon_months", 12)

    # Implied street return if street is right
    street_vs_current_pct = None
    if street_baseline and current:
        try:
            street_vs_current_pct = round((float(street_baseline) - float(current)) / float(current) * 100, 1)
        except Exception:
            pass

    sector = market.get("sector") or ""
    industry = market.get("industry") or ""

    def pct(v):
        if v is None:
            return "n/a"
        return f"{v:+.1f}%"

    def usd(v):
        if v is None:
            return "n/a"
        return f"${v:,.2f}"

    def signed_usd(v):
        if v is None:
            return "n/a"
        sign = "+" if v >= 0 else "-"
        return f"{sign}${abs(v):,.2f}"

    # H1 title: short, punchy. Full hypothesis goes in Step 2.
    title_short = title.split(".")[0].strip()
    if len(title_short) > 100:
        title_short = title_short[:97].rsplit(" ", 1)[0] + "..."

    card = f"""---
title: "{title[:90].replace('"', "'")}"
ticker: {ticker}
direction: {direction}
diamond_score: {score}
confidence: {confidence}
current_price: {current}
street_baseline_price: {street_baseline}
raw_adjustment_dollars: {raw_adj}
raw_adjustment_pct: {raw_adj_pct}
raw_hunter_predicted_price: {raw_hp}
confidence_weight: {conf_weight}
hunter_adjustment_dollars: {adj_dollars}
hunter_adjustment_pct: {adj_pct}
hunter_predicted_price: {hunter_price}
hunter_vs_current_pct: {vs_current}
hunter_vs_street_pct: {vs_street}
horizon_months: {horizon}
sector: "{sector}"
industry: "{industry}"
source_corpus: {source_table}
finding_id: {row.get("id")}
generated: {datetime.utcnow().strftime("%Y-%m-%d")}
---

# {dir_word} ${ticker}: {title_short}

**HUNTER diamond score:** {score} / 100 | **Confidence:** {confidence} | **Sector:** {sector} ({industry})

**Starting context:** {ticker} trades at **{usd(current)}**. Street consensus ({market.get('num_analysts') or '-'} analysts) has a 12-month target of **{usd(street_baseline)}** ({pct(street_vs_current_pct)} from here).

HUNTER has a cross-silo reason to disagree. Here's the logic, step by step, ending in the predicted price.

## Step 1: what street is pricing in

{street_view or "Street consensus on this name is built from standard sell-side models that do not ingest cross-silo information."}

Street baseline: **{usd(street_baseline)}** (12-month consensus target, implying {pct(street_vs_current_pct)} from current).

## Step 2: the HUNTER cross-silo adjustment

{summary}

{mech}

**Raw mechanism-implied adjustment:** {signed_usd(raw_adj) if raw_adj is not None else signed_usd(adj_dollars)} ({pct(raw_adj_pct) if raw_adj_pct is not None else pct(adj_pct)} of {usd(street_baseline)}).

**Key quantitative assumption:** {key_assumption}

### Confidence-weighted shrinkage

HUNTER self-reports **{valuation_conf}** confidence on this specific price translation (separate from the cross-silo thesis confidence, which is `{confidence}`). Raw adjustments are shrunk toward the street baseline by a confidence weight so weak theses don't drag portfolio accuracy. Weights are fixed in advance: high 1.00, medium 0.65, low 0.35.

**Confidence weight applied:** {conf_weight:.2f}

**Final weighted adjustment:** {signed_usd(adj_dollars)} ({pct(adj_pct)} of {usd(street_baseline)}).

## Step 3: the catalyst

{catalyst or "No near-term binary catalyst; structural repricing over the stated horizon."}

Horizon: **{horizon} months**.

---

## HUNTER predicted price

Street baseline {usd(street_baseline)} plus confidence-weighted HUNTER adjustment {signed_usd(adj_dollars)}:

# **{usd(hunter_price)}**

over the next {horizon} months.

| | |
|---|---|
| Implied return if HUNTER is right | **{pct(vs_current)}** vs current |
| Implied return if street is right | {pct(street_vs_current_pct)} vs current |
| HUNTER vs street spread (the tradeable edge) | **{pct(vs_street)}** |

*Pre-shrinkage, the raw mechanism implied {usd(raw_hp) if raw_hp is not None else usd(hunter_price)}. The weight {conf_weight:.2f} pulled the prediction toward street by {signed_usd(adj_dollars - raw_adj) if (raw_adj is not None and adj_dollars is not None) else 'n/a'}.*

Resolution on the [public board](https://johnmalpass.github.io/hunter-research) at the catalyst date above.

---

*Generated by [HUNTER](https://github.com/Johnmalpass/hunter-research), an autonomous cross-silo research engine.
Full thesis, cross-silo pathway, and kill-phase notes are public.
Predictions resolve on the [public board](https://johnmalpass.github.io/hunter-research).
This is a research note, not investment advice.*
"""
    return card


def format_twitter_thread(row, position, market, valuation):
    ticker = position.get("ticker")
    direction = position.get("direction", "long").upper()
    current = market.get("current_price")
    street_baseline = (valuation or {}).get("street_baseline_price") or market.get("target_mean")
    adj_dollars = (valuation or {}).get("hunter_adjustment_dollars")
    adj_pct = (valuation or {}).get("hunter_adjustment_pct")
    hunter_price = (valuation or {}).get("hunter_predicted_price")
    vs_current = (valuation or {}).get("hunter_vs_current_pct")
    vs_street = (valuation or {}).get("hunter_vs_street_pct")
    mech = (valuation or {}).get("mechanism_chain", "")
    catalyst = (valuation or {}).get("catalyst", "")
    key_assumption = (valuation or {}).get("key_assumption", "")
    horizon = (valuation or {}).get("horizon_months", 12)
    score = _get_score(row)
    title = _get_title(row)

    def clip(s, n):
        s = (s or "").strip()
        return s if len(s) <= n else s[:n-1].rstrip() + "..."

    def sdollars(v):
        if v is None:
            return "n/a"
        sign = "+" if v >= 0 else "-"
        return f"{sign}${abs(v):,.2f}"

    street_view_short = (valuation or {}).get("street_view_summary", "")

    tweets = [
        clip(
            f"{direction} ${ticker} thesis\n\n"
            f"Current: ${current}\n"
            f"Street 12mo target: ${street_baseline}\n\n"
            f"Street has one view. HUNTER's cross-silo corpus says they're missing a "
            f"specific mechanical chain. Thread with the logic and the final number.\n\n"
            f"1/6", 278),
        clip(
            f"Step 1: what street is pricing in.\n\n"
            f"${street_baseline} over 12 months. "
            f"{street_view_short}\n\n"
            f"That's the starting baseline.\n\n"
            f"2/6", 278),
        clip(
            f"Step 2: the HUNTER cross-silo adjustment.\n\n"
            f"{mech}\n\n"
            f"3/6", 278),
        clip(
            f"Step 2 (cont): the adjustment in numbers.\n\n"
            f"Applied to the street baseline: {sdollars(adj_dollars)} ({adj_pct}%).\n\n"
            f"Key assumption driving it: {key_assumption}\n\n"
            f"4/6", 278),
        clip(
            f"Step 3: the catalyst.\n\n{catalyst}\n\n"
            f"Horizon: {horizon} months.\n\n"
            f"5/6", 278),
        clip(
            f"HUNTER predicted price:\n\n"
            f"${street_baseline} {sdollars(adj_dollars)} = ${hunter_price}\n\n"
            f"That's {vs_current}% vs current and {vs_street}% vs street.\n\n"
            f"Diamond score {score}/100. Resolves on the public board.\n"
            f"github.com/Johnmalpass/hunter-research\n\n"
            f"6/6", 278),
    ]

    return {
        "ticker": ticker,
        "direction": position.get("direction"),
        "tweets": tweets,
        "thread_text": "\n\n---\n\n".join(tweets),
        "char_counts": [len(t) for t in tweets],
    }


# ============================================================
# Orchestration
# ============================================================

def publish_finding(row, source_table, dry_run=False, verbose=True):
    fid = row.get("id")
    title = _get_title(row)
    score = _get_score(row)

    if verbose:
        print(f"\n[{source_table}#{fid}] {title[:90]} (score={score})")

    # 1. Ticker extraction
    if verbose:
        print("  1/3 Extracting tradeable vehicles (Sonnet)...")
    extracted = extract_tickers(row)
    positions = extracted.get("positions", []) or []

    if not positions:
        if verbose:
            print("  [SKIP] No tradeable vehicle.")
        return None

    cards = []
    for pos in positions[:2]:
        ticker = (pos.get("ticker") or "").upper()
        direction = pos.get("direction", "long")
        if not ticker or ticker in ("MANUAL", "NONE", ""):
            continue

        # 2. Market snapshot
        if verbose:
            print(f"  2/3 Market snapshot for {ticker} (yfinance)...")
        market = fetch_market_snapshot(ticker)
        if not market or not market.get("current_price"):
            if verbose:
                print(f"  [SKIP {ticker}] No market data.")
            continue

        # 3. HUNTER-adjusted prediction (Opus) + confidence weighting
        if verbose:
            print(f"  3/3 HUNTER-adjusted prediction for {ticker} (Opus)...")
        valuation = compute_hunter_fair_value(row, ticker, market, direction)
        if not valuation:
            if verbose:
                print(f"  [SKIP {ticker}] Prediction computation failed.")
            continue

        # Shrink toward street based on HUNTER's own confidence
        valuation = apply_confidence_weighting(valuation, market)

        cp = market.get("current_price")
        street = valuation.get("street_baseline_price") or market.get("target_mean")
        raw_adj = valuation.get("raw_adjustment_dollars")
        w = valuation.get("confidence_weight")
        hpp = valuation.get("hunter_predicted_price")
        if verbose:
            conf = valuation.get("confidence", "medium")
            print(f"     {ticker}: current ${cp} | street ${street}")
            print(f"     raw adjustment {raw_adj} | confidence={conf} weight={w}")
            print(f"     HUNTER predicted price ${hpp} (vs current {valuation.get('hunter_vs_current_pct')}%, vs street {valuation.get('hunter_vs_street_pct')}%)")

        substack = format_substack_card(row, pos, market, valuation, source_table)
        twitter = format_twitter_thread(row, pos, market, valuation)

        card = {
            "finding_id": fid,
            "source_table": source_table,
            "ticker": ticker,
            "direction": direction,
            "reasoning": pos.get("reasoning", ""),
            "market": market,
            "valuation": valuation,
            "substack_md": substack,
            "twitter": twitter,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }
        cards.append(card)

    if not cards:
        if verbose:
            print("  [SKIP] No cards produced.")
        return None

    if not dry_run:
        ts = datetime.utcnow().strftime("%Y%m%d")
        for card in cards:
            tkr = card["ticker"]
            stem = f"{source_table}_{fid}_{tkr}_{ts}"
            md_path = PUBLICATIONS_DIR / f"SUBSTACK_{stem}.md"
            tw_path = PUBLICATIONS_DIR / f"TWITTER_{stem}.json"
            meta_path = PUBLICATIONS_DIR / f"META_{stem}.json"
            md_path.write_text(card["substack_md"])
            tw_path.write_text(json.dumps(card["twitter"], indent=2))
            meta_path.write_text(json.dumps(
                {k: v for k, v in card.items() if k not in ("substack_md",)},
                indent=2, default=str
            ))
            if verbose:
                print(f"     wrote {md_path.name}")
                print(f"     wrote {tw_path.name}")
                print(f"     wrote {meta_path.name}")

    return cards


def main():
    p = argparse.ArgumentParser(description="HUNTER Publication Agent")
    p.add_argument("--finding-id", type=int, help="Single finding ID to publish")
    p.add_argument("--all-diamonds", action="store_true", help="Publish all findings >= threshold")
    p.add_argument("--db", default=ZENODO_DB,
                   help=f"DB path (default: frozen Zenodo corpus at {ZENODO_DB})")
    p.add_argument("--threshold", type=int, default=65, help="Minimum score (default: 65)")
    p.add_argument("--limit", type=int, help="Cap number of findings processed")
    p.add_argument("--dry-run", action="store_true", help="Compute but do not write files")
    p.add_argument("--sleep", type=float, default=1.5,
                   help="Seconds between findings (rate-limit polite)")
    args = p.parse_args()

    db_path = args.db
    if not Path(db_path).exists():
        print(f"ERROR: DB not found at {db_path}")
        print(f"       Zenodo corpus expected at {ZENODO_DB}")
        print(f"       Live DB expected at {LIVE_DB}")
        sys.exit(1)

    print(f"DB: {db_path}")
    print(f"Publications dir: {PUBLICATIONS_DIR}")
    if args.dry_run:
        print("Mode: DRY RUN (no files written)")

    if args.finding_id:
        row, table = fetch_finding(db_path, args.finding_id)
        if not row:
            print(f"ERROR: finding id={args.finding_id} not found.")
            sys.exit(1)
        publish_finding(row, table, dry_run=args.dry_run)
        return

    if args.all_diamonds:
        rows = fetch_all_diamonds(db_path, threshold=args.threshold)
        if args.limit:
            rows = rows[:args.limit]
        print(f"Processing {len(rows)} findings >= {args.threshold}")
        ok = 0
        for i, (row, table) in enumerate(rows, 1):
            try:
                cards = publish_finding(row, table, dry_run=args.dry_run)
                if cards:
                    ok += 1
            except Exception as e:
                print(f"  [ERROR] finding {row.get('id')}: {e}")
            if i < len(rows) and args.sleep > 0:
                time.sleep(args.sleep)
        print(f"\nDone. {ok}/{len(rows)} findings produced publications.")
        return

    p.print_help()


if __name__ == "__main__":
    main()
