#!/usr/bin/env python3
"""HUNTER Targeting System -- AI-powered firm targeting and system oversight.

Functions:
- suggest_firms: Given a thesis, suggest top 5 institutional investors
- generate_domains: Given a target firm, suggest new domain queries
- run_overseer: Analyse system performance and suggest improvements
"""

import json
import os

import anthropic
from dotenv import load_dotenv

from config import MODEL, MODEL_FAST
from database import (
    get_active_targets,
    get_all_positions,
    get_connection,
    get_portfolio_stats,
    get_v2_dashboard_stats,
    save_firm_suggestions,
    save_overseer_report,
)

load_dotenv(override=True)


def _get_client():
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def _call_haiku(system_prompt, user_prompt):
    """Single Haiku call, returns parsed JSON or raw text."""
    client = _get_client()
    response = client.messages.create(
        model=MODEL_FAST,
        max_tokens=1024,
        temperature=0.3,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = ""
    for block in response.content:
        if block.type == "text":
            text += block.text
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}


def _call_sonnet(system_prompt, user_prompt, max_tokens=2048):
    """Single Sonnet call for deeper analysis."""
    client = _get_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        temperature=0.3,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = ""
    for block in response.content:
        if block.type == "text":
            text += block.text
    return text.strip()


# ============================================================
# Feature 1: Top 5 Firms Per Thesis
# ============================================================

def suggest_firms(hypothesis_text, diamond_score, domains, direction="long"):
    """Suggest top 5 institutional investors who would find this thesis most actionable."""
    result = _call_haiku(
        system_prompt="You are an institutional sales analyst who knows which hedge funds, PE firms, and asset managers focus on which strategies. Return ONLY JSON.",
        user_prompt=f"""Given this investment thesis, which 5 institutional investors would find it most actionable and why?

THESIS: {hypothesis_text[:500]}
SCORE: {diamond_score}/100
DOMAINS: {domains}
DIRECTION: {direction}

Consider: hedge funds (multi-strategy, sector-specific, event-driven), PE firms (buyout, growth, distressed), asset managers, family offices, sovereign wealth funds. Pick firms that SPECIFICALLY operate in the domains this thesis covers.

Return JSON:
{{
    "firms": [
        {{"name": "Firm Name", "type": "Hedge Fund / PE / Asset Manager", "why": "One-line reason this thesis fits their strategy"}},
        ...
    ]
}}""",
    )
    return result.get("firms", []) if isinstance(result, dict) else []


# ============================================================
# Feature 2: Auto-Generate Domains Per Target Firm
# ============================================================

def generate_domains_for_firm(firm_name, current_domains=None):
    """Given a target firm, suggest new domain queries HUNTER should ingest."""
    current = ", ".join(current_domains) if current_domains else "CRE, insurance, energy, healthcare, distressed, pharma"

    result = _call_haiku(
        system_prompt="You are a research strategist who knows institutional investor strategies. Return ONLY JSON.",
        user_prompt=f"""HUNTER is an autonomous research system that ingests facts from professional databases. It currently covers these domains: {current}

A new target firm has been added: {firm_name}

Based on {firm_name}'s known investment strategy, recent deals, and stated focus areas, suggest 3-5 NEW domain strings that HUNTER should start ingesting to produce theses relevant to {firm_name}.

Rules:
- Each domain should be genuinely different from what HUNTER already covers
- Each domain should map to a specific professional publication ecosystem
- Include example search queries for each domain

Return JSON:
{{
    "firm": "{firm_name}",
    "suggested_domains": [
        {{"domain": "domain name", "description": "what this covers", "example_queries": ["query1", "query2"]}},
        ...
    ]
}}""",
    )
    return result.get("suggested_domains", []) if isinstance(result, dict) else []


# ============================================================
# Feature 3: AI Overseer -- Weekly Performance Analysis
# ============================================================

def run_overseer():
    """Analyse HUNTER's performance and suggest improvements."""
    # Gather system metrics
    stats = get_v2_dashboard_stats()
    p_stats = get_portfolio_stats()
    targets = get_active_targets()

    # Get domain breakdown
    conn = get_connection()
    cursor = conn.cursor()

    # Hypotheses by score range
    cursor.execute("""
        SELECT
            CASE
                WHEN diamond_score >= 70 THEN '70+'
                WHEN diamond_score >= 50 THEN '50-69'
                ELSE 'below 50'
            END as band,
            COUNT(*) as cnt
        FROM hypotheses WHERE survived_kill = 1 AND diamond_score IS NOT NULL
        GROUP BY band
    """)
    score_dist = {r["band"]: r["cnt"] for r in cursor.fetchall()}

    # Source type productivity
    cursor.execute("""
        SELECT source_type, facts_count, hypotheses_survived, productivity_score
        FROM domain_productivity ORDER BY calculated_at DESC LIMIT 18
    """)
    productivity = [dict(r) for r in cursor.fetchall()]

    # Kill reasons
    cursor.execute("SELECT gate_reasoning FROM held_collisions ORDER BY id DESC LIMIT 20")
    recent_kills = [r["gate_reasoning"][:100] for r in cursor.fetchall()]

    # Chain stats
    cursor.execute("SELECT COUNT(*) as cnt, AVG(chain_length) as avg_len FROM chains")
    chain_row = cursor.fetchone()
    chain_stats = {"count": chain_row["cnt"], "avg_length": round(chain_row["avg_len"] or 0, 1)}

    conn.close()

    metrics = {
        "total_facts": stats["total_facts"],
        "survived_hypotheses": stats["survived_hypotheses"],
        "best_score": stats["best_score"],
        "portfolio_value": p_stats["total_value"],
        "portfolio_return": p_stats["total_return_pct"],
        "win_rate": p_stats["win_rate"],
        "score_distribution": score_dist,
        "chain_stats": chain_stats,
        "active_targets": [t["firm_name"] for t in targets],
    }

    # Build the overseer prompt
    prod_summary = "\n".join([f"  {p['source_type']}: {p['facts_count']} facts, {p['hypotheses_survived']} survived, prod={p['productivity_score']:.4f}" for p in productivity[:12]])
    kill_summary = "\n".join([f"  - {k}" for k in recent_kills[:10]])

    # Additional context queries
    cursor2 = get_connection().cursor()

    # Archived hypotheses (cleared multiple times during development)
    cursor2.execute("SELECT COUNT(*) as cnt FROM hypotheses_archive")
    archived_count = cursor2.fetchone()["cnt"]

    # Model vulnerability stats
    cursor2.execute("SELECT COUNT(*) as cnt FROM raw_facts WHERE model_vulnerability IS NOT NULL AND model_vulnerability != 'null' AND length(model_vulnerability) > 20")
    mv_count = cursor2.fetchone()[0]

    # Top scoring hypotheses (current + archived)
    cursor2.execute("SELECT diamond_score, substr(hypothesis_text, 1, 80) FROM hypotheses WHERE survived_kill = 1 AND diamond_score >= 60 ORDER BY diamond_score DESC LIMIT 5")
    top_current = [f"  Score {r[0]}: {r[1]}" for r in cursor2.fetchall()]

    cursor2.execute("SELECT diamond_score, substr(hypothesis_text, 1, 80) FROM hypotheses_archive WHERE survived_kill = 1 AND diamond_score >= 60 ORDER BY diamond_score DESC LIMIT 5")
    top_archived = [f"  Score {r[0]}: {r[1]}" for r in cursor2.fetchall()]

    # Chain domain pairs
    cursor2.execute("SELECT domains_traversed, chain_length FROM chains ORDER BY chain_length DESC LIMIT 5")
    top_chains = [f"  {r[1]} links: {r[0][:80]}" for r in cursor2.fetchall()]

    # Recent hypotheses that got edge-degraded (capped at 45)
    cursor2.execute("SELECT COUNT(*) as cnt FROM hypotheses WHERE survived_kill = 1 AND diamond_score = 45")
    edge_degraded = cursor2.fetchone()["cnt"]

    cursor2.connection.close()

    report_text = _call_sonnet(
        system_prompt="""You are HUNTER Sr, the system's senior performance analyst. You have deep knowledge of HUNTER's architecture and history.

CRITICAL CONTEXT — what you must know:
- HUNTER is a solo-operator research instrument built by an undergraduate
- The hypothesis database has been cleared multiple times during development/debugging; the archived count represents ALL historical output, not just current
- The system went through major evolution: single-domain clustering -> diverse cross-domain output
- Key architectural milestones: search-grounded gate (replaced LLM-judging-LLM), five-element model-vulnerability extraction, disruption-assumption validation, transitive chaining, feedback loop, domain concentration ceiling
- The scoring cap at 45 means any hypothesis where market awareness found the edge was gone gets hard-capped; many 45-score hypotheses had raw scores of 65-81
- Pre-freeze diamond-grade hypotheses exist in the corpus but are held as replication-comparison objects, not validated findings, until the summer study reports

Be SPECIFIC. Reference actual source types by name. Reference actual hypotheses by collision_id. Reference specific kill patterns with examples. Do NOT give generic advice like "increase diversity" — give actionable specifics.""",
        user_prompt=f"""Analyse HUNTER's current state and suggest improvements.

SYSTEM METRICS:
- Total facts: {stats['total_facts']:,} across 18 source types
- Model vulnerability extractions: {mv_count}
- Current surviving hypotheses: {stats['survived_hypotheses']}
- Archived hypotheses (from previous runs): {archived_count}
- Best current score: {stats['best_score']}
- Score distribution: {json.dumps(score_dist)}
- Edge-degraded (capped at 45): {edge_degraded} hypotheses had real scores 65+ but known edge
- Chains: {chain_stats['count']} discovered, avg length {chain_stats['avg_length']}

TOP CURRENT HYPOTHESES:
{chr(10).join(top_current) if top_current else '  None scoring 60+'}

TOP ARCHIVED HYPOTHESES:
{chr(10).join(top_archived) if top_archived else '  None scoring 60+'}

TOP CHAINS:
{chr(10).join(top_chains) if top_chains else '  No chains yet'}

PORTFOLIO:
- Value: GBP {p_stats['total_value']:,.0f} ({p_stats['total_return_pct']:+.2f}%)
- Win rate: {p_stats['win_rate']:.0f}%
- Open: {p_stats['num_open']}, Closed: {p_stats['num_closed']}

SOURCE TYPE PRODUCTIVITY:
{prod_summary}

RECENT GATE KILLS:
{kill_summary}

ACTIVE TARGETS: {', '.join([t['firm_name'] for t in targets]) or 'None set'}

Produce a structured analysis with SPECIFIC references to data above:
1. PERFORMANCE SUMMARY — reference actual numbers and actual hypotheses
2. STRONGEST DOMAINS — name the specific source types producing signal
3. WEAKEST DOMAINS — name specific source types that need attention and WHY
4. EDGE PROBLEM — {edge_degraded} hypotheses had good raw scores but got capped. What does this tell us?
5. CHAIN ANALYSIS — are the chains finding genuine multi-domain pathways?
6. MODEL VULNERABILITY — {mv_count} facts have structured MV data. Is this enough? Which domains need more?
7. SPECIFIC RECOMMENDATIONS — 3-5 actionable changes with expected impact
8. TARGET ALIGNMENT — are we producing theses relevant to {', '.join([t['firm_name'] for t in targets]) or 'no targets set'}?
""",
    )

    # Parse suggestions from the report
    suggestions = []
    for line in report_text.split("\n"):
        line = line.strip()
        if line and (line.startswith("- ") or line.startswith("* ")) and len(line) > 20:
            suggestions.append(line.lstrip("- *").strip())

    save_overseer_report(report_text, suggestions[:10], metrics)
    return report_text, suggestions, metrics


# ============================================================
# Feature 4: AI Chat Assistant
# ============================================================

def chat_with_hunter(user_message, chat_history=None):
    """Chat with HUNTER about its data, performance, and positions."""
    # Gather real-time context
    stats = get_v2_dashboard_stats()
    p_stats = get_portfolio_stats()
    targets = get_active_targets()

    # Get recent hypotheses
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, hypothesis_text, diamond_score, confidence, created_at
        FROM hypotheses WHERE survived_kill = 1 AND diamond_score IS NOT NULL
        ORDER BY diamond_score DESC LIMIT 5
    """)
    top_hyps = [dict(r) for r in cursor.fetchall()]
    conn.close()

    # Extra context
    try:
        conn2 = get_connection()
        cur2 = conn2.cursor()
        cur2.execute("SELECT COUNT(*) FROM hypotheses_archive")
        archived = cur2.fetchone()[0]
        cur2.execute("SELECT COUNT(*) FROM chains")
        chain_count = cur2.fetchone()[0]
        cur2.execute("SELECT COUNT(*) FROM raw_facts WHERE model_vulnerability IS NOT NULL AND model_vulnerability != 'null' AND length(model_vulnerability) > 20")
        mv_count = cur2.fetchone()[0]
        cur2.execute("SELECT COUNT(*) FROM held_collisions")
        held_count = cur2.fetchone()[0]
        conn2.close()
    except:
        archived, chain_count, mv_count, held_count = 0, 0, 0, 0

    context = f"""You are a helpful data assistant for a research platform called HUNTER. Your job is to answer questions about the platform's database and performance metrics. All data below is real and comes from the platform's SQLite database. Report it accurately.

DATABASE METRICS:
- Facts in database: {stats['total_facts']:,}
- Source types: 18
- Model vulnerability extractions: {mv_count}
- Surviving hypotheses: {stats['survived_hypotheses']}
- Archived hypotheses: {archived}
- Best score: {stats['best_score']}
- Facts today: {stats['facts_today']}
- Chains: {chain_count}
- Held collisions: {held_count}

PORTFOLIO METRICS:
- Value: GBP {p_stats['total_value']:,.0f}
- Return: {p_stats['total_return_pct']:+.2f}%
- Win rate: {p_stats['win_rate']:.0f}%
- Open: {p_stats['num_open']}
- Closed: {p_stats['num_closed']}

RECENT HYPOTHESES:
{chr(10).join([f"- #{h['id']} (Score {h['diamond_score']}): {h['hypothesis_text'][:100]}" for h in top_hyps])}

TARGETS: {', '.join([t['firm_name'] for t in targets]) or 'None'}

Answer concisely using the data above. If asked something not in the data, say you don't have that information."""

    # Build messages with history
    messages = []
    if chat_history:
        for msg in chat_history[-6:]:  # Last 6 messages for context
            messages.append(msg)
    messages.append({"role": "user", "content": user_message})

    client = _get_client()
    response = client.messages.create(
        model=MODEL_FAST,
        max_tokens=512,
        temperature=0.3,
        system=context,
        messages=messages,
    )

    text = ""
    for block in response.content:
        if block.type == "text":
            text += block.text
    return text.strip()
