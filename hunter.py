#!/usr/bin/env python3
"""HUNTER v2 -- Fact-Collision Autonomous Opportunity Engine.

Run with: python hunter.py
"""

import collections
import json
import os
import random
import re
import signal
import sys
import time
from datetime import datetime

import anthropic
from dotenv import load_dotenv

import config  # Module-level access for v3 Golden dynamic config
from config import (
    CORE_SOURCE_TYPES,
    DATA_SOURCES,
    DEEP_DIVE_THRESHOLD,
    DOMAINS,
    FOCUS_MODE,
    INGEST_RATIO,
    COLLISION_LOOKBACK_ANOMALIES,
    COLLISION_LOOKBACK_FACTS,
    KILL_SEARCH_COUNT,
    MODEL,
    MODEL_DEEP,
    MODEL_FAST,
    MAX_TOKENS_RESPONSE,
    MAX_TOKENS_RESPONSE_REPORT,
    MAX_TOKENS_RESPONSE_SEARCH,
    PRODUCTIVITY_BLEND_WEIGHT,
    PRODUCTIVITY_MIN_FACTS,
    ROTATION_PENALTY_FIRST,
    ROTATION_PENALTY_SECOND,
    ROTATION_WINDOW,
    SOURCE_ICONS,
    THEORY_RUN,
    THEORY_RUN_SELF_DOMAINS,
    TOKEN_CAP_NORMAL,
    TIMEOUT_NORMAL,
    compute_avg_domain_distance,
    compute_edge_decay_penalty,
    GAP_TARGETING_RATIO,
)
from database import (
    get_collision_to_hypothesis_rate,
    get_collisions_list,
    get_cycles_since_last_summary,
    get_facts_by_ids,
    get_findings_since,
    get_knowledge_base_stats,
    get_last_daily_summary_date,
    get_latest_domain_productivity,
    get_recent_anomalies,
    get_recent_facts,
    get_recent_facts_count,
    get_recent_hypothesis_domain_pairs,
    get_source_type_counts,
    init_db,
    is_collision_duplicate,
    mark_anomaly_attempted,
    save_anomaly,
    save_collision,
    save_cycle_log,
    save_daily_summary,
    save_chain,
    save_domain_productivity,
    save_finding,
    save_held_collision,
    save_hypothesis,
    save_knowledge_node,
    save_deep_dive,
    save_raw_fact,
    search_facts_by_entities,
    search_facts_by_implications,
    search_facts_by_keywords,
    search_facts_by_model_fields,
    search_facts_by_embedding,
    save_causal_edges,
    save_fact_embedding,
    find_causal_paths,
    find_contradictory_paths,
    get_collision_counts_by_source_pair,
    save_edge_recovery_event,
    save_theory_run_cycle,
    update_hypothesis_telemetry,
    search_facts_with_beliefs_for_asset,
    search_exogenous_facts_for_belief,
)
from prompts import (
    ANOMALY_DETECT_PROMPT,
    BATCH_ANOMALY_DETECT_PROMPT,
    BELIEF_REALITY_TEST_PROMPT,
    CHAIN_EXTEND_PROMPT,
    COLLISION_EVALUATE_PROMPT,
    DAILY_SYNTHESIS_PROMPT,
    DISRUPTION_ASSUMPTION_TEST_PROMPT,
    ENTITY_RESOLVE_PROMPT,
    HYPOTHESIS_FORM_PROMPT,
    HYPOTHESIS_REPORT_PROMPT,
    HYPOTHESIS_SCORE_PROMPT,
    INGEST_EXTRACT_PROMPT,
    DEEP_DIVE_PROMPT,
    KILL_PROMPT,
    NEGATIVE_SPACE_DETECT_PROMPT,
    SEARCH_GATE_QUERY_PROMPT,
    SEARCH_GATE_EVALUATE_PROMPT,
    SYSTEM_PROMPT,
)

# === Embedding Model (Branch 1) — lazy-loaded to avoid import cost ===
_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            return None
    return _embedding_model


def compute_fact_embedding(implications, model_vulnerability):
    """Compute embedding from implications + assumption + disruption.
    Returns numpy array (384-dim) or None."""
    model = get_embedding_model()
    if model is None:
        return None
    texts = []
    if isinstance(implications, list):
        texts.extend([imp for imp in implications if isinstance(imp, str) and len(imp) > 5])
    if isinstance(model_vulnerability, dict):
        for field in ("assumption", "disruption"):
            val = model_vulnerability.get(field, "")
            if val and isinstance(val, str) and len(val) > 5:
                texts.append(val)
    elif isinstance(model_vulnerability, str) and model_vulnerability != "null":
        try:
            mv = json.loads(model_vulnerability)
            if isinstance(mv, dict):
                for field in ("assumption", "disruption"):
                    val = mv.get(field, "")
                    if val and isinstance(val, str) and len(val) > 5:
                        texts.append(val)
        except (json.JSONDecodeError, TypeError):
            pass
    if not texts:
        return None
    combined = " | ".join(texts)
    embedding = model.encode(combined, normalize_embeddings=True)
    return embedding


import logging

# Set up file logging
log_dir = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'hunter.log')),
    ]
)
logger = logging.getLogger('hunter')

load_dotenv()

# ============================================================
# Terminal colours
# ============================================================

class C:
    """ANSI colour codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"

    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"

    @staticmethod
    def score_colour(score):
        if score >= 90:
            return C.BG_MAGENTA + C.WHITE + C.BOLD
        elif score >= 75:
            return C.GREEN + C.BOLD
        elif score >= 60:
            return C.CYAN + C.BOLD
        elif score >= 40:
            return C.YELLOW
        elif score >= 20:
            return C.DIM + C.YELLOW
        else:
            return C.RED + C.DIM

    @staticmethod
    def label(score):
        if score >= 90:
            return "LEGENDARY"
        elif score >= 75:
            return "DIAMOND"
        elif score >= 60:
            return "STRONG"
        elif score >= 40:
            return "NOTABLE"
        elif score >= 20:
            return "INTERESTING"
        else:
            return "NOISE"


def print_banner():
    print(f"""
{C.BOLD}{C.YELLOW}
  ██╗  ██╗██╗   ██╗███╗   ██╗████████╗███████╗██████╗
  ██║  ██║██║   ██║████╗  ██║╚══██╔══╝██╔════╝██╔══██╗
  ███████║██║   ██║██╔██╗ ██║   ██║   █████╗  ██████╔╝
  ██╔══██║██║   ██║██║╚██╗██║   ██║   ██╔══╝  ██╔══██╗
  ██║  ██║╚██████╔╝██║ ╚████║   ██║   ███████╗██║  ██║
  ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═╝
{C.RESET}
{C.DIM}  Autonomous Fact-Collision Engine v2.0{C.RESET}
{C.DIM}  Ingest facts. Detect anomalies. Collide. Hypothesize. Kill. Repeat.{C.RESET}
""")


def print_phase(phase, detail=""):
    icons = {
        "INGEST":    f"{C.BLUE}[INGEST]{C.RESET}",
        "EXTRACT":   f"{C.CYAN}[EXTRACT]{C.RESET}",
        "ANOMALY":   f"{C.MAGENTA}[ANOMALY]{C.RESET}",
        "COLLISION":  f"{C.YELLOW}[COLLISION]{C.RESET}",
        "ENTITY":    f"{C.BLUE}[ENTITY RESOLVE]{C.RESET}",
        "HYPOTHESIS": f"{C.BG_YELLOW}{C.WHITE}{C.BOLD}[HYPOTHESIS]{C.RESET}",
        "KILL":      f"{C.BG_RED}{C.WHITE}{C.BOLD}[KILL]{C.RESET}",
        "SCORE":     f"{C.GREEN}[SCORE]{C.RESET}",
        "REPORT":    f"{C.GREEN}[REPORT]{C.RESET}",
        "STATS":     f"{C.CYAN}[KNOWLEDGE BASE]{C.RESET}",
        "SYNTHESIS":  f"{C.BG_BLUE}{C.WHITE}{C.BOLD}[DAILY SYNTHESIS]{C.RESET}",
    }
    icon = icons.get(phase, f"[{phase}]")
    detail_str = f" {C.DIM}{detail}{C.RESET}" if detail else ""
    print(f"  {icon}{detail_str}")


def print_score(score, title, adjusted=None):
    colour = C.score_colour(score)
    label = C.label(score)
    adj_str = f" (adjusted: {adjusted:.1f})" if adjusted is not None else ""
    print(f"\n  {colour} {label} [{score}/100]{adj_str} {C.RESET}")
    print(f"  {C.BOLD}{title}{C.RESET}\n")
    logger.info(f"SCORE: {score}/100 - {title}")


def print_cycle_header(cycle_num, mode, detail=""):
    now = datetime.now().strftime("%H:%M:%S")
    mode_colour = C.BLUE if mode == "INGEST" else C.YELLOW
    print(f"\n{'='*70}")
    print(f"  {C.BOLD}Cycle #{cycle_num}{C.RESET}  {C.DIM}|{C.RESET}  {mode_colour}{mode}{C.RESET}  {C.DIM}|{C.RESET}  {C.DIM}{detail}{C.RESET}  {C.DIM}|{C.RESET}  {C.DIM}{now}{C.RESET}")
    print(f"{'='*70}")
    logger.info(f"Cycle #{cycle_num} | {mode} | {detail}")


def print_error(msg):
    print(f"  {C.RED}ERROR: {msg}{C.RESET}")
    logger.error(msg)


def print_info(msg):
    print(f"  {C.DIM}{msg}{C.RESET}")
    logger.info(msg)


# ============================================================
# Custom exceptions
# ============================================================

class JSONParseError(Exception):
    """Raised when no valid JSON can be extracted from a response."""
    pass


# ============================================================
# Anthropic API helpers
# ============================================================

client = None


def get_client():
    global client
    if client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print(f"\n{C.RED}ERROR: ANTHROPIC_API_KEY not set in .env file{C.RESET}")
            sys.exit(1)
        client = anthropic.Anthropic(api_key=api_key)
    return client


def extract_text_from_response(response):
    texts = []
    for block in response.content:
        if block.type == "text":
            texts.append(block.text)
    return "\n".join(texts)


def count_tokens(response):
    return response.usage.output_tokens


# ============================================================
# Smart rate limiting
# ============================================================

_request_timestamps = collections.deque(maxlen=200)
_input_token_timestamps = []  # list of (timestamp, token_count)


def rate_limit_pause(is_search=False, estimated_input_tokens=2000):
    """Only sleep if we're approaching the rate limit."""
    now = time.time()
    # Clean old timestamps (older than 60 seconds)
    while _request_timestamps and _request_timestamps[0] < now - 60:
        _request_timestamps.popleft()

    # Clean old token entries
    _input_token_timestamps[:] = [(t, c) for t, c in _input_token_timestamps if t > now - 60]

    # Check requests per minute (stay under 50 RPM)
    if len(_request_timestamps) >= 45:
        sleep_time = 60 - (now - _request_timestamps[0]) + 1
        if sleep_time > 0:
            time.sleep(sleep_time)

    # Check input tokens per minute (stay under 30k for Tier 1)
    if _input_token_timestamps:
        tokens_last_minute = sum(c for _, c in _input_token_timestamps)
        if tokens_last_minute + estimated_input_tokens > 25000:
            sleep_time = 60 - (now - _input_token_timestamps[0][0]) + 2
            if sleep_time > 0:
                time.sleep(min(sleep_time, 120))

    _request_timestamps.append(time.time())
    if is_search:
        _input_token_timestamps.append((time.time(), estimated_input_tokens * 3))  # web search inflates input
    else:
        _input_token_timestamps.append((time.time(), estimated_input_tokens))


def _update_token_tracking(response):
    """Update token tracking with actual usage from a response."""
    if hasattr(response, 'usage') and hasattr(response.usage, 'input_tokens'):
        actual_input = response.usage.input_tokens
        now = time.time()
        # Replace the last estimated entry with the actual count
        if _input_token_timestamps:
            last_ts, last_est = _input_token_timestamps[-1]
            # Only replace if it was added very recently (within 2 seconds)
            if now - last_ts < 2:
                _input_token_timestamps[-1] = (last_ts, actual_input)


# ============================================================
# Retry logic with exponential backoff
# ============================================================

def call_claude(messages, system=None, max_tokens=MAX_TOKENS_RESPONSE,
                tools=None, temperature=0.7, model=None):
    """Call Claude API with exponential backoff retry for transient errors."""
    c = get_client()
    kwargs = {
        "model": model or MODEL,
        "max_tokens": max_tokens,
        "messages": messages,
        "temperature": temperature,
    }
    if system:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = tools

    backoff_times = [2, 8, 32]
    last_error = None

    for attempt in range(3):
        try:
            return c.messages.create(**kwargs)
        except anthropic.RateLimitError as e:
            last_error = e
            if attempt < 2:
                sleep_time = backoff_times[attempt]
                print_info(f"Rate limit (429), retrying in {sleep_time}s (attempt {attempt + 1}/3)...")
                time.sleep(sleep_time)
            else:
                raise
        except anthropic.InternalServerError as e:
            last_error = e
            if attempt < 2:
                sleep_time = backoff_times[attempt]
                print_info(f"Server error (500), retrying in {sleep_time}s (attempt {attempt + 1}/3)...")
                time.sleep(sleep_time)
            else:
                raise
        except anthropic.APIStatusError as e:
            # Catch 503 and other transient status errors
            if hasattr(e, 'status_code') and e.status_code in (429, 500, 503):
                last_error = e
                if attempt < 2:
                    sleep_time = backoff_times[attempt]
                    print_info(f"API error ({e.status_code}), retrying in {sleep_time}s (attempt {attempt + 1}/3)...")
                    time.sleep(sleep_time)
                else:
                    raise
            else:
                raise
        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
            last_error = e
            if attempt < 2:
                sleep_time = backoff_times[attempt]
                print_info(f"Network error, retrying in {sleep_time}s (attempt {attempt + 1}/3)...")
                time.sleep(sleep_time)
            else:
                raise

    raise last_error


def call_with_web_search(user_prompt, system=None, max_tokens=MAX_TOKENS_RESPONSE, model=None):
    rate_limit_pause(is_search=True, estimated_input_tokens=25000)
    tools = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}]
    messages = [{"role": "user", "content": user_prompt}]
    response = call_claude(messages, system=system, max_tokens=max_tokens, tools=tools, model=model)
    _update_token_tracking(response)
    return response


def call_text(user_prompt, system=None, max_tokens=MAX_TOKENS_RESPONSE, temperature=0.7, model=None):
    rate_limit_pause(is_search=False, estimated_input_tokens=2000)
    messages = [{"role": "user", "content": user_prompt}]
    response = call_claude(messages, system=system, max_tokens=max_tokens, temperature=temperature, model=model)
    _update_token_tracking(response)
    return response


def call_kill_gate(user_prompt, system=None, max_tokens=MAX_TOKENS_RESPONSE, model=None):
    """Call the kill gate — uses web search normally, falls back to LLM-only in retrospective mode."""
    if getattr(config, 'RETROSPECTIVE_MODE', False) and getattr(config, 'RETROSPECTIVE_DISABLE_WEB_SEARCH', False):
        # Retrospective mode: LLM reasoning only, no web search
        return call_text(user_prompt, system=system, max_tokens=max_tokens,
                        temperature=getattr(config, 'RETROSPECTIVE_TEMPERATURE', 0.3), model=model)
    else:
        return call_with_web_search(user_prompt, system=system, max_tokens=max_tokens, model=model)


def retro_prompt_prefix():
    """Returns the retrospective mode header for prompts, or empty string if not active."""
    if getattr(config, 'RETROSPECTIVE_MODE', False):
        cutoff = getattr(config, 'RETROSPECTIVE_CUTOFF_DATE', '2024-12-31')
        return f"RETROSPECTIVE MODE. The current date is {cutoff}. Reason only from the facts provided. Do not use any knowledge of events after {cutoff}.\n\n"
    return ""


def parse_json_response(text):
    """Parse JSON from a Claude response. Bulletproof extraction.

    Raises JSONParseError if no valid JSON can be found.
    """
    text = text.strip()

    # Try extracting from ```json ... ``` blocks
    if "```json" in text:
        start = text.find("```json")
        if start != -1:
            start += 7
            end = text.find("```", start)
            if end != -1:
                candidate = text[start:end].strip()
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass  # Fall through to other strategies

    # Try extracting from ``` ... ``` blocks
    if "```" in text:
        start = text.find("```")
        if start != -1:
            start += 3
            end = text.find("```", start)
            if end != -1:
                candidate = text[start:end].strip()
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass

    # Try parsing the whole text as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object or array in the text
    brace_start = text.find("{")
    bracket_start = text.find("[")

    candidates = []
    if brace_start != -1:
        end = text.rfind("}") + 1
        if end > brace_start:
            candidates.append(text[brace_start:end])
    if bracket_start != -1:
        end = text.rfind("]") + 1
        if end > bracket_start:
            candidates.append(text[bracket_start:end])

    # Try the candidate that starts earliest
    candidates.sort(key=lambda c: text.find(c))
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise JSONParseError(f"No valid JSON found in response (first 200 chars): {text[:200]}")


# ============================================================
# Fact Validation -- reject garbage before it poisons the pipeline
# ============================================================

# Known approximate price ranges for commodities (per standard unit).
# If a scraped price is wildly outside these, the fact is bad data.
PRICE_SANITY = {
    "aluminum": {"unit": "per ton", "min": 1500, "max": 5000},
    "aluminium": {"unit": "per ton", "min": 1500, "max": 5000},
    "gold": {"unit": "per ounce", "min": 1500, "max": 5000},
    "silver": {"unit": "per ounce", "min": 15, "max": 120},
    "oil": {"unit": "per barrel", "min": 30, "max": 200},
    "crude": {"unit": "per barrel", "min": 30, "max": 200},
    "brent": {"unit": "per barrel", "min": 30, "max": 200},
    "wti": {"unit": "per barrel", "min": 30, "max": 200},
    "copper": {"unit": "per ton or per pound", "min": 3, "max": 15000},
    "natural gas": {"unit": "per mmbtu", "min": 1, "max": 20},
    "wheat": {"unit": "per bushel", "min": 3, "max": 15},
    "corn": {"unit": "per bushel", "min": 2, "max": 12},
    "platinum": {"unit": "per ounce", "min": 500, "max": 3000},
    "palladium": {"unit": "per ounce", "min": 500, "max": 4000},
    "bitcoin": {"unit": "per coin", "min": 10000, "max": 500000},
    "ethereum": {"unit": "per coin", "min": 500, "max": 20000},
}

def validate_fact(fact):
    """Validate a fact before saving. Returns (is_valid, reason).

    Checks for:
    1. Implausible prices (aluminum at $0.10/lb when real is $3200/ton)
    2. Empty or garbage content
    3. Opinion/prediction masquerading as fact
    """
    title = (fact.get("title") or "").lower()
    content = (fact.get("raw_content") or "").lower()
    combined = f"{title} {content}"

    # Check 1: Empty or too short to be useful
    if len(content) < 20:
        return False, "Content too short to be a verifiable fact"

    # Check 2: Reject commodity spot price facts -- they go stale fast and poison hypotheses
    # Price facts are useful for anomaly detection at ingest time but become dangerous
    # when they sit in the database for days/weeks with outdated numbers
    spot_price_patterns = [
        r"trading at \$\d+", r"price.*\$\d+.*per", r"spot price.*\$\d+",
        r"\$\d+.*per ounce", r"\$\d+.*per barrel", r"\$\d+.*per ton",
        r"▲.*WTD", r"▼.*WTD", r"week-to-date",
    ]
    is_spot_price = any(re.search(p, combined) for p in spot_price_patterns)
    source_type = fact.get("source_type", "")
    if is_spot_price and source_type == "commodity":
        # Only reject if it's JUST a price quote with no other substantive info
        word_count = len(combined.split())
        if word_count < 30:
            return False, "Bare commodity spot price — goes stale quickly"

    # Check 3: Price sanity check -- only reject OBVIOUSLY wrong prices
    # The challenge: commodities are quoted in different units (per ton, per pound, per ounce)
    # So we only reject prices that are wrong in ALL possible units
    price_matches = re.findall(r'\$([0-9,]+\.?\d*)', combined)
    if price_matches:
        for commodity, bounds in PRICE_SANITY.items():
            if commodity in combined:
                for price_str in price_matches:
                    try:
                        price = float(price_str.replace(",", ""))
                        # Only reject if price is absurdly low even for smallest unit
                        # e.g., aluminum at $0.10 is wrong whether per ton, per pound, or per ounce
                        if price < 0.50:
                            return False, f"Price ${price_str} for {commodity} is implausibly low"
                        # Reject if astronomically high (5x the per-ton max)
                        if price > bounds["max"] * 10:
                            return False, f"Price ${price_str} for {commodity} is implausibly high (expected max ~{bounds['max']} {bounds['unit']})"
                    except ValueError:
                        continue

    # Check 3: Not actually a fact (opinion/prediction/vague)
    opinion_phrases = [
        "experts predict", "is expected to", "could potentially",
        "may become", "is likely to", "analysts believe",
        "is transforming", "is revolutionizing", "is disrupting",
        "market is projected to", "is poised to",
    ]
    fact_has_specifics = bool(re.search(r'\d{4}', combined)) or bool(price_matches) or \
        any(c in combined for c in ["filed", "announced", "expired", "awarded", "approved", "published"])

    if not fact_has_specifics:
        opinion_count = sum(1 for phrase in opinion_phrases if phrase in combined)
        if opinion_count >= 2:
            return False, "Looks like opinion/prediction, not a verifiable fact"

    return True, "OK"


# ============================================================
# Credit balance detection -- stop wasting cycles on empty balance
# ============================================================

_consecutive_credit_errors = 0

def check_credit_error(error):
    """Check if an error is a credit balance error. Returns True if we should stop."""
    global _consecutive_credit_errors
    error_str = str(error)
    if "credit balance is too low" in error_str or "insufficient_quota" in error_str:
        _consecutive_credit_errors += 1
        if _consecutive_credit_errors >= 2:
            print(f"\n  {C.BG_RED}{C.WHITE}{C.BOLD} CREDITS EXHAUSTED — Stopping to avoid wasting cycles {C.RESET}")
            print(f"  {C.RED}Add credits at console.anthropic.com then restart.{C.RESET}\n")
            return True
    else:
        _consecutive_credit_errors = 0
    return False


# ============================================================
# INGEST CYCLE -- collect raw facts, detect anomalies
# ============================================================

class IngestCycle:
    """Ingests raw facts from a data source and detects anomalies."""

    def __init__(self, cycle_num):
        self.cycle_num = cycle_num
        self.source = self._pick_underused_source()
        self.tokens_used = 0
        self.facts_saved = 0
        self.anomalies_found = 0
        self.start_time = time.time()

    @staticmethod
    def _pick_underused_source():
        """Pick a source type weighted toward underused ones for diversity.

        Uses TOTAL counts (not just 24h) to push underrepresented sources
        like patents, bankruptcies, pharma, and academic up to parity with
        the saturated commodity and regulation feeds.
        """
        try:
            # Get total counts across all time
            from database import get_connection
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT source_type, COUNT(*) as cnt FROM raw_facts GROUP BY source_type")
            rows = cursor.fetchall()
            conn.close()
            total_counts = {r['source_type']: r['cnt'] for r in rows}

            source_types = list({s["type"] for s in DATA_SOURCES})
            max_count = max(total_counts.values()) if total_counts else 1

            # Deficit weights (existing — push underrepresented types to parity)
            deficit_weights = []
            for st in source_types:
                count = total_counts.get(st, 0)
                deficit = max(0, max_count - count)
                weight = (deficit + 1) ** 2
                deficit_weights.append(weight)

            # Normalise deficit weights
            total_d = sum(deficit_weights) or 1
            deficit_weights = [w / total_d for w in deficit_weights]

            # Productivity weights (feedback loop — push low-productivity types)
            productivity = get_latest_domain_productivity()
            if productivity:
                inv_prod_weights = []
                for st in source_types:
                    prod = productivity.get(st, 0.0)
                    # Inverse: low productivity = high weight (needs more/better facts)
                    inv_prod_weights.append(1.0 / (prod + 0.01))

                # Normalise
                total_p = sum(inv_prod_weights) or 1
                inv_prod_weights = [w / total_p for w in inv_prod_weights]

                # Blend deficit + productivity
                weights = []
                for i in range(len(source_types)):
                    blended = ((1 - PRODUCTIVITY_BLEND_WEIGHT) * deficit_weights[i] +
                               PRODUCTIVITY_BLEND_WEIGHT * inv_prod_weights[i])
                    weights.append(blended)

                # Log blended weights for diagnostics (top 5 by weight)
                indexed = sorted(zip(source_types, deficit_weights, inv_prod_weights, weights),
                                 key=lambda x: -x[3])
                for st, dw, pw, fw in indexed[:5]:
                    logger.info(f"Source weight: {st} deficit={dw:.3f} inv_prod={pw:.3f} blend={fw:.3f}")
            else:
                weights = deficit_weights

            # Normalise final weights
            total_w = sum(weights) or 1
            weights = [w / total_w for w in weights]
            chosen_type = random.choices(source_types, weights=weights, k=1)[0]
            candidates = [s for s in DATA_SOURCES if s["type"] == chosen_type]
            return random.choice(candidates)
        except Exception:
            return random.choice(DATA_SOURCES)

    def track_tokens(self, response):
        self.tokens_used += count_tokens(response)

    def _generate_dynamic_query(self):
        """Generate a novel search query that avoids facts already in the database."""
        try:
            from database import get_connection
            conn = get_connection()
            cursor = conn.cursor()
            # Get recent fact titles for this source type to avoid
            cursor.execute("""
                SELECT title FROM raw_facts
                WHERE source_type = ?
                ORDER BY id DESC LIMIT 20
            """, (self.source["type"],))
            recent_titles = [r["title"] for r in cursor.fetchall()]
            conn.close()

            recent_block = "\n".join(f"- {t}" for t in recent_titles[:15])
            base_query = self.source["query"]

            # Feedback loop: check if this source type needs model-targeting queries
            model_targeting_instruction = ""
            try:
                prod_scores = get_latest_domain_productivity()
                my_prod = prod_scores.get(self.source["type"], 0.0)
                if prod_scores and my_prod < 0.05:
                    model_targeting_instruction = """
CRITICAL — MODEL-TARGETING MODE:
This source type has very low collision productivity. Its facts are not producing cross-domain hypotheses because they lack specific model/methodology vocabulary.

Generate a query that will find a SPECIFIC NAMED MODEL, METHODOLOGY, or PRICING FRAMEWORK used by practitioners in this domain, including:
- The specific assumptions the model relies on
- When those assumptions were last calibrated or updated
- What data inputs the model uses

Examples of what we need:
- "ARGUS DCF model cap rate spread assumptions office REIT valuation 2025"
- "RSMeans cost database prevailing wage accuracy federal infrastructure 2025"
- "Baltic exchange FFA pricing methodology vessel supply model assumption 2025"
- "CECL credit loss model CRE portfolio vintage assumption recalibration 2025"

The goal is facts that NAME specific models with specific stale assumptions — this is what produces diamond-grade cross-domain collisions.
"""
            except Exception:
                pass

            response = call_text(
                f"""Generate 1 highly specific web search query to find NEW {self.source['type']} facts.

SOURCE TYPE: {self.source['type']}

FACTS WE ALREADY HAVE in this source type (search for COMPLETELY DIFFERENT subtopics):
{recent_block}

Rules:
- Look at what we already have and search for a COMPLETELY DIFFERENT INDUSTRY within this source type
- DO NOT go deeper into the same niche. If we have 3 facts about lumber, do NOT search for more lumber. Search for cement, or steel, or glass, or roofing materials.
- DO NOT search for topics related to our existing facts -- explore new territory
- BREADTH OVER DEPTH. Each query should target a different industry sector than what we already have. If we have pharma/biotech/FDA facts, search for construction, shipping, mining, agriculture, utilities, insurance, real estate, logistics, manufacturing, defense, or aerospace instead.
- NEVER generate queries about pharmaceutical, biotech, FDA, drug, generic, ANDA, clinical trial, or pharma-adjacent topics UNLESS this source type is specifically "pharmaceutical". Every other source type must explore non-pharma industries.
- Your query must be SPECIFIC and NICHE -- target obscure professional sources
- Include specific date ranges, classification codes, court names, database names, or technical terms
- Target facts that sit unread in professional silos -- the kind of thing only 50-500 people would see
- DO NOT generate generic queries like "latest news about X"

The goal is MAXIMUM INDUSTRY DIVERSITY. We want facts from as many different professional worlds as possible. Not depth in one niche.
{model_targeting_instruction}
Respond with ONLY a JSON object:
{{"query": "your specific search query"}}""",
                system="You are a research librarian who specialises in finding obscure, niche professional filings and records across MANY different industries. You never search the same industry twice in a row.",
                max_tokens=256,
                temperature=0.8,
                model=MODEL_FAST,
            )
            self.track_tokens(response)
            text = extract_text_from_response(response)
            data = parse_json_response(text)
            dynamic_query = data.get("query", "")
            if dynamic_query and len(dynamic_query) > 10:
                return dynamic_query
        except Exception as e:
            print_info(f"Dynamic query generation failed (using static): {e}")

        return self.source["query"]

    def _generate_gap_targeted_query(self):
        """Generate a search query targeting sparse regions of fact-space (Branch 5).
        Finds domain pairs with highest collision deficit (high distance, few collisions)
        and generates bridging queries to fill the gap."""
        try:
            pair_counts = get_collision_counts_by_source_pair()
            from config import get_domain_distance, DATA_SOURCES as _DS
            source_types = list({s["type"] for s in _DS})

            deficits = []
            for i in range(len(source_types)):
                for j in range(i + 1, len(source_types)):
                    pair = tuple(sorted([source_types[i], source_types[j]]))
                    distance = get_domain_distance(pair[0], pair[1])
                    count = pair_counts.get(pair, 0)
                    deficit = distance * (1.0 / (count + 1))
                    deficits.append((deficit, pair[0], pair[1]))

            deficits.sort(reverse=True)
            if not deficits:
                return None

            top_pair = random.choice(deficits[:5])
            _, type_a, type_b = top_pair

            response = call_text(
                f"""Generate 1 highly specific web search query that would find facts
in the "{type_a}" professional world that have IMPLICATIONS for the "{type_b}" professional world.

The query should target facts that sit in {type_a} publications but would be relevant to
{type_b} practitioners — the kind of cross-silo fact that produces diamond-grade collisions.

Examples:
- patent → insurance: "patent granted parametric weather index trigger mechanism 2026"
- commodity → pharmaceutical: "API active pharmaceutical ingredient raw material supply disruption 2026"
- bankruptcy → regulation: "Chapter 11 debtor regulatory compliance waiver environmental 2026"

Respond with ONLY: {{"query": "your specific search query", "target_source_type": "{type_a}"}}""",
                system="Generate cross-domain search queries.",
                max_tokens=256, temperature=0.8, model=MODEL_FAST,
            )
            self.track_tokens(response)
            data = parse_json_response(extract_text_from_response(response))
            query = data.get("query", "")
            if query and len(query) > 10:
                return (query, data.get("target_source_type", type_a))
        except Exception as e:
            print_info(f"Gap-targeted query failed (non-fatal): {e}")
        return None

    def run(self):
        source_icon = SOURCE_ICONS.get(self.source["type"], "📌")
        print_cycle_header(self.cycle_num, "INGEST", f"{source_icon} {self.source['type']}")

        self._gap_targeted = False
        try:
            # Phase 0: Query selection — 20% gap-targeted (Branch 5), 80% normal
            search_query = None
            if random.random() < GAP_TARGETING_RATIO:
                gap_result = self._generate_gap_targeted_query()
                if gap_result:
                    search_query, target_type = gap_result
                    # Override source to match the target type
                    matching_sources = [s for s in DATA_SOURCES if s["type"] == target_type]
                    if matching_sources:
                        self.source = random.choice(matching_sources)
                    self._gap_targeted = True
                    print_info(f"🎯 Gap-targeted query (bridging toward {target_type})")

            if search_query is None:
                search_query = self._generate_dynamic_query()

            # Phase 1: Search for raw data
            logger.info(f"Dynamic query: {search_query[:120]}")
            print_phase("INGEST", f"Searching: {search_query[:80]}...")
            response = call_with_web_search(
                f"Search for: {search_query}\n\nReturn ALL specific facts, dates, names, and numbers you find. Be thorough.",
                system=SYSTEM_PROMPT,
                max_tokens=MAX_TOKENS_RESPONSE_SEARCH,
                model=MODEL_FAST,  # Haiku for ingest -- fast and cheap
            )
            self.track_tokens(response)
            search_results = extract_text_from_response(response)

            if not search_results or len(search_results) < 100:
                print_info("No substantial results found")
                save_cycle_log("INGEST", self.source["type"], 1, self.tokens_used, 0,
                               time.time() - self.start_time, "completed")
                return

            # Phase 2: Extract discrete facts
            print_phase("EXTRACT", f"Extracting facts from {len(search_results)} chars...")
            prompt = INGEST_EXTRACT_PROMPT.format(
                search_results=search_results[:8000],
                source_type=self.source["type"],
            )
            response = call_text(prompt, system="You are a fact extraction engine. Return ONLY valid JSON. No commentary.", max_tokens=MAX_TOKENS_RESPONSE_REPORT, temperature=0.2)  # Sonnet -- Haiku summarises instead of generating cross-domain implications
            self.track_tokens(response)
            text = extract_text_from_response(response)

            try:
                data = parse_json_response(text)
                if isinstance(data, list):
                    facts = data
                else:
                    facts = data.get("facts", [])
            except (json.JSONDecodeError, JSONParseError) as e:
                # Try a more aggressive extraction -- find any JSON array in the response
                try:
                    import re
                    match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
                    if match:
                        facts = json.loads(match.group())
                    else:
                        print_error(f"Failed to parse facts JSON: {e}")
                        print_info(f"Raw response (first 300 chars): {text[:300]}")
                        facts = []
                except Exception:
                    print_error(f"Failed to parse facts JSON: {e}")
                    print_info(f"Raw response (first 300 chars): {text[:300]}")
                    facts = []

            if not facts:
                print_info("No verifiable facts extracted")
                save_cycle_log("INGEST", self.source["type"], 1, self.tokens_used, 0,
                               time.time() - self.start_time, "completed")
                return

            # Phase 3: Validate, save facts, and collect for batch anomaly detection
            saved_facts_with_ids = []
            rejected_count = 0
            for fact in facts:
                try:
                    # VALIDATE before saving
                    is_valid, reason = validate_fact(fact)
                    if not is_valid:
                        rejected_count += 1
                        print_info(f"Rejected: {reason} — {fact.get('title', '')[:50]}")
                        continue

                    fact_id = save_raw_fact(
                        source_type=self.source["type"],
                        source_url=fact.get("source_url", ""),
                        title=fact.get("title", "Unknown fact"),
                        raw_content=fact.get("raw_content", ""),
                        entities=fact.get("entities", []),
                        keywords=fact.get("keywords", ""),
                        domain=fact.get("domain", ""),
                        country=fact.get("country", ""),
                        date_of_fact=fact.get("date_of_fact", ""),
                        obscurity=fact.get("obscurity", "medium"),
                        implications=fact.get("implications", []),
                        model_vulnerability=fact.get("model_vulnerability"),
                        reflexivity_tag=fact.get("reflexivity_tag"),
                        market_belief=fact.get("market_belief"),
                    )

                    # Save causal edges (Branch 2)
                    if fact_id:
                        causal_edges = fact.get("causal_edges", [])
                        if causal_edges and isinstance(causal_edges, list):
                            try:
                                save_causal_edges(fact_id, causal_edges,
                                                  self.source["type"], fact.get("domain", ""))
                            except Exception as e:
                                logger.warning(f"Branch 2: save_causal_edges failed for fact {fact_id}: {e}")

                        # Compute and save embedding (Branch 1)
                        try:
                            _imp_list = fact.get("implications", [])
                            _mv_data = fact.get("model_vulnerability")
                            embedding = compute_fact_embedding(_imp_list, _mv_data)
                            if embedding is not None:
                                save_fact_embedding(fact_id, embedding.tobytes())
                        except Exception as e:
                            logger.warning(f"Branch 1: save_fact_embedding failed for fact {fact_id}: {e}")

                    # save_raw_fact returns None for duplicates
                    if fact_id is None:
                        print_info(f"Duplicate fact skipped: {fact.get('title', '')[:60]}")
                        continue

                    self.facts_saved += 1
                    saved_facts_with_ids.append((fact, fact_id))

                except Exception as e:
                    print_error(f"Failed to save fact: {e}")
                    continue

            if rejected_count > 0:
                print_info(f"Validation rejected {rejected_count}/{len(facts)} bad facts")

            # Phase 4: Batch anomaly detection on all saved facts
            if saved_facts_with_ids:
                self._batch_detect_anomalies(saved_facts_with_ids)

            duration = time.time() - self.start_time
            print(f"\n  {C.GREEN}✓ Ingested {self.facts_saved} facts, detected {self.anomalies_found} anomalies{C.RESET}")
            print(f"  {C.DIM}Tokens: {self.tokens_used:,} | Duration: {duration:.0f}s{C.RESET}")

            save_cycle_log("INGEST", self.source["type"], 1, self.tokens_used,
                           self.anomalies_found, duration, "completed",
                           gap_targeted=int(self._gap_targeted))

        except anthropic.RateLimitError as e:
            print_error(f"Rate limit hit: {e}")
            save_cycle_log("INGEST", self.source["type"], 0, self.tokens_used, 0,
                           time.time() - self.start_time, "rate_limit", str(e),
                           gap_targeted=int(self._gap_targeted))
            time.sleep(60)

        except Exception as e:
            print_error(f"Unexpected error: {e}")
            save_cycle_log("INGEST", self.source["type"], 0, self.tokens_used, 0,
                           time.time() - self.start_time, "error", str(e),
                           gap_targeted=int(getattr(self, '_gap_targeted', False)))
            if check_credit_error(e):
                raise SystemExit("Credits exhausted")

    def _batch_detect_anomalies(self, facts_with_ids):
        """Ask Claude which facts in a batch are anomalous. Single API call for all facts."""
        try:
            today = datetime.now().strftime("%B %d, %Y")

            # Build the facts block for the prompt
            facts_block = ""
            for i, (fact, fact_id) in enumerate(facts_with_ids):
                facts_block += f"\n[Fact #{i}] (id={fact_id}, source={self.source['type']})\n"
                facts_block += f"  Title: {fact.get('title', '')}\n"
                facts_block += f"  Content: {fact.get('raw_content', '')[:1500]}\n"
                facts_block += f"  Entities: {json.dumps(fact.get('entities', []))}\n"
                facts_block += f"  Domain: {fact.get('domain', '')}\n"
                facts_block += f"  Date: {fact.get('date_of_fact', '')}\n"

            prompt = BATCH_ANOMALY_DETECT_PROMPT.format(
                today_date=today,
                facts_block=facts_block[:12000],
            )
            response = call_text(prompt, max_tokens=2048, temperature=0.3, model=MODEL_FAST)
            self.track_tokens(response)
            text = extract_text_from_response(response)
            data = parse_json_response(text)

            anomalies_list = data.get("anomalies", [])
            for anomaly_entry in anomalies_list:
                fact_index = anomaly_entry.get("fact_index")
                if fact_index is None or fact_index < 0 or fact_index >= len(facts_with_ids):
                    continue

                fact, fact_id = facts_with_ids[fact_index]
                desc = (anomaly_entry.get("anomaly_description") or "").lower()

                # Hard filter: reject date-only anomalies
                date_noise_phrases = [
                    "future date", "future event", "hasn't occurred yet",
                    "hasn't happened yet", "months in the future",
                    "weeks in the future", "days in the future",
                    "not yet occurred", "impossible for a past event",
                    "dated in the future", "temporal contradiction",
                    "hasn't happened", "hasn't occurred",
                ]
                if any(phrase in desc for phrase in date_noise_phrases):
                    continue

                anomaly_id = save_anomaly(
                    raw_fact_id=fact_id,
                    anomaly_description=anomaly_entry.get("anomaly_description", ""),
                    weirdness_score=min(10, max(1, int(anomaly_entry.get("weirdness_score", 5)))),
                    anomaly_type=anomaly_entry.get("anomaly_type", "contradiction"),
                    entities=fact.get("entities", []),
                    domain=fact.get("domain", ""),
                )
                self.anomalies_found += 1
                print_phase("ANOMALY",
                    f"⚠️  Weirdness {anomaly_entry.get('weirdness_score', '?')}/10: {anomaly_entry.get('anomaly_description', '')[:80]}...")

        except Exception as e:
            # Anomaly detection is best-effort, don't fail the cycle
            # but DO log structurally so silent failures are visible later.
            print_error(f"Batch anomaly detection failed (non-fatal): {e}")
            logger.warning(f"batch_anomaly_fail cycle={self.cycle_num} facts={len(facts_with_ids)} err={type(e).__name__}: {e}")


# ============================================================
# COLLISION CYCLE -- collide facts, form hypotheses, kill them
# ============================================================

class CollisionCycle:
    """Collides anomalies with facts from different sources to find non-obvious insights."""

    def __init__(self, cycle_num):
        self.cycle_num = cycle_num
        self.tokens_used = 0
        self.collisions_found = 0
        self.hypotheses_formed = 0
        self.hypotheses_survived = 0
        self.start_time = time.time()
        self._processed_models_this_cycle = []  # Dedup: track broken model claims per cycle

        # Theory proof layer — observes pipeline outputs, never modifies them
        try:
            from theory import TheoryRecorder
            self._theory_recorder = TheoryRecorder()
        except Exception:
            self._theory_recorder = None

        # Theory layer agents — advanced per-collision/hypothesis evidence logging
        try:
            from theory_layer import TheoryTelemetry
            self._theory_telemetry = TheoryTelemetry()
        except Exception:
            self._theory_telemetry = None

    def track_tokens(self, response):
        self.tokens_used += count_tokens(response)

    def _compute_depth_metrics(self, matching_facts):
        """Module D: Compute depth-per-domain metrics from matching facts."""
        facts_per_domain = {}
        for f in (matching_facts or []):
            st = f.get("source_type", "unknown") if isinstance(f, dict) else "unknown"
            facts_per_domain[st] = facts_per_domain.get(st, 0) + 1
        total = sum(facts_per_domain.values()) or 1
        depths = list(facts_per_domain.values()) or [0]
        return {
            "facts_per_domain": facts_per_domain,
            "min_depth": min(depths),
            "max_depth": max(depths),
            "domain_count": len(facts_per_domain),
            "depth_concentration": round(max(depths) / total, 4),
        }

    def _lightweight_awareness_probe(self, thesis_text):
        """Module F: Single cheap web search for awareness level 0-4.
        Fires on EVERY scored hypothesis regardless of score.
        DISABLED in retrospective mode (Phase 1)."""
        if getattr(config, 'RETROSPECTIVE_MODE', False) and getattr(config, 'RETROSPECTIVE_DISABLE_WEB_SEARCH', False):
            return None  # Disabled in retrospective mode
        try:
            prompt = f"""Rate the market awareness of this specific thesis on a 0-4 scale:

THESIS: {thesis_text[:600]}

Scale:
0 = no published analysis found connecting these factors
1 = one obscure mention (single blog post, forum, tweet)
2 = multiple non-institutional sources discussing the connection
3 = at least one institutional research note or major publication
4 = consensus trade — multiple institutional sources, widely known

Respond with ONLY a JSON object:
{{"awareness_level": 0}}"""
            resp = call_with_web_search(prompt, system="Rate market awareness on 0-4 scale. Be precise.", max_tokens=256)
            self.track_tokens(resp)
            data = parse_json_response(extract_text_from_response(resp))
            return min(4, max(0, int(data.get("awareness_level", 0))))
        except Exception as e:
            print_info(f"Lightweight awareness probe failed (non-fatal): {e}")
            return None

    def _compute_entity_specificity(self, hypothesis_text):
        """Module H: Score 0-3 based on named entities in hypothesis."""
        try:
            prompt = f"""Analyze this thesis for specificity. For each field, answer 1 if a SPECIFIC named entity is present, 0 if generic:

THESIS: {hypothesis_text[:800]}

1. PRACTITIONERS: Does it name a specific firm/institution? (e.g. "Apollo Global" = 1, "life insurers" = 0)
2. METHODOLOGY: Does it name a specific model/framework? (e.g. "ARGUS DCF" = 1, "standard valuation" = 0)
3. ASSUMPTION: Does it name a specific number/threshold? (e.g. "7.2% cap rate" = 1, "high rates" = 0)

Respond with ONLY: {{"practitioners": 0, "methodology": 0, "assumption": 0}}"""
            resp = call_text(prompt, system="Score specificity 0/1 per field.", max_tokens=128, temperature=0.0)
            self.track_tokens(resp)
            data = parse_json_response(extract_text_from_response(resp))
            return sum(min(1, max(0, int(data.get(k, 0)))) for k in ("practitioners", "methodology", "assumption"))
        except Exception:
            return None

    def _compute_causal_chain_length(self, hypothesis_text, fact_chain):
        """Module J: Count causal links in the reasoning chain."""
        try:
            prompt = f"""How many distinct causal steps are in this reasoning chain?
Count each "A causes/leads to/enables B" as one step. A single fact stating something is 0 steps.

THESIS: {hypothesis_text[:600]}
FACTS: {json.dumps(fact_chain)[:400] if isinstance(fact_chain, list) else str(fact_chain)[:400]}

Respond with ONLY: {{"causal_chain_length": 1}}"""
            resp = call_text(prompt, system="Count causal steps. Return integer.", max_tokens=64, temperature=0.0)
            self.track_tokens(resp)
            data = parse_json_response(extract_text_from_response(resp))
            return max(0, int(data.get("causal_chain_length", 1)))
        except Exception:
            return None

    def _flush_theory_telemetry(self, hyp_id, matching_facts=None):
        """Flush all theory run telemetry for a saved hypothesis."""
        try:
            mf = matching_facts or getattr(self, "_matching_facts_for_telemetry", None) or []
            # Module D: Depth metrics
            depth = self._compute_depth_metrics(mf)
            kwargs = {
                "facts_per_domain": depth["facts_per_domain"],
                "min_depth": depth["min_depth"],
                "max_depth": depth["max_depth"],
                "domain_count": depth["domain_count"],
                "depth_concentration": depth["depth_concentration"],
            }

            # Module A: Market awareness telemetry
            if hasattr(self, "_awareness_telemetry") and self._awareness_telemetry:
                kwargs["market_awareness_telemetry"] = self._awareness_telemetry

            # Module G: Stage timestamps
            if hasattr(self, "_stage_timestamps") and self._stage_timestamps:
                kwargs["stage_timestamps"] = self._stage_timestamps

            # Module H: Entity specificity (computed earlier, stored on self)
            if hasattr(self, "_entity_specificity_score") and self._entity_specificity_score is not None:
                kwargs["entity_specificity_score"] = self._entity_specificity_score

            # Module J: Causal chain length (computed earlier, stored on self)
            if hasattr(self, "_causal_chain_length") and self._causal_chain_length is not None:
                kwargs["causal_chain_length"] = self._causal_chain_length

            # Module F: Lightweight awareness (computed earlier, stored on self)
            if hasattr(self, "_lightweight_awareness_level") and self._lightweight_awareness_level is not None:
                kwargs["lightweight_awareness_level"] = self._lightweight_awareness_level

            # Module K: Score components (stored on self during _score_and_save)
            for attr in ("mechanism_integrity", "domain_bonus", "chain_bonus",
                         "fact_confidence_adj", "actionability_multiplier", "confidence_penalty"):
                val = getattr(self, f"_score_{attr}", None)
                if val is not None:
                    kwargs[attr] = val

            # Stratum tracking
            if getattr(config, 'CURRENT_STRATUM', None):
                kwargs["stratum"] = config.CURRENT_STRATUM
            if getattr(config, 'CURRENT_COLLISION_MODE', None):
                kwargs["collision_mode"] = config.CURRENT_COLLISION_MODE

            # Module L: Cited fact IDs
            if hasattr(self, "_cited_fact_ids") and self._cited_fact_ids is not None:
                kwargs["cited_fact_ids"] = self._cited_fact_ids

            update_hypothesis_telemetry(hyp_id, **kwargs)

            # Bug 1 fix: read-back verification
            try:
                from database import get_connection
                _conn = get_connection()
                _row = _conn.execute("SELECT min_depth FROM hypotheses WHERE id = ?", (hyp_id,)).fetchone()
                _conn.close()
                if _row and _row[0] is None:
                    print_info(f"WARNING: Depth metrics read-back NULL for hyp {hyp_id}")
            except Exception:
                pass

            # Module B: Edge recovery event (deferred — needs hyp_id)
            if hasattr(self, "_pending_edge_recovery") and self._pending_edge_recovery:
                save_edge_recovery_event(hypothesis_id=hyp_id, **self._pending_edge_recovery)
                self._pending_edge_recovery = None
        except Exception as e:
            print_info(f"Theory telemetry flush failed (non-fatal): {e}")

    def run(self):
        print_cycle_header(self.cycle_num, "COLLISION", "Searching for fact collisions...")

        try:
            # Get recent anomalies and check fact count
            anomalies = get_recent_anomalies(COLLISION_LOOKBACK_ANOMALIES)
            facts_count = get_recent_facts_count(COLLISION_LOOKBACK_FACTS)

            if not anomalies:
                print_info("No anomalies to collide yet. Need more ingest cycles.")
                save_cycle_log("COLLISION", "none", 0, 0, 0,
                               time.time() - self.start_time, "completed")
                return

            if facts_count < 5:
                print_info(f"Only {facts_count} facts in database. Need more data for collisions.")
                save_cycle_log("COLLISION", "none", 0, 0, 0,
                               time.time() - self.start_time, "completed")
                return

            print_phase("COLLISION", f"Evaluating {len(anomalies)} anomalies against {facts_count} facts...")

            # PRIORITISE anomalies with cross-domain implication potential
            # Score each anomaly by how many different source types its implications could bridge to
            _pharma_keywords = {"pharmaceutical", "pharma", "fda", "drug", "generic",
                                "biotech", "anda", "nda", "cgmp", "clinical", "therapeutic",
                                "orphan", "biosimilar", "monoclonal", "oncology", "pdufa",
                                "gdufa", "505(b)", "paragraph iv", "orange book"}

            def _is_pharma_adjacent(anom):
                src = (anom.get("source_type", "") or "").lower()
                if src == "pharmaceutical":
                    return True
                desc = (anom.get("anomaly_description", "") or "").lower()
                fact_title = (anom.get("fact_title", "") or "").lower()
                text = desc + " " + fact_title
                return sum(1 for kw in _pharma_keywords if kw in text) >= 2

            # Classify each anomaly's professional world (not source type — actual content domain)
            _domain_keywords = {
                "pharma": {"pharmaceutical", "pharma", "fda", "drug", "generic", "anda", "biotech",
                           "cgmp", "clinical", "therapeutic", "orphan", "biosimilar", "oncology", "pdufa"},
                "insurance": {"insurance", "actuarial", "premium", "underwriting", "claims ratio",
                              "combined ratio", "loss ratio", "d&o", "liability coverage", "reinsurance"},
                "energy": {"ferc", "grid", "interconnection", "utility", "transmission", "megawatt",
                           "power plant", "renewable", "nerc", "cip-"},
                "cre": {"reit", "cmbs", "cap rate", "occupancy", "commercial real estate", "vacancy",
                        "tenant", "lease", "mortgage", "appraisal"},
                "construction": {"construction", "contractor", "lumber", "cement", "rsmeans", "prevailing wage",
                                 "building code", "surety bond"},
                "shipping": {"shipping", "freight", "vessel", "maritime", "port", "cargo", "baltic",
                             "container", "tanker", "bulk carrier"},
                "banking": {"bank", "loan", "fdic", "deposit", "dscr", "cecl", "credit loss",
                            "underwriting standard"},
                "healthcare": {"hospital", "cms", "medicare", "nursing", "hospice", "physician",
                               "outpatient", "senior living"},
                "distressed": {"bankruptcy", "chapter 11", "liquidation", "restructuring", "warn act",
                               "creditor", "receivership", "default"},
            }

            def _get_domain(anom):
                text = ((anom.get("anomaly_description", "") or "") + " " +
                        (anom.get("fact_title", "") or "")).lower()
                best_domain = "other"
                best_count = 0
                for domain, kws in _domain_keywords.items():
                    hits = sum(1 for kw in kws if kw in text)
                    if hits > best_count:
                        best_count = hits
                        best_domain = domain
                return best_domain if best_count >= 2 else "other"

            # Load recent domain pairs for rotation guarantee
            recent_domain_pairs = get_recent_hypothesis_domain_pairs(ROTATION_WINDOW)

            scored_anomalies = []
            for a in anomalies:
                score = a.get("weirdness_score", 5)
                # Boost anomalies that have implications (cross-domain bridge potential)
                try:
                    imps = json.loads(a.get("fact_implications", "[]") or "[]")
                    if imps and len(imps) >= 3:
                        score += 5
                    elif imps:
                        score += 2
                except (json.JSONDecodeError, TypeError):
                    pass
                a["_domain"] = _get_domain(a)

                # ROTATION: graduated penalty for domains that dominated recent hypotheses
                if recent_domain_pairs:
                    domain = a["_domain"]
                    appearances = sum(1 for pair in recent_domain_pairs if domain in pair)
                    if appearances >= 2:
                        score = int(score * ROTATION_PENALTY_SECOND)  # 0.4 — heavy penalty
                    elif appearances == 1:
                        score = int(score * ROTATION_PENALTY_FIRST)   # 0.7 — light penalty

                scored_anomalies.append((score, a))

            scored_anomalies.sort(key=lambda x: x[0], reverse=True)

            # ANOMALY SELECTION: depends on focus mode
            if FOCUS_MODE == "core":
                # ANCHOR/BRIDGE: 1 anchor from core domains + 2 bridges from non-core
                anchor_pool = [(s, a) for s, a in scored_anomalies
                               if a.get("source_type", "") in CORE_SOURCE_TYPES]
                bridge_pool = [(s, a) for s, a in scored_anomalies
                               if a.get("source_type", "") not in CORE_SOURCE_TYPES]

                anomalies_to_check = []

                # Pick best anchor
                if anchor_pool:
                    anomalies_to_check.append(anchor_pool[0][1])

                # Pick 2 bridges from different domains
                bridge_domains_used = set()
                for _, a in bridge_pool:
                    domain = a.get("_domain", "other")
                    if domain not in bridge_domains_used or domain == "other":
                        anomalies_to_check.append(a)
                        bridge_domains_used.add(domain)
                    if len(anomalies_to_check) >= 3:
                        break

                # Fallback: if bridge pool is too thin, relax and fill from any scored anomalies
                if len(anomalies_to_check) < 3:
                    print_info("Core mode: bridge pool thin, relaxing constraint for this cycle")
                    for _, a in scored_anomalies:
                        if a not in anomalies_to_check and len(anomalies_to_check) < 3:
                            anomalies_to_check.append(a)

            else:
                # DEFAULT MODE: domain concentration ceiling — no domain gets more than 1 of 3 slots
                anomalies_to_check = []
                domains_used = set()
                for _, a in scored_anomalies:
                    domain = a.get("_domain", "other")
                    if domain not in domains_used or domain == "other":
                        anomalies_to_check.append(a)
                        domains_used.add(domain)
                    if len(anomalies_to_check) >= 3:
                        break
                # Fill remaining from top scores if ceiling was too strict
                if len(anomalies_to_check) < 3:
                    for _, a in scored_anomalies:
                        if a not in anomalies_to_check and len(anomalies_to_check) < 3:
                            anomalies_to_check.append(a)

            if anomalies_to_check:
                top_sources = [a.get("source_type", "?") for a in anomalies_to_check]
                top_domains = [a.get("_domain", "?") for a in anomalies_to_check]
                mode_tag = f"[{FOCUS_MODE}]" if FOCUS_MODE != "default" else ""
                print_info(f"Priority anomalies {mode_tag}: {', '.join(top_sources)} (domains: {', '.join(top_domains)})")

            for anomaly in anomalies_to_check:
                self._process_anomaly(anomaly)

            duration = time.time() - self.start_time
            print(f"\n  {C.YELLOW}⚡ Collisions: {self.collisions_found} | Hypotheses: {self.hypotheses_formed} | Survived: {self.hypotheses_survived}{C.RESET}")
            print(f"  {C.DIM}Tokens: {self.tokens_used:,} | Duration: {duration:.0f}s{C.RESET}")

            save_cycle_log("COLLISION", "multi", 0, self.tokens_used,
                           self.hypotheses_survived, duration, "completed")

            # Theory proof layer: flush evidence to database
            if self._theory_recorder:
                try:
                    self._theory_recorder.flush_to_db()
                    summary = self._theory_recorder.get_session_summary()
                    if summary["total_collisions_observed"] > 0:
                        layer_hits = {k: v for k, v in summary.get("layer_evidence_counts", {}).items() if v > 0}
                        layers_active = summary.get("layers_with_evidence", 0)
                        auto_confirmed = summary.get("autopoiesis_confirmed", 0)
                        pred_acc = summary.get("prediction_accuracy", 0)
                        print(f"  {C.CYAN}📊 Theory: {summary['total_collisions_observed']} collision(s) → "
                              f"{layers_active}/13 layers, "
                              f"pred accuracy {pred_acc:.0%}, "
                              f"autopoiesis {auto_confirmed}{C.RESET}")
                except Exception:
                    pass

            # Update domain productivity scores (feedback loop)
            self._update_domain_productivity()

        except anthropic.RateLimitError as e:
            print_error(f"Rate limit hit: {e}")
            save_cycle_log("COLLISION", "none", 0, self.tokens_used, 0,
                           time.time() - self.start_time, "rate_limit", str(e))
            time.sleep(60)

        except Exception as e:
            print_error(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            save_cycle_log("COLLISION", "none", 0, self.tokens_used, 0,
                           time.time() - self.start_time, "error", str(e))
            if check_credit_error(e):
                raise SystemExit("Credits exhausted")

    def _update_domain_productivity(self):
        """Calculate and save domain productivity scores for the feedback loop."""
        try:
            from database import get_connection
            conn = get_connection()
            cursor = conn.cursor()

            # Get fact counts per source type
            cursor.execute("SELECT source_type, COUNT(*) as cnt FROM raw_facts GROUP BY source_type")
            fact_counts = {row["source_type"]: row["cnt"] for row in cursor.fetchall()}

            # Get survived hypothesis source types
            cursor.execute("""
                SELECT c.source_types
                FROM hypotheses h
                JOIN collisions c ON h.collision_id = c.id
                WHERE h.survived_kill = 1 AND c.source_types IS NOT NULL AND c.source_types != ''
            """)
            survived_by_type = collections.Counter()
            for row in cursor.fetchall():
                for st in row["source_types"].split(","):
                    st = st.strip()
                    if st:
                        survived_by_type[st] += 1
            conn.close()

            # Calculate productivity per source type
            metrics = []
            log_lines = []
            for st, fact_count in sorted(fact_counts.items()):
                if st == "test":
                    continue
                survived = survived_by_type.get(st, 0)
                # Only score types with enough facts to be meaningful
                if fact_count >= PRODUCTIVITY_MIN_FACTS:
                    prod_score = survived / fact_count
                else:
                    prod_score = 0.0  # Below threshold — falls through to deficit weights
                metrics.append({
                    "source_type": st,
                    "facts_count": fact_count,
                    "hypotheses_survived": survived,
                    "productivity_score": prod_score,
                })
                log_lines.append(f"{st}: {fact_count} facts, {survived} survived, prod={prod_score:.4f}")

            if metrics:
                save_domain_productivity(metrics)
                print_info(f"Domain productivity updated: {len(metrics)} types")
                for line in log_lines[:6]:  # Log top 6 for diagnostics
                    print_info(f"  {line}")

        except Exception as e:
            print_info(f"Productivity update failed (non-fatal): {e}")

    def _process_anomaly(self, anomaly):
        """Try to collide a single anomaly with facts from different sources."""
        # Extract entities from the anomaly's underlying fact
        try:
            entities = json.loads(anomaly.get("fact_entities", "[]") or "[]")
        except (json.JSONDecodeError, TypeError):
            entities = []

        # Extract implications from the anomaly's underlying fact
        try:
            fact_implications = json.loads(anomaly.get("fact_implications", "[]") or "[]")
        except (json.JSONDecodeError, TypeError):
            fact_implications = []

        if not entities and not fact_implications:
            mark_anomaly_attempted(anomaly['id'])
            return

        # === PARALLEL MATCHING: run all strategies simultaneously, reserve slots for each ===
        # This ensures every collision has breadth (implication bridges across domains)
        # AND specificity (same-entity facts for named companies and verifiable data).

        # Pool 1: IMPLICATION MATCHING (4 reserved slots — cross-domain breadth)
        implication_pool = []
        if fact_implications:
            implication_pool = search_facts_by_implications(
                fact_implications,
                exclude_source_type=anomaly.get("source_type"),
                days=COLLISION_LOOKBACK_FACTS,
            )
            if implication_pool:
                print_info(f"Implication matching found {len(implication_pool)} structural bridges")

        # Pool 2: ENTITY MATCHING (3 reserved slots — same-entity specificity)
        entity_pool = []
        if entities:
            entity_pool = search_facts_by_entities(
                entities,
                exclude_source_type=anomaly.get("source_type"),
                days=COLLISION_LOOKBACK_FACTS,
            )

        # Pool 3: KEYWORD + ACTOR MATCHING (3 reserved slots — middle ground)
        keyword_pool = []
        try:
            keywords = anomaly.get("fact_keywords", "").split(",")
            keywords = [k.strip() for k in keywords if k.strip()]
            if keywords:
                keyword_pool = search_facts_by_keywords(keywords, days=COLLISION_LOOKBACK_FACTS)
        except Exception:
            pass

        # Add cross-implication actors to keyword pool
        if fact_implications:
            try:
                import re as _re
                impl_text = " ".join(fact_implications)
                actor_candidates = _re.findall(r'(?:then|for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', impl_text)
                actor_candidates = [a.strip() for a in actor_candidates if len(a) > 5 and a.lower() not in
                    ("united states", "european union", "middle east", "south korea", "north america")][:5]
                if actor_candidates:
                    actor_facts = search_facts_by_entities(
                        actor_candidates,
                        exclude_source_type=anomaly.get("source_type"),
                        days=COLLISION_LOOKBACK_FACTS,
                    )
                    existing_ids = {f["id"] for f in keyword_pool}
                    for af in actor_facts:
                        if af["id"] not in existing_ids:
                            keyword_pool.append(af)
                            existing_ids.add(af["id"])
            except Exception:
                pass

        # Synonym expansion — always run, feeds into implication pool
        if fact_implications:
            try:
                longest_imp = max(fact_implications, key=len) if fact_implications else ""
                if len(longest_imp) > 30:
                    expand_response = call_text(
                        f"""Rewrite this mechanism in 3 alternative phrasings using vocabulary from DIFFERENT professional domains.

ORIGINAL: {longest_imp[:300]}

Rules:
- Each rephrasing must use terminology from a different professional field
- Keep the core mechanism the same, change the vocabulary
- Example: "actuarial repricing uncertainty" → "reinsurance rate adjustment" OR "climate risk premium recalculation" OR "underwriting model input invalidation"

Respond with ONLY a JSON object:
{{{{"alternatives": ["rephrasing 1", "rephrasing 2", "rephrasing 3"]}}}}""",
                        system="You are a cross-domain vocabulary translator.",
                        max_tokens=256,
                        temperature=0.5,
                        model=MODEL_FAST,
                    )
                    self.track_tokens(expand_response)
                    expand_text = extract_text_from_response(expand_response)
                    expand_data = parse_json_response(expand_text)
                    alternatives = expand_data.get("alternatives", [])
                    if alternatives:
                        synonym_facts = search_facts_by_implications(
                            alternatives,
                            exclude_source_type=anomaly.get("source_type"),
                            days=COLLISION_LOOKBACK_FACTS,
                        )
                        existing_ids = {f["id"] for f in implication_pool}
                        for sf in synonym_facts:
                            if sf["id"] not in existing_ids:
                                implication_pool.append(sf)
                                existing_ids.add(sf["id"])
            except Exception as _e:
                logger.info(f"pool_synonym_fail cycle={self.cycle_num} err={type(_e).__name__}: {_e}")

        # Module I: Single-domain control — filter pools by stratum constraints
        _anchor_source_type = anomaly.get("source_type", "unknown")
        if getattr(config, 'FORCE_SINGLE_DOMAIN', False):
            # Only keep facts with the same source_type as the anchor anomaly
            implication_pool = [f for f in implication_pool if f.get("source_type") == _anchor_source_type]
            entity_pool = [f for f in entity_pool if f.get("source_type") == _anchor_source_type]
            keyword_pool = [f for f in keyword_pool if f.get("source_type") == _anchor_source_type]
        elif getattr(config, 'COLLISION_MAX_DOMAINS', None) is not None:
            # Domain count will be enforced after blending
            pass

        # Pool 4: MODEL VULNERABILITY MATCHING (Strategy 4)
        # Two facts challenging the same ASSUMPTION or targeting the same METHODOLOGY
        # are exactly the cross-domain collisions HUNTER exists to find
        model_field_pool = []
        try:
            anchor_mv_raw = anomaly.get("fact_model_vulnerability")
            if anchor_mv_raw and anchor_mv_raw != "null":
                anchor_mv = json.loads(anchor_mv_raw) if isinstance(anchor_mv_raw, str) else anchor_mv_raw
                if isinstance(anchor_mv, dict):
                    model_field_pool = search_facts_by_model_fields(
                        anchor_mv,
                        exclude_source_type=anomaly.get("source_type"),
                        days=COLLISION_LOOKBACK_FACTS,
                    )
                    if model_field_pool:
                        print_info(f"Model vulnerability matching found {len(model_field_pool)} structural bridges")
        except Exception as _e:
            logger.info(f"pool_model_field_fail cycle={self.cycle_num} err={type(_e).__name__}: {_e}")

        # Pool 5: CAUSAL GRAPH TRAVERSAL (Branch 2)
        # Find facts connected through causal chains across domain boundaries
        causal_pool = []
        try:
            from database import get_connection as _gc2
            _conn2 = _gc2()
            _cur2 = _conn2.cursor()
            _cur2.execute("SELECT effect_node_lower FROM causal_edges WHERE source_fact_id = ?",
                          (anomaly.get("raw_fact_id"),))
            anchor_effects = [r[0] for r in _cur2.fetchall()]
            _cur2.execute("SELECT cause_node_lower FROM causal_edges WHERE source_fact_id = ?",
                          (anomaly.get("raw_fact_id"),))
            anchor_causes = [r[0] for r in _cur2.fetchall()]
            _conn2.close()

            if anchor_effects or anchor_causes:
                paths = find_causal_paths(
                    anchor_effects + anchor_causes,
                    max_hops=4,
                    exclude_source_type=anomaly.get("source_type"),
                )
                causal_fact_ids = set()
                for path in paths[:10]:
                    for edge in path:
                        causal_fact_ids.add(edge["source_fact_id"])
                if causal_fact_ids:
                    causal_pool = get_facts_by_ids(list(causal_fact_ids)[:20])
                    if causal_pool:
                        print_info(f"Causal graph traversal found {len(causal_pool)} connected facts via {len(paths)} paths")

                # Check for contradictory paths (opposite predictions on same node)
                _contradictions = []
                for node in (anchor_effects + anchor_causes)[:5]:
                    _contradictions.extend(find_contradictory_paths(node))
                if _contradictions:
                    self._causal_contradictions = _contradictions[:3]
                    print(f"  {C.MAGENTA}⚡ {len(_contradictions)} causal CONTRADICTIONS detected{C.RESET}")
        except Exception as e:
            logger.warning(f"Branch 2: causal pool construction failed: {e}")

        # Pool 6: EMBEDDING SIMILARITY (Branch 1)
        # Find facts with similar structural consequences but from distant source types
        embedding_pool = []
        try:
            anchor_implications = json.loads(anomaly.get("fact_implications", "[]") or "[]")
            anchor_mv_raw = anomaly.get("fact_model_vulnerability")
            anchor_mv = None
            if anchor_mv_raw and anchor_mv_raw != "null":
                anchor_mv = json.loads(anchor_mv_raw) if isinstance(anchor_mv_raw, str) else anchor_mv_raw

            query_embedding = compute_fact_embedding(anchor_implications, anchor_mv)
            if query_embedding is not None:
                embedding_pool = search_facts_by_embedding(
                    query_embedding.tobytes(),
                    exclude_source_type=anomaly.get("source_type"),
                    days=COLLISION_LOOKBACK_FACTS,
                    k=50,
                )
                if embedding_pool:
                    print_info(f"Embedding similarity found {len(embedding_pool)} structural matches")
        except Exception as e:
            logger.warning(f"Branch 1: embedding pool search failed: {e}")

        # Pool 7: BELIEF-REALITY MATCHING (Prompt 5)
        # When anchor has a market_belief, find exogenous facts that contradict it.
        # When anchor is exogenous, find market beliefs it might contradict.
        belief_reality_matches = []
        try:
            anchor_mb_raw = anomaly.get("fact_market_belief")
            anchor_reflexivity = anomaly.get("fact_reflexivity_tag")

            if anchor_mb_raw and anchor_mb_raw != "null":
                # Anchor HAS a market belief — find exogenous facts that contradict it
                anchor_mb = json.loads(anchor_mb_raw) if isinstance(anchor_mb_raw, str) else anchor_mb_raw
                if isinstance(anchor_mb, dict) and anchor_mb.get("asset"):
                    # Extract keywords from the belief's asset and text
                    belief_kws = []
                    for field in ("asset", "belief_text", "source_of_belief"):
                        val = anchor_mb.get(field, "")
                        if val:
                            belief_kws.extend([w for w in val.split() if len(w) > 3])
                    belief_kws = list(set(belief_kws))[:8]

                    if belief_kws:
                        exo_facts = search_exogenous_facts_for_belief(
                            belief_kws,
                            exclude_source_type=anomaly.get("source_type"),
                            days=COLLISION_LOOKBACK_FACTS,
                        )
                        # Test each exogenous fact against the belief
                        for ef in exo_facts[:5]:
                            try:
                                br_prompt = BELIEF_REALITY_TEST_PROMPT.format(
                                    belief_asset=anchor_mb.get("asset", "unknown"),
                                    belief_text=anchor_mb.get("belief_text", ""),
                                    belief_type=anchor_mb.get("belief_type", "unknown"),
                                    implied_value=anchor_mb.get("implied_value", "unknown"),
                                    source_of_belief=anchor_mb.get("source_of_belief", ""),
                                    belief_confidence=anchor_mb.get("confidence", 0.5),
                                    fact_id=ef["id"],
                                    fact_source_type=ef.get("source_type", "unknown"),
                                    fact_title=ef.get("title", ""),
                                    fact_content=ef.get("raw_content", "")[:400],
                                )
                                br_response = call_text(br_prompt, max_tokens=512, temperature=0.2, model=MODEL_FAST)
                                self.track_tokens(br_response)
                                br_text = extract_text_from_response(br_response)
                                br_data = parse_json_response(br_text)

                                if br_data.get("contradiction"):
                                    belief_reality_matches.append({
                                        "belief_fact_id": anomaly.get("raw_fact_id"),
                                        "reality_fact_id": ef["id"],
                                        "belief": anchor_mb,
                                        "direction": br_data.get("direction", "unknown"),
                                        "magnitude_pct": br_data.get("estimated_magnitude_pct", 0),
                                        "timeline_days": br_data.get("timeline_days", 0),
                                        "forcing_function": br_data.get("forcing_function", ""),
                                        "description": br_data.get("description", ""),
                                        "confidence": br_data.get("confidence", 0),
                                    })
                                    # Also add this fact to the collision pool
                                    existing_ids_all = {f["id"] for f in (implication_pool + entity_pool + keyword_pool
                                                        + model_field_pool + causal_pool + embedding_pool)}
                                    if ef["id"] not in existing_ids_all:
                                        embedding_pool.append(ef)  # Merge into embedding pool for blend
                                    print(f"  {C.MAGENTA}🎯 BELIEF-REALITY CONTRADICTION: {br_data.get('direction', '?')} "
                                          f"{br_data.get('estimated_magnitude_pct', '?')}% on {anchor_mb.get('asset', '?')}{C.RESET}")
                            except Exception:
                                pass

            elif anchor_reflexivity == "exogenous" and entities:
                # Anchor is exogenous — find market beliefs it might contradict
                belief_facts = search_facts_with_beliefs_for_asset(
                    entities[:5],
                    exclude_source_type=anomaly.get("source_type"),
                    days=COLLISION_LOOKBACK_FACTS,
                )
                for bf in belief_facts[:5]:
                    try:
                        bf_mb_raw = bf.get("market_belief")
                        if not bf_mb_raw or bf_mb_raw == "null":
                            continue
                        bf_mb = json.loads(bf_mb_raw) if isinstance(bf_mb_raw, str) else bf_mb_raw
                        if not isinstance(bf_mb, dict):
                            continue

                        # The anchor fact is the "reality", the belief fact has the "belief"
                        anchor_fact_content = anomaly.get("fact_raw_content", anomaly.get("anomaly_description", ""))
                        br_prompt = BELIEF_REALITY_TEST_PROMPT.format(
                            belief_asset=bf_mb.get("asset", "unknown"),
                            belief_text=bf_mb.get("belief_text", ""),
                            belief_type=bf_mb.get("belief_type", "unknown"),
                            implied_value=bf_mb.get("implied_value", "unknown"),
                            source_of_belief=bf_mb.get("source_of_belief", ""),
                            belief_confidence=bf_mb.get("confidence", 0.5),
                            fact_id=anomaly.get("raw_fact_id", 0),
                            fact_source_type=anomaly.get("source_type", "unknown"),
                            fact_title=anomaly.get("fact_title", ""),
                            fact_content=anchor_fact_content[:400],
                        )
                        br_response = call_text(br_prompt, max_tokens=512, temperature=0.2, model=MODEL_FAST)
                        self.track_tokens(br_response)
                        br_text = extract_text_from_response(br_response)
                        br_data = parse_json_response(br_text)

                        if br_data.get("contradiction"):
                            belief_reality_matches.append({
                                "belief_fact_id": bf["id"],
                                "reality_fact_id": anomaly.get("raw_fact_id"),
                                "belief": bf_mb,
                                "direction": br_data.get("direction", "unknown"),
                                "magnitude_pct": br_data.get("estimated_magnitude_pct", 0),
                                "timeline_days": br_data.get("timeline_days", 0),
                                "forcing_function": br_data.get("forcing_function", ""),
                                "description": br_data.get("description", ""),
                                "confidence": br_data.get("confidence", 0),
                            })
                            # Add the belief-carrying fact to collision pool
                            existing_ids_all = {f["id"] for f in (implication_pool + entity_pool + keyword_pool
                                                + model_field_pool + causal_pool + embedding_pool)}
                            if bf["id"] not in existing_ids_all:
                                embedding_pool.append(bf)
                            print(f"  {C.MAGENTA}🎯 BELIEF-REALITY CONTRADICTION: {br_data.get('direction', '?')} "
                                  f"{br_data.get('estimated_magnitude_pct', '?')}% on {bf_mb.get('asset', '?')}{C.RESET}")
                    except Exception:
                        pass

            if belief_reality_matches:
                print_info(f"Belief-reality matching found {len(belief_reality_matches)} contradiction(s)")
        except Exception:
            pass

        # Module I: Single-domain control — filter all pools
        if getattr(config, 'FORCE_SINGLE_DOMAIN', False):
            model_field_pool = [f for f in model_field_pool if f.get("source_type") == _anchor_source_type]
            causal_pool = [f for f in causal_pool if f.get("source_type") == _anchor_source_type]
            embedding_pool = [f for f in embedding_pool if f.get("source_type") == _anchor_source_type]

        # Check we have something to work with
        if not implication_pool and not entity_pool and not keyword_pool and not model_field_pool and not causal_pool and not embedding_pool:
            logger.info(
                f"pool_exhausted cycle={self.cycle_num} anomaly_id={anomaly.get('id')} "
                f"anchor_entities={len(getattr(self, '_anchor_entities', []))} "
                f"anchor_implications={len(getattr(self, '_anchor_implications', []))} "
                f"source_type={anomaly.get('source_type')}"
            )
            print_info(f"Pool exhausted for anomaly #{anomaly.get('id')} — no matches across 6 strategies")
            mark_anomaly_attempted(anomaly['id'])
            return

        # Store anchor reflexivity for scoring
        self._anchor_reflexivity_tag = anomaly.get("fact_reflexivity_tag")

        # Entity resolution — find related entities across all pools
        entity_connections = ""
        all_pool_facts = implication_pool + entity_pool + keyword_pool + model_field_pool + causal_pool + embedding_pool
        try:
            anomaly_entities_str = ", ".join(entities[:10])
            matching_entities = set()
            for f in all_pool_facts[:8]:
                try:
                    ents = json.loads(f.get("entities", "[]") or "[]")
                    matching_entities.update(ents[:5])
                except (json.JSONDecodeError, TypeError):
                    pass

            if matching_entities and entities:
                matching_entities_str = ", ".join(list(matching_entities)[:15])
                print_phase("ENTITY", f"Resolving entities: {anomaly_entities_str[:50]}...")

                prompt = ENTITY_RESOLVE_PROMPT.format(
                    source_a=anomaly.get("source_type", "unknown"),
                    entities_a=anomaly_entities_str,
                    source_b="mixed sources",
                    entities_b=matching_entities_str,
                )
                response = call_text(prompt, max_tokens=1024, temperature=0.3)
                self.track_tokens(response)
                text = extract_text_from_response(response)
                entity_data = parse_json_response(text)
                groups = entity_data.get("groups", [])
                if groups:
                    entity_connections = f"Related entity groups: {json.dumps(groups)}"
        except Exception:
            pass

        # === BLEND: 4 implication(+causal) + 2 model_field + 2 embedding + 1 entity + 1 keyword = 10 ===
        # Causal pool facts merge into implication pool (they ARE structural implications)
        existing_imp_ids = {f["id"] for f in implication_pool}
        for cf in causal_pool:
            if cf["id"] not in existing_imp_ids:
                implication_pool.append(cf)
                existing_imp_ids.add(cf["id"])
        matching_facts = []
        used_ids = set()

        # Determine if this anomaly is a core anchor (for fact pool prioritisation)
        _anomaly_is_core_anchor = (FOCUS_MODE == "core" and
                                    anomaly.get("source_type", "") in CORE_SOURCE_TYPES)

        def _pick_diverse(pool, n):
            """Pick up to n facts from pool, one per source type first, then backfill.
            Enforces minimum 2 distinct source types when pool has them and n >= 2.
            In Core mode: anchor anomalies prefer non-core facts (maximise bridge distance).
            """
            # In Core mode, sort pool to prefer cross-domain facts
            sorted_pool = pool
            if FOCUS_MODE == "core":
                if _anomaly_is_core_anchor:
                    sorted_pool = sorted(pool, key=lambda f: f.get("source_type", "") in CORE_SOURCE_TYPES)
                else:
                    sorted_pool = sorted(pool, key=lambda f: f.get("source_type", "") not in CORE_SOURCE_TYPES)

            picked = []
            by_source = {}
            for f in sorted_pool:
                if f["id"] in used_ids:
                    continue
                st = f.get("source_type", "unknown")
                if st not in by_source:
                    by_source[st] = f
            # Phase 1: one fact per source type (diversity-first)
            for f in list(by_source.values()):
                if len(picked) < n:
                    picked.append(f)
                    used_ids.add(f["id"])
            # Phase 2: backfill remaining slots, but skip same-type facts if we
            # haven't yet reached 2 distinct types and alternatives exist
            picked_types = {f.get("source_type", "unknown") for f in picked}
            for f in sorted_pool:
                if f["id"] not in used_ids and len(picked) < n:
                    ft = f.get("source_type", "unknown")
                    # If we only have 1 type so far and n >= 2, prefer a different type
                    if len(picked_types) < 2 and n >= 2 and ft in picked_types:
                        # Check if there's a different-type fact still available
                        alt = any(g["id"] not in used_ids and g.get("source_type", "unknown") != ft
                                  for g in sorted_pool)
                        if alt:
                            continue  # skip this same-type fact, pick a different-type one first
                    picked.append(f)
                    used_ids.add(f["id"])
                    picked_types.add(ft)
                    used_ids.add(f["id"])
            return picked

        # Implication facts get priority (includes causal pool merged in)
        imp_picks = _pick_diverse(implication_pool, 4)
        matching_facts.extend(imp_picks)

        # Model vulnerability facts — structural bridges (same assumption/methodology)
        mv_picks = _pick_diverse(model_field_pool, 2)
        matching_facts.extend(mv_picks)

        # Embedding facts — same consequence, different world
        emb_picks = _pick_diverse(embedding_pool, 2)
        matching_facts.extend(emb_picks)

        # Entity facts — same-company specificity
        ent_picks = _pick_diverse(entity_pool, 1)
        matching_facts.extend(ent_picks)

        # Keyword/actor facts — middle ground
        kw_picks = _pick_diverse(keyword_pool, 1)
        matching_facts.extend(kw_picks)

        pool_summary = f"Blend: {len(imp_picks)} imp + {len(mv_picks)} mv + {len(emb_picks)} emb + {len(ent_picks)} ent + {len(kw_picks)} kw = {len(matching_facts)} facts"
        print_info(pool_summary)

        if not matching_facts:
            mark_anomaly_attempted(anomaly['id'])
            return

        # Select top 10 facts MAXIMISING source type diversity
        # Pick one fact from each unique source type first, then fill remaining slots
        # No domain bans — the concentration ceiling on anomalies prevents monopolisation.
        # Any domain can participate in a collision as long as the collision is cross-domain.
        MAX_COLLISION_FACTS = 10
        by_source = {}
        for f in matching_facts:
            st = f.get("source_type", "unknown")
            if st not in by_source:
                by_source[st] = f
        diverse_picks = list(by_source.values())[:MAX_COLLISION_FACTS]

        if len(diverse_picks) < MAX_COLLISION_FACTS:
            existing_ids = {f["id"] for f in diverse_picks}
            for f in matching_facts:
                if f["id"] not in existing_ids and len(diverse_picks) < MAX_COLLISION_FACTS:
                    diverse_picks.append(f)
                    existing_ids.add(f["id"])

        matching_facts = diverse_picks if diverse_picks else matching_facts[:MAX_COLLISION_FACTS]

        # Module I: Enforce domain count constraints (MIN/MAX_DOMAINS)
        _fact_source_types = set(f.get("source_type", "unknown") for f in matching_facts)
        _fact_source_types.add(_anchor_source_type)  # include the anomaly's own source type
        _num_domains = len(_fact_source_types)
        _min_doms = getattr(config, 'COLLISION_MIN_DOMAINS', 2)
        _max_doms = getattr(config, 'COLLISION_MAX_DOMAINS', None)
        if _num_domains < _min_doms:
            print_info(f"Domain count {_num_domains} < min {_min_doms}. Skipping anomaly.")
            mark_anomaly_attempted(anomaly['id'])
            return
        if _max_doms is not None and _num_domains > _max_doms:
            # Trim to max domains: keep anchor + (_max_doms - 1) other source types
            _allowed_types = {_anchor_source_type}
            for f in matching_facts:
                st = f.get("source_type", "unknown")
                if st not in _allowed_types and len(_allowed_types) < _max_doms:
                    _allowed_types.add(st)
            matching_facts = [f for f in matching_facts if f.get("source_type", "unknown") in _allowed_types]

        # === DISRUPTION-ASSUMPTION VALIDATION ===
        # Check if any fact's model_vulnerability DISRUPTION logically invalidates
        # another fact's model_vulnerability ASSUMPTION. This is the structural test
        # that replaces vocabulary-based matching with logic-based matching.
        # Only runs on facts that have model_vulnerability data.

        # Collect facts with model_vulnerability from the anomaly's fact AND matching facts
        disruption_facts = []  # Facts that have a disruption (could break a model)
        assumption_facts = []  # Facts that describe a model with an assumption (could be broken)

        # Check the anomaly's source fact for model_vulnerability
        try:
            from database import get_connection as _gc
            _conn = _gc()
            _cur = _conn.cursor()
            _cur.execute("SELECT model_vulnerability FROM raw_facts WHERE id = ?", (anomaly.get("raw_fact_id"),))
            _row = _cur.fetchone()
            if _row and _row["model_vulnerability"] and _row["model_vulnerability"] != "null":
                _mv = json.loads(_row["model_vulnerability"]) if isinstance(_row["model_vulnerability"], str) else _row["model_vulnerability"]
                if isinstance(_mv, dict) and _mv.get("disruption"):
                    disruption_facts.append({"fact_id": anomaly.get("raw_fact_id"), "mv": _mv,
                                             "title": anomaly.get("fact_title", ""), "source_type": anomaly.get("source_type", "")})
                if isinstance(_mv, dict) and _mv.get("assumption"):
                    assumption_facts.append({"fact_id": anomaly.get("raw_fact_id"), "mv": _mv,
                                             "title": anomaly.get("fact_title", ""), "source_type": anomaly.get("source_type", "")})
            _conn.close()
        except Exception as _e:
            logger.warning(f"mv_anomaly_load_fail cycle={self.cycle_num} err={type(_e).__name__}: {_e}")

        # Check matching facts for model_vulnerability
        _mv_parse_failures = 0
        for f in matching_facts:
            mv_raw = f.get("model_vulnerability")
            if mv_raw and mv_raw != "null":
                try:
                    mv = json.loads(mv_raw) if isinstance(mv_raw, str) else mv_raw
                    if isinstance(mv, dict):
                        if mv.get("disruption"):
                            disruption_facts.append({"fact_id": f["id"], "mv": mv,
                                                     "title": f.get("title", ""), "source_type": f.get("source_type", "")})
                        if mv.get("assumption"):
                            assumption_facts.append({"fact_id": f["id"], "mv": mv,
                                                     "title": f.get("title", ""), "source_type": f.get("source_type", "")})
                except (json.JSONDecodeError, TypeError):
                    _mv_parse_failures += 1
        if _mv_parse_failures > 0:
            logger.info(f"mv_parse_failures cycle={self.cycle_num} count={_mv_parse_failures}")

        # Test disruption-assumption pairs across different source types
        validated_pairs = []
        if disruption_facts and assumption_facts:
            for d_fact in disruption_facts[:3]:  # Limit to 3 disruptions to control cost
                for a_fact in assumption_facts[:3]:  # Limit to 3 assumptions
                    # Skip same-source-type pairs (not cross-domain)
                    if d_fact["source_type"] == a_fact["source_type"]:
                        continue
                    # Skip self-pairing
                    if d_fact["fact_id"] == a_fact["fact_id"]:
                        continue

                    try:
                        da_prompt = DISRUPTION_ASSUMPTION_TEST_PROMPT.format(
                            fact_a_title=d_fact["title"][:100],
                            fact_a_content=d_fact["mv"].get("disruption", "")[:200],
                            fact_a_disruption=d_fact["mv"].get("disruption", ""),
                            fact_b_title=a_fact["title"][:100],
                            fact_b_content=a_fact["mv"].get("assumption", "")[:200],
                            fact_b_assumption=a_fact["mv"].get("assumption", ""),
                            fact_b_methodology=a_fact["mv"].get("methodology", "unknown"),
                            fact_b_practitioners=a_fact["mv"].get("practitioners", "unknown"),
                        )
                        da_response = call_text(da_prompt, max_tokens=256, temperature=0.2, model=MODEL_FAST)
                        self.track_tokens(da_response)
                        da_text = extract_text_from_response(da_response)
                        da_data = parse_json_response(da_text)

                        if da_data.get("invalidates"):
                            validated_pairs.append({
                                "disruption_fact": d_fact["fact_id"],
                                "assumption_fact": a_fact["fact_id"],
                                "disruption": d_fact["mv"].get("disruption", ""),
                                "disruption_source_type": d_fact.get("source_type", "unknown"),
                                "disruption_domain": d_fact["mv"].get("practitioners", "unknown"),
                                "broken_methodology": a_fact["mv"].get("methodology", ""),
                                "broken_assumption": a_fact["mv"].get("assumption", ""),
                                "practitioners": a_fact["mv"].get("practitioners", ""),
                                "assumption_domain": a_fact.get("source_type", "unknown"),
                                "explanation": da_data.get("explanation", ""),
                            })
                            print(f"  {C.GREEN}⚡ STRUCTURAL MATCH: {d_fact['title'][:50]} → breaks {a_fact['mv'].get('methodology', '')[:50]}{C.RESET}")
                    except Exception as _e:
                        logger.info(f"da_validation_fail cycle={self.cycle_num} err={type(_e).__name__}: {_e}")

        if validated_pairs:
            print_info(f"Disruption-assumption validation found {len(validated_pairs)} structural collision(s)")

        # === TRANSITIVE CHAIN EXTENSION ===
        # For each validated pair, trace the consequence forward:
        # "What does the broken assumption change, and whose methodology does that change break next?"
        # Extends the chain link by link across professional boundaries up to MAX_CHAIN_DEPTH.
        MAX_CHAIN_DEPTH = 5
        chains = []
        _skip_chain_extension = THEORY_RUN  # Skip in theory run — too slow, not needed for telemetry

        for vp in ([] if _skip_chain_extension else validated_pairs[:2]):  # Limit to 2 chains per collision to control cost
            chain_links = [{
                "link": 1,
                "disruption": vp["disruption"],
                "broken_methodology": vp["broken_methodology"],
                "broken_assumption": vp["broken_assumption"],
                "practitioners": vp["practitioners"],
                "explanation": vp["explanation"],
                "domain": vp.get("disruption_domain", vp.get("disruption_source_type", "unknown")),
            }]
            domains_traversed = [vp.get("disruption_source_type", "unknown")]

            current_disruption = vp["disruption"]
            current_methodology = vp["broken_methodology"]
            current_assumption = vp["broken_assumption"]
            current_practitioners = vp["practitioners"]
            current_explanation = vp["explanation"]

            for depth in range(2, MAX_CHAIN_DEPTH + 1):
                try:
                    extend_prompt = CHAIN_EXTEND_PROMPT.format(
                        disruption=current_disruption[:200],
                        broken_methodology=current_methodology[:150],
                        practitioners=current_practitioners[:100],
                        broken_assumption=current_assumption[:200],
                        explanation=current_explanation[:200],
                    )
                    extend_response = call_text(extend_prompt, max_tokens=512, temperature=0.3, model=MODEL_FAST)
                    self.track_tokens(extend_response)
                    extend_text = extract_text_from_response(extend_response)
                    extend_data = parse_json_response(extend_text)

                    if extend_data.get("no_extension", True):
                        break  # Chain terminates — no valid next link

                    next_domain = extend_data.get("next_domain", "unknown")
                    transmission = extend_data.get("transmission_pathway", "")

                    # Verify the next link crosses a professional boundary
                    if next_domain in domains_traversed:
                        break  # Would loop back to an already-visited domain

                    # Verify a specific transmission pathway was named
                    if not transmission or len(transmission) < 10 or "somehow" in transmission.lower() or "might" in transmission.lower():
                        print_info(f"Chain stopped at link {depth} — no specific transmission pathway named")
                        break

                    # MECHANISM VERIFY: two-step test
                    # Step 1: do both nodes operate as claimed? (search-based)
                    # Step 2: does the output logically flow to the input? (reasoning-based)
                    arrow_valid = True
                    try:
                        verify_resp = call_text(
                            f"""Two-step arrow test.

CLAIMED: "{transmission}"
OUTPUT FROM: {current_disruption[:150]}
INPUT TO: {extend_data.get('next_methodology', '')[:150]}

STEP 1 — NODE CHECK: Does the target system actually operate the way this arrow claims? Does it actually accept this type of input as part of its standard workflow? If the thesis mischaracterises how the system works, the arrow is broken regardless of logic.

STEP 2 — LOGIC CHECK (only if Step 1 passes): Given the systems work as described, does the output logically become an input? A novel but logically sound connection is VALID. A vague thematic connection is NOT.

JSON only: {{"valid": true/false, "step_failed": "node/logic/none", "reason": "what you found"}}""",
                            system="Step 1: verify nodes are correctly characterised. Step 2: verify logical connection. Category overlap is NOT verification. Novel but sound connections ARE valid.",
                            max_tokens=256, temperature=0.1, model=MODEL_FAST,
                        )
                        self.track_tokens(verify_resp)
                        verify_text = extract_text_from_response(verify_resp)
                        verify_data = parse_json_response(verify_text)
                        if not verify_data.get("valid", True):
                            arrow_valid = False
                            print(f"  {C.YELLOW}⚠️ Arrow broken: {verify_data.get('reason', '')[:60]}{C.RESET}")
                    except Exception as _e:
                        # Fail-safe: if verification fails, treat arrow as INVALID
                        # (previously assumed valid, which let broken chains through).
                        arrow_valid = False
                        logger.warning(f"arrow_verify_fail cycle={self.cycle_num} err={type(_e).__name__}: {_e}")

                    if not arrow_valid:
                        # ONE reroute attempt
                        print(f"  {C.YELLOW}🔄 Attempting reroute...{C.RESET}")
                        try:
                            reroute_resp = call_text(
                                f"""A chain link's transmission pathway is broken. Find ONE alternative.

From: {current_disruption[:200]}
To: {extend_data.get('next_methodology', '')[:200]}
Broken pathway: {transmission[:200]}

Find a DIFFERENT specific pathway (database, filing, workflow) that transmits this information.
Must be specific, verifiable, and non-obvious.
If no valid alternative exists, respond with found: false.

JSON only: {{"found": true/false, "pathway": "specific alternative", "evidence": "source"}}""",
                                system="Find alternative transmission pathways. Do NOT force connections.",
                                max_tokens=256, temperature=0.2, model=MODEL_FAST,
                            )
                            self.track_tokens(reroute_resp)
                            reroute_text = extract_text_from_response(reroute_resp)
                            reroute_data = parse_json_response(reroute_text)

                            if reroute_data.get("found") and len(reroute_data.get("pathway", "")) > 15:
                                transmission = reroute_data["pathway"]
                                print(f"  {C.GREEN}🔄 Rerouted: {transmission[:60]}{C.RESET}")
                            else:
                                print_info(f"Chain stopped at link {depth} — broken arrow, no alternative found")
                                break
                        except Exception:
                            print_info(f"Chain stopped at link {depth} — reroute failed")
                            break

                    new_link = {
                        "link": depth,
                        "output_change": extend_data.get("output_change", ""),
                        "disruption": extend_data.get("output_change", ""),
                        "broken_methodology": extend_data.get("next_methodology", ""),
                        "broken_assumption": extend_data.get("next_assumption", ""),
                        "practitioners": extend_data.get("next_practitioners", ""),
                        "transmission_pathway": transmission,
                        "explanation": f"Transmits via: {transmission[:100]}",
                        "domain": next_domain,
                    }
                    chain_links.append(new_link)
                    domains_traversed.append(next_domain)

                    print(f"  {C.MAGENTA}🔗 Chain link {depth}: {next_domain} — {extend_data.get('next_practitioners', '')[:50]} → via {transmission[:40]}{C.RESET}")

                    # Set up for next extension
                    current_disruption = extend_data.get("output_change", "")
                    current_methodology = extend_data.get("next_methodology", "")
                    current_assumption = extend_data.get("next_assumption", "")
                    current_practitioners = extend_data.get("next_practitioners", "")
                    current_explanation = f"Output change from link {depth-1}"

                except (json.JSONDecodeError, JSONParseError):
                    break
                except Exception:
                    break

            if len(chain_links) >= 2:
                chains.append({
                    "links": chain_links,
                    "domains": domains_traversed,
                    "length": len(chain_links),
                })
                print(f"  {C.GREEN}⛓️ CHAIN DISCOVERED: {len(chain_links)} links across {len(set(domains_traversed))} domains: {' → '.join(domains_traversed)}{C.RESET}")

        # Build collision evaluation input
        # Count domains by SOURCE TYPE not domain tag -- "Finance" and "Economics"
        # on the same commodity facts is still one silo. Different source types
        # (patent vs bankruptcy vs regulation) is genuinely different professional worlds.
        facts_text = ""
        fact_ids = []
        source_types_set = set()  # THIS is what counts for multi-domain priority
        domains_set = set()  # Keep for display
        for f in matching_facts:
            facts_text += f"\n[Fact #{f['id']}] ({f['source_type']}) {f['title']}: {f.get('raw_content', '')[:300]}\n"
            fact_ids.append(f["id"])
            source_types_set.add(f.get("source_type", "unknown"))
            if f.get("domain"):
                domains_set.add(f["domain"])

        # Add the anomaly's fact
        anomaly_fact_id = anomaly.get("raw_fact_id")
        if anomaly_fact_id and anomaly_fact_id not in fact_ids:
            fact_ids.insert(0, anomaly_fact_id)
        source_types_set.add(anomaly.get("source_type", "unknown"))
        if anomaly.get("domain"):
            domains_set.add(anomaly["domain"])

        anomaly_text = f"\n[Anomaly #{anomaly['id']}] (weirdness: {anomaly.get('weirdness_score', '?')}/10) {anomaly['anomaly_description']}"

        # DEDUP CHECK -- skip if we've already collided very similar fact sets
        if is_collision_duplicate(fact_ids):
            print_info("Skipping — similar collision already exists")
            mark_anomaly_attempted(anomaly['id'])
            return

        # SKIP single-source-type collisions -- these are inner-domain, not cross-silo
        if len(source_types_set) < 2:
            print_info(f"Skipping — single source type ({list(source_types_set)[0]}), no cross-silo edge")
            mark_anomaly_attempted(anomaly['id'])
            return

        print_phase("COLLISION", f"Evaluating {len(fact_ids)} facts across {len(source_types_set)} source types ({', '.join(source_types_set)})...")

        # Inject validated structural pairs into the context
        structural_context = ""
        if validated_pairs:
            structural_context = "\n\nSTRUCTURALLY VALIDATED CONNECTIONS (disruption-assumption pairs that passed logical test):\n"
            for vp in validated_pairs:
                structural_context += (
                    f"- Disruption (Fact #{vp['disruption_fact']}): {vp['disruption'][:150]}\n"
                    f"  BREAKS → {vp['broken_methodology']} used by {vp['practitioners']}\n"
                    f"  Because: {vp['explanation'][:150]}\n\n"
                )
            structural_context += "These connections have been VALIDATED as logically sound. Build your collision around them.\n"

        # Add chain context if any chains were discovered
        if chains:
            structural_context += "\n\nTRANSITIVE CAUSAL CHAINS (multi-link pathways across professional boundaries):\n"
            for chain in chains:
                structural_context += f"\n⛓️ {chain['length']}-LINK CHAIN across {len(set(chain['domains']))} domains: {' → '.join(chain['domains'])}\n"
                for link in chain["links"]:
                    structural_context += (
                        f"  Link {link['link']}: {link.get('practitioners', '?')} use {link.get('broken_methodology', '?')}\n"
                        f"    Assumption broken: {link.get('broken_assumption', '?')[:120]}\n"
                    )
                structural_context += "\nBuild your collision around the FULL CHAIN, not just the first link. The diamond is the complete pathway from the initial disruption to the final mispriced asset.\n"

        # Add belief-reality contradiction context
        if belief_reality_matches:
            structural_context += "\n\nBELIEF-REALITY CONTRADICTIONS (market belief vs physical-world evidence):\n"
            for brm in belief_reality_matches:
                structural_context += (
                    f"- Market believes: {brm['belief'].get('belief_text', '?')[:150]}\n"
                    f"  Asset: {brm['belief'].get('asset', '?')} | Direction: {brm.get('direction', '?')} | "
                    f"Magnitude: ~{brm.get('magnitude_pct', '?')}%\n"
                    f"  Reality contradicts: {brm.get('description', '')[:150]}\n"
                    f"  Forcing function: {brm.get('forcing_function', 'unknown')[:100]} "
                    f"(~{brm.get('timeline_days', '?')} days)\n\n"
                )
            structural_context += "These belief-reality gaps are QUANTIFIABLE mispricing signals. Build your collision around the specific magnitude and timeline.\n"

        # Evaluate collision
        today = datetime.now().strftime("%B %d, %Y")
        prompt = COLLISION_EVALUATE_PROMPT.format(
            today_date=today,
            facts_text=facts_text[:4000],
            anomalies_text=anomaly_text,
            entity_connections=(entity_connections or "None identified") + structural_context,
        )
        # Collision evaluation — filters which anomalies proceed to hypothesis formation.
        # Better filter = fewer wasted downstream calls. Use MODEL_DEEP for quality.
        response = call_text(prompt, system=SYSTEM_PROMPT, max_tokens=2048, temperature=0.5, model=MODEL_DEEP)
        self.track_tokens(response)
        text = extract_text_from_response(response)

        try:
            collision_data = parse_json_response(text)
        except (json.JSONDecodeError, JSONParseError):
            mark_anomaly_attempted(anomaly['id'])
            return

        if not collision_data.get("has_collision"):
            print_info("No meaningful collision detected")
            mark_anomaly_attempted(anomaly['id'])
            return

        # Compute temporal spread for scoring and storage
        _spread_days, _oldest_age_days, _temporal_details = self._compute_temporal_spread(matching_facts)
        # Store matching facts on self for scoring access later
        self._matching_facts_for_telemetry = matching_facts

        # Save collision
        self.collisions_found += 1
        collision_id = save_collision(
            fact_ids=fact_ids,
            anomaly_ids=[anomaly["id"]],
            collision_description=collision_data.get("collision_description", ""),
            num_facts=len(fact_ids),
            num_domains=len(domains_set),
            domains_involved=", ".join(domains_set),
            source_types=", ".join(source_types_set),
            temporal_spread_days=_spread_days,
            oldest_fact_age_days=_oldest_age_days,
        )
        print(f"  {C.YELLOW}💥 COLLISION detected: {collision_data.get('collision_description', '')[:80]}...{C.RESET}")

        # Theory proof layer: record collision evidence
        if self._theory_recorder:
            try:
                _avg_dist = compute_avg_domain_distance(list(source_types_set)) if len(source_types_set) >= 2 else 0.0
                self._theory_recorder.record_collision(
                    collision_id=collision_id,
                    collision_data=collision_data,
                    fact_ids=fact_ids,
                    source_types=source_types_set,
                    domain_distance=_avg_dist,
                    chains=chains,
                    belief_reality_matches=belief_reality_matches,
                    validated_pairs=validated_pairs,
                )
            except Exception:
                pass  # Theory recording never breaks trading pipeline

        # Theory layer agents: advanced evidence classification
        if self._theory_telemetry:
            try:
                self._theory_telemetry.log_collision(
                    collision_data=collision_data,
                    domain_pair=domains_set,
                    source_types=source_types_set,
                    negative_space_data=getattr(self, "_negative_space_data", None),
                    chains=chains,
                    belief_reality_matches=belief_reality_matches,
                    validated_pairs=validated_pairs,
                )
            except Exception:
                pass  # Theory agents never break trading pipeline

        # Save any discovered chains
        for chain in chains:
            save_chain(
                collision_id=collision_id,
                chain_links=chain["links"],
                domains_traversed=chain["domains"],
            )

        # ── SEARCH-GROUNDED GATE ──
        # Step 1: Check if collision evaluator generated a broken model claim
        broken_model = collision_data.get("broken_model")
        stale_assumption = collision_data.get("stale_assumption")
        silo_reason = collision_data.get("silo_reason")

        if not broken_model or not stale_assumption or not silo_reason:
            print_info(f"Collision held — no broken model claimed")
            save_held_collision(
                collision_id=collision_id,
                fact_ids=fact_ids,
                collision_description=collision_data.get("collision_description", ""),
                gate_reasoning="Collision evaluator could not identify a broken model.",
                domains_involved=", ".join(domains_set),
                source_types=", ".join(source_types_set),
            )
            mark_anomaly_attempted(anomaly['id'])
            return

        print(f"  {C.CYAN}🔧 Claimed broken model: {broken_model[:80]}{C.RESET}")

        # Step 2: Deduplication — skip if similar broken model already processed this cycle
        broken_model_lower = broken_model.lower()
        is_duplicate = False
        _dedup_stopwords = {
            "the", "a", "an", "is", "are", "for", "in", "on", "of",
            "to", "and", "that", "this", "with", "from", "by", "as",
            "at", "or", "not", "but", "be", "have", "has", "do",
            "does", "will", "would", "could", "should", "using",
            "based", "models", "model", "pricing", "assume", "assumes",
        }
        current_words = set(broken_model_lower.split()) - _dedup_stopwords
        for prev_model in self._processed_models_this_cycle:
            prev_words = set(prev_model.lower().split()) - _dedup_stopwords
            if not current_words or not prev_words:
                continue
            overlap = len(current_words & prev_words)
            union = len(current_words | prev_words)
            similarity = overlap / union if union > 0 else 0
            if similarity > 0.4:
                is_duplicate = True
                print_info(f"Collision DEDUPLICATED — similar to previous (similarity: {similarity:.2f})")
                break

        if is_duplicate:
            save_held_collision(
                collision_id=collision_id,
                fact_ids=fact_ids,
                collision_description=collision_data.get("collision_description", ""),
                gate_reasoning="DEDUPLICATED: Similar broken model already processed this cycle.",
                domains_involved=", ".join(domains_set),
                source_types=", ".join(source_types_set),
            )
            mark_anomaly_attempted(anomaly['id'])
            return

        # Step 3: Generate search queries to kill the broken model claim (Haiku)
        print_phase("COLLISION", "Search-grounded gate — generating kill queries...")
        query_prompt = SEARCH_GATE_QUERY_PROMPT.format(
            broken_model=broken_model,
            stale_assumption=stale_assumption,
            silo_reason=silo_reason,
        )
        query_response = call_text(query_prompt, system="Generate search queries designed to find evidence that a claimed market insight is already known.", max_tokens=512, temperature=0.3, model=MODEL_FAST)
        self.track_tokens(query_response)
        query_text = extract_text_from_response(query_response)

        try:
            queries = parse_json_response(query_text)
        except (json.JSONDecodeError, JSONParseError):
            queries = {"query_1": f"{broken_model} pricing update 2025 2026",
                       "query_2": f"{stale_assumption[:50]} already known",
                       "query_3": f"{silo_reason[:50]} research analysis"}

        # Step 4: Execute the three searches
        search_results = {}
        for i, key in enumerate(["query_1", "query_2", "query_3"], 1):
            query = queries.get(key, "")
            if not query:
                search_results[f"search_{i}"] = "No query generated"
                continue
            print_info(f"Search {i}: {query[:80]}")
            try:
                search_resp = call_with_web_search(
                    f"Search for: {query}\n\nReturn what you find. Focus on whether any results discuss the SPECIFIC CONNECTION between the search terms, not just the individual terms separately.",
                    system="You are a research assistant. Report what you find factually.",
                    max_tokens=1024,
                    model=MODEL_FAST,
                )
                self.track_tokens(search_resp)
                search_results[f"search_{i}"] = extract_text_from_response(search_resp)
            except Exception as e:
                print_info(f"Search {i} failed (non-fatal): {e}")
                search_results[f"search_{i}"] = f"Search failed: {e}"

        # Step 5: Evaluate search results against the broken model claim (Sonnet)
        print_phase("COLLISION", "Evaluating search results against claim...")

        # Build short versions for the evaluation prompt
        bm_short = broken_model[:100] if len(broken_model) > 100 else broken_model
        sa_short = stale_assumption[:100] if len(stale_assumption) > 100 else stale_assumption

        eval_prompt = SEARCH_GATE_EVALUATE_PROMPT.format(
            broken_model=broken_model,
            stale_assumption=stale_assumption,
            silo_reason=silo_reason,
            search_1_results=search_results.get("search_1", "No results")[:2000],
            search_2_results=search_results.get("search_2", "No results")[:2000],
            search_3_results=search_results.get("search_3", "No results")[:2000],
            broken_model_short=bm_short,
            stale_assumption_short=sa_short,
        )
        eval_response = call_text(eval_prompt, system=SYSTEM_PROMPT, max_tokens=1024, temperature=0.3, model=MODEL)
        self.track_tokens(eval_response)
        eval_text = extract_text_from_response(eval_response)

        try:
            gate_data = parse_json_response(eval_text)
        except (json.JSONDecodeError, JSONParseError):
            gate_data = {"survives": False, "kill_reason": "Failed to parse gate evaluation", "reasoning": "Parse error"}

        if not gate_data.get("survives", False):
            kill_reason = gate_data.get("kill_reason", "unknown")
            print(f"  {C.RED}🚫 SEARCH GATE KILLED: {kill_reason[:100]}{C.RESET}")
            save_held_collision(
                collision_id=collision_id,
                fact_ids=fact_ids,
                collision_description=collision_data.get("collision_description", ""),
                gate_reasoning=f"SEARCH KILLED: {kill_reason}",
                domains_involved=", ".join(domains_set),
                source_types=", ".join(source_types_set),
            )
            mark_anomaly_attempted(anomaly['id'])
            return

        # Survived the search gate — record this model for dedup
        self._processed_models_this_cycle.append(broken_model)
        print(f"  {C.GREEN}✅ SEARCH GATE SURVIVED: {broken_model[:80]}{C.RESET}")

        # Build gate_data in the format _form_hypothesis expects
        gate_data_for_hyp = {
            "has_broken_model": True,
            "broken_model": broken_model,
            "stale_assumption": stale_assumption,
            "silo_reason": silo_reason,
            "reasoning": gate_data.get("reasoning", ""),
        }

        # ── NEGATIVE SPACE CHECK ──
        # After collision survives the gate, check whether the expected market
        # reaction has occurred. If it hasn't, that's direct edge measurement.
        self._negative_space_data = None
        if not THEORY_RUN:
            try:
                facts_summary = "; ".join([
                    f"({f.get('source_type', '?')}) {f.get('title', '')[:80]}"
                    for f in matching_facts[:5]
                ])
                ns_prompt = NEGATIVE_SPACE_DETECT_PROMPT.format(
                    collision_description=collision_data.get("collision_description", "")[:500],
                    facts_summary=facts_summary[:1000],
                    broken_model=broken_model[:300] if broken_model else "Not identified",
                )
                ns_response = call_with_web_search(
                    ns_prompt,
                    system="You are a market microstructure analyst. Your job is to detect whether a market has reacted to information. Search for specific price movements, analyst coverage, and institutional positioning.",
                    max_tokens=1024,
                )
                self.track_tokens(ns_response)
                ns_text = extract_text_from_response(ns_response)
                self._negative_space_data = parse_json_response(ns_text)

                reacted = self._negative_space_data.get("reaction_occurred", None)
                gap = self._negative_space_data.get("gap_magnitude", "unknown")
                ns_score = int(self._negative_space_data.get("negative_space_score", 5))

                if reacted is False and gap in ("large", "total"):
                    print(f"  {C.GREEN}🕳️ NEGATIVE SPACE: Market has NOT reacted (gap: {gap}, score: {ns_score}/10){C.RESET}")
                elif reacted is True:
                    print(f"  {C.YELLOW}🕳️ Negative space: Market partially reacted (gap: {gap}, score: {ns_score}/10){C.RESET}")
                else:
                    print(f"  {C.DIM}🕳️ Negative space: {gap} (score: {ns_score}/10){C.RESET}")
            except Exception as e:
                print_info(f"Negative space check failed (non-fatal): {e}")

        # Form hypothesis -- pass SOURCE TYPE count for multi-domain priority
        self._form_hypothesis(collision_id, collision_data, matching_facts, anomaly, fact_ids,
                              num_domains=len(source_types_set), broken_model_data=gate_data_for_hyp,
                              chains=chains, source_types=source_types_set)

        # Mark anomaly as attempted after processing (whether collision found or not)
        mark_anomaly_attempted(anomaly['id'])

    def _verify_facts_before_hypothesis(self, facts):
        """Verify key numerical claims in facts before building a hypothesis.

        Uses a single web search to check prices/numbers in the fact chain.
        Returns corrected facts_detail string with verified numbers.
        """
        # Find facts with specific numbers that need verification (prices, quantities, percentages)
        numeric_facts = []
        for f in facts:
            content = (f.get('raw_content', '') + ' ' + f.get('title', ''))
            # Match prices, large numbers, percentages
            if re.search(r'\$[0-9,]+\.?\d*|\b\d{3,}[,\d]*\b|\d+%', content):
                numeric_facts.append(f)

        if not numeric_facts:
            return None  # No numbers to verify

        # Build a focused verification query
        verify_items = []
        for f in numeric_facts[:3]:  # Max 3 to verify
            content = f.get('raw_content', '')[:200]
            verify_items.append(f"- Fact #{f['id']}: {content}")

        verify_block = "\n".join(verify_items)

        print_phase("EXTRACT", "Verifying prices in fact chain...")
        try:
            verify_response = call_with_web_search(
                f"Verify these specific price claims. For each, find the CURRENT correct price from a reliable source:\n\n{verify_block}\n\n"
                f"For each fact, respond with the CORRECT current price and whether the claimed price is approximately right (within 15%) or wrong.\n\n"
                f"Respond with ONLY a JSON object:\n"
                f'{{"verifications": [{{"fact_id": <id>, "claimed_price": "...", "actual_price": "...", "is_correct": true/false, "source": "..."}}]}}',
                system="You are a fact-checker. Verify prices against current market data. Be precise.",
                max_tokens=1024,
            )
            self.track_tokens(verify_response)
            verify_text = extract_text_from_response(verify_response)

            try:
                verify_data = parse_json_response(verify_text)
                verifications = verify_data.get("verifications", [])
                bad_facts = [v for v in verifications if not v.get("is_correct", True)]
                if bad_facts:
                    bad_ids = {v.get("fact_id") for v in bad_facts}
                    corrections = {}
                    for v in bad_facts:
                        corrections[v.get("fact_id")] = f"[CORRECTED: claimed {v.get('claimed_price')}, actual is {v.get('actual_price')} per {v.get('source', 'web search')}]"
                        print_info(f"Price correction: Fact #{v.get('fact_id')}: {v.get('claimed_price')} → {v.get('actual_price')}")
                    return corrections
            except (json.JSONDecodeError, JSONParseError):
                pass
        except Exception as e:
            print_info(f"Verification failed (non-fatal): {e}")

        return None

    def _form_hypothesis(self, collision_id, collision_data, facts, anomaly, fact_ids, num_domains=1, broken_model_data=None, chains=None, source_types=None):
        """Form a hypothesis from a collision and attempt to kill it."""

        # VERIFY key prices/numbers in the fact chain BEFORE hypothesis formation
        corrections = self._verify_facts_before_hypothesis(facts)

        # Build detailed facts for hypothesis prompt, with corrections noted
        facts_detail = ""
        for f in facts:
            correction = ""
            if corrections and f['id'] in corrections:
                correction = f"\n  ⚠️ {corrections[f['id']]}"
            facts_detail += f"\nFact #{f['id']} ({f['source_type']}, {f.get('domain', 'unknown')}): {f['title']}\n{f.get('raw_content', '')[:400]}{correction}\n"

        anomaly_detail = f"\nAnomaly #{anomaly['id']} (weirdness: {anomaly.get('weirdness_score', '?')}/10):\n{anomaly['anomaly_description']}\n"

        # Build broken model context for hypothesis formation
        broken_model_context = ""
        if broken_model_data:
            broken_model_context = (
                f"\n\nBROKEN MODEL IDENTIFIED (your hypothesis MUST be built around this):\n"
                f"Model: {broken_model_data.get('broken_model', 'unknown')}\n"
                f"Stale Assumption: {broken_model_data.get('stale_assumption', 'unknown')}\n"
                f"Silo Reason: {broken_model_data.get('silo_reason', 'unknown')}\n"
                f"\nYour hypothesis should be: this model is wrong because of these facts, "
                f"it will correct when [specific catalyst], and the specific mispriced asset is [X]."
            )

        print_phase("HYPOTHESIS", "Forming hypothesis from collision...")
        today = datetime.now().strftime("%B %d, %Y")

        # Module L: In retrospective mode, add citation requirement
        _citation_suffix = ""
        if getattr(config, 'RETROSPECTIVE_MODE', False) and getattr(config, 'RETROSPECTIVE_ENFORCE_FACT_CITATION', False):
            _allowed_fact_ids = [f['id'] for f in facts]
            _citation_suffix = f"""

CITATION REQUIREMENT: You MUST cite the specific fact_ids you reasoned from.
Include a field "cited_fact_ids" in your JSON response containing an array of integer fact_ids from the facts provided above (allowed: {_allowed_fact_ids}).
Only cite facts present in the input. If you reason from information not in the provided facts, the hypothesis will be rejected."""

        prompt = retro_prompt_prefix() + HYPOTHESIS_FORM_PROMPT.format(
            today_date=today,
            collision_description=collision_data.get("collision_description", "") + broken_model_context,
            facts_detail=facts_detail[:4000],
            anomalies_detail=anomaly_detail,
        ) + _citation_suffix

        _temp = getattr(config, 'RETROSPECTIVE_TEMPERATURE', 0.5) if getattr(config, 'RETROSPECTIVE_MODE', False) else 0.5

        # Timeline-aware: during summer empirical phase, bias toward short (30-90d)
        # resolution windows so predictions resolve publicly within the same summer.
        try:
            from timeline import short_window_prompt_suffix
            prompt = prompt + short_window_prompt_suffix()
        except Exception:
            pass

        # Hypothesis formation — the generator. Upgraded model here reduces factual
        # hallucinations (the PJM/EIA/FOTIVDA-type errors) that otherwise force mechanism kills.
        response = call_text(prompt, system=SYSTEM_PROMPT, max_tokens=2048, temperature=_temp, model=MODEL_DEEP)
        self.track_tokens(response)
        text = extract_text_from_response(response)

        try:
            hyp_data = parse_json_response(text)
        except (json.JSONDecodeError, JSONParseError):
            print_error("Failed to parse hypothesis JSON")
            return

        # Module L: Compile-time citation hygiene check
        self._cited_fact_ids = None
        if getattr(config, 'RETROSPECTIVE_MODE', False) and getattr(config, 'RETROSPECTIVE_ENFORCE_FACT_CITATION', False):
            cited = hyp_data.get("cited_fact_ids", [])
            allowed = set(f['id'] for f in facts)
            if not cited or not set(cited).issubset(allowed):
                bad_ids = set(cited) - allowed if cited else set()
                print_info(f"Citation violation: cited {cited}, allowed {list(allowed)}, bad {bad_ids}")
                return  # Reject — do not save
            self._cited_fact_ids = cited

        # Reset theory run flags for this hypothesis
        self._theory_run_killed = False
        self._awareness_telemetry = None
        self._pending_edge_recovery = None
        # Bug 1 fix: Set matching facts for telemetry IMMEDIATELY after assembly,
        # before any early returns in kill phase can bypass it
        self._matching_facts_for_telemetry = facts  # 'facts' parameter from _form_hypothesis
        self._stage_timestamps = {  # Module G: per-stage timing
            "ts_collision_detected": datetime.now().isoformat(),
        }

        hypothesis_text = hyp_data.get("hypothesis", "")
        if not hypothesis_text:
            return

        # Module G: record timestamp after hypothesis formed
        self._stage_timestamps["ts_hypothesis_formed"] = datetime.now().isoformat()

        # DEDUP CHECK: does a structurally similar hypothesis already exist?
        try:
            from database import get_connection
            conn = get_connection()
            cursor = conn.cursor()
            # Check both live and archived hypotheses
            hyp_words = set(hypothesis_text.lower().split()[:20])
            cursor.execute("SELECT id, hypothesis_text FROM hypotheses ORDER BY id DESC LIMIT 50")
            for row in cursor.fetchall():
                existing_words = set(row["hypothesis_text"].lower().split()[:20])
                overlap = len(hyp_words & existing_words) / max(len(hyp_words), 1)
                if overlap > 0.6:
                    print_info(f"Dedup: similar to hypothesis #{row['id']} ({overlap:.0%} overlap). Skipping.")
                    conn.close()
                    return
            # Also check archive
            cursor.execute("SELECT id, hypothesis_text FROM hypotheses_archive ORDER BY id DESC LIMIT 50")
            for row in cursor.fetchall():
                existing_words = set(row["hypothesis_text"].lower().split()[:20])
                overlap = len(hyp_words & existing_words) / max(len(hyp_words), 1)
                if overlap > 0.6:
                    print_info(f"Dedup: similar to archived #{row['id']} ({overlap:.0%} overlap). Skipping.")
                    conn.close()
                    return
            conn.close()
        except Exception:
            pass  # Dedup is best-effort

        self.hypotheses_formed += 1
        fact_chain = hyp_data.get("fact_chain", [])
        action_steps = hyp_data.get("action_steps", "")
        time_window = hyp_data.get("time_window_days", 90)

        # PRE-FILTER: Observability check -- don't trust self-reporting, check keywords
        domains_crossed = hyp_data.get("domains_crossed", 1)
        structural = hyp_data.get("structural_or_event", "event")

        # Hard observability check -- if the hypothesis is about something on every front page, kill it
        hyp_lower = hypothesis_text.lower()
        front_page_topics = [
            "strait of hormuz", "hormuz", "iran strike", "iran attack", "khamenei",
            "oil price spike", "oil price surge", "brent crude surge",
            "trump tariff", "trade war", "tariff war",
            "bitcoin crash", "bitcoin surge", "crypto crash",
            "stock market crash", "recession",
        ]
        is_front_page = sum(1 for topic in front_page_topics if topic in hyp_lower)

        print(f"  {C.BOLD}Hypothesis: {hypothesis_text[:100]}...{C.RESET}")
        print(f"  {C.DIM}Domains: {domains_crossed} | Type: {structural} | Front-page hits: {is_front_page}{C.RESET}")

        if is_front_page >= 2:
            print(f"  {C.RED}✗ Auto-killed: front-page news ({is_front_page} matches) — no edge{C.RESET}")
            save_hypothesis(
                collision_id=collision_id, hypothesis_text=hypothesis_text,
                fact_chain=fact_chain, action_steps=action_steps,
                time_window_days=time_window,
                kill_attempts=[{"round": "pre-filter", "killed": True,
                    "reason": f"Front-page news ({is_front_page} topic matches) — no field-of-view edge",
                    "confidence": "strong", "kill_type": "observability"}],
                survived_kill=False,
            )
            return

        # Warn on event-driven (don't auto-kill, but flag it)
        if structural == "event" and domains_crossed < 3:
            print(f"  {C.YELLOW}⚠️  Event-driven with low domain spread — weak edge{C.RESET}")

        # DUPLICATE ROOT CAUSE CHECK — prevent the same upstream shock from generating
        # unlimited downstream theses. If 3+ active hypotheses share the same root driver,
        # new theses with that driver must demonstrate independent mechanisms.
        try:
            from database import get_connection as _gc2
            _conn2 = _gc2()
            _cur2 = _conn2.cursor()
            hyp_lower = hypothesis_text.lower()

            # Identify root cause keywords
            root_keywords = {
                "cmbs": ["cmbs", "office delinquency", "12.34%", "special servicing", "cre distress", "office loan"],
                "pharma_patent": ["patent expir", "generic entry", "anda", "exclusivity"],
                "insurance_pricing": ["actuarial", "d&o", "premium", "loss table"],
                "energy_grid": ["ferc", "interconnection", "transmission", "grid"],
            }

            for root_name, keywords in root_keywords.items():
                if any(kw in hyp_lower for kw in keywords):
                    # Count how many active hypotheses share this root
                    _cur2.execute("SELECT COUNT(*) FROM hypotheses WHERE survived_kill = 1 AND diamond_score >= 50")
                    # Check each one
                    _cur2.execute("SELECT hypothesis_text FROM hypotheses WHERE survived_kill = 1 AND diamond_score >= 50")
                    same_root = 0
                    for row in _cur2.fetchall():
                        if any(kw in row[0].lower() for kw in keywords):
                            same_root += 1

                    if same_root >= 3:
                        print(f"  {C.YELLOW}⚠️ ROOT CAUSE SATURATION: {root_name} appears in {same_root} active hypotheses{C.RESET}")
                        print(f"  {C.YELLOW}   New thesis must demonstrate independent mechanism to proceed{C.RESET}")
                        # Don't auto-kill but flag for mechanism kill to be extra strict
                    break

            _conn2.close()
        except Exception:
            pass

        # Module G: record timestamp before kill phase
        self._stage_timestamps["ts_kill_started"] = datetime.now().isoformat()

        # KILL PHASE -- 4 stages in strict order. First failure kills.
        kill_attempts = []
        fatal_flaw = False
        soft_kill_votes = 0

        # Extract key claims from hypothesis for targeted searching
        hyp_short = hypothesis_text[:150]

        # KILL GATE ARCHITECTURE — 4 stages in strict order. First failure kills.
        # Stage 1: Mechanism (two-step node+logic test) — runs FIRST, catches broken arrows
        # Stage 2: Factual (verify numbers/dates against primary sources)
        # Stage 3: Competitor (has someone already done this exact thing)
        # Stage 4: Barrier (why this specifically can't work)
        kill_angles = [
            {"query": f"verify mechanism: {hyp_short}", "type": "mechanism"},
            {"query": f"verify facts: {hyp_short}", "type": "fact_check"},
            {"query": f"{hyp_short} already exists solution", "type": "competitor"},
            {"query": f"why {hyp_short} impossible barriers failed", "type": "barrier"},
        ]

        for kill_round in range(min(KILL_SEARCH_COUNT, len(kill_angles))):
            angle = kill_angles[kill_round]
            print_phase("KILL", f"Kill attempt {kill_round + 1}/{KILL_SEARCH_COUNT} ({angle['type']})...")

            try:
                # Use mechanism-specific prompt for round 2
                if angle["type"] == "mechanism":
                    kill_prompt_text = f"""This thesis claims a causal chain. Test each arrow using TWO steps.

THESIS: {hypothesis_text[:800]}
FACT CHAIN: {json.dumps(fact_chain)[:800]}

For each arrow in the causal chain, apply BOTH steps:

STEP 1 — NODE VERIFICATION (search-based):
Search to verify that both the source system and the target system ACTUALLY OPERATE the way the thesis claims. Does the thesis correctly understand how each system works?
- "Thesis says LDI portfolios hold CMBS" -> Search: what do LDI portfolios actually hold? -> If they hold gilts/STRIPS/IG corporates/swaps and NOT CMBS -> NODE ERROR
- "Thesis says appraisers use CoStar comps" -> Search: do appraisers use CoStar? -> Yes, verified -> NODE CORRECT
- "Thesis says NYCECC applies to existing bonded projects" -> Search: does NYCECC apply retroactively? -> No, new filings only -> NODE ERROR
If EITHER node is mischaracterised, output: ARROW BROKEN — NODE ERROR. The thesis misunderstands how the system works.

STEP 2 — LOGIC VERIFICATION (reasoning-based, only if Step 1 passes):
Given that both systems are correctly characterised, does the output of system A logically flow into the input of system B?
- "If NAIC surplus is overstated, does that logically inflate AM Best ratings?" -> Yes, surplus IS the input to RBC ratio -> LOGIC VALID
- "If actuarial models have zero historical data for a new claim category, would they logically underprice it?" -> Yes, no data means no adjustment -> LOGIC VALID
- "If oil prices rise, would that logically affect semiconductor fab costs?" -> Needs specific mechanism, not obvious -> LOGIC WEAK
If the connection is a NOVEL but logically sound pathway between two correctly characterised systems: ARROW VALID — NOVEL PATHWAY
If the connection doesn't follow from how the systems actually work: ARROW BROKEN — LOGIC ERROR

Test the WEAKEST arrow first. One broken arrow kills the thesis.

Respond with ONLY a JSON object:
{{"killed": true/false, "kill_reason": "Which arrow failed, which step (NODE or LOGIC), and what you found", "confidence_in_kill": "strong/moderate/weak", "mechanism_tested": "the specific arrow tested", "kill_type": "mechanism", "step_failed": "node/logic/none"}}"""
                    # Mechanism kill is the critical arbiter — upgrading here reduces
                    # false-positive rejections of valid cross-silo theses.
                    kill_response = call_kill_gate(
                        retro_prompt_prefix() + kill_prompt_text,
                        system="Two-step mechanism test. Step 1: search to verify nodes operate as claimed. Step 2: reason whether the connection is logically sound. Node errors kill immediately. Novel but logical connections survive.",
                        max_tokens=1024,
                        model=MODEL_DEEP,
                    )
                else:
                    kill_response = call_kill_gate(
                        retro_prompt_prefix() + KILL_PROMPT.format(
                            hypothesis_text=hypothesis_text[:1000],
                            fact_chain=json.dumps(fact_chain)[:1000],
                            search_results=f"Search for: {angle['query']}",
                        ),
                        system="You are a ruthless hypothesis destroyer. Your ONLY job is to find reasons this hypothesis is WRONG. But you must have SPECIFIC EVIDENCE — not vibes. 'Competitors might exist' is NOT a kill. 'Company X at url Y does exactly this' IS a kill.",
                        max_tokens=1024,
                    )
                self.track_tokens(kill_response)
                kill_text = extract_text_from_response(kill_response)

                try:
                    kill_data = parse_json_response(kill_text)
                    is_killed = kill_data.get("killed", False)
                    confidence = kill_data.get("confidence_in_kill", "weak")
                    reason = kill_data.get("kill_reason", kill_data.get("survived_because", ""))

                    kill_attempts.append({
                        "round": kill_round + 1,
                        "killed": is_killed,
                        "reason": reason,
                        "confidence": confidence,
                        "kill_type": angle["type"],
                    })

                    # MECHANISM KILL — requires moderate+ confidence for instant death.
                    # Weak mechanism kills add a soft vote instead of instakilling.
                    # This is the fix for the "100% mechanism kill rate" issue — weak
                    # rejections no longer bypass the rest of the kill phase.
                    if is_killed and angle["type"] == "mechanism":
                        if confidence in ("strong", "moderate"):
                            fatal_flaw = True
                            print(f"  {C.BG_RED}{C.WHITE}{C.BOLD} MECHANISM KILL [{confidence}]: {reason[:80]} {C.RESET}")
                            kill_attempts.append({
                                "round": kill_round + 1, "killed": True, "reason": reason,
                                "confidence": confidence, "kill_type": "mechanism_fatal",
                            })
                            # THEORY RUN: bypass hard return — continue to scoring for telemetry collection
                            if THEORY_RUN:
                                print(f"  {C.YELLOW}[THEORY] Mechanism kill bypassed for telemetry — continuing to scoring{C.RESET}")
                                break  # Exit kill loop but don't return
                            else:
                                # Production: save and return immediately
                                save_hypothesis(
                                    collision_id=collision_id, hypothesis_text=hypothesis_text,
                                    fact_chain=fact_chain, action_steps=action_steps,
                                    time_window_days=time_window, kill_attempts=kill_attempts,
                                    survived_kill=False, diamond_score=None,
                                )
                                return  # HARD STOP — nothing else runs
                        else:
                            # Weak mechanism kill — add soft vote, continue to other kill rounds.
                            # A reviewer uncertain about mechanism shouldn't kill outright.
                            soft_kill_votes += 0.5
                            print(f"  {C.YELLOW}⚠️  WEAK MECHANISM KILL (soft vote {soft_kill_votes}): {reason[:80]}{C.RESET}")

                    if is_killed and confidence == "strong":
                        reason_lower = reason.lower()
                        kill_type = kill_data.get("kill_type", "other")

                        is_fact_kill = (
                            kill_type == "fact_wrong" or
                            angle["type"] == "fact_check" and is_killed or
                            "fundamentally" in reason_lower and ("wrong" in reason_lower or "incorrect" in reason_lower) or
                            "core fact" in reason_lower and "wrong" in reason_lower or
                            "data is wrong" in reason_lower or
                            "never happened" in reason_lower or
                            "does not exist" in reason_lower
                        )

                        if is_fact_kill:
                            fatal_flaw = True
                            print(f"  {C.BG_RED}{C.WHITE}{C.BOLD} FATAL FLAW: {reason[:80]} {C.RESET}")
                            break  # no point continuing
                        else:
                            soft_kill_votes += 1
                            print(f"  {C.RED}☠️  KILL VOTE ({soft_kill_votes}/2): {reason[:80]}{C.RESET}")
                    elif is_killed and confidence == "moderate":
                        soft_kill_votes += 0.5
                        print(f"  {C.YELLOW}⚠️  WEAK KILL ({soft_kill_votes}/2): {reason[:80]}{C.RESET}")
                    else:
                        print(f"  {C.GREEN}✓ Survived round {kill_round + 1}{C.RESET}")

                except (json.JSONDecodeError, JSONParseError):
                    kill_attempts.append({"round": kill_round + 1, "killed": False, "reason": "Parse error", "kill_type": angle["type"]})

            except Exception as e:
                kill_attempts.append({"round": kill_round + 1, "killed": False, "reason": f"Error: {e}", "kill_type": angle["type"]})

        # PIVOT ON FATAL FLAW: only attempt if the flaw is a QUANTITATIVE error (wrong number, wrong date, wrong jurisdiction)
        # NOT if the core premise is wrong (wrong company, wrong mechanism, fabricated connection)
        if fatal_flaw and len(kill_attempts) > 0:
            flaw_reason = ""
            for ka in kill_attempts:
                if ka.get("killed") and ka.get("confidence") == "strong":
                    flaw_reason = ka.get("reason", "")
                    break

            # Only pivot on quantitative errors, not premise errors
            flaw_lower = flaw_reason.lower()
            is_quantitative_error = any(phrase in flaw_lower for phrase in [
                "price is", "date is", "not until", "expired in", "expires in",
                "not march", "not 2026", "actually $", "actually is",
                "off by", "percentage is wrong", "number is wrong",
            ])
            is_premise_error = any(phrase in flaw_lower for phrase in [
                "does not exist", "is not a", "never happened", "completely wrong",
                "has nothing to do", "different company", "different domain",
                "fabricated", "no evidence", "is not related",
            ])

            if flaw_reason and is_quantitative_error and not is_premise_error:
                print_phase("KILL", "Attempting pivot on quantitative error...")
                try:
                    pivot_response = call_text(
                        f"""A hypothesis was killed because a specific number, date, or jurisdiction was wrong. The STRUCTURE may still be valid.

ORIGINAL HYPOTHESIS:
{hypothesis_text[:1500]}

QUANTITATIVE ERROR FOUND:
{flaw_reason[:500]}

STRICT RULES:
- You may ONLY adjust the specific number, date, or jurisdiction that was wrong
- You may NOT invent new facts, companies, or connections that weren't in the original hypothesis
- You may NOT claim a company operates in a domain it doesn't operate in
- You may NOT fabricate relationships between entities
- If correcting the error makes the thesis weaker but still valid, that's a valid pivot
- If correcting the error collapses the thesis entirely, say so honestly

Respond with ONLY a JSON object:
{{
    "can_pivot": true/false,
    "pivot_hypothesis": "The adjusted hypothesis with ONLY the quantitative error corrected (only if can_pivot)",
    "pivot_reasoning": "What specific number/date/jurisdiction was corrected",
    "adjusted_market": "What market/jurisdiction the corrected thesis applies to"
}}""",
                        system="You are a fact-checker. You may ONLY correct specific numbers, dates, or jurisdictions. You may NOT invent new claims or fabricate connections.",
                        max_tokens=1024,
                        temperature=0.2,
                    )
                    self.track_tokens(pivot_response)
                    pivot_text = extract_text_from_response(pivot_response)

                    try:
                        pivot_data = parse_json_response(pivot_text)
                        if pivot_data.get("can_pivot") and pivot_data.get("pivot_hypothesis"):
                            hypothesis_text = pivot_data["pivot_hypothesis"]
                            fatal_flaw = False
                            kill_attempts.append({
                                "round": "pivot",
                                "killed": False,
                                "reason": f"Pivoted: {pivot_data.get('pivot_reasoning', '')[:200]}. Market: {pivot_data.get('adjusted_market', '')}",
                                "confidence": None,
                                "kill_type": "pivot",
                            })
                            print(f"  {C.GREEN}{C.BOLD}🔄 PIVOT: {pivot_data.get('pivot_reasoning', '')[:80]}{C.RESET}")
                            print(f"  {C.CYAN}   Market: {pivot_data.get('adjusted_market', '')[:60]}{C.RESET}")
                        else:
                            print(f"  {C.RED}Pivot failed — thesis collapses with correction{C.RESET}")
                            kill_attempts.append({
                                "round": "pivot",
                                "killed": True,
                                "reason": "Thesis collapses entirely when flaw is corrected",
                                "confidence": "strong",
                                "kill_type": "pivot",
                            })
                    except (json.JSONDecodeError, JSONParseError):
                        pass

                except Exception as e:
                    print_info(f"Pivot attempt failed (non-fatal): {e}")

        # MULTI-DOMAIN PRIORITY: more domains = more runway, more nurturing
        # 1-2 domains: standard treatment
        # 3 domains: need 3 soft kills, get steelman round
        # 4 domains: need all 3 rounds to agree on kill, get steelman
        # 5+ domains: near-unkillable on soft kills, only fatal flaw can kill
        is_multi_domain = num_domains >= 3
        if num_domains >= 5:
            soft_kill_threshold = 99  # effectively unkillable by soft kills
        elif num_domains >= 4:
            soft_kill_threshold = 4  # need unanimous + extra
        elif num_domains >= 3:
            soft_kill_threshold = 3
        else:
            soft_kill_threshold = 2

        # Check if mechanism kill fired (node error = no steelman allowed)
        mechanism_killed = any(ka.get("kill_type") == "mechanism" and ka.get("killed") for ka in kill_attempts)

        if is_multi_domain and not fatal_flaw and not mechanism_killed and soft_kill_votes > 0 and soft_kill_votes < soft_kill_threshold:
            # STEELMAN ROUND: before killing a multi-domain hypothesis on soft kills,
            # give it one more chance -- actively try to STRENGTHEN the thesis
            # NOTE: steelman NEVER runs after a mechanism node error
            print_phase("KILL", f"Steelman round (3+ domain priority)...")
            try:
                steelman_response = call_text(
                    f"""This hypothesis crosses {num_domains} professional domains, making it potentially very valuable.
It received soft kill votes but no fatal flaw was found. Before we kill it, try to SAVE it.

HYPOTHESIS:
{hypothesis_text[:1000]}

KILL CONCERNS:
{json.dumps([k for k in kill_attempts if k.get('killed')], indent=2)[:1000]}

Can this hypothesis be REFRAMED or NARROWED to address the kill concerns while keeping the core cross-domain insight?
- If the competitor concern is about a broad industry, can the thesis be narrowed to a specific niche?
- If the barrier concern is about one approach, is there an alternative approach?
- Can the thesis be strengthened by focusing on the most defensible subset of the original claim?

Respond with ONLY a JSON object:
{{
    "can_save": true/false,
    "refined_hypothesis": "The reframed hypothesis that addresses kill concerns (only if can_save)",
    "reasoning": "Why this version survives the concerns"
}}""",
                    system="You are trying to SAVE a valuable multi-domain hypothesis. Find a way to reframe it that addresses the kill concerns while preserving the core insight.",
                    max_tokens=1024,
                    temperature=0.4,
                )
                self.track_tokens(steelman_response)
                steelman_text = extract_text_from_response(steelman_response)

                try:
                    steelman_data = parse_json_response(steelman_text)
                    if steelman_data.get("can_save") and steelman_data.get("refined_hypothesis"):
                        hypothesis_text = steelman_data["refined_hypothesis"]
                        soft_kill_votes = soft_kill_votes / 2  # Halve soft kills -- thesis was reframed but carries the scar
                        kill_attempts.append({
                            "round": "steelman",
                            "killed": False,
                            "reason": f"Reframed: {steelman_data.get('reasoning', '')[:200]}",
                            "confidence": None,
                            "kill_type": "steelman",
                        })
                        print(f"  {C.GREEN}💎 Steelman saved {num_domains}-domain hypothesis{C.RESET}")
                    else:
                        kill_attempts.append({
                            "round": "steelman",
                            "killed": True,
                            "reason": "Could not be saved even with reframing",
                            "confidence": "moderate",
                            "kill_type": "steelman",
                        })
                        print(f"  {C.RED}Steelman failed — hypothesis cannot be saved{C.RESET}")
                except (json.JSONDecodeError, JSONParseError):
                    pass
            except Exception as e:
                print_info(f"Steelman round failed (non-fatal): {e}")

        # FATAL FLAW = instant death. Soft kills need majority (higher bar for multi-domain).
        killed = fatal_flaw or soft_kill_votes >= soft_kill_threshold

        if is_multi_domain and not killed:
            print(f"  {C.CYAN}💎 Multi-domain ({num_domains}) hypothesis given priority treatment{C.RESET}")

        if killed:
            if THEORY_RUN:
                # THEORY RUN: bypass kill return — continue to scoring for telemetry
                # The hypothesis is flagged as killed in kill_attempts, survived_kill will be False
                print(f"  {C.YELLOW}[THEORY] Kill bypassed for telemetry — proceeding to scoring{C.RESET}")
                self._theory_run_killed = True  # Flag so we save with survived_kill=False
            else:
                # Production: save and return
                save_hypothesis(
                    collision_id=collision_id,
                    hypothesis_text=hypothesis_text,
                    fact_chain=fact_chain,
                    action_steps=action_steps,
                    time_window_days=time_window,
                    kill_attempts=kill_attempts,
                    survived_kill=False,
                )
                return

        # Module G: record timestamp after kill phase
        self._stage_timestamps["ts_kill_completed"] = datetime.now().isoformat()

        # SURVIVED! Now refine it.
        if not getattr(self, "_theory_run_killed", False):
            self.hypotheses_survived += 1
            print(f"\n  {C.GREEN}{C.BOLD}✅ HYPOTHESIS SURVIVED {len(kill_attempts)} KILL ATTEMPTS{C.RESET}")

        # REFINEMENT PHASE -- stress-test the conclusion and find the optimal play
        # Skip refinement in theory run for killed hypotheses (saves API cost)
        if not getattr(self, "_theory_run_killed", False):
            hypothesis_text, action_steps, kill_attempts = self._refine_hypothesis(
                hypothesis_text, fact_chain, action_steps, kill_attempts
            )

        # Module G: record timestamp before scoring
        import time as _time_mod
        self._stage_timestamps["ts_scoring_started"] = datetime.now().isoformat()
        self._score_and_save(collision_id, hypothesis_text, fact_chain,
                             action_steps, time_window, kill_attempts,
                             num_domains=num_domains, chains=chains,
                             source_types=source_types)

    def _refine_hypothesis(self, hypothesis_text, fact_chain, action_steps, kill_attempts):
        """Refine a surviving hypothesis -- stress-test the conclusion and find the optimal play.

        The facts survived. The collision is real. But is the CONCLUSION correct?
        This phase checks whether the proposed trade/action is the best one given the facts,
        and corrects it if not. Especially important for financial instrument hypotheses.
        """
        hyp_lower = hypothesis_text.lower()
        financial_keywords = ["warrant", "option", "future", "convertible", "derivative",
                              "short", "long", "exercise", "strike", "put", "call",
                              "arbitrage", "mispriced", "mispricing", "spread",
                              "futures", "swap", "hedge", "margin", "premium"]
        is_financial = any(kw in hyp_lower for kw in financial_keywords)

        if not is_financial:
            return hypothesis_text, action_steps, kill_attempts

        print_phase("SCORE", "Refining financial mechanics...")
        try:
            refine_response = call_text(
                f"""A hypothesis survived fact-checking and kill attempts. The FACTS are verified correct.
But the CONCLUSION — the specific trade or action — may not be optimal or may contain a reasoning error.

HYPOTHESIS:
{hypothesis_text[:2000]}

PROPOSED ACTION:
{action_steps[:1000]}

Your job:
1. Walk through the mechanics of every financial instrument mentioned step by step.
   - For warrants/options: what is intrinsic value? What will rational holders do? Is the market price consistent with rational exercise?
   - For futures/commodities: is the proposed direction correct given the facts?
   - For shorts: what is the actual risk? Who is on the other side of the trade?
   - For arbitrage: does the spread actually exist after transaction costs?

2. Is the proposed trade the BEST play given these facts? Or is there a better one?
   - Maybe the facts are real but the trade direction is wrong
   - Maybe the instrument is right but the timing is wrong
   - Maybe there's a second-order effect that's more profitable (e.g., dilution from warrant exercise instead of the warrant itself)

3. If the conclusion or trade is wrong, REWRITE it with the correct mechanics. Keep the same facts and collision — just fix the conclusion.

Respond with ONLY a JSON object:
{{
    "original_logic_correct": true/false,
    "reasoning": "Step by step mechanics analysis",
    "logic_error": "What was wrong with the original conclusion, if anything",
    "refined_hypothesis": "The corrected hypothesis with same facts but better conclusion (only if original was wrong, otherwise null)",
    "refined_action_steps": "The corrected action steps (only if original was wrong, otherwise null)",
    "better_play": "Description of a better trade if one exists (or null)"
}}""",
                system="You are a quantitative finance expert and trader. You understand derivatives mechanics, exercise decisions, intrinsic value, sunk costs, dilution effects, and market microstructure. Find the BEST trade given the facts, even if the original hypothesis proposed the wrong one.",
                max_tokens=1536,
                temperature=0.3,
            )
            self.track_tokens(refine_response)
            refine_text = extract_text_from_response(refine_response)

            try:
                refine_data = parse_json_response(refine_text)

                if not refine_data.get("original_logic_correct", True):
                    # Logic was wrong -- use the refined version
                    refined_hyp = refine_data.get("refined_hypothesis")
                    refined_actions = refine_data.get("refined_action_steps")
                    logic_error = refine_data.get("logic_error", "")
                    better_play = refine_data.get("better_play", "")

                    if refined_hyp:
                        print(f"  {C.YELLOW}🔧 REFINED: Original logic had error: {logic_error[:80]}{C.RESET}")
                        if better_play:
                            print(f"  {C.CYAN}💡 Better play: {better_play[:80]}{C.RESET}")
                        hypothesis_text = refined_hyp
                        if refined_actions:
                            action_steps = refined_actions

                        kill_attempts.append({
                            "round": "refinement",
                            "killed": False,
                            "reason": f"Logic corrected: {logic_error}. Better play: {better_play or 'see refined hypothesis'}",
                            "confidence": None,
                            "kill_type": "refinement",
                        })
                    else:
                        print(f"  {C.YELLOW}⚠️  Logic error found but no refined version provided{C.RESET}")
                else:
                    print(f"  {C.GREEN}✓ Financial mechanics confirmed correct{C.RESET}")
                    kill_attempts.append({
                        "round": "refinement",
                        "killed": False,
                        "reason": refine_data.get("reasoning", "Mechanics verified correct"),
                        "confidence": None,
                        "kill_type": "refinement",
                    })

            except (json.JSONDecodeError, JSONParseError):
                print_info("Refinement parse failed (non-fatal)")

        except Exception as e:
            print_info(f"Refinement failed (non-fatal): {e}")

        return hypothesis_text, action_steps, kill_attempts

    def _chain_weakest_link_ok(self, chain):
        """Check if every link in a chain has a specific, verified transmission pathway.
        If any single link is weak (vague, short, or uses hedge words), the chain bonus
        is zeroed. A 5-link chain with one garbage link is worse than a solid 3-link chain.

        Returns True if all links pass, False if any link fails."""
        if not chain or "links" not in chain:
            return True  # No links to check = pass by default
        HEDGE_WORDS = {"somehow", "might", "could", "possibly", "perhaps", "may affect",
                       "generally", "typically", "tends to", "likely", "probably"}
        MIN_PATHWAY_LENGTH = 20  # A real pathway needs at least 20 chars of specificity
        for link in chain["links"]:
            pathway = link.get("transmission_pathway", "") or link.get("explanation", "")
            if not pathway or len(pathway) < MIN_PATHWAY_LENGTH:
                print(f"  {C.YELLOW}⚠️ Weak chain link {link.get('link', '?')}: pathway too short ({len(pathway)} chars){C.RESET}")
                return False
            pathway_lower = pathway.lower()
            for hedge in HEDGE_WORDS:
                if hedge in pathway_lower:
                    print(f"  {C.YELLOW}⚠️ Weak chain link {link.get('link', '?')}: hedge word '{hedge}' in pathway{C.RESET}")
                    return False
        return True

    def _compute_temporal_spread(self, facts):
        """Compute temporal spread between participating facts in days.

        Uses ingested_at as primary (reliable), falls back to date_of_fact if valid.
        Returns (spread_days, oldest_fact_age_days, details_dict).

        The signal: oldest fact has been public for N days and nobody connected it.
        That duration IS the signal of how deep in negative space the thesis lives.
        """
        dates = []
        now = datetime.now()

        for f in facts:
            ingested_at = f.get("ingested_at")
            if ingested_at:
                try:
                    dt = datetime.fromisoformat(ingested_at.replace("Z", "").split("+")[0])
                    dates.append(dt)
                    continue
                except (ValueError, TypeError):
                    pass
            date_of_fact = f.get("date_of_fact")
            if date_of_fact and date_of_fact != "unknown":
                try:
                    dt = datetime.strptime(date_of_fact, "%Y-%m-%d")
                    dates.append(dt)
                except (ValueError, TypeError):
                    pass

        if len(dates) < 2:
            return 0, 0, {}

        oldest = min(dates)
        newest = max(dates)
        spread_days = (newest - oldest).days
        oldest_age_days = (now - oldest).days

        return spread_days, oldest_age_days, {
            "oldest_date": oldest.isoformat(),
            "newest_date": newest.isoformat(),
            "spread_days": spread_days,
            "oldest_age_days": oldest_age_days,
            "num_facts_with_dates": len(dates),
        }

    def _compute_fact_confidence(self, fact_chain):
        """Compute a confidence multiplier based on the source types of contributing facts.

        SEC filings, FDA databases, patent records = high confidence (1.0)
        Regulation, government contracts = medium-high (0.9)
        Earnings, bankruptcy = medium (0.85)
        Academic, pharmaceutical = medium (0.85)
        App rankings, job listings, other = lower (0.75)
        """
        source_weights = {
            "sec_filing": 1.0, "patent": 1.0, "regulation": 0.95,
            "government_contract": 0.9, "pharmaceutical": 0.9,
            "bankruptcy": 0.85, "earnings": 0.85, "academic": 0.85,
            "job_listing": 0.75, "app_ranking": 0.7, "commodity": 0.8,
            "other": 0.7,
        }
        try:
            fact_ids = [f.get("fact_id") for f in fact_chain if f.get("fact_id")]
            if not fact_ids:
                return 1.0

            from database import get_connection
            conn = get_connection()
            cursor = conn.cursor()
            placeholders = ",".join(["?"] * len(fact_ids[:10]))
            cursor.execute(f"SELECT source_type FROM raw_facts WHERE id IN ({placeholders})",
                          [int(fid) for fid in fact_ids[:10]])
            sources = [r["source_type"] for r in cursor.fetchall()]
            conn.close()

            if not sources:
                return 1.0

            avg_weight = sum(source_weights.get(s, 0.75) for s in sources) / len(sources)
            return avg_weight
        except Exception:
            return 1.0

    def _score_and_save(self, collision_id, hypothesis_text, fact_chain,
                        action_steps, time_window, kill_attempts, num_domains=1, chains=None,
                        source_types=None):
        """Score a surviving hypothesis and save it."""
        print_phase("SCORE", "Scoring surviving hypothesis...")

        # In theory run, mechanism-killed hypotheses continue here for telemetry
        # but must be saved with survived_kill=False
        _survived = not getattr(self, "_theory_run_killed", False)

        prompt = HYPOTHESIS_SCORE_PROMPT.format(
            hypothesis_text=hypothesis_text[:2000],
            fact_chain=json.dumps(fact_chain)[:2000],
            kill_attempts=json.dumps(kill_attempts)[:1000],
        )
        # ADVERSARIAL SCORING: fresh context, no HUNTER identity, skeptical framing
        # The scorer has zero knowledge of the generation process to prevent self-grading bias
        response = call_text(prompt, system="You are a skeptical senior analyst at a quantitative hedge fund. Your job is to find reasons this score should be LOWER. You have never seen this hypothesis before. You have no investment in whether it succeeds. Score conservatively. Calibration anchors: [LIFE INSURANCE CRE 5-SILO CHAIN WITH NAIC CATALYST DATE = 92] [GRID CONNECTION ARBITRAGE WITH FERC QUEUE DATA = 88] [GENERIC SECTOR ROTATION = 35] [OBVIOUS SINGLE-HOP EARNINGS PLAY = 25]. For each dimension, state the strongest reason the score should be lower BEFORE assigning the number.", max_tokens=1024, temperature=0.3, model=MODEL_DEEP)
        self.track_tokens(response)
        text = extract_text_from_response(response)

        try:
            score_data = parse_json_response(text)
        except (json.JSONDecodeError, JSONParseError):
            print_error("Failed to parse score JSON")
            # Save without score
            save_hypothesis(
                collision_id=collision_id,
                hypothesis_text=hypothesis_text,
                fact_chain=fact_chain,
                action_steps=action_steps,
                time_window_days=time_window,
                kill_attempts=kill_attempts,
                survived_kill=_survived,
            )
            return

        # Extract scores (6 dimensions now)
        novelty = min(20, max(0, int(score_data.get("novelty", 0))))
        feasibility = min(20, max(0, int(score_data.get("feasibility", 0))))
        timing = min(20, max(0, int(score_data.get("timing", 0))))
        asymmetry = min(20, max(0, int(score_data.get("asymmetry", 0))))
        intersection = min(20, max(0, int(score_data.get("intersection", 0))))
        mechanism_integrity = min(20, max(0, int(score_data.get("mechanism_integrity", 10))))

        # Mechanism integrity cap: if mechanism score is low, total can't exceed 60
        # regardless of how good other dimensions look
        total_score = novelty + feasibility + timing + asymmetry + intersection
        if mechanism_integrity <= 5:
            total_score = min(50, total_score)
            print(f"  {C.YELLOW}⚠️ Low mechanism integrity ({mechanism_integrity}/20) — score capped at 50{C.RESET}")
        elif mechanism_integrity <= 10:
            total_score = min(70, total_score)
            print(f"  {C.YELLOW}⚠️ Moderate mechanism integrity ({mechanism_integrity}/20) — score capped at 70{C.RESET}")

        # DOMAIN BONUS: weighted by actual information-theoretic distance between silos
        # Patent + Bankruptcy (0.80 distance) is worth more than SEC Filing + Earnings (0.15)
        # Bonus = round(avg_pairwise_distance * 10), so:
        #   avg 0.2 (adjacent) = +2, avg 0.5 (moderate) = +5, avg 0.8 (distant) = +8
        domain_bonus = 0
        avg_distance = 0.0
        if source_types and len(source_types) >= 2:
            avg_distance = compute_avg_domain_distance(source_types)
            domain_bonus = round(avg_distance * 10)
            domain_bonus = min(10, max(0, domain_bonus))  # cap at 10
        elif num_domains >= 3:
            # Fallback if source_types not available (shouldn't happen in normal flow)
            domain_bonus = 2
        total_score = min(100, total_score + domain_bonus)
        if domain_bonus > 0:
            print(f"  {C.CYAN}+{domain_bonus} domain bonus (avg distance: {avg_distance:.2f} across {num_domains} domains){C.RESET}")

        # NARRATIVE PENALTY: empirical finding (r = -0.49 narrative → survival).
        # High-narrative hypotheses are the "obvious" ones kill-rounds can find.
        # Cheap local scoring; no LLM call.
        narrative_penalty = 0
        try:
            from narrative_detector import score_narrative
            ns = score_narrative(hypothesis_text or "")
            narrative_strength = ns.get("narrative_strength", 0.0)
            if narrative_strength >= 0.6:
                narrative_penalty = 5
            elif narrative_strength >= 0.45:
                narrative_penalty = 2
            if narrative_penalty > 0:
                print(f"  {C.YELLOW}-{narrative_penalty} narrative penalty "
                      f"(strength {narrative_strength:.2f}; high narrative correlates with kill){C.RESET}")
                total_score = max(0, total_score - narrative_penalty)
        except Exception as _e:
            logger.info(f"narrative_penalty_unavailable: {_e}")

        # FACT CONFIDENCE: weight score by quality of source types
        fact_confidence = self._compute_fact_confidence(fact_chain)
        confidence_adj = 0  # Module K: track for decomposition
        if fact_confidence < 0.85:
            confidence_adj = int((fact_confidence - 1.0) * 20)  # penalty up to -5
            total_score = max(0, total_score + confidence_adj)
            print(f"  {C.YELLOW}{confidence_adj} fact confidence penalty (avg source quality: {fact_confidence:.2f}){C.RESET}")
        elif fact_confidence > 0.95:
            confidence_adj = int((fact_confidence - 0.9) * 20)  # bonus up to +2
            total_score = min(100, total_score + confidence_adj)
            print(f"  {C.GREEN}+{confidence_adj} fact confidence bonus (high-quality sources: {fact_confidence:.2f}){C.RESET}")

        mult = max(0.7, min(1.3, float(score_data.get("actionability_multiplier", 1.0))))
        penalty = max(-15, min(0, float(score_data.get("confidence_penalty", 0))))

        # CHAIN BONUS: multi-link chains get scoring multiplier
        # 2 links = +5, 3 links = +10, 4+ links = +15. Capped at +15.
        # WEAKEST LINK CHECK: if ANY link in the chain is vague/unverified, bonus = 0.
        # A 5-link chain with one garbage link is worse than a solid 3-link chain.
        chain_bonus = 0
        if chains:
            best_chain = max(chains, key=lambda c: c["length"])
            if self._chain_weakest_link_ok(best_chain):
                if best_chain["length"] >= 4:
                    chain_bonus = 15
                elif best_chain["length"] >= 3:
                    chain_bonus = 10
                elif best_chain["length"] >= 2:
                    chain_bonus = 5
                if chain_bonus:
                    total_score = min(100, total_score + chain_bonus)
                    print(f"  {C.MAGENTA}⛓️ +{chain_bonus} chain bonus ({best_chain['length']} links across {len(set(best_chain['domains']))} domains){C.RESET}")
            else:
                print(f"  {C.YELLOW}⛓️ Chain bonus ZEROED — weakest link failed quality check{C.RESET}")

        # TEMPORAL ASYMMETRY BONUS: wider time gap = deeper in negative space
        # A patent from 8 months ago colliding with a bankruptcy from yesterday
        # is more interesting than two facts from the same week
        temporal_bonus = 0
        try:
            _telemetry_facts = getattr(self, "_matching_facts_for_telemetry", [])
            if _telemetry_facts:
                spread_days, oldest_age_days, _td = self._compute_temporal_spread(_telemetry_facts)
                # Spread bonus: how far apart are the oldest and newest facts?
                if spread_days >= 180:
                    temporal_bonus = 5
                elif spread_days >= 90:
                    temporal_bonus = 3
                elif spread_days >= 30:
                    temporal_bonus = 1
                # Age bonus: how long has the oldest fact been public and unconnected?
                if oldest_age_days >= 270:
                    temporal_bonus += 3
                elif oldest_age_days >= 180:
                    temporal_bonus += 2
                elif oldest_age_days >= 90:
                    temporal_bonus += 1
                temporal_bonus = min(8, temporal_bonus)
                if temporal_bonus > 0:
                    total_score = min(100, total_score + temporal_bonus)
                    print(f"  {C.CYAN}⏳ +{temporal_bonus} temporal asymmetry bonus (spread: {spread_days}d, oldest: {oldest_age_days}d){C.RESET}")
        except Exception:
            pass

        # NEGATIVE SPACE BONUS: market hasn't reacted = edge confirmed
        negative_space_bonus = 0
        _ns_data = getattr(self, "_negative_space_data", None)
        if _ns_data:
            try:
                ns_score = int(_ns_data.get("negative_space_score", 5))
                reacted = _ns_data.get("reaction_occurred", None)
                gap = _ns_data.get("gap_magnitude", "unknown")
                if reacted is False:
                    if gap == "total":
                        negative_space_bonus = 7
                    elif gap == "large":
                        negative_space_bonus = 5
                    elif gap == "medium":
                        negative_space_bonus = 3
                elif reacted is True and gap == "small":
                    negative_space_bonus = -2
                if negative_space_bonus > 0:
                    total_score = min(100, total_score + negative_space_bonus)
                    print(f"  {C.GREEN}🕳️ +{negative_space_bonus} negative space bonus (gap: {gap}){C.RESET}")
                elif negative_space_bonus < 0:
                    total_score = max(0, total_score + negative_space_bonus)
                    print(f"  {C.YELLOW}🕳️ {negative_space_bonus} negative space penalty (market already reacted){C.RESET}")
            except Exception as e:
                logger.warning(f"Negative space bonus failed: {e}")

        # REFLEXIVITY BONUS (Branch 3): exogenous fact contradicting endogenous = market delusion
        reflexivity_bonus = 0
        try:
            _telemetry_facts = getattr(self, "_matching_facts_for_telemetry", [])
            _anchor_reflex = getattr(self, "_anchor_reflexivity_tag", None)
            if _telemetry_facts and _anchor_reflex:
                exo_count = sum(1 for f in _telemetry_facts if f.get("reflexivity_tag") == "exogenous")
                endo_count = sum(1 for f in _telemetry_facts if f.get("reflexivity_tag") == "endogenous")
                if (_anchor_reflex == "exogenous" and endo_count >= 1) or \
                   (_anchor_reflex == "endogenous" and exo_count >= 1):
                    reflexivity_bonus = 3
                    total_score = min(100, total_score + reflexivity_bonus)
                    print(f"  {C.MAGENTA}🔄 +{reflexivity_bonus} reflexivity bonus (exogenous vs endogenous contradiction){C.RESET}")
        except Exception as e:
            logger.warning(f"Branch 3: reflexivity bonus failed: {e}")

        adjusted = max(0, min(100, (total_score * mult) + penalty))

        # Module G: record timestamp after scoring
        self._stage_timestamps["ts_scoring_completed"] = datetime.now().isoformat()

        # Module K: Store score components for decomposition analysis
        self._score_mechanism_integrity = mechanism_integrity
        self._score_domain_bonus = domain_bonus
        self._score_chain_bonus = chain_bonus
        self._score_fact_confidence_adj = confidence_adj
        self._score_actionability_multiplier = mult
        self._score_confidence_penalty = penalty
        self._score_temporal_bonus = temporal_bonus
        self._score_negative_space_bonus = negative_space_bonus
        self._score_reflexivity_bonus = reflexivity_bonus

        title = score_data.get("title", hypothesis_text[:60])
        summary = score_data.get("summary", "")
        confidence = score_data.get("confidence_level", "Medium")

        print_score(total_score, title, adjusted)

        # MARKET AWARENESS CHECK -- for high-scoring hypotheses, verify the edge still exists
        if total_score >= 60:
            print_phase("SCORE", "Market awareness check — verifying edge hasn't been arbitraged...")
            try:
                market_response = call_with_web_search(
                    f"""This hypothesis scored {total_score}/100 and proposes a specific trade or action:

HYPOTHESIS: {hypothesis_text[:1000]}
ACTION: {action_steps[:500]}

Before this can be confirmed as a real opportunity, AGGRESSIVELY search for whether this thesis already exists:
1. Search for the EXACT company name + the core thesis claim. Example: "[Company] patent expiry generic delay" or "[Company] PTAB challenge revenue impact"
2. Search major financial publications: Bloomberg, Reuters, Seeking Alpha, Barron's, FiercePharma, BioPharma Dive. Has ANY article made this argument?
3. If you can reach this thesis by reading ONE article from a major publication, the edge is GONE. Kill it.
4. Has the price already moved to reflect this thesis? If the stock is up/down 20%+ in the direction the thesis implies, institutional money already found this.
5. Check if any sell-side analyst has published a note with this thesis in the last 90 days.

The bar is HIGH. If this thesis could be assembled by a single analyst reading their normal information sources within their domain, it has no edge. The only hypotheses that survive are ones that REQUIRE reading across multiple professional silos simultaneously — something no single analyst does.

Respond with ONLY a JSON object:
{{
    "edge_still_exists": true/false,
    "current_price_action": "What the asset is doing right now",
    "already_published": "Has this exact thesis been published? By whom?",
    "price_already_moved": "Has the price already moved to reflect this?",
    "reasoning": "Why the edge does or doesn't still exist",
    "discovery_channels": ["list of specific publications/platforms where this thesis or its components have been discussed — e.g. 'Goldman Sachs research note', 'Seeking Alpha article', 'Reddit r/wallstreetbets post'. Empty list if no coverage found."],
    "time_since_discovery_days": "integer — approximate number of days since the EARLIEST known publication or discussion of this thesis or its key components. 0 if no coverage found, otherwise estimate from the oldest relevant source."{'''
    ,
    "awareness_level": "integer 0-4 using EXACTLY this scale: 0 = no published analysis found connecting these factors, 1 = one obscure mention (single blog post, forum thread, or tweet), 2 = multiple non-institutional sources discussing the connection, 3 = at least one institutional research note or major publication, 4 = consensus trade — multiple institutional sources, appears widely known",
    "awareness_sources": [
        {{
            "url": "URL of the source (or 'no_url' if not available)",
            "source_type": "blog/research_note/institutional/news/social/forum",
            "recency_days": "approximate age in days",
            "specificity": "exact_connection/one_side_only/tangential",
            "excerpt": "1-2 sentence quote showing what the source says about this thesis"
        }}
    ],
    "search_queries_used": ["the actual search queries you ran"]''' if THEORY_RUN else ''}
}}""",
                    system="You are a market analyst. Check whether a proposed trade thesis has already been discovered and priced in by the market.",
                    max_tokens=2048 if THEORY_RUN else 1024,
                )
                self.track_tokens(market_response)
                market_text = extract_text_from_response(market_response)

                try:
                    market_data = parse_json_response(market_text)

                    # ── THEORY RUN: Module A — Graded Market Awareness Telemetry ──
                    _awareness_telemetry = None
                    if THEORY_RUN:
                        from datetime import datetime as _dt
                        _awareness_sources = market_data.get("awareness_sources", [])
                        if not isinstance(_awareness_sources, list):
                            _awareness_sources = []

                        # Module C: Self-Contribution Probe
                        _self_flags = []
                        _hunter_vocab = {"practitioners", "methodology", "assumption", "calibration", "disruption"}
                        for src in _awareness_sources:
                            if not isinstance(src, dict):
                                continue
                            flag_reasons = []
                            # Check URL against user's domains
                            src_url = str(src.get("url", "")).lower()
                            for domain in THEORY_RUN_SELF_DOMAINS:
                                if domain.lower() in src_url:
                                    flag_reasons.append(f"url_matches_self_domain:{domain}")
                            # Check for HUNTER's distinctive 5-field vocabulary
                            excerpt = str(src.get("excerpt", "")).lower()
                            vocab_hits = sum(1 for v in _hunter_vocab if v in excerpt)
                            if vocab_hits >= 3:
                                flag_reasons.append(f"hunter_vocab_match:{vocab_hits}/5")
                            if flag_reasons:
                                _self_flags.append({
                                    "url": src.get("url", ""),
                                    "reasons": flag_reasons,
                                    "excerpt": src.get("excerpt", "")[:200],
                                })

                        _awareness_telemetry = {
                            "awareness_level": int(market_data.get("awareness_level", 0)),
                            "awareness_sources": _awareness_sources,
                            "search_queries_used": market_data.get("search_queries_used", []),
                            "awareness_check_timestamp": _dt.utcnow().isoformat() + "Z",
                            "edge_still_exists": market_data.get("edge_still_exists", True),
                            "self_contribution_flags": _self_flags,
                        }
                        # Store on self for later use by edge recovery and save
                        self._awareness_telemetry = _awareness_telemetry

                    # INFORMATION DECAY (Branch 4): continuous channel-weighted penalty
                    # Applied whether or not edge_still_exists — partial coverage = partial decay
                    _disc_channels = market_data.get("discovery_channels", [])
                    _time_since = market_data.get("time_since_discovery_days", 0)
                    if isinstance(_disc_channels, list) and _disc_channels and _time_since:
                        try:
                            _decay_penalty = compute_edge_decay_penalty(_disc_channels, _time_since)
                            if _decay_penalty < 0:
                                total_score = max(0, total_score + _decay_penalty)
                                adjusted = max(0, min(100, (total_score * mult) + penalty))
                                _ch_short = ", ".join(str(c)[:30] for c in _disc_channels[:3])
                                print(f"  {C.YELLOW}📉 {_decay_penalty} information decay ({_ch_short}, {_time_since}d ago){C.RESET}")
                        except Exception as e:
                            logger.warning(f"Branch 4: decay penalty failed: {e}")

                    if not market_data.get("edge_still_exists", True):
                        old_score = total_score
                        already_by = market_data.get("already_published", "market")
                        published_reasoning = market_data.get("reasoning", "")

                        # ── EDGE RECOVERY ROUND ──
                        # The broad category is known. But is there a specific element
                        # (quantification, chain link, mechanism) that IS novel?
                        # If yes, reframe around the novel element and partially recover the score.
                        print(f"  {C.YELLOW}🔍 Edge Recovery — searching for novel sub-element...{C.RESET}")
                        try:
                            recovery_response = call_with_web_search(
                                f"""The BROAD CATEGORY of this thesis has been published:

THESIS: {hypothesis_text[:800]}
WHAT'S ALREADY KNOWN: {published_reasoning[:500]}
PUBLISHED BY: {already_by[:200]}

However, the thesis may contain SPECIFIC elements that are NOT in the published research. Search for whether these specific sub-elements are novel:

1. SPECIFIC QUANTIFICATION — Has anyone published the exact numbers/ratios this thesis cites? (e.g., "28x differential" or "12.34% vs 0.43%" or specific dollar amounts)
2. SPECIFIC MECHANISM — Has anyone published the specific causal chain this thesis describes? Not just "A affects B" but the specific multi-step pathway?
3. SPECIFIC TRADE STRUCTURE — Has anyone published the specific trade this thesis recommends? (e.g., the specific convergence trade, the specific pair trade, the specific instrument)
4. SPECIFIC CHAIN LINK — If this thesis traces a multi-domain chain, has anyone published the LATER links? (The first link may be known but links 3-5 may be novel)

Search specifically for each element. For each one, state whether it IS or IS NOT in published research.

Respond with ONLY a JSON object:
{{
    "has_novel_element": true/false,
    "novel_element": "Description of what specifically is novel (only if true)",
    "reframed_thesis": "The thesis reframed around the novel element (only if true)",
    "recovery_reasoning": "Why this specific element hasn't been published"
}}""",
                                system="You are checking whether a specific sub-element of an investment thesis is novel, even though the broad category is known. Be precise about what IS and ISN'T published.",
                                max_tokens=1024,
                            )
                            self.track_tokens(recovery_response)
                            recovery_text = extract_text_from_response(recovery_response)
                            recovery_data = parse_json_response(recovery_text)

                            if recovery_data.get("has_novel_element"):
                                novel_el = recovery_data.get("novel_element", "")
                                reframed = recovery_data.get("reframed_thesis", "")
                                # Bug 3 fix: save original text BEFORE overwrite
                                _original_thesis_for_recovery = hypothesis_text
                                print(f"  {C.GREEN}✅ NOVEL ELEMENT FOUND: {novel_el[:80]}{C.RESET}")

                                # RE-SCORE the reframed thesis from scratch
                                # The reframed version is a different thesis — it deserves its own score
                                print(f"  {C.YELLOW}📊 Re-scoring reframed thesis...{C.RESET}")
                                reframe_text = reframed if reframed else f"{hypothesis_text}\n\nNOVEL EDGE: {novel_el}"

                                try:
                                    rescore_prompt = HYPOTHESIS_SCORE_PROMPT.format(
                                        hypothesis_text=reframe_text[:2000],
                                        fact_chain=json.dumps(fact_chain)[:1500],
                                        kill_attempts=json.dumps(kill_attempts)[:800],
                                    )
                                    rescore_response = call_text(rescore_prompt, system="You are a skeptical senior analyst at a quantitative hedge fund. Your job is to find reasons this score should be LOWER. You have never seen this hypothesis before. You have no investment in whether it succeeds. Score conservatively. Calibration anchors: [LIFE INSURANCE CRE 5-SILO CHAIN WITH NAIC CATALYST DATE = 92] [GRID CONNECTION ARBITRAGE WITH FERC QUEUE DATA = 88] [GENERIC SECTOR ROTATION = 35] [OBVIOUS SINGLE-HOP EARNINGS PLAY = 25]. For each dimension, state the strongest reason the score should be lower BEFORE assigning the number.", max_tokens=1024, temperature=0.3, model=MODEL_DEEP)
                                    self.track_tokens(rescore_response)
                                    rescore_text = extract_text_from_response(rescore_response)
                                    rescore_data = parse_json_response(rescore_text)

                                    new_novelty = int(rescore_data.get("novelty", novelty))
                                    new_feasibility = int(rescore_data.get("feasibility", feasibility))
                                    new_timing = int(rescore_data.get("timing", timing))
                                    new_asymmetry = int(rescore_data.get("asymmetry", asymmetry))
                                    new_intersection = int(rescore_data.get("intersection", intersection))
                                    new_mechanism = int(rescore_data.get("mechanism_integrity", 10))
                                    new_total = new_novelty + new_feasibility + new_timing + new_asymmetry + new_intersection
                                    # Apply mechanism cap
                                    if new_mechanism <= 5:
                                        new_total = min(50, new_total)
                                    elif new_mechanism <= 10:
                                        new_total = min(70, new_total)

                                    # Apply same bonuses as original (domain distance + chain + temporal + negative space + reflexivity)
                                    if source_types and len(source_types) >= 2:
                                        _avg_dist = compute_avg_domain_distance(source_types)
                                        _db = min(10, max(0, round(_avg_dist * 10)))
                                        new_total = min(100, new_total + _db)
                                    if chains:
                                        best_chain = max(chains, key=lambda c: c["length"])
                                        _chain_ok = self._chain_weakest_link_ok(best_chain)
                                        if _chain_ok:
                                            if best_chain["length"] >= 4: new_total = min(100, new_total + 15)
                                            elif best_chain["length"] >= 3: new_total = min(100, new_total + 10)
                                            elif best_chain["length"] >= 2: new_total = min(100, new_total + 5)
                                    # Re-apply temporal, negative space, and reflexivity bonuses (computed earlier)
                                    if temporal_bonus > 0:
                                        new_total = min(100, new_total + temporal_bonus)
                                    if negative_space_bonus != 0:
                                        new_total = max(0, min(100, new_total + negative_space_bonus))
                                    if reflexivity_bonus > 0:
                                        new_total = min(100, new_total + reflexivity_bonus)

                                    new_adjusted = max(0, min(100, (new_total * mult) + penalty))

                                    print(f"  {C.GREEN}📊 REFRAMED SCORE: {new_total}/100 (was {old_score} before cap){C.RESET}")
                                    print(f"  {C.GREEN}   N:{new_novelty} F:{new_feasibility} T:{new_timing} A:{new_asymmetry} I:{new_intersection}{C.RESET}")

                                    # Use the reframed score — no cap since the novel element has confirmed edge
                                    total_score = new_total
                                    adjusted = new_adjusted
                                    novelty = new_novelty
                                    feasibility = new_feasibility
                                    timing = new_timing
                                    asymmetry = new_asymmetry
                                    intersection = new_intersection

                                    # Update the hypothesis text to include the reframe
                                    if reframed:
                                        hypothesis_text = reframed

                                    title = rescore_data.get("title", title)

                                    # Auto-generate PDF if reframed score is diamond-grade (skip in theory run)
                                    if new_total >= 65 and not THEORY_RUN:
                                        print(f"  {C.MAGENTA}📄 Auto-generating PDF for edge-recovered diamond...{C.RESET}")
                                        try:
                                            import subprocess as _sp
                                            # Save hypothesis first so enrich can find it
                                            # (it will be saved below in the normal flow)
                                            # Instead, queue for PDF generation after save
                                            self._queued_pdf_generation = True
                                        except Exception:
                                            pass

                                except Exception as e:
                                    # Re-scoring failed — use 70% fallback
                                    total_score = max(45, int(old_score * 0.7))
                                    adjusted = max(0, min(100, (total_score * mult) + penalty))
                                    print(f"  {C.YELLOW}Re-score failed ({e}), using 70% fallback: {total_score}{C.RESET}")

                                kill_attempts.append({
                                    "round": "market_awareness",
                                    "killed": False,
                                    "reason": f"Edge degraded on broad category ({already_by[:80]}), but RECOVERED on novel sub-element: {novel_el[:200]}",
                                    "confidence": None,
                                    "kill_type": "market_check",
                                })
                                kill_attempts.append({
                                    "round": "edge_recovery",
                                    "killed": False,
                                    "reason": f"Novel element: {novel_el}. Reframed thesis re-scored at {total_score}/100. Original broad category scored {old_score} before cap.",
                                    "confidence": None,
                                    "kill_type": "edge_recovery",
                                })

                                # ── THEORY RUN: Module B — Edge Recovery Telemetry (success path) ──
                                if THEORY_RUN:
                                    _orig_awareness = _awareness_telemetry.get("awareness_level", 0) if _awareness_telemetry else 0
                                    # Fresh graded awareness check on the RECOVERED thesis
                                    _recovered_awareness = 0
                                    try:
                                        _recheck_resp = call_with_web_search(
                                            f"""Rate the market awareness level of this SPECIFIC thesis (not the broad category):

THESIS: {(reframed or hypothesis_text)[:800]}

Using EXACTLY this scale:
0 = no published analysis found connecting these factors
1 = one obscure mention (single blog post, forum thread, or tweet)
2 = multiple non-institutional sources discussing the connection
3 = at least one institutional research note or major publication
4 = consensus trade — multiple institutional sources, appears widely known

Respond with ONLY a JSON object:
{{"awareness_level": 0-4, "reasoning": "why this level"}}""",
                                            system="Rate market awareness on the 0-4 scale provided. Be precise.",
                                            max_tokens=512,
                                        )
                                        self.track_tokens(_recheck_resp)
                                        _recheck_data = parse_json_response(extract_text_from_response(_recheck_resp))
                                        _recovered_awareness = int(_recheck_data.get("awareness_level", 0))
                                    except Exception:
                                        _recovered_awareness = 0

                                    # Bug 3 fix: use saved original text, not overwritten hypothesis_text
                                    _recovery_original = _original_thesis_for_recovery[:2000]
                                    _recovery_recovered = (reframed or hypothesis_text)[:2000]
                                    # Sanity check: are they actually different?
                                    from difflib import SequenceMatcher as _SM
                                    _sim = _SM(None, _recovery_original, _recovery_recovered).ratio()
                                    if _sim >= 0.9:
                                        print_info(f"Edge recovery text similarity {_sim:.2f} >= 0.9 — logging as recovery_failed")
                                        self._pending_edge_recovery = {
                                            "original_thesis_text": _recovery_original,
                                            "original_awareness_level": _orig_awareness,
                                            "killed_at_score": old_score,
                                            "novel_subelement_found": False,
                                            "recovered_thesis_text": None,
                                            "recovered_awareness_level": None,
                                            "recovered_score": None,
                                            "delta_awareness": None,
                                            "delta_score": None,
                                        }
                                    else:
                                        self._pending_edge_recovery = {
                                            "original_thesis_text": _recovery_original,
                                            "original_awareness_level": _orig_awareness,
                                            "killed_at_score": old_score,
                                            "novel_subelement_found": True,
                                            "recovered_thesis_text": _recovery_recovered,
                                            "recovered_awareness_level": _recovered_awareness,
                                            "recovered_score": total_score,
                                            "delta_awareness": _orig_awareness - _recovered_awareness,
                                            "delta_score": total_score - old_score,
                                        }

                            else:
                                # No novel element found — hard cap at 45
                                total_score = min(45, total_score)
                                adjusted = max(0, min(100, (total_score * mult) + penalty))
                                print(f"  {C.RED}🚫 EDGE GONE — no novel sub-element found{C.RESET}")
                                print(f"  {C.RED}   Score capped: {old_score} → {total_score}{C.RESET}")
                                kill_attempts.append({
                                    "round": "market_awareness",
                                    "killed": False,
                                    "reason": f"Edge degraded: {published_reasoning}",
                                    "confidence": None,
                                    "kill_type": "market_check",
                                })

                                # ── THEORY RUN: Module B — Edge Recovery Telemetry (failure path) ──
                                if THEORY_RUN:
                                    _orig_awareness = _awareness_telemetry.get("awareness_level", 0) if _awareness_telemetry else 0
                                    self._pending_edge_recovery = {
                                        "original_thesis_text": hypothesis_text[:2000],
                                        "original_awareness_level": _orig_awareness,
                                        "killed_at_score": old_score,
                                        "novel_subelement_found": False,
                                        "recovered_thesis_text": None,
                                        "recovered_awareness_level": None,
                                        "recovered_score": None,
                                        "delta_awareness": None,
                                        "delta_score": None,
                                    }

                        except Exception as e:
                            # Recovery failed — fall back to hard cap
                            total_score = min(45, total_score)
                            adjusted = max(0, min(100, (total_score * mult) + penalty))
                            print(f"  {C.RED}🚫 EDGE GONE — recovery failed: {e}{C.RESET}")
                            print(f"  {C.RED}   Score capped: {old_score} → {total_score}{C.RESET}")
                            kill_attempts.append({
                                "round": "market_awareness",
                                "killed": False,
                                "reason": f"Edge degraded: {published_reasoning}",
                                "confidence": None,
                                "kill_type": "market_check",
                            })
                    else:
                        print(f"  {C.GREEN}✓ Edge confirmed — not yet priced in{C.RESET}")
                        kill_attempts.append({
                            "round": "market_awareness",
                            "killed": False,
                            "reason": f"Edge confirmed: {market_data.get('reasoning', '')}",
                            "confidence": None,
                            "kill_type": "market_check",
                        })
                except (json.JSONDecodeError, JSONParseError):
                    pass
            except Exception as e:
                print_info(f"Market awareness check failed (non-fatal): {e}")

        # ── THEORY RUN: Unconditional modules (fire on every scored hypothesis) ──
        if THEORY_RUN:
            # Module F: Lightweight awareness probe (DISABLED in retrospective mode)
            self._lightweight_awareness_level = self._lightweight_awareness_probe(hypothesis_text)
            self._stage_timestamps["ts_awareness_probed"] = datetime.now().isoformat()

            # Module H: Entity specificity scoring
            self._entity_specificity_score = self._compute_entity_specificity(hypothesis_text)

            # Module J: Causal chain depth
            self._causal_chain_length = self._compute_causal_chain_length(hypothesis_text, fact_chain)

        # Hard floor: below 65 is not worth reporting
        if total_score < 65:
            print(f"  {C.DIM}Below threshold (65). Saved but no report.{C.RESET}")
            _hyp_id_low = save_hypothesis(
                collision_id=collision_id, hypothesis_text=hypothesis_text,
                fact_chain=fact_chain, action_steps=action_steps,
                time_window_days=time_window, kill_attempts=kill_attempts,
                survived_kill=_survived, diamond_score=total_score,
                novelty=novelty, feasibility=feasibility, timing=timing,
                asymmetry=asymmetry, intersection=intersection, confidence=confidence,
            )
            # ── THEORY RUN: Module D + telemetry flush for below-threshold ──
            if THEORY_RUN and _hyp_id_low:
                self._flush_theory_telemetry(_hyp_id_low)
            # Theory proof layer: record hypothesis evidence (all scores, not just high)
            if self._theory_recorder and _hyp_id_low:
                try:
                    self._theory_recorder.record_hypothesis(
                        hypothesis_id=_hyp_id_low, collision_id=collision_id,
                        diamond_score=total_score, survived_kill=_survived,
                        source_types=source_types or set(),
                        domain_distance=avg_distance,
                        score_components={"novelty": novelty, "feasibility": feasibility,
                                          "timing": timing, "asymmetry": asymmetry,
                                          "intersection": intersection,
                                          "mechanism_integrity": mechanism_integrity,
                                          "domain_bonus": domain_bonus, "chain_bonus": chain_bonus,
                                          "temporal_bonus": temporal_bonus,
                                          "negative_space_bonus": negative_space_bonus,
                                          "reflexivity_bonus": reflexivity_bonus},
                        negative_space_data=getattr(self, "_negative_space_data", None),
                        chains=chains,
                    )
                except Exception:
                    pass
            # Theory layer agents: log hypothesis evidence
            if self._theory_telemetry and _hyp_id_low:
                try:
                    _chain_len = max((c.get("length", 0) for c in chains), default=0) if chains else 0
                    self._theory_telemetry.log_hypothesis(
                        hypothesis_data={"hypothesis_id": _hyp_id_low},
                        final_score=total_score,
                        kill_results=kill_attempts,
                        chain_length=_chain_len,
                        domain_pair=source_types_set,
                        source_types=source_types,
                    )
                except Exception:
                    pass
            return

        # Write full report if score >= 65 (skip in theory run — we only need telemetry)
        full_report = None
        if total_score >= 65 and not THEORY_RUN:
            print_phase("REPORT", f"Writing report (score: {total_score})...")
            try:
                report_prompt = HYPOTHESIS_REPORT_PROMPT.format(
                    score=total_score,
                    hypothesis_text=hypothesis_text[:2000],
                    novelty=novelty, feasibility=feasibility, timing=timing,
                    asymmetry=asymmetry, intersection=intersection,
                    confidence=confidence,
                    fact_chain=json.dumps(fact_chain)[:3000],
                    kill_attempts=json.dumps(kill_attempts)[:1500],
                    title=title,
                    time_window=time_window,
                )
                report_response = call_text(report_prompt, system=SYSTEM_PROMPT,
                                            max_tokens=MAX_TOKENS_RESPONSE_REPORT)
                self.track_tokens(report_response)
                full_report = extract_text_from_response(report_response)
                print_info(f"Report saved ({len(full_report)} chars)")
            except Exception as e:
                print_error(f"Report generation failed: {e}")

        # Save hypothesis
        hyp_id = save_hypothesis(
            collision_id=collision_id,
            hypothesis_text=hypothesis_text,
            fact_chain=fact_chain,
            action_steps=action_steps,
            time_window_days=time_window,
            kill_attempts=kill_attempts,
            survived_kill=_survived,
            diamond_score=total_score,
            novelty=novelty,
            feasibility=feasibility,
            timing=timing,
            asymmetry=asymmetry,
            intersection=intersection,
            confidence=confidence,
            full_report=full_report,
        )

        # ── THEORY RUN: Module D + telemetry flush for above-threshold ──
        if THEORY_RUN and hyp_id:
            self._flush_theory_telemetry(hyp_id)

        # Theory proof layer: record hypothesis evidence
        if self._theory_recorder and hyp_id:
            try:
                self._theory_recorder.record_hypothesis(
                    hypothesis_id=hyp_id, collision_id=collision_id,
                    diamond_score=total_score, survived_kill=_survived,
                    source_types=source_types or set(),
                    domain_distance=avg_distance,
                    score_components={"novelty": novelty, "feasibility": feasibility,
                                      "timing": timing, "asymmetry": asymmetry,
                                      "intersection": intersection,
                                      "mechanism_integrity": mechanism_integrity,
                                      "domain_bonus": domain_bonus, "chain_bonus": chain_bonus,
                                      "temporal_bonus": temporal_bonus,
                                      "negative_space_bonus": negative_space_bonus,
                                      "reflexivity_bonus": reflexivity_bonus},
                    negative_space_data=getattr(self, "_negative_space_data", None),
                    chains=chains,
                )
            except Exception:
                pass

        # Theory layer agents: log hypothesis evidence
        if self._theory_telemetry and hyp_id:
            try:
                _chain_len = max((c.get("length", 0) for c in chains), default=0) if chains else 0
                self._theory_telemetry.log_hypothesis(
                    hypothesis_data={"hypothesis_id": hyp_id},
                    final_score=total_score,
                    kill_results=kill_attempts,
                    chain_length=_chain_len,
                    domain_pair=source_types or set(),
                    source_types=source_types,
                )
            except Exception:
                pass

        # Also save to findings table for knowledge graph compatibility (skip in theory run)
        if total_score >= 65 and not THEORY_RUN:
            finding_id = save_finding(
                title=title,
                domain="Cross-Domain Collision",
                score=total_score,
                novelty_score=novelty,
                feasibility_score=feasibility,
                timing_score=timing,
                asymmetry_score=asymmetry,
                intersection_score=intersection,
                actionability_multiplier=mult,
                confidence_penalty=penalty,
                personal_fit_bonus=0,
                adjusted_score=adjusted,
                summary=summary,
                full_report=full_report,
                evidence_urls="[]",
                action_steps=action_steps,
                confidence=confidence,
                time_sensitivity=score_data.get("time_sensitivity", f"{time_window} days"),
            )
            print_info(f"Saved as finding #{finding_id} and hypothesis #{hyp_id}")

            # Auto-generate PDF for edge-recovered diamonds (skip in theory run)
            if getattr(self, '_queued_pdf_generation', False) and hyp_id and not THEORY_RUN:
                try:
                    import subprocess as _sp
                    print(f"  {C.MAGENTA}📄 Generating PDF for edge-recovered diamond #{hyp_id}...{C.RESET}")
                    _sp.Popen(
                        ["python", "enrich_thesis.py", str(hyp_id)],
                        cwd=os.path.dirname(os.path.abspath(__file__)),
                        stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
                    )
                    print(f"  {C.MAGENTA}📄 PDF generation started in background{C.RESET}")
                    self._queued_pdf_generation = False
                except Exception as e:
                    print_info(f"PDF generation failed (non-fatal): {e}")

            # Trigger deep dive for high-scoring hypotheses (skip in theory run)
            if total_score >= 75 and not THEORY_RUN:
                print(f"\n  {C.BG_MAGENTA}{C.WHITE}{C.BOLD} DIAMOND DETECTED! Entering Deep Dive... {C.RESET}")
                self._deep_dive(hyp_id, finding_id, title, hypothesis_text, fact_chain)

        return hyp_id

    def _deep_dive(self, hypothesis_id, finding_id, title, hypothesis_text, fact_chain):
        """Deep dive into a high-scoring hypothesis from multiple angles."""
        print_phase("SCORE", f"Deep diving: {title[:60]}...")

        try:
            prompt = DEEP_DIVE_PROMPT.format(
                score=75,
                title=title,
                domain_name="Cross-Domain Collision",
                analysis=hypothesis_text[:3000],
            )

            response = call_with_web_search(
                prompt,
                system=SYSTEM_PROMPT,
                max_tokens=MAX_TOKENS_RESPONSE_REPORT,
            )
            self.track_tokens(response)
            deep_dive_text = extract_text_from_response(response)

            # Save deep dive
            save_deep_dive(
                finding_id=finding_id,
                additional_searches=3,
                total_tokens=count_tokens(response),
                validation_notes=deep_dive_text[:2000],
                competitor_analysis="See full report",
                market_size="See full report",
                action_plan="See full report",
                final_recommendation="PURSUE" if True else "MONITOR",
            )

            print(f"  {C.GREEN}✓ Deep dive complete ({len(deep_dive_text)} chars){C.RESET}")

        except Exception as e:
            print_error(f"Deep dive failed (non-fatal): {e}")


# ============================================================
# Daily synthesis
# ============================================================

def run_daily_synthesis():
    """Run a daily synthesis cycle to review all findings and plan next steps."""
    print_phase("SYNTHESIS", "Running daily synthesis...")

    try:
        today = datetime.now().strftime("%Y-%m-%d")
        last_summary_date = get_last_daily_summary_date()

        # Get findings since last summary (or last 24 hours)
        if last_summary_date:
            findings = get_findings_since(last_summary_date)
        else:
            findings = get_findings_since(
                (datetime.now().replace(hour=0, minute=0, second=0)).isoformat()
            )

        cycles_run = get_cycles_since_last_summary()

        # Build findings summary
        findings_summary = ""
        diamonds_found = 0
        for f in findings[:50]:
            score = f.get("adjusted_score", f.get("score", 0))
            findings_summary += f"\n[#{f['id']}] Score: {score} | {f.get('domain', '?')} | {f['title']}\n"
            findings_summary += f"  {f.get('summary', '')[:200]}\n"
            if score >= 75:
                diamonds_found += 1

        if not findings_summary:
            findings_summary = "No findings recorded in this period."

        prompt = DAILY_SYNTHESIS_PROMPT.format(
            date=today,
            total_cycles=cycles_run,
            total_findings=len(findings),
            diamonds_found=diamonds_found,
            findings_summary=findings_summary[:6000],
            evolutions_summary="None tracked in this period.",
            cross_refs_summary="None tracked in this period.",
        )

        response = call_text(prompt, system=SYSTEM_PROMPT, max_tokens=MAX_TOKENS_RESPONSE_REPORT, temperature=0.5)
        text = extract_text_from_response(response)
        tokens_used = response.usage.output_tokens

        try:
            synthesis_data = parse_json_response(text)
        except (json.JSONDecodeError, JSONParseError):
            # If JSON parse fails, save raw text as synthesis
            synthesis_data = {
                "most_promising_thread": "",
                "missed_connections": "",
                "tomorrow_priorities": "",
                "full_synthesis": text,
            }

        best_finding_id = findings[0]["id"] if findings else None

        save_daily_summary(
            summary_date=today,
            total_cycles=cycles_run,
            total_findings=len(findings),
            diamonds_found=diamonds_found,
            best_finding_id=best_finding_id,
            most_promising_thread=synthesis_data.get("most_promising_thread", ""),
            missed_connections=synthesis_data.get("missed_connections", ""),
            tomorrow_priorities=synthesis_data.get("tomorrow_priorities", ""),
            full_synthesis=synthesis_data.get("full_synthesis", text),
            tokens_used=tokens_used,
        )

        print(f"  {C.GREEN}✓ Daily synthesis complete{C.RESET}")
        if synthesis_data.get("most_promising_thread"):
            print(f"  {C.BOLD}Most promising: {str(synthesis_data['most_promising_thread'])[:100]}{C.RESET}")

    except Exception as e:
        print_error(f"Daily synthesis failed: {e}")


# ============================================================
# Knowledge base stats (every 50 ingest cycles)
# ============================================================

def print_knowledge_base_stats():
    stats = get_knowledge_base_stats()
    print(f"\n  {C.CYAN}📊 Knowledge base: {stats['total_facts']} facts | {stats['total_anomalies']} anomalies | {stats['unique_entities']} unique entities{C.RESET}\n")


# ============================================================
# Main loop
# ============================================================

_shutdown = False


def signal_handler(sig, frame):
    global _shutdown
    print(f"\n{C.YELLOW}Shutting down gracefully after current cycle...{C.RESET}")
    _shutdown = True


def main():
    global _shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    init_db()

    # Verify API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print(f"{C.RED}ERROR: Set ANTHROPIC_API_KEY in .env file{C.RESET}")
        sys.exit(1)

    logger.info("HUNTER v2 starting...")
    print_banner()
    print(f"  {C.DIM}Model: {MODEL}{C.RESET}")
    print(f"  {C.DIM}Ingest ratio: {INGEST_RATIO*100:.0f}% ingest / {(1-INGEST_RATIO)*100:.0f}% collision{C.RESET}")
    print(f"  {C.DIM}Data sources: {len(DATA_SOURCES)}{C.RESET}")
    print(f"  {C.DIM}Kill attempts per hypothesis: {KILL_SEARCH_COUNT}{C.RESET}")
    print()

    cycle_num = 0
    ingest_count = 0
    last_synthesis_time = time.time()

    while not _shutdown:
        cycle_num += 1

        try:
            if random.random() < INGEST_RATIO:
                # INGEST MODE
                ingest_count += 1
                IngestCycle(cycle_num).run()

                # Every 50 ingest cycles, print knowledge base stats
                if ingest_count % 50 == 0:
                    print_knowledge_base_stats()

            else:
                # COLLISION MODE
                CollisionCycle(cycle_num).run()

        except SystemExit as e:
            print(f"\n{C.YELLOW}Stopping: {e}{C.RESET}")
            break

        except Exception as e:
            print_error(f"Cycle failed: {e}")
            import traceback
            traceback.print_exc()

        # Daily synthesis check: every 100 ingest cycles OR every 24 hours
        hours_since_synthesis = (time.time() - last_synthesis_time) / 3600
        if ingest_count > 0 and (ingest_count % 100 == 0 or hours_since_synthesis >= 24):
            run_daily_synthesis()
            last_synthesis_time = time.time()

        # Theory layer periodic tasks (decay tracking, cycle detection, backtesting)
        try:
            from theory_layer import run_periodic_theory_tasks
            run_periodic_theory_tasks()
        except Exception:
            pass  # Theory agents never break main loop

        if _shutdown:
            break

        # Pause between cycles
        pause = random.uniform(3, 5)
        print(f"  {C.DIM}Next cycle in {pause:.0f}s...{C.RESET}")
        time.sleep(pause)

    # Final stats
    print(f"\n{C.BOLD}Session complete.{C.RESET}")
    stats = get_knowledge_base_stats()
    logger.info(f"Session complete. Facts: {stats['total_facts']}")
    print(f"  Facts: {stats['total_facts']} | Anomalies: {stats['total_anomalies']} | Entities: {stats['unique_entities']}")
    rate = get_collision_to_hypothesis_rate()
    print(f"  Collision → Hypothesis rate: {rate*100:.1f}%")


if __name__ == "__main__":
    main()
