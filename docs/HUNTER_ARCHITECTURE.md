# hunter.py architecture

A walkthrough of the main engine file. `hunter.py` is the long file in the repo (4,534 lines). This document is the map. Read it before opening the file.

The file is one of seven covered by the SHA-256 pre-registration lock (`f39d2f5ff6b3e695`). It cannot be split or refactored during the summer 2026 study window without breaking the manifest. Splitting into a proper package is scheduled for v2 (September 2026 onwards).

## File layout

The file is organised into thirteen internal sections, each marked by a `# ===` header. Every section has a single responsibility. Use this table to jump:

| Lines | Section | What lives here |
|---:|---|---|
| 1–116 | Imports and config | Standard library, third-party, internal config and prompts |
| 117–177 | Embedding model | Lazy-loaded sentence-transformers wrapper. `compute_fact_embedding()` produces 384-dim vectors from implications + model-vulnerability fields |
| 178–294 | Terminal colours and pretty-print helpers | `class C` for ANSI codes; `print_info`, `print_error`, `print_warning`, `print_collision`, etc. |
| 295–303 | Custom exceptions | `JSONParseError` and friends |
| 304–333 | Smart rate limiting | Token-bucket throttle around the Anthropic client |
| 334–385 | Retry logic with exponential backoff | Wraps every API call with retry + jitter |
| 386–549 | Fact validation | Schema enforcement on extracted facts, type coercion, rejection logging |
| 550–644 | Credit balance detection | Detects "you're out of API credits" and halts cleanly |
| 645–665 | Helpers | Small utilities |
| 666–1119 | `class IngestCycle` | One ingest pass against one source type. Pulls facts via web search, extracts entities/implications/causal arrows, persists to DB, runs anomaly detection |
| 1120–4346 | `class CollisionCycle` | The main work. Looks at recent anomalies, runs seven matching strategies in parallel, evaluates and blends matches, forms hypotheses, runs the four-round adversarial kill phase plus financial-mechanics check, scores survivors with a fresh-context adversarial reviewer |
| 4347–4429 | Daily synthesis | End-of-day rollup: cycle counts, top findings, daily summary insert |
| 4430–4438 | Knowledge base stats | Periodic reporting (every 50 ingest cycles) |
| 4439–end | Main loop | The `__main__` entry point: instantiates orchestrator, runs cycles, handles SIGINT |

## The two big classes

### `IngestCycle` (lines 666–1119, ~450 lines)

One instance per ingest pass. Constructor takes a source-type config (e.g. `{"type": "patent", "queries": [...], "weight": 1.0}`). The lifecycle is:

1. `run()` → entry point
2. Pick a query template, web search via the Anthropic API's `web_search_tool`
3. Extract facts from the search result via `INGEST_EXTRACT_PROMPT`
4. For each fact: validate schema, save to `raw_facts`, save fact embedding, save causal edges
5. Batch anomaly detection across all saved facts (one API call for the batch, not per-fact)
6. Persist `cycle_logs` row with telemetry

Failure modes are explicit. `anthropic.RateLimitError` triggers a 60-second backoff. Credit exhaustion raises `SystemExit`. Unexpected exceptions log and continue.

### `CollisionCycle` (lines 1120–4346, ~3,200 lines)

This is the actual brain. One instance per collision pass. The lifecycle:

1. Pull recent anomalies from the database (lookback governed by `COLLISION_LOOKBACK_ANOMALIES`)
2. For each anomaly, run **seven matching strategies in parallel** to find candidate facts in other silos:
   - Implication matching
   - Entity matching
   - Keyword matching (fallback)
   - Model-field matching
   - Causal-graph traversal
   - Embedding similarity
   - Belief-reality contradiction
3. Blend matches into a 10-fact pool, evaluate the collision via `COLLISION_EVALUATE_PROMPT`
4. If the collision is promising, form a hypothesis via `HYPOTHESIS_FORM_PROMPT` with explicit asset, direction, and resolution date
5. Run the four-round kill phase via `KILL_PROMPT`:
   - Mechanism check (does each causal arrow name a transmission pathway?)
   - Fact-check (does each underlying fact verify against the live web?)
   - Competitor check (has someone published this thesis already?)
   - Barrier check (is there a regulatory or structural obstacle?)
6. If kill survives, run financial-mechanics validation
7. Score the surviving hypothesis with `HYPOTHESIS_SCORE_PROMPT` against four calibration anchors
8. If score ≥ 65, save to `findings` and post to the prediction board

Each step has its own try/except with specific failure logging. The class is large because the matching strategies, the kill rounds, and the scoring all happen in one place.

**Why not split CollisionCycle into separate files.** The collision pipeline is one logical operation that flows through seven matching strategies, a blend, a kill gauntlet, and a scorer. Splitting it across five files would force a lot of state-passing or session globals. The current monolith keeps it readable as one flow at the cost of file length. Re-evaluation point: v2 in September.

## Where the kill rounds actually live

Every kill round is a method on `CollisionCycle`:

- `_kill_round_mechanism()` — the most important round; verifies each causal arrow has a named pathway
- `_kill_round_fact_check()` — verifies underlying facts against live web
- `_kill_round_competitor()` — searches for prior published versions of the thesis
- `_kill_round_barrier()` — checks for regulatory or structural barriers
- `_market_check()` — sanity check on whether the trade direction is correct given current pricing

Each kill round is wired through `KILL_PROMPT` in `prompts.py` with a different round-name parameter. The prompts handle the actual instructions; the methods handle orchestration, retries, and result parsing.

## Logging

All log output goes to `hunter.log` in the repo root via `logging.basicConfig`. Console output uses the `print_*` helpers in lines 178–294. The two channels are intentionally separate: the file log is for forensics, the console is for the human watching the run.

## Imports inside `hunter.py`

Top of file imports are grouped:
- Standard library (lines 7–15)
- Third-party (`anthropic`, `dotenv`, lines 17–18)
- Internal config (lines 20–48): explicit names imported from `config.py` to make dependencies legible
- Internal database (lines 49–...): explicit names from `database.py`
- Internal prompts (lines 99–115): explicit names from `prompts.py`

`logging` is imported at line 163, mid-file, in front of the logging setup block. This is a stylistic wart that will be cleaned up in v2.

## What's NOT in hunter.py

The pipeline depends on these external modules:

- `database.py` — schema, all SQL, all DB accessors. Not in the SHA hash.
- `prompts.py` — all 26 LLM prompts. In the SHA hash (changing a prompt would change the engine's behaviour).
- `config.py` — domain definitions, constants, model routing, distance matrix. In the hash.
- `theory.py` — collision-formula math, depth-value math. In the hash.
- `cycle_detector.py` — Tarjan SCC over the causal graph. In the hash.
- `thesis_dedup.py` — dedup logic. In the hash.
- `portfolio_feedback.py` — closes the loop from realised P&L back to scoring. In the hash.
- All analyser modules (`narrative_detector.py`, `halflife_estimator.py`, `phase_transition_detector.py`, etc.) — not in the hash. Each is single-responsibility.
- `orchestrator.py` — schedules the theory agents. Not in the hash.

## How to read the file efficiently

If you're trying to understand a specific behaviour, start by knowing which section it lives in (the table at the top of this doc). Most of the file is `IngestCycle` and `CollisionCycle`. The internal `# ===` headers are real section markers; jump to them with grep:

```bash
grep -n "^# ===" hunter.py
```

If you want to know what an LLM prompt does, the prompt itself lives in `prompts.py`. `hunter.py` only contains orchestration around the prompt.

## Why it's a monolith

Honest answer. I built this solo over six months as a research instrument. I knew the hot path (collision + kill + score) had to share a lot of state, and I didn't want to introduce session-passing boilerplate before I had the right abstractions. So `CollisionCycle` grew. Every section was added when I needed it, and the class structure preserves the logical flow as one readable pass.

A senior engineer would split this into a `collision/` package with separate modules for each matching strategy and each kill round. That's the v2 refactor. For the summer 2026 study, the monolith is what runs against the locked SHA hash.

## Pre-registration lock

`hunter.py`, `prompts.py`, `config.py`, `theory.py`, `thesis_dedup.py`, `portfolio_feedback.py`, and `cycle_detector.py` are hashed together via `preregister.py:_code_hash()`. The combined SHA-256 (truncated to 16 hex chars) is `f39d2f5ff6b3e695`, locked 2026-04-19. Any change to any of these seven files invalidates the lock and breaks the pre-registration.

The lock will be released and recomputed for v2 after the summer 2026 study reports in September.
