#!/usr/bin/env python3
"""HUNTER — single entry point.

One command instead of remembering which script to run. Every subcommand
documents what it needs (API budget, corpus state, time) so you never
accidentally run the live pipeline when you meant to analyse offline.

Usage:
    python run.py <command> [args]

Core commands:
    live              Run the main ingest/collision engine continuously (requires API key).
                      This is the hunter.py main loop via the orchestrator.
    live-no-theory    Same as live but without theory-layer agents (cheaper).
    dashboard         Launch the unified Streamlit dashboard at localhost:8501.
    status            Print one-screen state of the system (free, offline).

Analysis commands (all free, no API calls):
    analyse           Run all 15 analyser modules in the correct order.
                      Populates every empirical table and refreshes the dashboard.
    analyse quick     Run only the three fastest analysers (no embedding models).
    cycles            Re-detect epistemic cycles (Tarjan SCC over causal graph).
    tam               Recompute TAM scenarios.
    formula           Validate collision formula against observed collisions.
    reinforcement     Measure reinforcement/correction rates per domain.
    halflife          Estimate fact half-life per source type.
    narrative         Score narrative structure of all hypotheses.
    kill_topology     Map kill-failure topology.
    phase             Detect phase-transition risk per domain.
    adversarial       Classify hypotheses as accidental vs adversarial residual.
    obscurity         Audit finding obscurity scores.

Data-fixup commands:
    backfill          Run all safe backfills (embeddings, telemetry, causal edges).
    causal_edges      Extract causal edges from chains.
    embeddings        Backfill missing fact embeddings.

Governance commands:
    preregister       Show pre-registration status / preview strata.
    preregister lock  Lock the pre-registration manifest (frozen corpus + code hash).
    preregister check Verify the current state against the locked manifest.

Reports:
    report            Generate the latest PDF intelligence report.
    pitch             Open the internal pitch markdown if present (uses $PAGER).

Inverse HUNTER (alpha flywheel):
    decompose '<belief text>' [asset] [source]
                      Decompose a published belief into testable assumptions.
    inverse run       Process all active beliefs; emit signals (API cost).
    inverse simulate  Show candidate facts per belief (no API, free).
    inverse audit     List open inverse signals awaiting resolution.

Research diary (brand flywheel):
    diary             Generate incremental diary entries for new findings.
    diary weekly      Build this week's rollup.
    diary monthly     Build this month's rollup.
    diary preprint    Compile all monthlies into a preprint draft.

Cycle portfolio (theory-validating trades):
    cycle plan        Propose positions for each detected cycle (no trade).
    cycle open        Open paper positions for new cycles.
    cycle review      Refresh current prices on open cycle positions.
    cycle close       Close positions past target_exit_date.
    cycle audit       Full cycle portfolio report.

Reflexivity layer (moat flywheel):
    edge              Compute differential-edge score for all surviving hypotheses.
    edge audit        Rank hypotheses by how much only YOUR hunter finds them.

Temporal collisions:
    expire scan       Extract expiration events from corpus.
    expire calendar   Show next 90 days of expirations.
    expire collide    Find date-clustered expirations (temporal collisions).

Negative inference:
    gaps              Find missing-but-expected facts (source-type + chain-lag gaps).

Adversarial integrity:
    adversarial sim   Generate synthetic fake collisions (no LLM cost).
    adversarial run   Run synthetics through real kill phase (API cost).
    adversarial report  Show integrity test history.

Signal execution:
    orders today      Generate today's paper-trade instructions.
    orders pending    Show recent order files.

Performance attribution:
    attribution       Break down P&L by thesis / cycle / inverse portfolio.

Theory & research (full compendium integration):
    canon             Show canon v2 summary (25 domains, 300 pairs, clusters).
    frontier          Run all 6 frontier hypothesis detectors (F1-F6).
    frontier <fn>     Run a single detector (e.g. frontier f1).
    research          Show 95-item research space status.
    research <sect>   Show a specific sector (A-G).
    research next     Show next actionable items.
    moat              Show current moat strength (5 layers).
    moat targets      Show action items to strengthen each layer.
    coin              Show HUNTER Coin economics.
    coin scaling      Show scaling curve across 1-20k nodes.
    self              Show self-improvement goal + progress.
    self plan         Propose self-improvement changes.
    self apply        Apply Level 0 proposals automatically.

Meta:
    help              Show this message.
    doctor            Check environment, dependencies, DB health, pending work.
    clean             Remove temporary files (safe — never touches the DB).
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()
DB_PATH = HERE / "hunter.db"


def _run(cmd, env=None, check=True):
    """Run a subprocess and stream output."""
    print(f"\n→ {cmd}\n")
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    try:
        r = subprocess.run(cmd, shell=True, cwd=HERE, env=merged_env)
        if check and r.returncode != 0:
            print(f"\n[!] Exit code {r.returncode}")
        return r.returncode
    except KeyboardInterrupt:
        print("\n[stopped by user]")
        return 130


# ═══════════════════════════════════════════════════════════════════════
# CORE COMMANDS
# ═══════════════════════════════════════════════════════════════════════

def cmd_live(args):
    """Run the main live pipeline (API required)."""
    _check_api_key()
    return _run("python orchestrator.py")


def cmd_live_no_theory(args):
    """Run the main loop without theory agents."""
    _check_api_key()
    return _run("python hunter.py")


def cmd_dashboard(args):
    """Launch the master dashboard."""
    return _run("streamlit run master_dashboard.py")


def cmd_status(args):
    """One-screen system health."""
    import sqlite3
    if not DB_PATH.exists():
        print("No database yet. Run `python run.py live` first.")
        return 1

    conn = sqlite3.connect(DB_PATH)
    def q(sql):
        try:
            return conn.execute(sql).fetchone()[0]
        except Exception:
            return None

    print("=" * 70)
    print("  HUNTER — system state")
    print("=" * 70)
    print()
    print("CORPUS")
    print(f"  Facts:                 {q('SELECT COUNT(*) FROM raw_facts') or 0:,}")
    print(f"  Entity index:          {q('SELECT COUNT(*) FROM fact_entities') or 0:,}")
    print(f"  Model-field extract:   {q('SELECT COUNT(*) FROM fact_model_fields') or 0:,}")
    print(f"  Anomalies:             {q('SELECT COUNT(*) FROM anomalies') or 0:,}")
    print()
    print("REASONING")
    print(f"  Collisions:            {q('SELECT COUNT(*) FROM collisions') or 0:,}")
    print(f"  Chains:                {q('SELECT COUNT(*) FROM chains') or 0:,}")
    print(f"  Causal edges:          {q('SELECT COUNT(*) FROM causal_edges') or 0:,}")
    print(f"  Hypotheses:            {q('SELECT COUNT(*) FROM hypotheses') or 0:,}")
    print(f"  Surviving:             {q('SELECT COUNT(*) FROM hypotheses WHERE survived_kill=1') or 0:,}")
    print(f"  Detected cycles:       {q('SELECT COUNT(*) FROM detected_cycles') or 0:,}")
    print()
    print("THEORY & ANALYSIS")
    print(f"  Theory evidence rows:  {q('SELECT COUNT(*) FROM theory_evidence') or 0:,}")
    print(f"  Formula validations:   {q('SELECT COUNT(*) FROM formula_validation') or 0:,}")
    r_row = conn.execute(
        "SELECT pearson_r, p_value, verdict FROM formula_validation "
        "ORDER BY date DESC LIMIT 1"
    ).fetchone()
    if r_row:
        print(f"  Latest formula r:      {r_row[0]:+.3f}  (p={r_row[1]:.3f}, verdict={r_row[2]})")
    print()
    print("PORTFOLIO")
    open_sql = "SELECT COUNT(*) FROM portfolio_positions WHERE status='open'"
    closed_sql = "SELECT COUNT(*) FROM portfolio_positions WHERE status='closed'"
    print(f"  Open positions:        {q(open_sql) or 0:,}")
    print(f"  Closed positions:      {q(closed_sql) or 0:,}")
    print()
    print("PRE-REGISTRATION")
    if (HERE / "preregistration.json").exists():
        print("  Manifest:              LOCKED")
    else:
        print("  Manifest:              NOT LOCKED  (run `python run.py preregister lock`)")
    conn.close()
    return 0


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS COMMANDS
# ═══════════════════════════════════════════════════════════════════════

ANALYSERS_FAST = [
    ("formula", "formula_validator.py write"),
    ("narrative", "narrative_detector.py write"),
    ("obscurity", "obscurity_filter.py audit"),
    ("kill_topology", "kill_failure_mapper.py write"),
    ("adversarial", "adversarial_residual_classifier.py write"),
    ("phase", "phase_transition_detector.py write"),
    ("causal_edges", "chain_to_causal_edges.py write"),
    ("cycles", "cycle_detector.py run"),
    ("tam", "residual_tam.py write"),
]

ANALYSERS_SLOW = [
    ("reinforcement", "reinforcement_measurer.py write"),
    ("halflife", "halflife_estimator.py write"),
    ("telemetry", "backfill_telemetry.py write"),
]


def cmd_analyse(args):
    quick = args and args[0] == "quick"
    suite = ANALYSERS_FAST if quick else (ANALYSERS_FAST + ANALYSERS_SLOW)
    print(f"\nRunning {'fast' if quick else 'full'} analyser suite ({len(suite)} modules)\n")
    failed = []
    for name, cmd in suite:
        print(f"━━━ {name} " + "━" * (60 - len(name)))
        rc = _run(f"python {cmd}", check=False)
        if rc != 0:
            failed.append(name)
    print(f"\n{'━' * 70}")
    if failed:
        print(f"⚠ {len(failed)} analysers failed: {', '.join(failed)}")
        return 1
    print(f"✓ All {len(suite)} analysers complete. Refresh dashboard to view.")
    return 0


# Each individual analyser has its own small command for targeted runs
INDIVIDUAL_ANALYSERS = {
    "cycles":         "cycle_detector.py run",
    "tam":            "residual_tam.py write",
    "formula":        "formula_validator.py write",
    "reinforcement":  "reinforcement_measurer.py write",
    "halflife":       "halflife_estimator.py write",
    "narrative":      "narrative_detector.py write",
    "kill_topology":  "kill_failure_mapper.py write",
    "phase":          "phase_transition_detector.py write",
    "adversarial":    "adversarial_residual_classifier.py write",
    "obscurity":      "obscurity_filter.py audit",
}


def cmd_individual_analyser(name, args):
    cmd = INDIVIDUAL_ANALYSERS.get(name)
    if not cmd:
        print(f"Unknown analyser: {name}")
        return 1
    return _run(f"python {cmd}")


# ═══════════════════════════════════════════════════════════════════════
# BACKFILL / FIXUP
# ═══════════════════════════════════════════════════════════════════════

def cmd_backfill(args):
    print("\nRunning all safe backfills (no API calls).\n")
    _run("python chain_to_causal_edges.py write")
    _run("python backfill_telemetry.py write")
    print("\n(Note: backfill_embeddings.py requires sentence-transformers; run it separately if facts lack embeddings.)")
    return 0


def cmd_causal_edges(args):
    return _run("python chain_to_causal_edges.py write")


def cmd_embeddings(args):
    return _run("python backfill_embeddings.py")


# ═══════════════════════════════════════════════════════════════════════
# PRE-REGISTRATION
# ═══════════════════════════════════════════════════════════════════════

def cmd_preregister(args):
    sub = args[0] if args else "status"
    if sub == "lock":
        return _run("python preregister.py write")
    if sub == "check":
        return _run("python preregister.py verify")
    return _run("python preregister.py status")


# ═══════════════════════════════════════════════════════════════════════
# REPORTS
# ═══════════════════════════════════════════════════════════════════════

def cmd_report(args):
    """Generate the latest intelligence report PDF."""
    _check_api_key()
    return _run("python generate_report.py")


def cmd_pitch(args):
    path = HERE / "HUNTER_PITCH.md"
    if not path.exists():
        print("HUNTER_PITCH.md not found.")
        return 1
    pager = os.environ.get("PAGER", "less")
    return _run(f"{pager} {path}")


# ═══════════════════════════════════════════════════════════════════════
# META
# ═══════════════════════════════════════════════════════════════════════

def cmd_doctor(args):
    """Check environment, dependencies, DB health."""
    import shutil
    problems = []
    # API key
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key and not (HERE / ".env").exists():
        problems.append("No ANTHROPIC_API_KEY in env or .env file.")
    # Python deps
    required = ["anthropic", "streamlit", "pandas", "numpy", "yfinance",
                "dotenv", "sentence_transformers"]
    missing = []
    for mod in required:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        problems.append(f"Missing Python packages: {', '.join(missing)}")
    # DB
    if not DB_PATH.exists():
        problems.append("No hunter.db yet. Run `python run.py live` to create it.")
    else:
        size_mb = DB_PATH.stat().st_size / 1024 / 1024
        print(f"  Database size: {size_mb:.1f} MB")
    # Disk
    total, used, free = shutil.disk_usage(HERE)
    print(f"  Disk free:     {free / 1024 / 1024 / 1024:.1f} GB")

    if problems:
        print("\n⚠ Issues found:")
        for p in problems:
            print(f"   - {p}")
        return 1
    print("\n✓ All checks pass.")
    return 0


def cmd_clean(args):
    """Remove *.pyc, __pycache__, .log tmp files. Never touches DB."""
    import shutil
    removed = 0
    for pyc in HERE.rglob("__pycache__"):
        shutil.rmtree(pyc, ignore_errors=True)
        removed += 1
    for log in HERE.glob("*.log"):
        # Keep the real hunter.log, remove any backups
        if log.name == "hunter.log":
            continue
        log.unlink()
        removed += 1
    print(f"✓ Removed {removed} temporary items.")
    return 0


def cmd_help(args):
    print(__doc__)
    return 0


# ═══════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def _check_api_key():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key and not (HERE / ".env").exists():
        print("[!] ANTHROPIC_API_KEY not set. Either export it or put it in .env")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════
# DISPATCH
# ═══════════════════════════════════════════════════════════════════════

def cmd_decompose(args):
    """Decompose a published belief into testable assumptions."""
    _check_api_key()
    if not args:
        print("Usage: python run.py decompose '<belief text>' [asset] [source]")
        return 1
    import subprocess
    return subprocess.run(
        ["python", "belief_decomposer.py"] + args, cwd=HERE
    ).returncode


def cmd_inverse(args):
    """Run / simulate / audit inverse HUNTER."""
    sub = args[0] if args else "audit"
    if sub == "run":
        _check_api_key()
        return _run("python inverse_hunter.py run")
    if sub == "simulate":
        return _run("python inverse_hunter.py simulate")
    return _run("python inverse_hunter.py audit")


def cmd_diary(args):
    """Generate research diary entries (incremental / weekly / monthly / preprint)."""
    sub = args[0] if args else "incremental"
    if sub in ("weekly", "monthly", "preprint", "--all"):
        return _run(f"python research_diary.py {sub}")
    return _run("python research_diary.py")


def cmd_cycle(args):
    sub = args[0] if args else "plan"
    return _run(f"python cycle_portfolio.py {sub}")

def cmd_edge(args):
    sub = args[0] if args else "audit"
    if sub == "audit":
        return _run("python meta_hunter.py audit")
    return _run("python meta_hunter.py write")

def cmd_expire(args):
    sub = args[0] if args else "calendar"
    return _run(f"python expiration_tracker.py {sub}")

def cmd_gaps(args):
    sub = args[0] if args else "scan"
    return _run(f"python negative_inference.py {sub}")

def cmd_adversarial(args):
    sub = args[0] if args else "simulate"
    if sub == "sim" or sub == "simulate":
        return _run("python adversarial_self_test.py simulate")
    if sub == "run":
        _check_api_key()
        return _run("python adversarial_self_test.py run")
    return _run("python adversarial_self_test.py report")

def cmd_orders(args):
    sub = args[0] if args else "today"
    return _run(f"python signal_to_order.py {sub}")

def cmd_attribution(args):
    return _run("python performance_attribution.py")

def cmd_canon(args):
    return _run("python theory_canon_v2.py")

def cmd_frontier(args):
    sub = args[0] if args else "all"
    return _run(f"python frontier_hypotheses.py {sub}")

def cmd_research(args):
    sub = args[0] if args else "status"
    return _run(f"python research_space.py {sub}")

def cmd_moat(args):
    sub = args[0] if args else "status"
    return _run(f"python moat_tracker.py {sub}")

def cmd_coin(args):
    sub = args[0] if args else "status"
    return _run(f"python hunter_coin.py {sub}")

def cmd_self(args):
    sub = args[0] if args else "status"
    if sub == "apply":
        _check_api_key()
    return _run(f"python self_improve.py {sub}")


COMMANDS = {
    "live":          cmd_live,
    "live-no-theory": cmd_live_no_theory,
    "dashboard":     cmd_dashboard,
    "status":        cmd_status,
    "analyse":       cmd_analyse,
    "analyze":       cmd_analyse,   # alias
    "backfill":      cmd_backfill,
    "causal_edges":  cmd_causal_edges,
    "embeddings":    cmd_embeddings,
    "preregister":   cmd_preregister,
    "report":        cmd_report,
    "pitch":         cmd_pitch,
    "doctor":        cmd_doctor,
    "clean":         cmd_clean,
    "decompose":     cmd_decompose,
    "inverse":       cmd_inverse,
    "diary":         cmd_diary,
    "cycle":         cmd_cycle,
    "edge":          cmd_edge,
    "expire":        cmd_expire,
    "gaps":          cmd_gaps,
    "adversarial":   cmd_adversarial,
    "orders":        cmd_orders,
    "attribution":   cmd_attribution,
    "canon":         cmd_canon,
    "frontier":      cmd_frontier,
    "research":      cmd_research,
    "moat":          cmd_moat,
    "coin":          cmd_coin,
    "self":          cmd_self,
    "help":          cmd_help,
    "-h":            cmd_help,
    "--help":        cmd_help,
}


def main():
    if len(sys.argv) < 2:
        cmd_help(None)
        return 0
    cmd = sys.argv[1]
    rest = sys.argv[2:]
    if cmd in COMMANDS:
        return COMMANDS[cmd](rest)
    if cmd in INDIVIDUAL_ANALYSERS:
        return cmd_individual_analyser(cmd, rest)
    print(f"Unknown command: {cmd}\n")
    cmd_help(None)
    return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
