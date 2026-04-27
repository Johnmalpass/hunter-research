"""HUNTER Quant operator CLI.

Examples:

  # Ingestion
  python -m quant ingest edgar --tickers AAPL,MET
  python -m quant ingest fred --series DGS10,DGS2
  python -m quant ingest faers --drugs METFORMIN,ATORVASTATIN
  python -m quant ingest polygon --tickers AAPL,MET --from-date 2024-01-01
  python -m quant ingest tiingo --tickers AAPL,MET --days 7
  python -m quant ingest gdelt --queries "cmbs delinquency,office distress"

  # Query the unified store
  python -m quant query                                  # list assets
  python -m quant query --asset AAPL                     # list AAPL fields
  python -m quant query --asset DGS10 --field value --history 30
  python -m quant query --runs                           # adapter run history

  # System status
  python -m quant status

  # Synergy / compositional alpha analyses
  python -m quant synergy --demo                         # XOR/Mirror/Independent
  python -m quant synergy --synthetic                    # HUNTER-shaped synthetic demo
  python -m quant synergy --hunter-db hunter.db          # live db
  python -m quant synergy --hunter-db /path/to/zenodo.sqlite

  # Compiled mechanism backtest (Phase 3)
  python -m quant mechanisms list                        # registered theses
  python -m quant mechanisms check thesis_328            # required-data check
  python -m quant backtest thesis_328 --start 2018-01-01 --end 2025-04-27
  python -m quant backtest pharma_adverse_spike_metformin_pfe --start 2024-01-01

  # Macro regime detector
  python -m quant regime                                 # current regime
  python -m quant regime --asof 2020-03-23                # specific date
  python -m quant regime --history 2018-01-01 --step 30   # walk through time

  # LLM mechanism compiler (Phase 3 follow-up)
  python -m quant compile --dry-run --thesis-id 328 --title "..." --text "<thesis>"
  python -m quant compile --live    --thesis-id 328 --title "..." --text "<thesis>"

  # Inquiry queue (system asks; you answer daily)
  python -m quant inquiries list
  python -m quant inquiries answer 7 "exit"
  python -m quant inquiries dismiss 7

  # TRADER agent: one full cycle through every registered mechanism
  python -m quant trade cycle --nav 10000              # dry-run, prints decision
  python -m quant trade cycle --nav 10000 --live       # log signals to ledger

  # Compositional-depth analyses (three independent lenses on alpha quality)
  python -m quant analyze --demon-index <mechanism_id>     # thermodynamic bound
  python -m quant analyze --demon-index-all                # all mechanisms
  python -m quant analyze --k-score --thesis "..." --facts "f1|f2|f3"

Each adapter explains how to enable itself if a key is missing. Exit codes:
0 success, 1 runtime issue, 2 usage error.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure repo root on path and load .env when invoked as a script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv

    # override=True so values in .env take precedence over an empty/stale shell var
    load_dotenv(_ROOT / ".env", override=True)
except ImportError:
    pass


# ============================================================
# ingest
# ============================================================

def _csv(s: str | None) -> list[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def _ingest(args: argparse.Namespace) -> int:
    name = args.adapter

    if name == "edgar":
        from quant.data.adapters.edgar import EdgarAdapter

        a = EdgarAdapter(db_path=args.db)
        rows = a.run(
            tickers=_csv(args.tickers),
            ciks=_csv(args.ciks),
            count=args.count,
        )

    elif name == "fred":
        from quant.data.adapters.fred import FredAdapter

        try:
            a = FredAdapter(db_path=args.db)
        except ValueError as e:
            print(f"FRED skipped: {e}", file=sys.stderr)
            return 1
        kwargs = {"observation_start": args.from_date}
        if args.series:
            kwargs["series"] = _csv(args.series)
        rows = a.run(**kwargs)

    elif name == "faers":
        from quant.data.adapters.faers import FaersAdapter

        a = FaersAdapter(db_path=args.db)
        rows = a.run(drugs=_csv(args.drugs), date_from=args.from_date)

    elif name == "polygon":
        from quant.data.adapters.polygon import PolygonAdapter

        try:
            a = PolygonAdapter(db_path=args.db)
        except ValueError as e:
            print(f"Polygon skipped: {e}", file=sys.stderr)
            return 1
        rows = a.run(tickers=_csv(args.tickers), from_date=args.from_date)

    elif name == "tiingo":
        from quant.data.adapters.tiingo import TiingoAdapter

        try:
            a = TiingoAdapter(db_path=args.db)
        except ValueError as e:
            print(f"Tiingo skipped: {e}", file=sys.stderr)
            return 1
        rows = a.run(tickers=_csv(args.tickers), days_back=args.days)

    elif name == "gdelt":
        from quant.data.adapters.gdelt import GdeltAdapter

        a = GdeltAdapter(db_path=args.db)
        rows = a.run(queries=_csv(args.queries), days_back=args.days)

    else:
        print(f"Unknown adapter: {name}", file=sys.stderr)
        return 2

    print(f"{name}: {rows} rows ingested")
    return 0


# ============================================================
# query
# ============================================================

def _query(args: argparse.Namespace) -> int:
    from quant.data.base import MarketState

    with MarketState(args.db) as state:
        if args.runs:
            runs = state.adapter_runs(adapter=args.adapter, limit=args.limit)
            if not runs:
                print("  no runs recorded yet")
                return 0
            for r in runs:
                status = (
                    f"OK rows={r['rows_written']}"
                    if not r["error"]
                    else f"ERR {r['error'][:60]}"
                )
                print(f"  {r['adapter']:10s}  {r['started_at'][:19]}  {status}")
            return 0

        if args.asset:
            if args.field:
                if args.history:
                    start = datetime.now(timezone.utc) - timedelta(days=args.history)
                    points = state.history(args.asset, args.field, start)
                    for p in points:
                        print(
                            f"  {p.timestamp.isoformat()[:19]}  "
                            f"{p.field}  {p.value}  src={p.source}"
                        )
                    print(f"\n  {len(points)} points")
                else:
                    p = state.latest(args.asset, args.field)
                    if p:
                        print(
                            f"  {p.timestamp.isoformat()[:19]}  "
                            f"{p.field}  {p.value}  src={p.source}"
                        )
                    else:
                        print(f"  no data for {args.asset}/{args.field}")
            else:
                fields = state.fields(args.asset)
                print(f"  {args.asset}: {len(fields)} fields")
                for f in sorted(fields):
                    p = state.latest(args.asset, f)
                    if p is None:
                        continue
                    val_repr = str(p.value)
                    if len(val_repr) > 40:
                        val_repr = val_repr[:37] + "..."
                    print(
                        f"    {f}  {p.timestamp.isoformat()[:19]}  "
                        f"{val_repr}  src={p.source}"
                    )
            return 0

        # No --asset and no --runs: list assets
        assets = sorted(state.assets())
        print(f"  {len(assets)} distinct assets in store")
        for a in assets[:30]:
            fields = state.fields(a)
            print(f"    {a}  ({len(fields)} fields)")
        if len(assets) > 30:
            print(f"    ... and {len(assets) - 30} more")
        return 0


# ============================================================
# status
# ============================================================

def _status(args: argparse.Namespace) -> int:
    from quant.data.base import DEFAULT_DB, MarketState

    db_path = args.db or DEFAULT_DB
    print(f"Database: {db_path}")
    if not Path(db_path).exists():
        print("  (does not exist yet — run an ingestion first)")
        return 0

    with MarketState(db_path) as state:
        assets = state.assets()
        runs = state.adapter_runs(limit=50)
        sources = {r["adapter"] for r in runs}
        ok_runs = sum(1 for r in runs if not r["error"])
        err_runs = sum(1 for r in runs if r["error"])
        total_rows = sum(r["rows_written"] or 0 for r in runs)

    print(f"Assets:          {len(assets)} distinct")
    print(f"Adapters used:   {', '.join(sorted(sources)) or 'none'}")
    print(f"Recent runs:     {len(runs)}  ({ok_runs} ok, {err_runs} err)")
    print(f"Rows in window:  {total_rows}")
    return 0


# ============================================================
# synergy
# ============================================================

def _synergy(args: argparse.Namespace) -> int:
    if args.demo:
        from quant.research.synergy import _self_demo

        _self_demo()
        return 0

    if args.synthetic:
        from quant.research.hunter_bridge import cli_demo_on_synthetic

        result = cli_demo_on_synthetic()
        print(json.dumps(result, indent=2, default=str))
        return 0

    if args.regime_split_demo:
        from quant.research.regime_synergy import cli_demo_regime_split

        print(json.dumps(cli_demo_regime_split(), indent=2, default=str))
        return 0

    if args.hunter_db and args.regime_conditional:
        from quant.research.regime_synergy import (
            compute_regime_conditional_synergy,
        )

        result = compute_regime_conditional_synergy(args.hunter_db, quant_db=args.db)
        print(json.dumps(result, indent=2, default=str))
        return 0

    if args.hunter_db:
        from quant.research.hunter_bridge import compute_collision_synergy

        result = compute_collision_synergy(args.hunter_db)
        print(json.dumps(result, indent=2, default=str))
        return 0

    print(
        "Specify one of --demo, --synthetic, --regime-split-demo, "
        "--hunter-db PATH [--regime-conditional]",
        file=sys.stderr,
    )
    return 2


# ============================================================
# mechanisms / backtest
# ============================================================

def _mechanisms(args: argparse.Namespace) -> int:
    from quant.data.base import MarketState
    from quant.research.mechanism import (
        _import_all_mechanisms,
        get_mechanism,
        list_mechanisms,
    )

    _import_all_mechanisms()

    if args.action == "list":
        names = list_mechanisms()
        if not names:
            print("  (no mechanisms registered)")
            return 0
        for n in names:
            print(f"  {n}")
        return 0

    if args.action == "check":
        try:
            cls = get_mechanism(args.thesis_id)
        except KeyError as e:
            print(str(e), file=sys.stderr)
            return 1
        m = cls()
        with MarketState(args.db) as state:
            check = m.check_data(state)
        print(f"Mechanism: {m.thesis_id}  ({m.name})")
        print(f"Universe:  {', '.join(m.universe)}")
        print(f"Holding:   {m.holding_period_days} days")
        print("")
        print("Data requirements:")
        for req in m.requirements:
            key = f"{req.asset_id}/{req.field}"
            count = check.get(key, 0)
            marker = "ok " if count > 0 else "MISSING"
            line = f"  {marker}  {key}  ({count} rows)  via {req.suggested_adapter}"
            if count == 0 and req.note:
                line += f"\n            -> {req.note}"
            print(line)
        return 0

    print(f"Unknown mechanisms action: {args.action}", file=sys.stderr)
    return 2


def _backtest(args: argparse.Namespace) -> int:
    from datetime import datetime as _dt, timezone as _tz

    from quant.data.base import MarketState
    from quant.research.backtest import backtest_mechanism, format_result
    from quant.research.mechanism import _import_all_mechanisms, get_mechanism

    _import_all_mechanisms()

    try:
        cls = get_mechanism(args.thesis_id)
    except KeyError as e:
        print(str(e), file=sys.stderr)
        return 1

    mechanism = cls()
    start = _dt.fromisoformat(args.start).replace(tzinfo=_tz.utc)
    end = (
        _dt.fromisoformat(args.end).replace(tzinfo=_tz.utc)
        if args.end
        else _dt.now(_tz.utc)
    )

    with MarketState(args.db) as state:
        result = backtest_mechanism(
            mechanism,
            state,
            start_date=start,
            end_date=end,
            eval_freq_days=args.freq,
            cost_bps=args.cost_bps,
            slippage_bps=args.slippage_bps,
        )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        print(format_result(result, max_trades_shown=args.show_trades))
    return 0


# ============================================================
# regime
# ============================================================

def _regime(args: argparse.Namespace) -> int:
    from datetime import datetime as _dt, timezone as _tz

    from quant.data.base import MarketState
    from quant.research.regime import RegimeDetector
    from quant.research.regime_forecast import RegimeForecaster, cli_forecast

    asof_dt = (
        _dt.fromisoformat(args.asof).replace(tzinfo=_tz.utc)
        if args.asof
        else _dt.now(_tz.utc)
    )

    if args.fit_forecaster:
        with MarketState(args.db) as state:
            f = RegimeForecaster(state, db_path=args.db)
            M = f.fit(lookback_years=args.lookback_years)
            print(f"Fitted daily transition matrix from {args.lookback_years}y FRED history.")
            for i, r in enumerate(["risk_on", "late_cycle", "risk_off", "crisis"]):
                row = "  ".join(f"{x:.4f}" for x in M[i])
                print(f"  {r:<11} -> {row}")
            stationary = f.stationary_distribution()
            print()
            print("Long-run stationary distribution:")
            for r, p in sorted(stationary.items(), key=lambda x: -x[1]):
                print(f"  {r:<11}  {p:.2%}")
            tau = f.characteristic_persistence_days()
            print()
            print(f"Characteristic persistence: {tau:.1f} days")
        return 0

    if args.forecast is not None:
        result = cli_forecast(
            db_path=args.db,
            horizons=(1, 7, 30, 60, 90, 180, 365)
            if args.forecast == 0
            else (int(args.forecast),),
            asof=asof_dt,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0

    with MarketState(args.db) as state:
        det = RegimeDetector(state)

        if args.history:
            start = _dt.fromisoformat(args.history).replace(tzinfo=_tz.utc)
            end = (
                _dt.fromisoformat(args.history_end).replace(tzinfo=_tz.utc)
                if args.history_end
                else asof_dt
            )
            states = det.detect_history(start, end, step_days=args.step)
            if not states:
                print("  no regime states (insufficient FRED data?)", file=sys.stderr)
                return 1
            print(
                f"{'date':<12}  {'regime':<11}  "
                f"{'risk_on':>8}  {'late':>8}  {'risk_off':>8}  {'crisis':>8}  "
                f"{'YC slope':>8}  {'HY OAS':>8}"
            )
            for rs in states:
                p = rs.probabilities
                yc = rs.inputs["yield_curve_slope"]
                hy = rs.inputs["high_yield_spread"]
                print(
                    f"{rs.asof.date()}  {rs.regime:<11}  "
                    f"{p['risk_on']:>8.2%}  {p['late_cycle']:>8.2%}  "
                    f"{p['risk_off']:>8.2%}  {p['crisis']:>8.2%}  "
                    f"{yc:>+8.2f}  {hy:>8.2f}"
                )
            return 0

        rs = det.detect(asof_dt)
        if rs is None:
            print(
                "  no regime state — ingest FRED first:\n"
                "    python -m quant ingest fred",
                file=sys.stderr,
            )
            return 1
        if args.json:
            print(json.dumps(rs.to_dict(), indent=2, default=str))
        else:
            print(f"As of: {rs.asof.date()}")
            print(f"Regime: {rs.regime}")
            print("Probabilities:")
            for r, p in sorted(rs.probabilities.items(), key=lambda x: -x[1]):
                bar = "#" * int(p * 40)
                print(f"  {r:<11}  {p:>6.2%}  {bar}")
            print("Inputs:")
            for k, v in rs.inputs.items():
                if "pct" in k:
                    print(f"  {k:<28}  {v:>6.2%}")
                else:
                    print(f"  {k:<28}  {v:>+7.2f}")
        return 0


# ============================================================
# doctor
# ============================================================

def _doctor(args: argparse.Namespace) -> int:
    """One-shot system-health check: keys, db, mechanisms, regime, inquiries."""
    import os
    from datetime import datetime as _dt, timezone as _tz
    from pathlib import Path

    from quant.agents.inquiry import list_open_inquiries
    from quant.data.base import DEFAULT_DB, MarketState
    from quant.research.mechanism import (
        _import_all_mechanisms,
        list_mechanisms,
    )
    from quant.research.regime import RegimeDetector

    print("HUNTER Quant System Health Check")
    print("=" * 50)

    # API keys
    print("\n[API keys]")
    keys = {
        "ANTHROPIC_API_KEY": "(needed for compile + agents)",
        "FRED_API_KEY": "(needed for FRED ingestion)",
        "POLYGON_API_KEY": "(optional; equity prices)",
        "TIINGO_API_KEY": "(optional; news + sentiment)",
        "OPENFDA_API_KEY": "(optional; FAERS speed-up)",
    }
    for k, note in keys.items():
        v = os.environ.get(k)
        print(f"  {k:<25}  {'set' if v else 'NOT SET':<8}  {note}")

    # Data store
    print("\n[Quant data store]")
    db_path = Path(args.db) if args.db else DEFAULT_DB
    if not db_path.exists():
        print(f"  database: {db_path} (does not exist yet)")
    else:
        with MarketState(db_path) as state:
            assets = state.assets()
            runs = state.adapter_runs(limit=10)
            sources = sorted({r["adapter"] for r in runs}) if runs else []
            print(f"  database:    {db_path}")
            print(f"  assets:      {len(assets)} distinct")
            print(f"  adapters used: {', '.join(sources) or 'none yet'}")
            print(f"  recent runs: {len(runs)}")

    # Mechanisms
    print("\n[Compiled mechanisms]")
    _import_all_mechanisms()
    names = list_mechanisms()
    if not names:
        print("  no mechanisms registered")
    else:
        for n in names:
            print(f"  registered: {n}")

    # Inquiries
    print("\n[Inquiries]")
    open_inq = list_open_inquiries(db_path=args.db)
    if not open_inq:
        print("  no open inquiries")
    else:
        print(f"  {len(open_inq)} open:")
        for i in open_inq[:5]:
            print(f"    #{i.id} [{i.urgency}] {i.body[:60]}")
        if len(open_inq) > 5:
            print(f"    ... and {len(open_inq) - 5} more")

    # Regime
    print("\n[Macro regime]")
    if not db_path.exists():
        print("  no data; ingest FRED first")
    else:
        with MarketState(db_path) as state:
            det = RegimeDetector(state)
            rs = det.detect(_dt.now(_tz.utc))
            if rs is None:
                print("  no regime data; ingest FRED first")
            else:
                top = sorted(rs.probabilities.items(), key=lambda x: -x[1])
                top_three = ", ".join(
                    f"{r}={p:.0%}" for r, p in top[:3]
                )
                print(f"  current: {rs.regime} ({top_three})")
                print(
                    f"  YC slope: {rs.inputs['yield_curve_slope']:+.2f}; "
                    f"BAA10Y: {rs.inputs['high_yield_spread']:.2f}"
                )

    print("\n" + "=" * 50)
    return 0


# ============================================================
# compile
# ============================================================

def _compile(args: argparse.Namespace) -> int:
    from quant.research.compile import compile_thesis

    res = compile_thesis(
        thesis_id=args.thesis_id,
        thesis_title=args.title or args.thesis_id,
        thesis_text=args.text,
        silos=[s.strip() for s in (args.silos or "").split(",") if s.strip()],
        thesis_score=args.score,
        dry_run=not args.live,
        overwrite=args.overwrite,
    )
    if res.status == "dry_run":
        print(res.prompt)
        print("\n[dry-run] no API call made. Pass --live to call Opus.", file=sys.stderr)
        return 0
    if res.status == "ok":
        print(f"compiled mechanism written to: {res.written_to}")
        print(
            f"input tokens: {res.input_tokens}  "
            f"output tokens: {res.output_tokens}  "
            f"cost: ${res.cost_usd:.4f}"
        )
        return 0
    if res.status == "validation_failed":
        print(f"validation failed: {res.validation_message}", file=sys.stderr)
        if res.code:
            print("---- generated code ----", file=sys.stderr)
            print(res.code, file=sys.stderr)
        return 1
    if res.status == "api_error":
        print(f"API error: {res.error}", file=sys.stderr)
        return 1
    print(f"unknown status: {res.status}", file=sys.stderr)
    return 1


# ============================================================
# inquiries
# ============================================================

def _inquiries(args: argparse.Namespace) -> int:
    from quant.agents.inquiry import (
        answer_inquiry,
        dismiss_inquiry,
        list_open_inquiries,
        open_inquiry,
    )

    if args.action == "list":
        items = list_open_inquiries(db_path=args.db)
        if not items:
            print("  (no open inquiries)")
            return 0
        for i in items:
            print(
                f"  #{i.id:<4} [{i.urgency:<8}] [{i.inquiry_type:<10}] "
                f"{i.created_at.isoformat()[:19]}"
            )
            print(f"        {i.body}")
            if i.options:
                print(f"        options: {', '.join(i.options)}")
            if i.related_files:
                print(f"        files:   {i.related_files}")
            print()
        return 0

    if args.action == "answer":
        answer_inquiry(args.id, args.text, db_path=args.db)
        print(f"  answered #{args.id}")
        return 0

    if args.action == "dismiss":
        dismiss_inquiry(args.id, db_path=args.db)
        print(f"  dismissed #{args.id}")
        return 0

    if args.action == "open":
        iid = open_inquiry(
            inquiry_type=args.type,
            body=args.body,
            urgency=args.urgency,
            db_path=args.db,
        )
        print(f"  opened #{iid}")
        return 0

    print(f"unknown action: {args.action}", file=sys.stderr)
    return 2


# ============================================================
# analyze (compositional depth lenses)
# ============================================================

def _analyze(args: argparse.Namespace) -> int:
    if args.demon_index:
        from quant.research.demon_index import compute_demon_index

        di = compute_demon_index(args.demon_index, db_path=args.db)
        print(json.dumps(di.to_dict(), indent=2, default=str))
        return 0

    if args.demon_index_all:
        from quant.research.demon_index import compute_demon_index_all

        results = compute_demon_index_all(db_path=args.db)
        if not results:
            print("  no mechanisms with completed trades found", file=sys.stderr)
            return 1
        out = [r.to_dict() for r in results]
        print(json.dumps(out, indent=2, default=str))
        return 0

    if args.k_score:
        from quant.research.k_score import k_score

        if not args.thesis:
            print("--thesis is required with --k-score", file=sys.stderr)
            return 2
        facts = [f.strip() for f in (args.facts or "").split("|") if f.strip()]
        result = k_score(args.thesis, facts)
        print(json.dumps(result.to_dict(), indent=2, default=str))
        return 0

    if args.translate:
        from quant.research.audience_translator import (
            DEFAULT_PROFILES,
            translate_for_all_audiences,
            translate_for_audience,
        )

        if not args.thesis:
            print("--thesis is required with --translate", file=sys.stderr)
            return 2
        if args.audience and args.audience != "all":
            t = translate_for_audience(
                args.thesis,
                args.audience,
                thesis_id=args.thesis_id or "untitled",
                dry_run=not args.live,
            )
            print(json.dumps(t.to_dict(), indent=2, default=str))
        else:
            out = translate_for_all_audiences(
                args.thesis,
                thesis_id=args.thesis_id or "untitled",
                dry_run=not args.live,
            )
            print(json.dumps({k: v.to_dict() for k, v in out.items()}, indent=2, default=str))
        return 0

    if args.dialect_kl:
        from quant.research.dialect_kl import compute_silo_kl_matrix

        # Read demo statements from --statements arg as 'silo:stmt|silo:stmt|...'
        if not args.statements:
            print(
                "--statements required: 'silo:stmt|silo:stmt|...'",
                file=sys.stderr,
            )
            return 2
        by_silo: dict[str, list[str]] = {}
        for chunk in args.statements.split("|"):
            if ":" not in chunk:
                continue
            silo, stmt = chunk.split(":", 1)
            by_silo.setdefault(silo.strip(), []).append(stmt.strip())
        matrix = compute_silo_kl_matrix(by_silo)
        out = matrix.to_dict()
        out["top_pairs"] = matrix.top_pairs(k=10)
        print(json.dumps(out, indent=2, default=str))
        return 0

    if args.articulation_lead:
        from datetime import datetime as _dt, timezone as _tz

        from quant.research.articulation_lead import compute_articulation_lead

        keywords = [k.strip() for k in (args.keywords or "").split(",") if k.strip()]
        if not keywords:
            print("--keywords required with --articulation-lead", file=sys.stderr)
            return 2
        asof = (
            _dt.fromisoformat(args.asof).replace(tzinfo=_tz.utc)
            if args.asof
            else None
        )
        record = compute_articulation_lead(
            args.thesis_id or "untitled",
            thesis_keywords=keywords,
            hunter_articulated_at=asof,
            db_path=args.db,
        )
        print(json.dumps(record.to_dict(), indent=2, default=str))
        return 0

    print(
        "Specify one of: --demon-index, --demon-index-all, --k-score, "
        "--translate, --dialect-kl, --articulation-lead",
        file=sys.stderr,
    )
    return 2


# ============================================================
# trade
# ============================================================

def _trade(args: argparse.Namespace) -> int:
    from datetime import datetime as _dt, timezone as _tz

    from quant.agents.trader import run_cycle

    if args.action == "cycle":
        result = run_cycle(
            nav=args.nav,
            asof=(
                _dt.fromisoformat(args.asof).replace(tzinfo=_tz.utc)
                if args.asof
                else None
            ),
            db_path=args.db,
            dry_run=not args.live,
        )
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, default=str))
        else:
            r = result
            print(f"\nTRADER cycle  {r.asof.isoformat()[:19]}")
            print(f"  NAV:                {r.nav:,.0f}")
            print(f"  Regime:             {r.regime or 'unknown'}")
            if r.regime_probabilities:
                top = sorted(
                    r.regime_probabilities.items(), key=lambda x: -x[1]
                )[:3]
                pretty = ", ".join(f"{k}={v:.0%}" for k, v in top)
                print(f"    probabilities:    {pretty}")
            print(f"  Mechanisms evaluated: {r.n_mechanisms_evaluated}")
            print(f"  Signals emitted:      {r.n_signals_emitted}")
            print(f"  Coalition votes:      {r.n_coalition_votes}")
            print(f"  Orders proposed:      {r.n_orders_proposed}")
            print(f"    approved:           {r.n_approved}")
            print(f"    vetoed:             {r.n_vetoed}")
            print(f"    inquiries opened:   {r.n_inquiries_opened}")
            print()
            for step in r.steps:
                msg = (
                    f"  evaluated {step.mechanism_id:<30} "
                    f"signals={step.n_signals}  {step.elapsed_ms}ms"
                )
                if step.error:
                    msg += f"  ERROR: {step.error}"
                print(msg)
            if r.orders:
                print()
                print("  Orders:")
                for o in r.orders:
                    line = (
                        f"    {o.verdict.value:<11} "
                        f"{o.order.direction:<5} "
                        f"{o.order.asset:<8} "
                        f"size={o.order.size_pct_of_nav:.2%}  "
                        f"conf={o.order.confidence:.2f}  "
                        f"mechs={','.join(o.order.contributing_mechanisms)}"
                    )
                    print(line)
                    if o.verdict_reason:
                        print(f"        reason: {o.verdict_reason}")
            print()
            print(f"  {r.rationale}")
        return 0

    print(f"unknown trade action: {args.action}", file=sys.stderr)
    return 2


# ============================================================
# main
# ============================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="quant", description="HUNTER Quant CLI")
    p.add_argument("--db", help="Override quant data store path")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ingest <adapter>
    ing = sub.add_parser("ingest", help="Run an ingestion adapter")
    ing_sub = ing.add_subparsers(dest="adapter", required=True)

    edgar = ing_sub.add_parser("edgar", help="SEC EDGAR filings index")
    edgar.add_argument("--tickers", help="Comma-separated tickers")
    edgar.add_argument("--ciks", help="Comma-separated CIKs")
    edgar.add_argument("--count", type=int, default=40)

    fred = ing_sub.add_parser("fred", help="FRED macro time series")
    fred.add_argument("--series", help="Comma-separated FRED series IDs")
    fred.add_argument("--from-date", default="2010-01-01", dest="from_date")

    faers = ing_sub.add_parser("faers", help="FDA adverse event counts")
    faers.add_argument("--drugs", required=True, help="Comma-separated drug names")
    faers.add_argument("--from-date", default="2020-01-01", dest="from_date")

    polygon = ing_sub.add_parser("polygon", help="Polygon.io equity bars")
    polygon.add_argument("--tickers", required=True)
    polygon.add_argument("--from-date", default="2020-01-01", dest="from_date")

    tiingo = ing_sub.add_parser("tiingo", help="Tiingo news + sentiment")
    tiingo.add_argument("--tickers", required=True)
    tiingo.add_argument("--days", type=int, default=7)

    gdelt = ing_sub.add_parser("gdelt", help="GDELT global news events")
    gdelt.add_argument("--queries", required=True, help="Comma-separated queries")
    gdelt.add_argument("--days", type=int, default=7)

    # query
    q = sub.add_parser("query", help="Query the unified data store")
    q.add_argument("--asset", help="Asset id to inspect")
    q.add_argument("--field", help="Field to query")
    q.add_argument("--history", type=int, help="Show N days of history")
    q.add_argument("--runs", action="store_true", help="Show recent adapter runs")
    q.add_argument("--adapter", help="Filter --runs by adapter name")
    q.add_argument("--limit", type=int, default=20)

    # status
    sub.add_parser("status", help="One-screen state of store")

    # doctor
    sub.add_parser("doctor", help="System health check (keys, db, mechanisms, regime, inquiries)")

    # synergy
    syn = sub.add_parser("synergy", help="Synergy estimator analyses")
    syn.add_argument("--demo", action="store_true", help="XOR/Mirror/Independent")
    syn.add_argument(
        "--synthetic",
        action="store_true",
        help="Synthetic HUNTER-shaped (synergistic / redundant / independent) demo",
    )
    syn.add_argument(
        "--regime-split-demo",
        action="store_true",
        dest="regime_split_demo",
        help="Synthetic demo: pooled II hides regime-conditional synergy",
    )
    syn.add_argument(
        "--hunter-db",
        help="Path to HUNTER corpus sqlite (live or Zenodo) for II analysis",
    )
    syn.add_argument(
        "--regime-conditional",
        action="store_true",
        dest="regime_conditional",
        help="Combine with --hunter-db to compute II per macro regime",
    )

    # mechanisms
    mech = sub.add_parser("mechanisms", help="List or inspect compiled HUNTER theses")
    mech_sub = mech.add_subparsers(dest="action", required=True)
    mech_sub.add_parser("list", help="List registered mechanism IDs")
    chk = mech_sub.add_parser("check", help="Show data requirements for a mechanism")
    chk.add_argument("thesis_id", help="ID such as thesis_328")

    # backtest
    bt = sub.add_parser("backtest", help="Run a compiled mechanism over history")
    bt.add_argument("thesis_id", help="Mechanism ID (see `mechanisms list`)")
    bt.add_argument("--start", required=True, help="YYYY-MM-DD")
    bt.add_argument("--end", help="YYYY-MM-DD (default: today)")
    bt.add_argument("--freq", type=int, default=7, help="Evaluation cadence in days")
    bt.add_argument("--cost-bps", type=float, default=5.0, dest="cost_bps")
    bt.add_argument("--slippage-bps", type=float, default=5.0, dest="slippage_bps")
    bt.add_argument("--show-trades", type=int, default=10, dest="show_trades")
    bt.add_argument("--json", action="store_true", help="Emit JSON instead of table")

    # regime
    rg = sub.add_parser("regime", help="Macro regime detector + forecaster")
    rg.add_argument("--asof", help="YYYY-MM-DD (default: today)")
    rg.add_argument("--history", help="Walk regime over time from YYYY-MM-DD")
    rg.add_argument("--history-end", dest="history_end")
    rg.add_argument("--step", type=int, default=14)
    rg.add_argument("--json", action="store_true")
    rg.add_argument(
        "--forecast", type=int, default=None,
        help="Horizon days (e.g. 30); pass 0 for the standard 1/7/30/60/90/180/365 set",
    )
    rg.add_argument(
        "--fit-forecaster", action="store_true", dest="fit_forecaster",
        help="Refit the Markov transition matrix from FRED history",
    )
    rg.add_argument(
        "--lookback-years", type=int, default=10, dest="lookback_years",
    )

    # compile
    cmp = sub.add_parser("compile", help="LLM-compile a HUNTER thesis into a Mechanism")
    cmp.add_argument("--thesis-id", required=True, dest="thesis_id")
    cmp.add_argument("--title", help="Human-readable title")
    cmp.add_argument("--text", required=True, help="Thesis description")
    cmp.add_argument("--silos", help="Comma-separated silos for context")
    cmp.add_argument("--score", type=float, help="Diamond score if known")
    cmp.add_argument("--live", action="store_true",
                     help="Actually call Opus (default: dry-run prints prompt)")
    cmp.add_argument("--overwrite", action="store_true",
                     help="Replace existing mechanism file")

    # analyze
    az = sub.add_parser("analyze", help="Compositional-depth analyses (Demon Index, K-Score)")
    az.add_argument("--demon-index", dest="demon_index", help="mechanism_id to analyze")
    az.add_argument("--demon-index-all", action="store_true", dest="demon_index_all",
                    help="Demon index for every mechanism with closed trades")
    az.add_argument("--k-score", action="store_true", dest="k_score",
                    help="K-score for a thesis (requires --thesis)")
    az.add_argument("--thesis", help="Thesis text for K-score / translation")
    az.add_argument("--facts", help="Pipe-separated constituent facts for K-score")
    az.add_argument("--translate", action="store_true",
                    help="Audience-translate a thesis (use --audience or 'all')")
    az.add_argument("--audience",
                    choices=["substack", "ssrn", "sell_side", "treasury", "twitter", "all"],
                    default="all", help="Audience for --translate")
    az.add_argument("--thesis-id", dest="thesis_id", help="Thesis identifier")
    az.add_argument("--live", action="store_true",
                    help="Make actual API calls for --translate (default: dry-run)")
    az.add_argument("--dialect-kl", action="store_true", dest="dialect_kl",
                    help="Compute Jensen-Shannon dialect KL matrix from --statements")
    az.add_argument("--statements", help="silo:stmt|silo:stmt|... for --dialect-kl")
    az.add_argument("--articulation-lead", action="store_true", dest="articulation_lead",
                    help="Articulation lead time for thesis (requires --thesis-id, --keywords)")
    az.add_argument("--keywords", help="Comma-separated keywords for --articulation-lead")
    az.add_argument("--asof", help="YYYY-MM-DD HUNTER articulation date")

    # trade
    tr = sub.add_parser("trade", help="TRADER agent (run a full cycle)")
    tr_sub = tr.add_subparsers(dest="action", required=True)
    cycle = tr_sub.add_parser("cycle", help="Run one TRADER cycle end-to-end")
    cycle.add_argument("--nav", type=float, required=True, help="Account NAV")
    cycle.add_argument("--asof", help="YYYY-MM-DD (default: now)")
    cycle.add_argument(
        "--live", action="store_true",
        help="Actually log signals + open inquiries (default: dry-run)",
    )
    cycle.add_argument("--json", action="store_true", help="Emit JSON")

    # inquiries
    inq = sub.add_parser("inquiries", help="System inquiry queue (operator feedback)")
    inq_sub = inq.add_subparsers(dest="action", required=True)
    inq_sub.add_parser("list", help="Show open inquiries")
    a = inq_sub.add_parser("answer", help="Answer an inquiry")
    a.add_argument("id", type=int)
    a.add_argument("text", help="Your answer")
    d = inq_sub.add_parser("dismiss", help="Dismiss an inquiry without answering")
    d.add_argument("id", type=int)
    o = inq_sub.add_parser("open", help="Manually open an inquiry (debug/testing)")
    o.add_argument("--type", required=True,
                   choices=["decision", "data", "validation", "review"])
    o.add_argument("--body", required=True)
    o.add_argument("--urgency", default="medium",
                   choices=["low", "medium", "high", "critical"])

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "ingest":
        return _ingest(args)
    if args.cmd == "query":
        return _query(args)
    if args.cmd == "status":
        return _status(args)
    if args.cmd == "doctor":
        return _doctor(args)
    if args.cmd == "synergy":
        return _synergy(args)
    if args.cmd == "mechanisms":
        return _mechanisms(args)
    if args.cmd == "backtest":
        return _backtest(args)
    if args.cmd == "regime":
        return _regime(args)
    if args.cmd == "compile":
        return _compile(args)
    if args.cmd == "inquiries":
        return _inquiries(args)
    if args.cmd == "trade":
        return _trade(args)
    if args.cmd == "analyze":
        return _analyze(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
