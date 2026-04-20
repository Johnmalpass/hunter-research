# HUNTER

**An autonomous research instrument for cross‑silo financial inference, with a pre‑registered 12‑week empirical study and a public prediction ledger.**

![status: early](https://img.shields.io/badge/status-early%20research-orange) ![license: MIT](https://img.shields.io/badge/code-MIT-blue) ![corpus: CC-BY-4.0](https://img.shields.io/badge/corpus-CC--BY--4.0-green) ![preregistered](https://img.shields.io/badge/preregistered-2026--04--19-blueviolet)

---

## What this is

HUNTER is a continuously‑running Python pipeline that ingests specific, dated facts from eighteen professional financial silos — patent filings, SEC disclosures, NAIC insurance reserves, OSHA enforcement, CMBS delinquency and surveillance, Federal Register rule changes, commodity inventories, analyst targets, academic preprints, pharmaceutical approvals, distressed credit, healthcare REIT filings, energy‑infrastructure filings, specialty real estate, government contracts, earnings transcripts, job‑listing signals, and app‑ranking signals — cross‑matches them across seven parallel strategies, forms compositional hypotheses, subjects them to a mechanism‑verified adversarial kill gauntlet, and posts survivors to a public prediction board with named assets, directions, and resolution dates.

It is also the measurement platform for a pre‑registered empirical study of *compositional alpha* — cross‑silo information asymmetries that no single specialist captures — running out‑of‑sample from June through August 2026.

## Status

**Early research.** The pipeline is built, tested, and operational. The corpus is frozen. The prediction board is live and empty by design. The twelve‑week pre‑registered summer 2026 study is the first full, out‑of‑sample run under the upgraded three‑tier model routing (Opus 4.7 for mechanism kill and adversarial review, Sonnet 4.5 for standard extraction, Haiku 4.5 for high‑volume ingestion). Zero predictions have resolved as of launch; the ledger fills beginning June 1, 2026, with first resolutions in mid‑July.

Pre‑freeze empirical observations (the mechanism/audience kill asymmetry; the hub‑and‑spoke methodology graph; the bimodal diamond‑score distribution; the negative narrative‑survival correlation; the nine Tarjan cycles) are held back as *hypotheses to be tested*, not findings, until summer replication completes under the frozen pre‑registration.

## Provenance

This repository is the public release of a six‑month private solo build (November 2025 – April 2026). The git history shows a single initial commit because the artifact audit trail was never intended to live in git: it lives in the SHA‑256‑locked pre‑registration manifest (`preregistration.json`, code hash `f39d2f5ff6b3e695`, locked 2026‑04‑19), the timestamped frozen corpus on Zenodo (DOI pending), and the public, timestamped prediction board. Anyone who wants to audit a claim runs `python run.py preregister check` against the manifest; independent replication runs against the frozen Zenodo corpus and the locked code hash, not against the commit history. Papers 5 and 6 in `/papers/` are framing‑priority drafts whose central empirical claims are pending summer 2026 replication — see the banner at the top of each file before citing.

## A note on the operator

This project is built and run by a single person: John Malpass, second‑year BSc Economics at University College Dublin. That is relevant context for how the work should be read. The repository is public for a reason — priority of discovery is claimed at the moment of posting, and honest public critique is worth more to the project than private reassurance. Prior‑art pointers, design criticism, and replication attempts are all welcome. Contact details are below.

## Key artifacts

- **Corpus (Zenodo, CC‑BY‑4.0)** — 12,030 facts across 18 silos, 77 countries, 30,967 normalised entity‑index entries, 11,835 distinct entities, 6,670 model‑field extractions, 474 cross‑silo collisions, 171 directed causal edges with named transmission pathways, 52 multi‑link chains, 1,570 detected anomalies, 61 formed hypotheses with completed adversarial review, and 1,155 theory‑evidence records across 13 framework layers. DOI: *pending Zenodo reserve*.
- **Methods paper (Paper 0, SSRN)** — Instrument description, pipeline stages, model‑field extraction, implication matching, differential edge, and kill‑phase design. *Submission pending April 2026.*
- **Theoretical and empirical drafts (SSRN, rolling)** — drafts on the mechanism‑assembly bottleneck (Paper 2), the non‑zero compositional residual (Paper 3), and the cross‑silo composition test (Paper 4). Empirical claims in the framework papers are presented as pre‑registered hypotheses until summer replication completes.
- **Prediction board** — public, timestamped, resolvable. URL: `https://johnjosephmalpass-droid.github.io/hunter-research/`
- **Methodology brief (PDF, 2 pages)** — free, publicly downloadable, linked on the prediction board.
- **Pre‑registration manifest** — `preregistration.json`, SHA‑256 locked at hash `f39d2f5ff6b3e695`, corpus frozen 2024‑12‑31, code hash locked 2026‑04‑19.

## Quick start

```bash
git clone https://github.com/johnjosephmalpass-droid/hunter-research.git
cd hunter-research
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your API keys; see below for tiers

# dashboard (no API required)
python run.py dashboard

# one-screen state of the whole system (no API)
python run.py status

# full pipeline (requires API budget; honours the preregistration freeze)
python run.py live

# verify no drift against the preregistration manifest
python run.py preregister check

# run all analyser modules against the current corpus (no API)
python run.py analyse
```

A three‑tier model routing configuration is used: Opus 4.7 for critical reasoning (mechanism kill, adversarial review), Sonnet 4.5 for standard extraction, and Haiku 4.5 for high‑volume ingestion. Full budgets and cycle throttles are documented in `config.py`.

## Pipeline architecture

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  INGEST  │──▶│  EXTRACT │──▶│  DETECT  │──▶│  COLLIDE │
│ 18 silos │   │  fact +  │   │ anomalies│   │ 7 strat. │
│ 220 q.   │   │ implica. │   │ weirdness│   │ matching │
└──────────┘   └──────────┘   └──────────┘   └──────────┘
                                                   │
                                                   ▼
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  LEDGER  │◀──│  SCORE   │◀──│   KILL   │◀──│  FORM    │
│ public   │   │ 6 dims + │   │ mech +   │   │ hypo. +  │
│ board    │   │ anchors  │   │ fact +   │   │ resol'n  │
│          │   │ (fresh   │   │ compet + │   │ date     │
│          │   │ context) │   │ barrier  │   │          │
└──────────┘   └──────────┘   └──────────┘   └──────────┘
```

Each stage is defined in `prompts.py` (26 LLM prompts, one per pipeline step) and routed through `config.py` (18 source types, 153‑pair hand‑calibrated domain distance matrix, 220 ingest queries). The causal graph, model‑field extractions, and adversarial review traces are all persisted to the 43 database tables defined in the schema.

The seven collision strategies run in parallel per anomaly: implication matching, entity matching, keyword matching, model‑field matching, causal‑graph traversal, embedding similarity, and belief‑reality contradiction. Matches are blended, evaluated by an LLM, gated against prior publication, and promoted to hypotheses with resolution dates. The kill gauntlet then runs four adversarial rounds (mechanism, fact‑check, competitor, barrier) and a final market‑check; surviving hypotheses are scored in a fresh context by an adversarial reviewer against four calibration anchors.

## Module inventory

**Ingestion & extraction**: `hunter.py`, `fact_extractor.py` (inline), `prompts.py`.

**Matching & collision**: `hunter.py` (CollisionCycle), seven parallel strategies — implication bigrams, entity match, keyword match, model‑field match, causal‑path BFS, embedding similarity, belief‑reality contradiction.

**Adversarial kill phase**: kill rounds embedded in `hunter.py` with `formula_validator.py`, `kill_failure_mapper.py`, and `financial_mechanics_check` within the scoring module.

**Hypothesis formation, scoring, ledger**: `hunter.py`, `prediction_board.py`, `portfolio.py`, `portfolio_feedback.py`.

**Analysis & graph**: `cycle_detector.py` (Tarjan SCC), `cycle_chain_detector.py`, `chain_to_causal_edges.py`, `narrative_detector.py`, `obscurity_filter.py`, `halflife_estimator.py`, `reinforcement_measurer.py`, `phase_transition_detector.py`, `adversarial_residual_classifier.py`, `thesis_dedup.py`, `chain_decay_fitter.py`, `residual_tam.py`.

**Self‑improvement**: `adversarial_self_test.py`, `self_improve.py`, `moat_tracker.py`, `reinforcement_measurer.py`, `meta_hunter.py`, `inverse_hunter.py`, `frontier_hypotheses.py`, `belief_decomposer.py`.

**Study infrastructure**: `preregister.py`, `orchestrator.py`, `calibration.py`, `theory_layer.py`, `theory.py`, `theory_canon_v2.py`.

**Dashboards**: `master_dashboard.py` (unified five‑tab Streamlit dashboard: overview, findings, knowledge, theory, ops), `hunter_dashboard.py` (legacy), `theory_dashboard.py` (legacy), `public_dashboard.py`.

**Reports and artifacts**: `generate_report.py`, `enrich_thesis.py`, `build_story_pdf.py`, `targeting.py`.

Total: ~32,000 lines of Python across 60+ modules, 43 DB tables, 26 LLM prompts.

## Pre‑registered summer 2026 study

A twelve‑week out‑of‑sample study runs June 1 – August 31, 2026, on the frozen corpus. The pre‑registration manifest is locked at SHA‑256 hash `f39d2f5ff6b3e695`.

**Primary hypothesis.** Median portfolio alpha over SPY total return increases monotonically across strata defined by the number of distinct professional silos in each hypothesis: A ≤ B ≤ C ≤ D, with D − A > 0 at p < 0.05 under a 10,000‑resample paired bootstrap. Strata definitions are fixed in `config.py` and `preregistration.json`.

**Secondary hypotheses.**
- **H2 (cycle persistence).** Detected cycles with reinforcement ≥ 0.5 persist uncorrected in the market for ≥ 14 days in ≥ 2 of the 9 currently detected cycles.
- **H3 (domain distance).** Cross‑silo collisions (domain distance ≥ 0.60) produce higher adjusted scores than within‑silo matches (< 0.30) by ≥ 10 points on average.
- **H4 (chain depth).** Chain‑depth‑3 hypotheses outperform chain‑depth‑1 hypotheses in realised alpha with Cohen's d ≥ 0.3.

**Null baselines (pre‑committed).**
- **B1 random‑pair.** Facts drawn from distinct source types at random; full pipeline run.
- **B2 within‑silo.** Same‑source‑type pairing forced.
- **B3 shuffled‑label.** Source‑type labels shuffled before pipeline execution.

**Decision rules** (pre‑committed and visible in `preregistration.json`):
- Primary wins → accept compositional alpha hypothesis; publish empirical paper.
- Primary loses (D ≤ B or monotonicity violated) → reject; publish null‑result paper.
- No post‑hoc corpus additions, no scoring‑weight changes, no primary/secondary swap, no retroactive exclusion. All four strata outcomes reported regardless of sign.

Any drift in code or corpus during the study is automatically flagged by `python run.py preregister check` and reported in the final paper regardless of outcome.

Pre‑freeze empirical observations — the mechanism‑vs‑audience kill asymmetry, the hub‑and‑spoke methodology graph concentrated on ARGUS Enterprise DCF cap‑rate assumptions, the bimodal distribution of diamond scores, the negative correlation between narrative strength and kill survival (r = −0.49), the nine closed Tarjan cycles — are held back as secondary hypotheses to be tested on the frozen corpus under the upgraded pipeline tier during summer 2026. The summer study is their test.

## What this repository is *not*

- Not a fund, not a product, not a pitch. Nothing here solicits capital or clients.
- Not a commentary Substack dressed as code. The code is the primary artifact; the writing is the explanation.
- Not a claim of a specific hit rate, return, or market‑size opportunity. There is no track record yet. The ledger is how that gets established, publicly, starting June 2026.

## How to cite

If you use the corpus, cite:

> Malpass, J. (2026). *HUNTER Cross‑Silo Financial Corpus v1 (frozen April 2026)* [Data set]. Zenodo. https://doi.org/XXXXX

If you reference the instrument or methodology, cite:

> Malpass, J. (2026). *HUNTER: An Autonomous Research Instrument for Cross‑Silo Financial Inference* (Working Paper 0). SSRN.

## License

- **Code**: MIT License (see `LICENSE`). Use, fork, and run freely.
- **Corpus and derived data**: CC‑BY‑4.0 (see `LICENSE_DATA`). Redistribution with attribution.
- **Working papers and Substack posts**: CC‑BY‑4.0 unless explicitly marked otherwise.

## Contact

Honest critique, prior‑art pointers, and replication attempts are welcome. No cold pitches, please; this is a research project.

John Malpass · University College Dublin · Email on Zenodo record and Substack About page.
