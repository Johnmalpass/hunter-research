# HUNTER — Engineering Evolution

*How the instrument changed between the first version and the pre-registration-locked April 2026 build. What was added, what was removed, what was kept, and what the trade-offs were.*

The v1 Zenodo corpus contains outputs from at least two distinct pipeline iterations. The earlier run (March 28 – April 3, 2026, `hypotheses_archive` table, 263 hypotheses) was produced under a substantially different engineering configuration than the later run (April 1 – April 4, 2026, `hypotheses` table, 61 hypotheses). This document is a cross-walk so readers can interpret the output by the pipeline that produced it. A snapshot of the earlier configuration is preserved in `archive/config_backup.py`.

---

## The two engineering regimes

| Dimension | Original HUNTER (pre-April 19) | Current HUNTER (April 19 onward) |
|---|---|---|
| Model routing | Single Sonnet-4 model for reasoning, Haiku-4 for extraction | **Three-tier:** Opus 4.7 (critical reasoning), Sonnet 4.5 (extraction), Haiku 4.5 (high-volume ingestion) |
| Domain scope | **14 generalist domains** (Mathematics, Economics, Tech / AI, History, Science, Medicine, Geopolitics, etc.) | **18 professional financial silos** (patents, SEC filings, NAIC reserves, OSHA, CMBS, Federal Register, commodity, analyst targets, academic, pharma, distressed credit, healthcare REITs, energy infra, specialty RE, government contracts, earnings, job listings, app rankings) |
| Ingest/collision ratio | 0.6 / 0.4 (60% ingest, 40% collision) | 0.5 / 0.5 |
| Diamond scale | 6-band: noise / interesting / notable / strong / diamond / legendary | Continuous 0–100; findings threshold at 65 |
| Kill phase | Fact-check + competitor + barrier rounds | **Adds mechanism kill:** every causal arrow must name a specific transmission pathway (filing, database, workflow) or the arrow is rejected |
| Scoring | LLM self-scored, single context | Fresh-context adversarial scoring against four calibrated anchors (92 / 88 / 35 / 25) |
| Query templates | 160+ templates across 25 APIs | 220 templates, 18 silos, 153-pair hand-calibrated distance matrix |
| Self-improvement | Manual prompt iteration | 7 theory agents (TheoryTelemetry / DecayTracker / CycleDetector / CollisionFormulaValidator / ChainDepthProfiler / BacktestReconciler / ResidualEstimator) |
| Deep-dive threshold | ≥ 75 | ≥ 75 (unchanged) |
| Pre-registration | None | SHA-256-locked manifest (`f39d2f5ff6b3e695`), frozen corpus, fixed decision rules, three null baselines |

## What was added (and why)

**Mechanism kill.** The most substantive upgrade. Under the original pipeline, a hypothesis survived the kill phase if adversarial web-search could not find a direct fact-check refutation, a competitor already doing it, or a structural barrier. Under the new pipeline, the reviewer additionally demands that every causal arrow in the hypothesis name the specific filing, database, or workflow through which the output of one silo becomes an input to another. Arrows without a named pathway are rejected, regardless of whether fact-check succeeds. This is the single biggest reason output per cycle is lower in the new pipeline — claims that survived the old pipeline on argument quality can fail the new pipeline on lack of named mechanism.

Trade-off: fewer but harder-to-refute hypotheses in the new pipeline, versus more but less-mechanism-grounded hypotheses in the old. The 263-vs-61 volume split reflects this.

**Three-tier model routing.** Opus 4.7 is routed to mechanism kill and adversarial scoring; Sonnet 4.5 to standard fact extraction and collision evaluation; Haiku 4.5 to anomaly detection and implication tagging. Cost efficiency improved substantially — ~$15–25/day under the three-tier routing versus ~$50–80/day under all-Sonnet. Critically, Opus at the mechanism-kill stage catches flaws that Sonnet missed, which tightens the kill phase independently of the mechanism-kill logic.

**Fresh-context adversarial scoring with four calibration anchors.** Under the original pipeline, the same LLM that generated a hypothesis often scored it, creating scoring inflation. Under the current pipeline, a fresh LLM context with no memory of the generation step scores each surviving hypothesis against four pre-calibrated anchor scores (92 for a reference-quality hypothesis, 88 for a solid one, 35 for a thin one, 25 for a throwaway). This roughly eliminates self-grading inflation.

**Theory agents.** Seven agents (`TheoryTelemetry`, `DecayTracker`, `CycleDetector`, `CollisionFormulaValidator`, `ChainDepthProfiler`, `BacktestReconciler`, `ResidualEstimator`) attach to the orchestrator on fixed schedules and write framework-auditing output into dedicated tables. These did not exist in the original pipeline; they are how the instrument watches itself for framework drift.

**Pre-registration machinery.** The original pipeline had no formal study frame. The current build locks a corpus cutoff, hashes the eligible fact-ID set, hashes the code state, pre-commits three null baselines (random-pair, within-silo, shuffled-label), and fixes decision rules — including an explicit commitment to publish null results. This is the difference between an exploratory research instrument and a pre-registered empirical study.

## What was narrowed (and why)

**Domain scope: 14 → 18 professional financial silos.** The old 14 domains included Mathematics, Economics, Technology / AI, History, Science, Medicine, Geopolitics — a generalist's toolbox. The narrowing to 18 financial-adjacent silos was a deliberate focusing move once the operator's research interest concentrated on compositional alpha in finance. The archive output still shows traces of the old generalism (e.g. pre-pivot findings involving gaming platforms, medieval guilds, AI hardware — see the `findings` table limitation note in `docs/LIMITATIONS.md` L7). The new pipeline would not produce those.

**Collision strategies expanded: 3 → 7.** The original collision engine used entity matching, implication matching, and keyword fallback. The current engine runs seven strategies in parallel: those three plus model-field matching, causal-graph BFS, embedding similarity, and belief-reality contradiction. Matches from all seven are blended into a 10-fact pool before collision evaluation.

**Diamond scale consolidated.** The six-band original scale (noise / interesting / notable / strong / diamond / legendary) was replaced by a continuous 0–100 score with a single operational threshold at 65 for the findings table. The bands are still referenced in archived documents but the scoring is continuous in the current code.

## What was not changed

- The **collision formula** has the same functional form in both pipelines; only the weights refit.
- The **database schema** is append-only; every table that existed in March exists today.
- The **226-table count**, **26 LLM prompts**, and **153-pair distance matrix** are the current version; earlier versions had different counts but the architecture is the same.
- **The seven matching strategies**, **four-round kill gauntlet**, and **adversarial scoring** are present in both pipelines; the mechanism-kill expansion is the upgrade.

## Which pipeline's output should a reader trust?

Neither, without summer replication.

**The archive (263 hypotheses, old pipeline)** has broader thematic range, higher absolute yield of diamond-tier output (28 at ≥ 65, 13 at ≥ 75, 9 at ≥ 85), but was produced under a kill phase that did not require named transmission pathways. Several of the top archive hypotheses are clearly structural and well-reasoned (the NYC municipal underwriting cluster, the pension discount-rate calibration cluster, the PJM forward-curve cluster, the CMBS → LDI cascade). Others are less rigorously killed than the current pipeline would tolerate. Filter by creation date and examine the kill_attempts JSON before citing any specific archive hypothesis.

**The main table (61 hypotheses, new pipeline)** has a higher hit rate per hypothesis (18% at ≥ 65 vs 10.6% in archive), survived the mechanism-kill discipline, and carries narrative-scoring metadata the archive lacks. But the sample is small and the April 1–4 window does not cover the thematic breadth the archive does.

**The honest reading for now:** treat the archive as the reservoir of candidate themes and the main table as the reservoir of mechanism-tight survivors. The summer 2026 study re-runs the upgraded pipeline on the frozen corpus; at that point, either set can be compared against the out-of-sample run to decide which pipeline better captures genuine cross-silo residual.

## What might be brought back from old HUNTER

Three elements of the original pipeline are worth reconsidering for v2 (post-summer):

1. **Generalist domain scope.** Restricting to 18 financial silos improves focus but loses cross-domain edge with science, medicine, climate, geopolitics. A post-summer v2 could add a "generalist mode" that runs a reduced-weight version of the old 14 domains alongside the core 18. Keeps the focus for the primary test; regains the breadth for exploratory output.

2. **Six-band diamond scale.** The continuous 0–100 score is convenient for threshold-setting but loses the intuition of *noise / interesting / notable / strong / diamond / legendary*. Bringing the band labels back as display-only annotations (not as scoring bins) would help dashboard legibility without breaking the continuous score.

3. **Idea-evolution tracking.** The original pipeline tracked how hypotheses mutated across cycles (5 records in `idea_evolutions`). The current pipeline does not explicitly track this. Extending the lineage tracking so every current-pipeline hypothesis carries a pointer to its closest archive predecessor would make the archive usefully searchable by "did this theme already come up under the old pipeline?".

None of these changes can happen during the summer run — the code is hash-locked. All three are v2 candidates for October 2026 onward.

## Where the old code lives

- **`archive/config_backup.py`** — full configuration snapshot from March 29, 2026. Contains the 14-domain definitions with their descriptions and example searches, the six-band diamond thresholds, the old token caps, and the old ingest ratio.
- **`archive/v3_golden/`** — the v3 Golden retrospective experiment, which ran the original pipeline against past facts with `RETROSPECTIVE_DISABLE_WEB_SEARCH = True`. That experiment produced Stratum D < Stratum B, which is the contradictory pilot acknowledged in the README's Pre-registered Study section. It is preserved as a reminder that the primary-endpoint test is not a foregone conclusion.
- **`archive/old_db_snapshots/`** — historical database snapshots; reference material only.

The original HUNTER was not deleted. It was deliberately frozen into an archive so the evolution is legible and the earlier output remains interpretable in context.

---

*John Malpass · University College Dublin · April 2026.*
