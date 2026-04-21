# Three patterns from the pre-freeze corpus

*Patterns the summer study will test, not confirmed findings. Two-page summary.*

The HUNTER corpus produced several empirical observations during the pre-freeze development period (before the 2024-12-31 cutoff, under the earlier pipeline tier). Full details are in `docs/EMPIRICAL_FINDINGS.md`. Three are interesting enough to call out, and all three are held back as hypotheses for the summer 2026 study to replicate or refute.

## 1. Narrative strength negatively predicts kill survival (r = −0.49)

**What I predicted.** Hypotheses with clean narrative structure (clear protagonist, catalyst, resolution) would persist longer through adversarial review, because a smooth story is harder to dislodge.

**What the pre-freeze data showed.** The opposite. Hypotheses with high narrative strength (score ≥ 0.6) died in the kill rounds every time (0% survival, n = 3). Hypotheses with low narrative strength (< 0.4) survived 58% of the time (n = 24). Pearson r = −0.49 between narrative strength and survival, r = −0.52 against diamond score.

**Reading.** If this replicates, the claim is: a cross-silo hypothesis that already has a clean story has already been articulated somewhere, so a web-searched kill round can find the counter-evidence. The structurally uncorrectable core survives precisely because nobody has written the story yet — there is no audience, no correction infrastructure, no search terms that return disconfirming papers. Weak narrative becomes a proxy for genuine structural opacity.

**What summer tests.** Whether the sign and magnitude hold on the out-of-sample run under the upgraded pipeline. Pre-registered threshold: r ≤ −0.3 at p < 0.05 counts as replication.

## 2. Audience-focused kill rounds find zero kills (0 of 34)

**What I predicted.** Cross-silo hypotheses would sometimes fail because a competitor had already published the thesis, or because a regulatory/structural barrier made the trade impossible, or because the market had already priced the edge.

**What the pre-freeze data showed.** Across 34 adversarial kill attempts focused on competitor existence, structural barriers, and prior market awareness, **zero succeeded.** The only rounds that killed hypotheses at rate were mechanism-focused (verifying that each named causal arrow corresponds to a real transmission pathway).

**Reading.** If this replicates, it refines Shleifer-Vishny's limits-to-arbitrage framework for the cross-silo regime. The binding constraint on cross-silo claims is not competition, capital, or noise traders. It is the cost of assembling a verifiable causal mechanism that names each arrow's transmission pathway. I call this the *mechanism-assembly bottleneck*.

**What summer tests.** The asymmetry replicates if mechanism kill rate exceeds 85% and audience kill rate stays below 15% at p < 0.001 (one-sided binomial). A sample of n ≥ 128 combined review attempts is targeted. Refutation: audience kill rate above 30% or mechanism below 70%.

## 3. Bankruptcy × regulation is the top structural-incompleteness pair

**What I predicted.** Nothing specific. The kill-failure topology was an exploratory measurement.

**What the pre-freeze data showed.** Of 138 silo-pair combinations where adversarial review was attempted, the highest "structural incompleteness" signal lands at **bankruptcy × regulation** (87% kill failure rate, 80% survival rate, n = 5). Close behind: `other × regulation`, `app_ranking × commodity`, `cre_credit × energy_infra`, `insurance × other`, `distressed × regulation`. Every top pair includes a regulatory or enforcement dimension.

**Reading.** If this replicates, cross-silo alpha in finance is primarily *regulatory-transition alpha*. The connective tissue between silos is the government publication system (Federal Register, state commissioners, PACER, federal agencies) because government publications are designed to be read by multiple constituencies. Corporate entities concentrate in one silo each. This sharpens the practical prescription for anyone operating a HUNTER-style pipeline: oversample regulatory transitions at silo boundaries.

**What summer tests.** Whether the ranked pair list remains stable out-of-sample. Pre-registered: at least 4 of the top 6 pre-freeze pairs remain in the top 10 after summer re-computation.

---

## What these are not

Not findings. The pre-freeze corpus was produced under an earlier pipeline tier; the upgraded summer pipeline (Opus 4.7 for mechanism and adversarial review) re-runs the analysis on the frozen corpus. Until that re-run reports, every number above is provisional. The pre-registration manifest (SHA-256 `f39d2f5ff6b3e695`) locks the test rules in advance. Decision rules for replication, partial replication, and refutation are fixed.

If the summer study contradicts any of the three patterns, the null result ships in Paper 1. The framework is designed to survive being wrong about specific patterns as long as the methodological triad (implication matching, model-field extraction, differential edge) itself produces non-random signal. Even that is empirically testable through the three pre-registered null baselines: random-pair, within-silo, shuffled-label.

---

*John Malpass · University College Dublin · April 2026. Pre-freeze summary · not for citation as established findings.*
