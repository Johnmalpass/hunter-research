# Statistical methods

The full inference framework for the summer 2026 study. This document fills the gap a sophisticated reviewer would flag: the empirical work in `MATH_VERIFICATION.md` and `EMPIRICAL_FINDINGS.md` is descriptive. The methods below are inferential. They cover the identification strategy, the causal-inference framework, power analysis, Bayesian posterior framework, multiple-testing correction, robustness, and sensitivity.

The summer study primary endpoint is the only test on which a strict frequentist decision is made. Everything else (Bayesian posteriors, sensitivity analyses) is published as supporting evidence. The pre-registration manifest does not require Bayesian methods; this document adds them as a methodological supplement that any reviewer would expect.

---

## 1. Identification strategy

The primary causal claim is: increasing the number of distinct professional silos a hypothesis combines causes higher realised alpha. The challenge is separating this from confounders.

### Confounders to rule out

- **Topic effect.** Cross-silo hypotheses might just happen to be in topics where alpha is higher (CRE, regulatory transitions). Solution: stratify the primary test by topic cluster; report A vs D within each cluster as a secondary check.
- **Selection effect.** The kill phase is more aggressive on cross-silo claims, so survivors are pre-selected for quality. Solution: the within-silo control baseline (B2) runs the same kill phase on within-silo facts. The differential between cross-silo and within-silo alpha at the same kill-survival rate is the identified effect.
- **Look-ahead leakage.** A fact dated after the cutoff but ingested into a hypothesis would create look-ahead alpha. Solution: pre-registration locks the eligible fact set at the SHA-256 hash. `python run.py preregister check` re-verifies during the study.
- **Timing effect.** Hypotheses generated at different times in the run face different market conditions. Solution: control for cycle-day fixed effects in the realised-alpha regression.

### What identifies the cross-silo effect

The three null baselines collectively isolate the cross-silo channel:

- **B1 random-pair** removes the structural-collision component but keeps the multi-silo character. If B1 alpha equals stratum B alpha, the structural-collision logic adds nothing and the result is just diversification.
- **B2 within-silo** removes the cross-silo character but keeps structural collision logic. If B2 alpha equals stratum A alpha, within-silo collisions add nothing and the cross-silo channel is the binding constraint.
- **B3 shuffled-label** destroys the source-type information entirely while preserving every other feature of the pipeline. If B3 alpha is non-zero, the pipeline's signal is not coming from the silo identity; it's coming from something else (text length, formatting, fact age) that should be controlled for.

The cross-silo identifying assumption is: stratum D alpha minus the maximum of (B1, B2, B3) alpha is attributable to cross-silo composition specifically, not to multi-fact aggregation, structural collision logic, or pipeline artefacts.

### Plain-English version

If random-pair, within-silo, and shuffled-label baselines all produce no alpha and the four strata produce monotonically increasing alpha A ≤ B ≤ C ≤ D with D − A > 0 at p < 0.05, the cross-silo channel is doing the work. If any baseline shows non-trivial alpha, the source of that alpha must be identified before the primary result can be interpreted as evidence for compositional alpha.

---

## 2. Power analysis

Pre-registration `power_analysis` field commits to: Cohen's d ≥ 0.3, alpha = 0.05, power = 0.80, minimum n per stratum = 30.

### Sample-size justification

Under a two-sample t-test for D vs A:

- Effect size d = 0.3, alpha = 0.05, power = 0.80 → required n per group ≈ 175 (Cohen 1988 tables).
- Effect size d = 0.5 (medium), same alpha and power → required n per group ≈ 64.
- Effect size d = 0.8 (large), same alpha and power → required n per group ≈ 26.

The pre-registration requires n ≥ 30 per stratum, which corresponds to detecting effect sizes of approximately d ≥ 0.7. **This is a strong-effect-only design.** If compositional alpha exists at d < 0.5, the summer study is underpowered.

### Honest acknowledgement

Detecting d ≥ 0.7 is plausible if the framework is right and the effect is concentrated in the highest stratum. Detecting smaller effects requires more cycles, which means more compute. Cycles are budget-constrained.

The decision rule is: if the summer produces n < 30 per stratum, the result is reported as inconclusive rather than null. If n ≥ 30 per stratum and D − A > 0 at p < 0.05, the result is reported as positive. If n ≥ 30 per stratum and the test fails, the null paper ships.

### Bootstrap confidence

The pre-registered test uses a 10,000-resample paired bootstrap of the median, not a parametric t-test. This handles the non-normality of return distributions and small-sample tail behaviour. Bootstrap confidence intervals on the D − A median difference are reported alongside the p-value.

---

## 3. Bayesian framework

The Bayesian posteriors below complement the frequentist primary test. They are not the primary decision rule, but they answer the question a quant actually wants to know: given the data, what is the probability that compositional alpha is real?

### Prior

For each stratum k ∈ {A, B, C, D}, the prior on median realised alpha μ_k is:

```
μ_k ~ Normal(0, σ_prior²)
```

with σ_prior = 0.05 (i.e. the prior 95% interval is roughly ±10% annualised excess return). This is a **weakly informative prior** that places non-trivial mass on zero (the efficient-markets hypothesis) but does not rule out non-zero effects.

### Likelihood

For each hypothesis i in stratum k with realised alpha r_ik, the likelihood is:

```
r_ik ~ Normal(μ_k, σ_obs²)
```

with σ_obs estimated empirically from the dispersion of returns within each stratum.

### Posterior quantities of interest

- **P(μ_D > μ_A | data)** — the Bayesian analogue of the pre-registered primary test. A value above 0.95 is the Bayesian "supports compositional alpha" threshold.
- **Posterior mean and 95% credible interval on (μ_D − μ_A).** The effect size with uncertainty.
- **Bayes factor BF₁₀** comparing the compositional-alpha model against the null (μ_A = μ_B = μ_C = μ_D). BF > 10 is "strong evidence for the alternative" by Kass-Raftery convention.

### Why both

The frequentist test is what the manifest commits to. The Bayesian analysis is what a sophisticated reader will want regardless. Publishing both pre-empts the "you cherry-picked the test that worked" critique because both are committed in advance and both run on the same data.

The Bayesian re-analysis script `bayesian_alpha.py` runs the full posterior inference against the frozen Zenodo corpus and reports all three quantities above.

---

## 4. Multiple-testing correction

The pre-registration commits to one primary test (A ≤ B ≤ C ≤ D monotonic) and three secondary tests (H2 cycle persistence, H3 cross-silo collision scoring, H4 chain-depth-3 outperformance). Without correction, the family-wise type-I error rate is approximately 0.18 across the four tests (1 − (0.95)^4).

### Correction procedure

The four tests use a hierarchical correction:

- **The primary test** is reported at α = 0.05 uncorrected. The pre-registration designates it as primary precisely so that it does not need correction.
- **The three secondary tests** are reported with **Holm-Bonferroni** correction at family-wise α = 0.05. Sorted by p-value, the three are tested at α/3, α/2, α/1. This controls family-wise error at 0.05 for the secondary family.

This is more conservative than Benjamini-Hochberg FDR control and is preferred when the cost of a single false positive is high.

### Pre-freeze patterns reported separately

The patterns reported in `MATH_VERIFICATION.md` (narrative correlation r = −0.27, hump-curve at d=2, alpha = 0.94 vs 0.27 refutation, 9/9 cycle stability, kill-failure topology) are descriptive observations on the pre-freeze corpus. They are **not** pre-registered tests and are not subject to multiple-testing correction. They are framed as hypotheses for the summer to test, not as findings.

The summer study tests one primary and three secondary endpoints. Everything else is supporting descriptive analysis.

---

## 5. Robustness

The primary test is computed under the canonical specification but reported under three additional specifications:

### Robustness 1: Stratum boundary sensitivity

The pre-registration defines strata as A=1, B=2, C=3, D≥4 silos. Robustness specifications:

- A=1, B=2-3, C=4-5, D≥6 (looser strata)
- A=1-2, B=3, C=4, D≥5 (tighter strata)
- Continuous regression of alpha on silo count, no strata

If the primary signal disappears under reasonable stratum re-binning, the result is fragile and should be reported as such.

### Robustness 2: Outlier handling

The primary test uses the median, which is robust to outliers. Two robustness checks:

- Trimmed mean (5% trim from each tail) instead of median
- Mean with one outlier (the highest-alpha hypothesis in stratum D) removed

Both should give the same qualitative answer. If they don't, a single hypothesis is driving the result and the result is fragile.

### Robustness 3: Survivorship and exclusion

Two checks:

- Including hypotheses that did not resolve before August 31 (treated as zero-alpha)
- Excluding hypotheses with structural breaks during the study window (e.g. M&A, regime change)

Both should produce qualitatively similar results.

### What gets reported

All four specifications (canonical + three robustness checks) are reported in the summer paper regardless of outcome. Disagreement among them is explicitly flagged.

---

## 6. Standard errors and inference

### Standard errors

Realised alpha for each hypothesis is computed as (hypothesis return − SPY return) over the position holding window. The standard error of the median for stratum k is computed via the bootstrap (10,000 resamples). This is more robust than analytic standard errors under non-normality.

### Hypothesis-level uncertainty

Each hypothesis has a "score uncertainty" from the adversarial reviewer (the reviewer reports a confidence band around its diamond score). This uncertainty is **not** propagated into the realised-alpha test because the score and the realised alpha are measured in different units. The score is used only to threshold which hypotheses get posted to the prediction board.

### Cross-cycle correlation

Hypotheses produced in adjacent cycles share market context (same week, same macro regime). The bootstrap is performed at the cycle-day level (block bootstrap with block size = 1 day) to account for within-day correlation in returns.

---

## 7. Sensitivity analyses

The primary test is reported with sensitivity to:

- **Diamond threshold.** Test with threshold ≥ 60, ≥ 65 (canonical), ≥ 70, ≥ 75. Lower thresholds include more hypotheses; higher thresholds restrict to higher-conviction calls.
- **Position holding window.** Canonical is the resolution date in the hypothesis. Sensitivity windows: 30 days, 60 days, 90 days regardless of stated resolution date.
- **Benchmark.** SPY total return is the canonical benchmark. Sensitivity: equal-weight S&P 500, CRSP value-weighted, sector-adjusted return for the hypothesis's primary sector.

Three sensitivity tables are reported in the summer paper.

---

## 8. What this document does not do

This document specifies the inference framework for the summer 2026 study. It does not:

- Specify the trade execution rules (those are in the public prediction board posting protocol)
- Specify the corpus generation rules (those are in `preregistration.json`)
- Specify the kill-phase logic (that is in `prompts.py` and `hunter.py`, both SHA-locked)
- Pre-commit to specific Bayesian priors beyond the weakly informative ones above (the priors are defensible but a reviewer can argue for tighter or wider priors; that is a robustness check)

Any methodological change to this document during the summer study window is logged in `CHANGELOG.md` with a justification. The frequentist primary test is the immutable decision rule.

---

*John Malpass · University College Dublin · April 2026.*
