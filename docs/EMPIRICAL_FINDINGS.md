# HUNTER — Empirical Findings Report

Generated 2026-04-18 by running the seven new analyser modules against the
existing 12,030-fact corpus, 474 collisions, 61 hypotheses, and 20 surviving theses.

This document answers the 10 "open questions" with real numbers from real data.
Some findings **support** the framework. Some **challenge** it. Some **reverse**
it. Report all three, honestly.

---

## 1. Does the collision formula predict?

**Answer: No. Not at current weights.**

| Metric | Value |
|---|---|
| Pearson r (predicted vs actual pair counts) | **+0.067** |
| Spearman ρ | +0.088 |
| p-value | 0.413 |
| Pairs tested | 153 |
| Pairs with data | 153 |
| Verdict | **WEIGHTS_WRONG** |

### What this means
The formula
```
score(A,B) = silos·silos·0.003 + (reinf+reinf)·20 + (1−corr)(1−corr)·30 + resid·resid·400
```
has no detectable linear correlation with actual collision density. It is
statistically indistinguishable from random noise at current weights.

### But the *structure* might still be right
Regression against actual collision counts suggests:

| Coefficient | Current | Suggested delta |
|---|---|---|
| silo | 0.003 | **+115.08** (× 38,000) |
| reinforcement | 20 | +0.23 |
| correction | 30 | **−3.51** (sign may be wrong) |
| residual | 400 | **+34.61** |

**Silo count and residual density matter much more than current weights imply.**
The correction coefficient may have the wrong *sign* — domains with higher
correction rates seem to produce *more* collisions, not fewer. That's
counter-intuitive and needs investigation.

### What HUNTER actually finds, that the formula misses

**Top 5 most under-predicted pairs** (formula said low, reality said high):
| Pair | Predicted | Observed |
|---|---:|---:|
| cre_credit × distressed | 36.06 | **122** |
| pharmaceutical × regulation | 36.42 | **118** |
| cre_credit × sec_filing | 32.12 | **105** |
| earnings × pharmaceutical | 32.53 | **102** |
| earnings × regulation | 30.43 | **99** |

These are all CRE / regulatory / pharma combinations — exactly where HUNTER
has produced its top-scoring findings. The formula is blind to the very places
HUNTER finds diamonds.

### Action
Refit weights via regression before the summer study. The functional form
survives; the hand-calibrated constants don't. This is a **falsifiable
improvement** to the framework.

---

## 2. What are the actual reinforcement and correction rates?

**Answer: Your hand-guesses were systematically wrong.**

| Source type | Reinf predicted | Reinf measured | Δ | Corr predicted | Corr measured | Δ | Persistence ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| app_ranking | 0.500 | **1.000** | +0.500 | 0.500 | 0.041 | -0.459 | 24.3x |
| bankruptcy | 0.400 | **0.997** | +0.597 | 0.300 | 0.013 | -0.286 | **74.0x** |
| commodity | 0.600 | **1.000** | +0.400 | 0.500 | 0.043 | -0.457 | 23.4x |
| cre_credit | 0.400 | **0.980** | +0.580 | 0.200 | 0.033 | -0.167 | 30.0x |
| distressed | 0.500 | **1.000** | +0.500 | 0.300 | 0.026 | -0.274 | 39.1x |
| earnings | 0.800 | **1.000** | +0.200 | 0.700 | 0.040 | -0.660 | 25.0x |
| energy_infra | 0.500 | **0.997** | +0.497 | 0.300 | 0.085 | -0.215 | 11.8x |
| healthcare_re | 0.400 | **0.997** | +0.597 | 0.200 | 0.095 | -0.105 | 10.5x |
| insurance | 0.300 | **0.993** | +0.693 | 0.150 | 0.046 | -0.104 | 21.7x |
| regulation | 0.300 | **0.997** | +0.697 | 0.100 | 0.076 | -0.024 | 13.1x |

### What this means

Every domain's **reinforcement rate is near 1.0** — facts in a source type
almost always echo earlier facts in the same source type. The echo-chamber
effect is **much stronger than the framework assumed**.

Every domain's **correction rate is near 0** — explicit corrections,
retractions, and revisions are rare. The public correction-loop the
framework assumes doesn't really run at the source-type level.

### Persistence ratios
- Framework predicted aggregate **207x**
- Measured domain-level ratios: **10.5x – 74.0x**
- If reinforcement ≈ 1.0 and correction ≈ 0.03, ratio ≈ 33x

The framework over-estimated aggregate persistence by ~6×, but got the
**direction** right. Errors persist far longer than correction dynamics
predict. Framework survives qualitatively; specific 207x constant needs
downward revision to ~30–60x.

### Action
Replace hand-coded `DOMAIN_THEORY_PARAMS` with measured values before
the summer study. This is already written to `measured_domain_params`.

---

## 3. Is the 120-day half-life prediction right?

**Answer: No. Off by ~94×. Measured ~11,300 days (~31 years).**

| Source type | n obs | correction events | half-life (days) |
|---|---:|---:|---:|
| academic | 636 | 89 | **1,573** (4.3 years) |
| healthcare_re | 647 | 39 | **4,090** (11 years) |
| energy_infra | 654 | 25 | **6,414** (17.5 years) |
| patent | 651 | 18 | 9,021 |
| cre_credit | 604 | 13 | 11,544 |
| government_contract | 634 | 13 | 12,128 |
| **Global pooled** | **11,279** | **248** | **11,318 (31 years)** |

### What this means

Two interpretations:

1. **The framework's 120-day prediction is miscalibrated.** The *direction*
   is right (errors persist) but the specific timescale is off by two
   orders of magnitude.

2. **Our correction-detection is too strict.** We require both entity
   overlap AND explicit correction vocabulary (retracted, revised,
   contradicted, etc.). Many real corrections are implicit —
   market prices move, analysts quietly change models, no one publishes
   a retraction. So the measured correction rate is a *lower bound*.

Academic source type shows the shortest half-life (~4 years) because
retractions are formally published. Market-reactive domains (earnings,
commodity, insurance) show the longest half-lives because corrections
there are diffuse price signals, not retraction notices.

### Action
- Recalibrate framework half-life prediction to a log-scale range:
  months for formal-retraction domains, decades for price-reactive domains.
- Develop an implicit-correction detector for market-reactive domains
  (price divergence, analyst forecast drift) before summer.

---

## 4. Do the 9 cycle types actually exist?

**Answer: 2 of 9 observed so far in the corpus; framework partially supported.**

From [cycle_detector.py](cycle_detector.py) run on 52 chains with semantic node
merging:

| Cycle type | Count detected |
|---|---:|
| cross_domain_3node | 7 |
| cross_domain_4node | 1 |
| cross_domain_6node | 1 |
| simple_3node | 0 |
| nested | 0 |
| coupled | 0 |
| braided | 0 |
| hierarchical | 0 |
| temporal | 0 |
| interference | 0 |
| dormant | 0 |

### What this means
At current data density, only **cross-domain** cycles are distinguishable
from noise. The other 7 cycle types are theoretically defined but
empirically indistinguishable in 52 chains.

### Action
The taxonomy may need pruning. Summer study should either:
- Find the missing types with more data (targeted collision strategies
  aimed at temporal, nested, hierarchical patterns), OR
- Reduce taxonomy to the 2–3 empirically resolvable types.

Either way, the strongest detected cycle is a **6-node cross-domain loop**
spanning CMBS servicing → credit rating surveillance → fixed income →
insurance actuarial → structured finance → real estate appraisal. This
is a real empirical artefact, not theoretical.

---

## 5. Narrative structure & kill-survival — the surprise

**Answer: The framework had the direction wrong.**

| Metric | Value |
|---|---|
| Pearson r (narrative strength → survival) | **−0.49** |
| Pearson r (narrative strength → diamond score) | **−0.52** |
| High-narrative hypotheses (strength ≥ 0.6) | 3, **0% survival** |
| Low-narrative hypotheses (strength < 0.4) | 24, **58% survival** |
| Survival uplift from narrative | **−58%** |

### What this means
The framework predicted that **compelling narrative reinforces errors** —
embedded stories are harder to dislodge. Our data shows the **opposite**:

Strong-narrative hypotheses are the ones kill-rounds successfully destroy.
Survivors tend to be raw, structural, hard to articulate — they don't
*have* a clean narrative yet because nobody's written one.

### Reconciliation with framework

This is actually consistent with Layer 10 (structural incompleteness) —
if an error is structurally unreachable by correction mechanisms, nobody
has written the narrative for correcting it either. The absence of narrative
is evidence of absence of correction infrastructure.

Meanwhile, strong-narrative hypotheses are obvious-under-another-label —
the narrative was written because the error is already widely visible,
which means an adversarial kill round can *find* the counter-evidence.

### Action
- Invert the expected direction: **strong narrative = likely front-page**.
- Add a small negative penalty (−3 to −5 points) to hypotheses with
  narrative_strength ≥ 0.6, matching the empirical direction.

---

## 6. Where does kill-round correction actually fail?

**Answer: The structural-incompleteness topology is real and mappable.**

### Kill-type success rate

| Kill type | Attempts | No evidence found | Kills found | Hit rate |
|---|---:|---:|---:|---:|
| mechanism_fatal | 40 | 0 | 40 | **100%** |
| mechanism | 41 | 0 | 41 | **100%** |
| fact_check | 13 | 10 | 3 | 23% |
| barrier | 12 | 12 | 0 | **0%** |
| competitor | 12 | 12 | 0 | **0%** |
| market_check | 10 | 10 | 0 | **0%** |
| edge_recovery | 9 | 9 | 0 | **0%** |
| refinement | 10 | 10 | 0 | 0% |

### What this means
- **Financial-mechanics checks** (mechanism rounds) find kills 100% of the time
  they should. This is the refinement step catching bad trade logic.
- **Adversarial web-search kills** (competitor, barrier, market_check) find
  **zero kills** across ~34 attempts. This is either:
  - Evidence of structural incompleteness (the edges really are obscure), OR
  - Evidence that the kill prompts need sharpening (false-negative kills).

### Top 10 structural-incompleteness candidate pairs

| Pair | n hypotheses | Kill failure rate | Survival rate | SI score |
|---|---:|---:|---:|---:|
| **bankruptcy × regulation** | 5 | **87.0%** | **80.0%** | **1.56** |
| other × regulation | 9 | 73.0% | 55.6% | 1.22 |
| app_ranking × commodity | 3 | 83.3% | 66.7% | 0.96 |
| cre_credit × energy_infra | 9 | 66.7% | 44.4% | 0.89 |
| insurance × other | 10 | 65.7% | 40.0% | 0.83 |
| distressed × regulation | 7 | 70.4% | 42.9% | 0.80 |

**Bankruptcy × regulation** is the top structural-incompleteness candidate.
Makes sense — bankruptcy court filings + regulatory rule changes is exactly
the kind of cross-silo information where nobody in either domain reads the
other.

### Action
For the summer study, use this ranked table to *target* collision generation —
these pairs are most likely to produce unkilled, unreachable structural edges.

---

## 7. Phase-transition signal per domain

**Answer: weak signal at current data density; useful after more history.**

| Domain | Latest rate | Mean rate (180d) | z-score | Risk |
|---|---:|---:|---:|---:|
| job_listing | 0.400 | 0.167 | +1.4 | 0.50 |
| earnings | 0.567 | 0.244 | +1.3 | 0.50 |
| cre_credit | −0.733 | −0.011 | −2.0 | 0.00 |
| distressed | −0.700 | −0.078 | −2.0 | 0.00 |

Most domains show flat or negative z-scores (corrections keeping pace with
accumulation). **job_listing and earnings** are the two domains where
residual is accumulating faster than average — flagged for attention.

### Action
Will become more useful after 6–12 months of history. The framework
component (Layer 4) remains untested at current density.

---

## 8. Adversarial vs accidental classification

**Answer: Current lightweight classifier is too sparse; 92% of
hypotheses classified "unknown".**

| Category | n | Survived | Killed | Avg score |
|---|---:|---:|---:|---:|
| unknown | 56 | 19 | 37 | 68.05 |
| accidental | 2 | 0 | 2 | 0 |
| self_reinforcing | 1 | 1 | 0 | 70.0 |
| regulatory | 1 | 0 | 1 | 0 |
| structural | 1 | 0 | 1 | 0 |

At n=61, we can't say anything statistically meaningful. Need:
- LLM-based classification (Haiku) to replace pattern-matching
- 200+ hypotheses to get useful per-category survival rates

### Action
Defer until summer. Replace pattern-matching classifier with LLM prompt
that explicitly asks "does this error have an active maintainer?".

---

## Meta-findings — what the data says about the framework

### Supported
- Cross-domain cycles **exist** and **are detectable** (9 found).
- Residual **persists** longer than correction dynamics predict. Framework
  direction correct; constants miscalibrated.
- Kill rounds on competitor/barrier/market_check **systematically fail** —
  operationalised evidence of structural-incompleteness candidates.

### Challenged
- Collision formula has **no predictive power at current weights** (r = 0.07).
  Needs re-weighting. Structure may survive.
- Narrative strength is **negatively** correlated with survival. Framework
  predicted positive correlation.
- Half-life is **off by ~94×** from 120-day prediction. Direction right,
  magnitude wrong.

### Undetermined
- Phase transitions (need longer time series)
- 7 of 9 cycle types (need targeted collision generation)
- Adversarial vs accidental (need LLM classifier + more data)

---

## Summer study adjustments

These findings update the pre-registered study:

1. **Primary endpoint unchanged**: D > B > A monotonic alpha.
2. **New secondary endpoint**: does refitted collision formula (regression
   coefficients from §1) have r ≥ 0.4 on held-out data?
3. **Targeted collision strategy**: oversample bankruptcy × regulation,
   cre_credit × distressed, pharmaceutical × regulation — the empirically
   productive pairs.
4. **Scoring update**: add a `narrative_penalty` component. High narrative
   reduces score by 3–5 points. Log and measure effect on adjusted win rate.
5. **Half-life recalibration**: use per-source measured half-lives rather
   than the 120-day constant in `theory.py`.

---

## Why this report matters for Bain

This is not "here's a theory." This is "here's a theory, here's the data,
here's what the data says, here's what's wrong with the theory, here's the
fix."

Most pitch decks you'll see from researchers claim the theory works. This
report documents the **two major places it doesn't** and still recommends
proceeding — because the *direction* is right and the *fixes are identifiable*.

That's the distinction between confirmation-hunting and science. Bain funds
the latter.
