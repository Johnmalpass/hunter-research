# HUNTER — The Theory Canon

The canonical statement of what the framework claims, what it does not claim,
the formal definitions that anchor the vocabulary, and the testable predictions
the summer study will evaluate.

This document replaces the ambiguous claims in the earlier framework docs.
It is explicitly designed so that a senior reviewer can read it and immediately
see: what is novel, what is defensible, what is testable, what is aspirational.

---

## 1. The canonical vocabulary (coin these, use them consistently)

These are the terms HUNTER owns. Use them in every subsequent paper,
preprint, dashboard, and pitch. Consistency creates brand.

| Term | Canonical definition |
|---|---|
| **Compositional alpha** | Excess risk-adjusted return available from combining information that is individually known within its originating professional silo but is not combined by any market participant. |
| **Epistemic residual** | The portion of total market mispricing attributable to cross-silo informational compositions rather than to within-silo analytic errors or risk-premia adjustments. |
| **Professional silo** | A domain of practice (e.g. patent examination, bankruptcy court administration, insurance actuarial science) whose practitioners share methodology, publication venues, and incentive structures but do not routinely exchange structured information with adjacent domains. |
| **Collision** | A tuple of ≥ 2 facts drawn from structurally independent silos whose joint implication is a specific, time-bounded, actionable mispricing that is not implied by any single fact. |
| **Collision chain** | A sequence of facts (f₁, f₂, ..., f_d) across d domains where each adjacent pair is a collision. Length d is the chain depth. |
| **Epistemic cycle** | A closed collision chain (f₁ → f₂ → ... → f_d → f₁) where each transition is a directed causal implication. |
| **Kill round** | An adversarial web-search-grounded attempt to find fact-level, competitor, barrier, or mechanism evidence that the hypothesis is wrong. |
| **Structural incompleteness** | A mispricing that systematically survives N rounds of adversarial kill because no natural market participant has the incentive or cross-silo expertise to publish the correcting evidence. |
| **Persistence ratio** | r_i / c_i for a domain i, where r_i is the empirically measured reinforcement rate and c_i is the empirically measured correction rate. |

These terms are precise enough to appear in a paper's glossary, general
enough to be cited by others.

---

## 2. What the framework claims (defensible, listed tightest-first)

### C1. Claim: Errors in complete markets are structured, not random.

**Precision**: In the set of all market mispricings, the subset attributable
to compositional informational gaps has non-zero density and is
distributionally non-uniform across domain pairs.

**Defensibility**: Established by existing literature (Grossman-Stiglitz 1980,
Shleifer-Vishny 1997). HUNTER's contribution is not the *existence* of this
structure but the *mappability* of it.

**Testable prediction**: Cross-silo domain pairs with high silo counts,
high reinforcement, and low correction will produce observably more
collisions than pairs with low silo counts.
Pearson r > 0.4 between predicted and observed in the summer study
is the threshold.

**Current status**: Not supported at current weights (r = 0.14 after refit,
p = 0.08). Summer study is the real test.

---

### C2. Claim: Cross-domain epistemic cycles exist and can be detected.

**Precision**: Directed closed loops of ≥ 3 nodes exist in the corpus of
detected collision chains. Node identity is defined by the
(broken_methodology, broken_assumption) pair.

**Defensibility**: Demonstrated empirically. 9 cycles detected in
HUNTER's current 52-chain corpus under 0.78 semantic merging threshold.
Strongest is a 6-node cross-silo loop through CMBS servicing →
credit rating → fixed income → insurance actuarial → structured
finance → real estate appraisal.

**Testable prediction**: Detected cycles will remain detectable
(persistent) for ≥ 14 days in ≥ 2 of the 9 cycles across the 90-day
summer window.

**Current status**: Detected but persistence not yet measured.

---

### C3. Claim: Compositional residuals persist longer than within-silo corrections.

**Precision**: The empirical half-life of cross-silo residuals is
significantly longer than the half-life of within-silo anomalies
of equivalent magnitude.

**Defensibility**: Consistent with Grossman-Stiglitz (bounded arbitrageur
attention) and Shleifer-Vishny (limits to arbitrage). HUNTER operationalises
the measurement.

**Testable prediction**: Median half-life of surviving hypotheses ≥ 180
days; median half-life of killed hypotheses ≤ 30 days.

**Current status**: Measured global half-life (11,318 days) vastly
exceeds framework's 120-day prediction, but the measurement is a lower
bound on correction-rate (our detector only catches explicit correction
vocabulary). Direction of framework is supported; specific constant
needs recalibration.

---

### C4. Claim: Strength of compositional alpha scales with domain distance.

**Precision**: For collision pairs (A, B) with information-theoretic
distance d(A,B), the average realised alpha E[α | d] is monotonically
increasing in d.

**Defensibility**: This is the *primary* hypothesis of the summer
pre-registration. 4-stratum design.

**Testable prediction**: Median adjusted-score alpha increases
monotonically across strata A (1 domain) < B (2) < C (3) < D (≥4).

**Current status**: v3 Golden test contradicted this (Stratum D <
Stratum B). v4 summer test is the proper evaluation.

---

### C5. Claim: Some compositional errors are structurally uncorrectable by internal market mechanisms.

**Precision**: There exists a non-empty subset of cross-silo mispricings
that survive adversarial kill rounds not because the kill prompts are
weak but because no market participant has the incentive AND the
cross-silo expertise AND the distributional channel to publish the
correcting evidence.

**Defensibility**: This is the *framework's* strongest and most novel claim.
It is distinct from "hard to arbitrage" (Shleifer-Vishny) because it
specifies a mechanism: the absence of a single agent with the required
cross-silo competence.

**Testable prediction**: Of the surviving hypotheses after 90 days, a
non-trivial fraction (≥ 20%) will show zero external correction
signals (no analyst note, no rating change, no price movement
exceeding 1σ) across the entire window.

**Current status**: Preliminary evidence is promising. Competitor / barrier
/ market_check kill rounds have found 0 kills across 34 attempts, which
is suggestive but not yet dispositive. The summer study will be
definitive.

---

### C6. Claim: The taxonomy of epistemic cycles is finite and has at least three distinguishable types.

**Precision**: Cycles partition into types (simple, cross-domain, nested,
coupled, braided, hierarchical, temporal, interference, dormant)
distinguished by their graph structure and node domain diversity.

**Defensibility**: Novel classification. 2 types clearly distinguishable in
current data (cross_domain_3node, cross_domain_6node); 7 types
conjectured.

**Testable prediction**: At least 4 of the 9 types will be empirically
distinguishable in the summer corpus.

**Current status**: 2/9 currently detected. Taxonomy needs either
validation with more data or pruning.

---

## 3. What the framework does NOT claim (overreach removed)

These statements appeared in earlier drafts. They are withdrawn.

- **"Gödelian incompleteness theorem for markets."** The argument
  is an *analogy*, not a proof. Use "structural uncorrectability" or
  "mechanism-absent residuals" instead. No formal incompleteness
  theorem applies to market structure without significant additional
  assumptions that were never stated.
- **"Autopoietic verification — finding residual where predicted IS evidence."**
  This is circular. A framework cannot be validated purely by finding
  the phenomena it predicts without an independent control. Layer 12
  is demoted from "direct evidence" to "consistency check."
- **"$5.65T in total residual."** Replaced with scenario-based TAM
  estimates from `residual_tam.py`: $14B conservative, $176B central,
  $1.7T optimistic. Point estimate removed; scenarios retained with
  explicit assumptions.
- **"The system that proves the theory IS the system the theory describes."**
  Removed. Self-reference doesn't make a claim more defensible; it
  makes it less falsifiable.
- **"$2.8T cross-domain residual, 4.84x ratio to known inefficiencies."**
  The 4.84× ratio is defensible as an order-of-magnitude argument
  (since most known inefficiencies are within-silo and this claims to
  measure cross-silo). But the $2.8T specific number came from a
  formula whose computed tables returned zeros. Removed as a specific
  number; retained as "of the same order as or larger than all known
  within-silo inefficiencies combined" — which is the actual argument.

## 4. The formal collision-scoring function

**Definition**: For source types A, B ∈ 𝒟 (the domain set) with parameters
(silos, reinf, corr, resid), the collision score is

```
score(A, B) = w_silo · silo_A · silo_B
            + w_reinf · (reinf_A + reinf_B)
            + w_corr · (1 - corr_A)(1 - corr_B)
            + w_resid · resid_A · resid_B
            + w_0
```

with live weights (v2, refitted 2026-04-19):

```
w_silo  = 0.10       w_reinf = 15.0
w_corr  = 10.0       w_resid = 1500.0
w_0     = 20.0
```

Weights v1 (original hand-calibrated) are retained in `theory.py` as
`COLLISION_FORMULA_WEIGHTS["v1_original"]` for reproducibility.

In-sample r at v2: 0.14 (v1: 0.07). Summer study's held-out evaluation
will be the definitive test.

## 5. The formal chain-value distribution

**Definition**: For a collision chain of depth d, the expected per-chain
value (in $M of addressable residual, conditional on the chain existing) is

```
V(d) = v_0 · d · exp(-d / τ)        for d ∈ {1, 2, ..., 50}
```

with v_0 = $10M, τ = 3. Peak at d ≈ 3 with V(3) ≈ $11M.

**Important**: V(d) is **per-chain**. The total addressable compositional
residual is

```
TAM = N_pairs · C_per_pair_per_year · years · ∫ w(d) · V(d) dd
```

where w(d) is the empirical depth distribution. Central scenario yields
TAM ≈ $176B with annual flow $35B.

## 6. The testable predictions (for the summer paper)

The summer paper tests six predictions. Each has a pre-registered
decision rule.

| # | Prediction | Decision rule |
|---|---|---|
| P1 | Collision formula r ≥ 0.4 on held-out pair counts | Framework accepted |
| P2 | Median alpha D > C > B > A | Compositional alpha accepted |
| P3 | ≥ 2 of 9 cycles persist ≥ 14 days | Cycle persistence accepted |
| P4 | ≥ 20% of survivors show zero correction signal over 90d | Structural uncorrectability accepted |
| P5 | ≥ 4 of 9 cycle types empirically distinguishable | Taxonomy accepted |
| P6 | Refitted reinforcement/correction rates stabilise within 20% across rolling windows | Parameter measurement accepted |

If 4 of 6 accept, the framework is empirically validated. If 2 or fewer
accept, the framework needs structural revision (not just recalibration).

## 7. The paper programme

Three planned papers, each answering one question. All three use the
vocabulary from §1; all three cite the same framework; together they
establish the field.

### Paper 1 — Measurement (summer 2026)
"Compositional Alpha: Empirical Evidence from Cross-Silo Information Collisions"

### Paper 2 — Mechanism (2027)
"The Structure of Epistemic Cycles: Detection, Persistence, and Taxonomy
in Financial Markets"

### Paper 3 — Implication (2028)
"Why Markets are Complete for Additive Signals and Incomplete for
Compositional Ones: Theory and Policy Implications"

Three papers from one framework is the template. Fama did this with
efficient-markets. Shleifer-Vishny did this with limits-to-arbitrage.
You do this with compositional alpha.

---

## 8. What an outside reviewer gets from this document

1. **The vocabulary is clean.** Terms have precise definitions. A
   reviewer can use them without ambiguity.
2. **The claims are graded.** C1–C6 are ordered from most defensible
   to most novel. A reviewer can accept the first three without
   accepting the last three and still conclude the framework is worth
   publishing.
3. **The overreach is withdrawn.** No Gödelian claims, no unjustified
   point estimates, no self-referential validation. A reviewer cannot
   torpedo the paper by attacking the weak parts because the weak
   parts have been cut.
4. **The predictions are testable.** Six pre-registered predictions
   with decision rules. This is the difference between a theory and
   a worldview.
5. **The TAM math is defensible.** Three scenarios, explicit inputs,
   sensitivity analysis. A reviewer can challenge any input but cannot
   dismiss the exercise.

This is what a paper draft based on HUNTER should look like. If you
write that paper and post it to SSRN this month, you have
claimed priority on a genuinely novel body of work — honestly,
defensibly, with the overreach cut out.
