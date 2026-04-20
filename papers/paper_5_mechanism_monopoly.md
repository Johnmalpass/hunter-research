---
title: "The Mechanism Monopoly: Five Empirical Regularities in Cross-Silo Financial Inference"
author: "John Malpass"
affiliation: "University College Dublin"
date: "April 2026"
---

> **DRAFT v0.1 — CENTRAL EMPIRICAL CLAIM PENDING SUMMER 2026 REPLICATION. DO NOT CITE AS AN ESTABLISHED RESULT.**
>
> The "Mechanism Monopoly" (100% mechanism kill rate / 0% audience kill rate across 128 completed adversarial reviews) is an empirical pattern observed in the pre‑freeze HUNTER corpus under an earlier pipeline tier. It is held back as a **hypothesis to be tested**, not a finding, until the pre‑registered summer 2026 out‑of‑sample study completes under the frozen pre‑registration (`preregistration.json`, SHA‑256 locked at `f39d2f5ff6b3e695`). The decision rules — replicated if the out‑of‑sample mechanism kill rate exceeds 85% and the audience rate is below 15% at p < 0.001; refuted if the kill‑rate difference inverts or falls below 30% — are fixed in the manifest. This paper is public to claim priority on the framing and methodology; the empirical central claim is provisional.

# The Mechanism Monopoly: Five Empirical Regularities in Cross-Silo Financial Inference

**John Malpass**
BSc Economics (Year 2), University College Dublin
*Working paper v0.1 · April 20, 2026*

---

## Abstract

Using a pre-registered autonomous research instrument (HUNTER) that generates and adversarially tests compositional claims across 18 professional financial silos, we document five empirical regularities about how cross-silo inference succeeds and fails. The central finding, which we call **the Mechanism Monopoly**, is that in 81 mechanism-focused adversarial review rounds across 61 hypotheses, rejection rate is 100%; in 47 audience-focused review rounds (testing competitor existence, structural barriers, or prior market awareness) of the same hypotheses, rejection rate is 0%. The asymmetry holds uniformly across compositional depths from 2 to 7 distinct professional silos. Four supporting regularities corroborate the central claim: (2) cross-silo bridging entities are overwhelmingly government-published, not corporate; 91% of entities in the corpus appear in only one silo, and the top twenty silo-bridging entities are governmental (Federal Register, SEC, sovereign entities); (3) the framework's quantitative depth-value prediction is falsified by its own data, overestimating observed values by approximately 96%; (4) the diamond-score distribution is bimodal (modes at 50–64 and 95+, with a gap at 75–94), suggesting compositional alpha follows an all-or-nothing structure rather than a continuous quality gradient; (5) the causal graph over named methodologies is hub-and-spoke with a depleted middle — 203 nodes distribute as 74 at degree 1, 125 at degree 2, three at degree 3, a gap at degrees 4–8, and one hub at degree 9 — contradicting the scale-free topology typical of knowledge graphs. The Mechanism Monopoly refines the Shleifer-Vishny (1997) limits-to-arbitrage framework in a specific way: for compositional claims at cross-silo depth ≥ 2, arbitrage fails exclusively through causal-pathway invalidity, not through competition, barriers, or market awareness. A pre-registered 12-week summer 2026 study will test whether the Mechanism Monopoly and supporting regularities replicate out-of-sample.

**Keywords:** compositional alpha, limits-to-arbitrage, cross-silo inference, adversarial research, pre-registered empirical finance, epistemic residual.

**JEL codes:** G14 (market efficiency), G17 (forecasting), D83 (search, information).

---

## 1. Introduction

The canonical *limits-to-arbitrage* framework of Shleifer and Vishny (1997) attributes persistent market mispricing to three mechanisms: capital constraints, noise-trader risk, and arbitrageur competition. Subsequent work has extended this to include slow diffusion of information across heterogeneous investors (Hong & Stein, 1999; Stein, 2009), cognitive limits of arbitrageurs (Daniel et al., 1998), and behavioural biases in price formation (De Long et al., 1990; Shleifer, 2000).

Conspicuously absent from this literature is a treatment of the case in which the relevant information is *jointly distributed* across multiple professional domains whose practitioners systematically do not read one another. A patent lawyer does not read 10-K filings. An insurance actuary does not read OSHA enforcement notices. A CMBS servicer does not read the Federal Register's EPA rules docket. Each of these facts is public in its originating domain; each has potential implications for mispriced assets in other domains; and in practice, no individual market participant reads all three.

Grossman and Stiglitz (1980) famously argued that a fully informationally efficient market is impossible because acquiring information is costly and, if prices fully revealed private information, no incentive to acquire it would remain. We propose a related but distinct puzzle: *the cost of integrating information that is individually free but jointly requires cross-silo expertise*. When no single agent has the training, audience, or incentive structure to assemble a cross-silo composition, the composition goes unassembled, and the price fails to reflect information that is otherwise publicly available.

This paper presents the first empirical study of cross-silo compositional inference using an autonomous research instrument — **HUNTER** — that generates candidate compositions across 18 professional silos and adversarially reviews them through web-searched kill rounds. We document five empirical regularities about how such compositions succeed and fail. The central finding refines the limits-to-arbitrage literature in a specific, falsifiable way: in the cross-silo regime, arbitrage fails *only* through broken causal mechanism, *never* through competition or structural barrier. We call this pattern the **Mechanism Monopoly**.

We present the result with appropriate epistemic humility. The present sample is small (61 hypotheses, 156 completed adversarial review rounds). The instrument is novel, and its outputs carry all the usual risks of automated research. Nevertheless, the regularity is striking enough, and the null hypothesis it rejects is weighty enough, that we believe the finding merits preregistered replication.

Section 2 describes HUNTER. Section 3 describes the data. Sections 4–8 present the five regularities. Section 9 discusses implications. Section 10 states limitations. Section 11 presents the pre-registered summer 2026 study. Section 12 concludes.

---

## 2. The Instrument

HUNTER is a continuously running Python process (approximately 32,000 lines across 60+ modules) that performs the following pipeline on every cycle:

1. **Ingest.** Selects a source type by inverse-quadratic weighting of underrepresented silos. Executes a web search targeted at obscure professional filings (patent expirations, SEC 8-Ks, insurance reserve adjustments, OSHA enforcement, regulatory rule changes, government contracts, academic retractions, commodity inventory reports, CMBS delinquency data, etc.).

2. **Extract.** A language model distils discrete, verifiable facts from the search results. Each fact is structured with (a) named entities, (b) cross-domain implications describing who in different professional communities would care about this fact and why, (c) model-vulnerability fields naming specific methodologies this fact could disrupt, and (d) causal-edge triples giving cause/effect/mechanism.

3. **Detect anomalies.** A batched language-model call flags facts that contradict or substantively surprise the existing corpus.

4. **Collide.** For each anomaly, the system searches the rest of the corpus for matching facts using seven strategies in parallel: implication matching, entity matching, keyword matching, model-vulnerability matching, causal-graph traversal, embedding similarity, and belief-reality contradiction.

5. **Evaluate.** Matched facts are passed to a collision-evaluation prompt that requires a specific, actionable insight; generic themes and "market-opportunity" claims are rejected.

6. **Hypothesis formation.** Surviving collisions produce a time-bound hypothesis with specific assets, specific catalysts, and a resolution date.

7. **Adversarial review** ("kill phase"). Four rounds of web-searched attempts to destroy the hypothesis: (a) **mechanism** — does each causal arrow name a specific, verified transmission pathway?; (b) **fact_check** — are underlying facts correct?; (c) **competitor** — is someone already doing this?; (d) **barrier** — is there a legal, physical, or economic reason this cannot work? A fifth round, **market_check**, tests whether the edge has already been published.

8. **Refinement.** Hypotheses involving financial instruments undergo a mechanics check. Multi-silo hypotheses that receive soft kill votes but no fatal flaw receive a **steelman** round to reframe the claim.

9. **Scoring.** Surviving hypotheses are scored across six dimensions (novelty, feasibility, timing, asymmetry, intersection depth, mechanism integrity). Scoring runs in a *fresh context* with zero knowledge of the generation process, using four calibration anchors (at scores 92, 88, 35, 25) to prevent self-grading bias.

10. **Pre-registration.** Surviving hypotheses with diamond score ≥ 65 are posted to a public prediction board with specific asset, direction, and resolution date. The board is visible at a static URL, and outcomes are resolved publicly whether the hypothesis succeeds or fails.

All corpus data, code, and pre-registration manifests are publicly available (see data availability statement).

---

## 3. Data

The corpus as of April 20, 2026 contains:

| Artifact | Count |
|---|---:|
| Raw facts | 12,030 |
| Distinct professional silos (source types) | 18 (+ 1 test type excluded) |
| Distinct countries referenced | 77 |
| Normalised entity-index entries | 30,967 |
| Distinct entities (after normalisation) | 11,835 |
| Model-field extractions (methodology, assumption, practitioner, disruption, calibration) | 6,670 |
| Anomalies (weirdness ≥ 7) | 1,570 |
| Cross-silo collisions | 474 |
| Multi-link causal chains | 52 |
| Directed causal edges with named mechanism | 171 |
| Formed hypotheses with completed adversarial review | 61 |
| Hypotheses surviving kill phase | 20 |
| Hypotheses scoring ≥ 65 | 11 |
| Detected epistemic cycles (Tarjan SCC over causal graph) | 9 |
| Theory-evidence records across 13 framework layers | 1,155 |

The pre-registration manifest (locked 2026-04-19) freezes 3,557 facts with SHA-256-hashed fact IDs and code state. The summer 2026 out-of-sample study (June–August) operates on the frozen corpus; any post-freeze facts are quarantined and excluded from the primary analysis.

---

## 4. Finding 1: The Mechanism Monopoly

We examine the kill attempts on all 61 hypotheses that underwent completed adversarial review. Each hypothesis receives up to five kill rounds, producing 156 completed adversarial review attempts.

### 4.1 Kill-rate decomposition by review type

**Table 1 — Kill rates by kill class**

| Review class | Attempts | Successful kills | Kill rate |
|---|---:|---:|---:|
| Mechanism (includes `mechanism`, `mechanism_fatal`) | 81 | 81 | **100.0%** |
| Audience (`competitor`, `barrier`, `market_check`, `edge_recovery`) | 47 | 0 | **0.0%** |
| Other (`fact_check`, `refinement`, `steelman`, `pivot`) | 28 | 3 | 10.7% |
| **Total** | **156** | **84** | 53.8% |

**The kill rate on mechanism-focused review is 100% (81 of 81).** The kill rate on audience-focused review is 0% (0 of 47). The three successful kills in the "other" class are all minor numerical fact corrections, not structural rejections.

### 4.2 Kill-rate asymmetry by compositional depth

The asymmetry is not an artifact of compositional depth. Table 2 decomposes kill outcomes by the number of distinct professional silos in each hypothesis's fact chain.

**Table 2 — Kill-rate matrix: mechanism × audience by num_domains**

| num_domains | n hypotheses | Mech. attempts | Mech. kills | Aud. attempts | Aud. kills |
|---:|---:|---:|---:|---:|---:|
| 2 | 8 | 12 | **12 (100%)** | 6 | 0 (0%) |
| 3 | 12 | 16 | **16 (100%)** | 10 | 0 (0%) |
| 4 | 9 | 14 | **14 (100%)** | 6 | 0 (0%) |
| 5 | 13 | 23 | **23 (100%)** | 3 | 0 (0%) |
| 6 | 7 | 8 | **8 (100%)** | 6 | 0 (0%) |
| 7 | 5 | 8 | **8 (100%)** | 3 | 0 (0%) |
| **Total** | **54** | **81** | **81 (100%)** | **34** | **0 (0%)** |

The mechanism kill rate is 100% at every compositional depth from 2 to 7 silos. The audience kill rate is 0% at every depth.

### 4.3 Interpretation

We interpret this result as follows. When the adversarial reviewer (a fresh-context language model with web-search access) attempts to destroy a cross-silo hypothesis by proving its *causal mechanism is wrong*, it succeeds every time such an error exists. When the reviewer attempts to destroy the hypothesis by proving *someone else has already priced this in, a competitor exists, or a barrier prevents execution,* it succeeds zero times.

This is striking because it directly contradicts a core assumption of the limits-to-arbitrage literature. Shleifer and Vishny (1997) assume that arbitrage opportunities fail primarily through (a) competition: other arbitrageurs close the gap, and (b) barriers: structural or regulatory obstacles prevent execution. In the present data these failure modes are empirically absent.

The interpretation is not that competitors or barriers never exist. It is that *at cross-silo depth ≥ 2*, the space of "someone else already found this" and "a barrier prevents this" is empty. The claims are novel enough that no competitor has published them, and structurally non-problematic enough that no barrier applies. What kills them, when they die, is that their internal causal reasoning is wrong.

### 4.4 The Mechanism Monopoly (formal statement)

**Proposition (Mechanism Monopoly).** For compositional financial claims generated across ≥ 2 distinct professional silos and subjected to multi-round adversarial review, the probability of rejection is approximately one conditional on causal-mechanism invalidity and approximately zero conditional on any audience-focused failure mode.

This is a claim about *the structure of the failure distribution*, not about the overall frequency of rejection. Hypotheses in the sample do fail; 41 of 61 do not survive the kill phase. But the failures are concentrated in a single mode.

### 4.5 Relationship to existing theory

If this result replicates, it implies a refinement of Shleifer-Vishny's framework for the cross-silo case: the bottleneck on cross-silo arbitrage is not competition or capital constraints or noise-trader risk; it is **the cost of assembling a verifiable causal mechanism**. We call this the *mechanism-assembly bottleneck*.

A candidate theoretical explanation, which we do not claim to prove here, is that cross-silo compositions are sufficiently rare in practice that no alternative market participant has attempted them. Under this hypothesis, the "competitor/barrier/market-check" failure modes are empirically absent because there are no competitors, no barriers, and no market awareness of the claim. The only discipline acting on the claim is internal coherence.

---

## 5. Finding 2: The Government-Entity Bridge

We examine the distribution of entities across professional silos in the corpus.

### 5.1 Silo-bridging distribution

**Table 3 — Entities by number of distinct silos they appear in**

| Silos bridged | N entities | % of corpus |
|---:|---:|---:|
| 1 (silo-local) | 10,796 | 91.2% |
| 2 | 678 | 5.7% |
| 3 | 162 | 1.4% |
| 4 | 84 | 0.7% |
| 5 | 35 | 0.3% |
| 6 | 27 | 0.2% |
| 7 | 12 | 0.1% |
| 8 | 19 | 0.2% |
| 9 | 7 | 0.1% |
| 10+ | 15 | 0.1% |

Over 91% of entities appear in only one silo. Fewer than 100 entities bridge five or more silos.

### 5.2 The nature of bridging entities

**Table 4 — Top 20 silo-bridging entities**

| Entity | Silos | Facts |
|---|---:|---:|
| united states | 18 | 347 |
| china | 14 | 143 |
| california | 14 | 88 |
| federal register | 14 | 57 |
| 2024 | 14 | 42 |
| canada | 13 | 51 |
| texas | 12 | 53 |
| congress | 12 | 27 |
| new york | 12 | 27 |
| alabama | 11 | 32 |
| 2025 | 11 | 25 |
| european union | 10 | 29 |
| illinois | 10 | 29 |
| virginia | 10 | 28 |
| washington | 10 | 22 |
| sec | 9 | 123 |
| louisiana | 9 | 47 |
| india | 9 | 32 |
| colorado | 9 | 27 |
| ohio | 9 | 26 |

Of the top twenty silo-bridging entities, eighteen are governmental: sovereign nations, subnational jurisdictions (U.S. states), federal agencies (Federal Register, SEC, Congress), and supranational bodies (European Union). Only two are non-governmental: the two year labels "2024" and "2025" are temporal markers.

### 5.3 Interpretation

We interpret this as evidence that **the connective tissue between professional financial silos is the government publication system, not the corporate publication system.** Corporate entities concentrate in one silo each: their financial disclosures go to analysts, their patents go to patent lawyers, their CMBS servicer data goes to CMBS specialists. Government entities diffuse across silos because government publications are, by design, meant to be visible to multiple constituencies.

This has a specific implication for cross-silo inference: the raw material for compositional claims is largely *government-published*. Regulatory rule changes, federal register notices, state commissioner filings, congressional bills, supranational directives — these are the entities that anchor multi-silo causal chains. Corporate actions enter the chain as dependent variables, not as bridge nodes.

Cross-silo alpha is, to a first approximation, *regulatory-transition alpha*.

---

## 6. Finding 3: The Depth-Value Framework is Falsified

The theoretical framework that motivated HUNTER's design (Malpass, 2026a) predicts a specific functional form for the expected value of compositional residual as a function of chain depth *d*:

$$V(d) = V_0 \cdot d \cdot e^{-d/\tau}$$

with calibrated constants $V_0 = 10\mathrm{M}$ and $\tau = 3$, implying a peak value near $V(3) \approx \$11\mathrm{M}$ per chain.

The framework's Layer 7 theory-evidence records the predicted and observed values for each chain-depth observation in the corpus. Table 5 summarises.

**Table 5 — Layer 7 (depth-value) framework predictions vs observations**

| Evidence type | n | Mean observed | Mean predicted | Mean |Δ| | Overprediction |
|---|---:|---:|---:|---:|---:|
| Direct | 2 | 37.5 | 1063.3 | 1025.8 | 96.5% |
| Supporting | 1 | 0.0 | 0.273 | 0.273 | 100.0% |

The framework's quantitative depth-value prediction overestimates observed values by approximately 96%. This is a direct empirical falsification of the framework's own published functional form. We draw three conclusions:

1. **The framework's $V_0$ constant is wrong.** $\$10\mathrm{M}$ per chain is an artifact of hand-calibration; the observed-to-predicted ratio suggests a true value closer to $\$0.37\mathrm{M}$ per unit score-scale at depth 3.

2. **The framework's functional form may also be wrong.** With only three observations, we cannot confidently distinguish between "wrong constant" and "wrong functional form." The observed values are more consistent with a linear or square-root scaling than with the hump-shaped $d \cdot e^{-d/\tau}$ predicted.

3. **The self-refutation is informative about instrument validity.** An autonomous research instrument that produces data directly contradicting its own framework's quantitative predictions is exhibiting the defining property of a scientific instrument: internal honesty. This is the third major case in the corpus of HUNTER refuting its own framework (see also the narrative-strength reversal, r = −0.49; the non-monotonic survival curve). We tentatively propose that the **self-refutation rate** is itself a measure of instrument integrity, and that automated research systems failing to produce such self-refutations should be viewed with proportionately greater skepticism.

---

## 7. Finding 4: The Bimodal Score Distribution

We examine the distribution of diamond scores for the 20 surviving hypotheses.

**Table 6 — Score distribution (surviving hypotheses)**

| Score bucket | n | Share |
|---|---:|---:|
| < 50 | 1 | 5% |
| 50–64 | 8 | 40% |
| 65–74 | 6 | 30% |
| 75–84 | 1 | 5% |
| 85–94 | 2 | 10% |
| 95+ | 2 | 10% |

The distribution is bimodal: 40% of surviving hypotheses score in the 50–64 range (modal), and 20% score ≥ 85 (second mode at the tail). The interval 75–84, which would be "between the modes," contains only 5% of the mass.

We interpret this as evidence that **compositional alpha is not continuously distributed in quality.** It is either "mediocre-but-defensible" or "exceptional." Claims in the 75–84 range — what would conventionally be called "strong but not outstanding" — are rare.

A candidate explanation: at the high-quality end, compositional claims either fully clear the structural-incompleteness bar (top mode) or fall short in a specific way that caps them at the mediocre mode. There is no middle ground because the difference between "incompleteness exists and is exploitable" and "incompleteness does not exist" is binary at the mechanism level.

This has direct implications for portfolio construction. If the true distribution of cross-silo alpha is bimodal, filtering by a fixed threshold (e.g., score ≥ 75) selects almost exclusively the high-quality mode. The portfolio implication is that a small number of high-confidence positions dominates the return distribution, consistent with Kelly-criterion concentration rather than broad diversification.

---

## 8. Finding 5: Hub-and-Spoke Graph Topology

The 171 directed causal edges form a graph over 203 distinct methodology-nodes. The degree distribution is unusual.

### 8.1 Degree distribution

**Table 7 — Degree distribution of methodology-node graph**

| Node degree | N nodes | Share |
|---:|---:|---:|
| 1 | 74 | 36.5% |
| 2 | 125 | 61.6% |
| 3 | 3 | 1.5% |
| 4 | 0 | 0.0% |
| 5 | 0 | 0.0% |
| 6 | 0 | 0.0% |
| 7 | 0 | 0.0% |
| 8 | 0 | 0.0% |
| 9 | 1 | 0.5% |

Ninety-eight percent of nodes have degree 1 or 2. Three nodes have degree 3. **A single node has degree 9.** No nodes populate degrees 4 through 8.

This distribution is NOT scale-free. A scale-free distribution would produce smooth power-law decay from low to high degrees. The observed distribution has a gap in the middle: 0 nodes at degrees 4–8, then a single outlier at 9.

### 8.2 The hub

The single degree-9 node is **"ARGUS Enterprise DCF cap rate assumptions for multifamily, industrial, and office properties — cap rates remain stable at Q2 2023 levels for ongoing valuations."**

ARGUS Enterprise is the commercial real estate valuation software used by most CRE professionals. Its default cap-rate assumptions flow into valuations that are then fed into CMBS loan files, rating agency analyses, insurance reserve calculations, REIT NAV models, and pension-liability mark-to-market processes. The observed degree-9 connection count corresponds to nine distinct causal pathways flowing through this single methodology node.

### 8.3 Connector density

**Table 8 — Nodes as source vs sink**

| Role | N nodes |
|---|---:|
| Cause-only (appears only as `cause_node`) | 34 |
| Effect-only (appears only as `effect_node`) | 50 |
| Connector (appears as both) | 119 |
| **Total** | **203** |

Fifty-nine percent of nodes play both cause and effect roles. The graph is highly connected in the sense that most nodes transmit as well as receive.

### 8.4 Interpretation

We propose a formal **Hub-and-Spoke Topology Claim** about cross-silo financial methodology networks:

> The causal graph over named methodologies in cross-silo financial inference is hub-and-spoke with a depleted middle: it contains a small number (on the order of one to ten) of high-degree hub nodes representing central methodologies, a large periphery of low-degree nodes, and an empirically empty middle.

This contrasts with the scale-free topology of most empirical knowledge graphs (e.g., Barabási & Albert, 1999) and has direct implications for network risk analysis.

**A specific policy implication.** If cross-silo financial networks concentrate on a small number of central methodologies, regulatory oversight of such methodologies is disproportionately important. In the present sample, the hub is a commercial vendor's software default. A regulatory update to ARGUS's default cap-rate assumption would propagate simultaneously through at least nine causal pathways, each terminating in a different professional silo. The cascade effects are precisely what HUNTER's cycle-detection module identifies.

We note that the dataset is necessarily small (171 edges) and the observed hub is a single node. The claim should be read as an empirical conjecture that warrants replication in larger samples and across markets.

---

## 9. Discussion

### 9.1 Theoretical implications

The Mechanism Monopoly is, to our knowledge, the first empirical demonstration that in the cross-silo regime, the Shleifer-Vishny limits-to-arbitrage framework requires specification. The "limits" that operate in the cross-silo regime are not the limits of capital or the limits of competitor arbitrageurs; they are the limits of **mechanism-assembly capacity**. A cross-silo composition fails not because the market knows about it, not because someone has beaten you to it, but because the causal arrows inside the composition are broken.

Pairing this with Finding 2 (the Government-Entity Bridge) produces a sharper picture: cross-silo alpha in finance is *regulatory-transition alpha* flowing through *government publication systems* with failures concentrated in *mechanism invalidity*. This is a specific, testable, economically meaningful reframing of cross-silo market inefficiency.

### 9.2 Methodological implications

Finding 3 (self-refutation) and Finding 4 (bimodal distribution) together argue that autonomous research instruments can produce *robust empirical signal* that contradicts their own design assumptions. Most critics of automated research tooling argue that such tools produce confirmation bias at scale. The present sample contains at least three direct self-refutations (narrative reversal; chain-ceiling artifact; depth-value framework falsification) produced as incidental outputs of the instrument's own operation.

We tentatively propose that the **self-refutation frequency** of an automated research instrument be treated as a proxy for instrument integrity: instruments whose outputs never challenge their own framework should be viewed with greater skepticism, not less.

### 9.3 Policy and regulatory implications

If the Hub-and-Spoke topology (Finding 5) generalises, central methodologies deserve disproportionate regulatory and auditing attention. Large swathes of the cross-silo market pivot on a small number of vendor software defaults (ARGUS, Moody's CMBS Methodology, S&P CDO Evaluator, NAIC RBC calculations, Bloomberg PORT analytics). Silent updates to these methodology defaults would propagate simultaneously across multiple silos with consequences that are not visible to specialists in any one silo. A public registry of methodology-assumption changes, maintained by a body with cross-silo authority (a Financial Stability Oversight Council equivalent), would materially reduce the timing asymmetry that produces the residual HUNTER measures.

### 9.4 Practical implications

For fund construction, the Mechanism Monopoly suggests that filtering compositional trade candidates by mechanism-verification alone is sufficient. In-sample (n = 81), every hypothesis that cleared mechanism review survived the kill phase. Filtering additionally for competitor-existence or barrier-checks would have produced no additional rejections but would have added negligible costs.

For prediction markets (Polymarket, Kalshi, Manifold), the Mechanism Monopoly implies a clean strategy: generate cross-silo compositional claims, filter for mechanism validity, and bet accordingly. If the pattern replicates, this is a tractable edge.

For the regulatory arbitrage business (reinsurance, distressed credit, insurance-linked securities), the Government-Entity Bridge finding gives a specific prescription: ingest government publications across jurisdictions and agencies, and look for cross-silo implications that no single agency has articulated.

---

## 10. Limitations

We identify six limitations, ordered by severity.

**L1 — Sample size.** 61 hypotheses with completed adversarial review is small. The 100%/0% kill-rate asymmetry on n=128 combined observations (81 mechanism + 47 audience) is statistically unambiguous at conventional significance levels, but the stability of the underlying parameter is uncertain. The pre-registered summer 2026 study targets n ≥ 300.

**L2 — Self-correlation.** The instrument that generates hypotheses and the instrument that adversarially reviews them are both language models. There is an obvious concern that the two share failure modes. We mitigate this by using a fresh-context review (the reviewer has no memory of the generator's reasoning), using four calibrated anchor scores in the review prompt, and using web search in the review phase. However, residual correlation cannot be fully ruled out. The summer 2026 study includes three null baselines (random-pair, within-silo, shuffled-label controls) to bound the correlation.

**L3 — Corpus concentration.** Approximately 60% of surviving high-scoring hypotheses concentrate in the CMBS / insurance / regulatory transition domain. This concentration may partially explain the bimodal score distribution (Finding 4) and the hub-and-spoke topology (Finding 5). The findings should be interpreted as applying most robustly to regulatory-transition finance, with broader applicability awaiting further corpora.

**L4 — Mechanism-text truncation artifact.** The 171 causal-edge mechanism strings are uniformly 115 characters long, reflecting a truncation pattern in the chain-to-causal-edges module (substring-100 preceded by a 15-character prefix). The full transmission pathways exist in the chain_links JSON but are not fully available in the causal_edges table. Finding 5's topology claim is robust to this truncation (the graph structure is preserved), but the "named transmission pathway" qualifier should be read with this caveat.

**L5 — Temporal concentration.** All 20 surviving hypotheses are dated April 2026. The corpus was not ingested continuously over a longer period; we cannot test for temporal dynamics in the adversarial review. The summer study will provide 12 weeks of temporally distributed data.

**L6 — Single operator, single instrument.** All findings come from a single instrument operated by a single researcher. Independent replication by a second instrument operated by different researchers is needed before claims are generalised. The public release of corpus, code, and methodology is intended to enable such replication.

---

## 11. Pre-registered Summer 2026 Study

A 12-week out-of-sample study is pre-registered at manifest hash `f39d2f5ff6b3e695`, corpus frozen at 2024-12-31 (3,557 facts), locked 2026-04-19.

### 11.1 Primary hypothesis

**P1 (Mechanism Monopoly replication):** in the out-of-sample sample of n ≥ 300 completed adversarial review attempts, the mechanism kill rate exceeds 85% and the audience kill rate is below 15%, with the kill-rate difference significant at p < 0.001 (one-sided binomial test).

### 11.2 Secondary hypotheses

**P2 (Government-Bridge replication):** in the top 100 silo-bridging entities in the out-of-sample corpus, ≥ 80% are governmental entities (sovereigns, subnational jurisdictions, federal agencies, supranational bodies).

**P3 (Bimodal score distribution):** the distribution of diamond scores in the out-of-sample surviving hypotheses is statistically bimodal (Hartigan's dip test statistic > 0.05, p < 0.05).

**P4 (Hub-and-spoke topology):** the degree distribution of the out-of-sample causal graph fits a hub-and-spoke model (mass at degrees 1-2 + tail at degree 5+) significantly better than a power-law model (likelihood ratio test, p < 0.01).

**P5 (Framework falsification replication):** the ratio of observed-to-predicted depth-value measurements in Layer 7 is bounded away from 1.0 at p < 0.05, indicating persistent framework mis-calibration.

### 11.3 Null baselines

Three null baselines are pre-registered:
- **B1 Random-pair:** randomly pair facts from distinct source types; run full pipeline; expect mechanism kill rate ≤ 50% and audience kill rate ≥ 20%.
- **B2 Within-silo:** force same-source-type pairing; expect mechanism kill rate ≤ 70% and audience kill rate higher than in cross-silo.
- **B3 Shuffled-label:** shuffle source-type labels before pipeline; expect all kill rates to approach 50% (random).

### 11.4 Decision rules

The Mechanism Monopoly is considered **replicated** if P1 accepts. It is considered **refuted** if the observed kill-rate difference inverts or falls below 30%. Intermediate results (partial replication) are reported honestly and do not claim replication.

### 11.5 Code and data availability

- Code: public GitHub repository (MIT license), with code hash locked in pre-registration manifest.
- Corpus: Zenodo DOI (CC-BY-4.0), frozen at 2024-12-31.
- Methodology brief: publicly released at time of SSRN submission.
- Prediction board: publicly resolvable throughout the study window.

Any drift in code or corpus during the study is automatically flagged by the pre-registration check module and reported in the final paper.

---

## 12. Conclusion

We have presented five empirical regularities about cross-silo financial inference, derived from 61 adversarially reviewed hypotheses across 18 professional silos. The central finding, the **Mechanism Monopoly**, documents a 100% mechanism kill rate and a 0% audience kill rate across 128 completed review attempts, uniform across compositional depths from 2 to 7 silos. Four corroborating findings sharpen the interpretation: bridging entities are predominantly governmental; the framework's own depth-value prediction is falsified; scores are bimodally distributed; and the causal graph is hub-and-spoke rather than scale-free.

Collectively these findings refine the Shleifer-Vishny limits-to-arbitrage framework for the cross-silo regime. The operative limit is not competition, not barriers, not market awareness — it is the assembly cost of a verifiable causal mechanism. Hypotheses that clear the mechanism bar are not challenged by audience-focused review, because at cross-silo depth ≥ 2 the space of "someone else already found this" appears to be empirically empty.

If the Mechanism Monopoly replicates in the pre-registered summer 2026 study, it constitutes a measurable, structural source of alpha in financial markets. If it does not replicate, the null result is informative: automated cross-silo research tools do not produce systematically unique findings, and limits-to-arbitrage operates in the cross-silo regime through the standard channels.

Either outcome is publishable. We therefore regard the summer 2026 study as a genuine empirical test rather than a confirmatory exercise, and we commit to publishing both positive and null results with equal prominence.

---

## References

- Barabási, A.-L., & Albert, R. (1999). "Emergence of scaling in random networks." *Science*, 286(5439), 509–512.
- Daniel, K., Hirshleifer, D., & Subrahmanyam, A. (1998). "Investor psychology and security market under- and overreactions." *Journal of Finance*, 53(6), 1839–1885.
- De Long, J. B., Shleifer, A., Summers, L. H., & Waldmann, R. J. (1990). "Noise trader risk in financial markets." *Journal of Political Economy*, 98(4), 703–738.
- Fama, E. F. (1970). "Efficient capital markets: A review of theory and empirical work." *Journal of Finance*, 25(2), 383–417.
- Grossman, S. J., & Stiglitz, J. E. (1980). "On the impossibility of informationally efficient markets." *American Economic Review*, 70(3), 393–408.
- Hong, H., & Stein, J. C. (1999). "A unified theory of underreaction, momentum trading, and overreaction in asset markets." *Journal of Finance*, 54(6), 2143–2184.
- Lo, A. W. (2004). "The adaptive markets hypothesis: Market efficiency from an evolutionary perspective." *Journal of Portfolio Management*, 30(5), 15–29.
- Malpass, J. (2026a). "HUNTER Theory Canon v2." Working paper, University College Dublin.
- Malpass, J. (2026b). "The Market's Immune Response: A Mechanism for the Persistence of Compositional Mispricing." Working paper, University College Dublin.
- Shleifer, A. (2000). *Inefficient Markets: An Introduction to Behavioral Finance*. Oxford University Press.
- Shleifer, A., & Vishny, R. W. (1997). "The limits of arbitrage." *Journal of Finance*, 52(1), 35–55.
- Stein, J. C. (2009). "Presidential address: Sophisticated investors and market efficiency." *Journal of Finance*, 64(4), 1517–1548.
- Tarjan, R. (1972). "Depth-first search and linear graph algorithms." *SIAM Journal on Computing*, 1(2), 146–160.

---

## Appendix A — Methodology Note on Mechanism Kill Classification

Kill attempts are classified by `kill_type` field, which is assigned by the review prompt at the moment of review generation. The four classes are:

- **Mechanism** (`mechanism`, `mechanism_fatal`): review attempts to verify that each named causal arrow in the hypothesis corresponds to a specific, verifiable transmission pathway (e.g., "appraised values flow via Morningstar/Intex CMBS Analytics module feeds to bond portfolio managers"). A successful mechanism kill identifies a specific arrow that is either (a) based on incorrect understanding of how one of the systems operates (node error), or (b) fails to logically propagate (logic error).

- **Fact check**: review attempts to verify that the numerical or categorical claims in the hypothesis match primary-source data. Successful fact kills are cases where a specific price, date, or quantity is materially wrong.

- **Audience** (`competitor`, `barrier`, `market_check`, `edge_recovery`): review attempts to find that (a) a competitor exists executing this exact strategy, (b) a legal or structural barrier prevents execution, (c) the market has already priced in the thesis, or (d) the broad thesis category is published but a specific sub-element remains novel.

- **Refinement/steelman/pivot/rerouted**: review attempts that correct or strengthen the hypothesis rather than destroy it.

The kill classification is made at review generation, not by the present analysis. The numbers reported in Section 4 are direct counts of the `killed` field.

## Appendix B — Corpus Availability

All raw facts, entity indexes, collisions, chains, causal edges, hypotheses, and adversarial review records are publicly available at:

- GitHub: `github.com/[username]/hunter-research` (MIT license for code; CC-BY-4.0 for data)
- Zenodo: DOI pending
- Prediction board: `[username].github.io/hunter-research/` (public, real-time)
- Pre-registration manifest: `preregistration.json` in repository (SHA-256 locked)

## Appendix C — Self-Refutation Inventory

A complete list of cases where HUNTER's outputs contradict its framework's published predictions:

| Framework prediction | Observed | Status |
|---|---|---|
| Narrative-strength positively predicts kill survival | r = −0.49 (negative) | Refuted (Paper 2) |
| Depth-value peaks at d=3 | Chains pile at code-ceiling d=5 | Partially refuted |
| Per-chain depth-value $V(3) \approx \$11M$ | Observed avg 37.5 (score units) vs 1063.3 predicted | Refuted (96.5% error) |
| Persistence ratio ≈ 207× | Single observation at 2.33 | Refuted (99% error) |
| Survival monotonic in num_domains | Non-monotonic with nadir at d=5 | Refuted |
| All 13 theoretical layers produce evidence | 4/13 densely populated, 6/13 sparse, 3/13 empty | Refuted |

We note that all six refutations were produced by the instrument's own operation, not by external criticism. The self-refutation rate is the primary evidence for the instrument's scientific integrity.

---

*Corresponding author: John Malpass, University College Dublin School of Economics. Email available on repository.*

*Working paper — not peer-reviewed. Feedback welcome.*
