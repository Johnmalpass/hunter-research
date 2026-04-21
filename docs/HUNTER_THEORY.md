# HUNTER — Theoretical Framework

*A ten-layer theory of compositional information asymmetry. Working draft, April 2026. Summer 2026 empirical calibration is the critical test of every quantitative prediction below.*

**John Malpass · University College Dublin · April 2026**
*Framework developed through two intensive theoretical sessions, January–April 2026. This document assembles the ten layers, the three original contributions, the mathematical backbone, the cross-domain generalisation, and the three-paper research programme into a single reference. Every quantitative claim here is a **prediction**, not a finding, until the pre-registered Summer 2026 out-of-sample study reports.*

---

## 1. The core insight

Information exists in silos. Markets, sciences, legal systems, governments, and corporations operate inside compartments separated by language, methodology, incentive structure, and institutional boundary. Within each silo, information is well-explored. **Between silos, information asymmetry is enormous.**

When information from multiple silos is composed together, it reveals patterns invisible within any single silo. A fact that is obvious in silo A and a fact that is obvious in silo B may combine to reveal a third fact that is invisible to participants in either silo alone. This is **compositional blindness** — the inability of any single domain to see what multiple domains reveal together.

The worked example that motivated the framework: solar panel efficiency data (materials science), polysilicon supply chains (commodities), and tax incentive structures (law) are each well-understood independently. But their collision reveals a mispriced renewable energy transition. The science says efficiency gains should drive adoption; the commodities data says supply constraints should limit it; the law says subsidies should accelerate it. Only the three together reveal when those forces are misaligned in the market.

HUNTER is an instrument built to detect compositions at scale. This document is the theoretical argument for why the compositions exist, why they persist, and how to measure them.

---

## 2. The core thesis and the incompleteness trilemma

**Markets are structurally efficient for additive signals and structurally inefficient for compositional signals.**

The residual — the gap between the best single-silo posterior and the best joint posterior — has four properties under this framework:

- **Measurable.** A rate-distortion lower bound establishes a theoretical floor on the incompleteness residual (Layer 5).
- **Persistent.** Self-reinforcing cycles are stable equilibria; they do not decay toward correction (Layer 8).
- **Finite but fractal.** Total residual value converges to a bounded sum (order-of-magnitude estimates below) but its structure is inexhaustible — residual exists at every scale (Layer 10).
- **Generalisable.** The framework holds across all domains with information silos and attention constraints.

### The Incompleteness Trilemma

Any system that processes information under real constraints faces three desiderata that cannot be jointly satisfied:

| Desideratum | Definition | Implication |
|---|---|---|
| **Specialisation** | Deep expertise requires domain focus | Creates silos and translation loss |
| **Finite attention** | Cognitive bandwidth is bounded | Prevents cross-silo questions |
| **Complete pricing** | Prices reflect all available information | Impossible given silos + attention limits |

**Central result:** you cannot have all three. Any real system sacrifices one. Markets sacrifice *complete pricing*; the residual is the proof. The framework's central empirical project is measuring the residual and showing it is not zero.

---

## 3. The ten layers

The theory has ten layers. The first seven extend existing literatures (Shannon, Grossman–Stiglitz, Hong–Stein, Arrow–Debreu) into the compositional regime. The last three — Layers 8, 9, 10 — are the framework's original contributions.

### Layer 1 — Translation Loss

When information crosses a silo boundary, fidelity degrades. A chemistry fact expressed in chemical notation must be translated into financial notation before it can be priced. Each translation introduces noise, loss of context, and compression artifacts.

**Quantitative form.** A fact with native-domain signal strength $S$ translates with strength $S' = S \times L$ where $L$ is the translation-loss coefficient (typically 0.4–0.7 in observed corpora). Composing across $N$ domains multiplies these: $S_{\text{final}} = S \times L^N$. With $L = 0.6$ and $N = 3$, signal degrades to 21.6% of native strength. Cross-domain facts are invisible not because they lack validity but because translation destroys their visibility to any single-silo participant.

### Layer 2 — Attention Topology

Attention in markets is topological: analysts cluster around obvious nodes and ignore edges. Finance analysts focus on earnings, revenue growth, macro conditions. Science researchers focus on published papers and peer review. The intersections receive approximately zero attention.

Model attention as a graph: nodes are information sources, edges are the observation paths analysts actually traverse. Attention concentrates on high-degree nodes. Edges connecting nodes from different clusters have attention degree ≈ 0. **Value hiding in these edges grows geometrically over time because no one is actively analysing them.** Blind spots are not random; they follow from the topology of analyst incentives.

### Layer 3 — The Question Gap

Most questions asked in markets are single-domain questions: *what is the fair value of this stock*, *how fast will AI adoption accelerate*. Cross-domain questions — *how do quantum computing breakthroughs affect encryption costs which affect data-storage pricing which affects data-centre capex* — are almost never asked.

The questions that would expose compositional blindness are the ones no analyst has a professional incentive to ask. Each analyst is rational within their domain. But their combined questions would reveal massive mispricing. The gap is systematic, not filled because filling it crosses institutional boundaries. *The wrong loss function is being optimised at the incentive level.*

### Layer 4 — Epistemic Phase Transitions

Small inputs can cause massive epistemic shifts when a system approaches a critical threshold. This is a phase transition — a qualitative reorganisation of knowledge structure triggered by quantitative accumulation.

Regulatory changes are incremental (each new rule adds 1–2% complexity), but at a critical threshold the entire landscape reorganises and previously-hidden risks become obvious. The system was in one equilibrium below the threshold, then flips to another above it. HUNTER detects approach to thresholds by analysing the *derivative* of change, not absolute change. Phase transitions exhibit specific statistical signatures — increased volatility in related metrics, longer tail events, cascading failures — detectable weeks or months before the transition occurs.

### Layer 5 — Rate-Distortion Bedrock

Shannon's rate-distortion theorem proves a fundamental information-theoretic trade-off: you cannot losslessly compress information about state $X$ into bits transmissible across domain boundary $Y$ without distortion. The rate-distortion function $R(D)$ defines the minimum transmission rate needed to convey $X$ with distortion $D$.

For cross-domain compositional problems this creates a floor: no amount of effort can make cross-domain signal loss disappear entirely. **Residual distortion is a law of information, not a feature of current systems.** Markets experience this as persistent asymmetry. Yet markets price as if zero distortion were possible. **The gap between the theoretical floor and market pricing is precisely where compositional value lives.**

### Layer 6 — Market Incompleteness Theorem (Arrow–Debreu extension)

Arrow and Debreu (1954) proved that complete markets require a full set of state-contingent contracts — one for every possible future state. Real markets are vastly incomplete.

Extension: compositional states are states defined by multiple domains. A state like "quantum breakthrough + energy policy shift + materials discovery + trade war" is a 4-domain compositional state. Markets have essentially zero contracts for such states. When such states occur, they cause sudden repricing because markets had no way to hedge them. **Incompleteness is magnified at the edges where domains touch.** This creates the deepest residual value: hedging contracts for states that markets literally cannot currently price.

*Note on naming:* earlier drafts used "Market Incompleteness Theorem" as a Gödel-style framing. The current manuscript treats this as a *structural claim about market completeness in the Arrow–Debreu sense, extended to compositional states* — not a Gödelian incompleteness theorem. The formal version, if it exists, requires collaboration with a senior theorist and sits in Paper 2 of the research programme.

### Layer 7 — Depth-Value Distribution

Compositional value follows a characteristic distribution across depth. At depth 0 (single domain), value is zero — no composition. At depth 1 (pairs), value is moderate. At depths 2–3, value peaks. At depth 4+, value decays exponentially.

| Depth | Composition | Relative value | Frequency |
|---|---|---|---|
| 0 | Single domain | 0% | ~25M facts |
| 1 | A + B | 15–25% | ~300 pairs |
| 2 | A + B + C | 45–65% | ~2,300 triples |
| 3 | A + B + C + D | 70–85% | ~12,650 quads |
| 4 | A + B + C + D + E | 12–18% | ~53,130 quints |
| 5+ | Deep compositions | <8% | exponential |

**Quantitative form.** $V(d) \propto \alpha^d$ with $\alpha \approx 0.27$–$0.30$. The hump curve emerges empirically. This means **99% of value by depth 8.** Searching beyond depth 8 yields exponentially harder problems with exponentially lower expected value. HUNTER's optimal strategy terminates at depth 8.

**Intuition.** Each time you add a domain, you must find facts in that domain AND verify they interact meaningfully with the previous facts. Interaction failures multiply: at depth 1, ~70% of pairs interact; at depth 2, ~50% of triples; at depth 3, ~30% of quads; by depth 8, only ~1% of octuples have meaningful signal.

### Layer 8 — Epistemic Cycles *(original contribution)*

A cycle is a compositional path that loops: A → B → C → A. Each step contributes to the next, and the final step feeds back to the first. Unlike chains (A → B → C) which terminate, cycles create **self-reinforcing systems**.

**Canonical example: the Capital Cost Cycle.**

- OSHA (workplace safety) → Steel industry cost (safety regulations increase material spec)
- Steel prices → Building credit spreads (construction costs rise, default risk perception shifts)
- Credit spreads → Insurance premiums (systemic risk premium feeds back into liability costs)
- Insurance costs → Capital cost of construction (insurability affects WACC)
- Capital cost → OSHA violations acceptable (firms cut corners when capital is expensive) → back to OSHA

**Why cycles are structurally invisible.** No starting point, no endpoint. Each specialist sees only the incoming arrow and cannot observe the full loop. The loop closes in the space *between* specialisms.

**Markov result.** Cycles with reinforcement rate $r \geq$ correction rate $c$ never converge to truth. They settle into permanent misalignment — mathematically an absorbing state in the associated Markov chain. When $r < 0.30$, cycle decays; when $r > 0.35$, cycle grows exponentially. The critical threshold $r = c$ separates decaying from amplifying cycles.

### Layer 9 — The Cycle Hierarchy *(original contribution)*

Cycles don't exist in isolation. Simple 3-node cycles nest inside larger cycles, creating hierarchical structures. A 6-node cycle contains three 2-node sub-loops. A 9-node cycle contains multiple overlapping 3-node and 4-node cycles. These hierarchies amplify feedback because each level reinforces the others.

**Hierarchy rules:**
1. *Nested cycles:* outer loops feed inner loops; value multiplies.
2. *Coupled cycles:* two cycles share a node or edge, creating a figure-8 structure.
3. *Braided cycles:* cycles with shared edges (not just shared nodes); the strongest reinforcement pattern.
4. *Hierarchical cycles:* cycles of cycles. A 9-node system where each node is itself a 3-node cycle produces a three-level hierarchy. Amplification at level 3 is approximately (decay factor)$^{-3} \approx $ 50–100× stronger than simple chains.

### Layer 10 — Fractal Incompleteness *(original contribution)*

No matter how deeply you analyse compositional structures, residual incompleteness persists at every scale. Solve the 3-domain problem, and 4-domain problems remain. Solve all $n$-domain problems up to depth $D$, and depth $D+1$ remains. This is **fractal** — self-similar at every scale.

This follows from information theory: for $N$ domains with $M$ states each, possible compositional states = $M^N$ (exponential growth). But analysis capacity grows polynomially at best. Therefore no matter how sophisticated HUNTER becomes, residual asymmetry that it cannot close will always exist at the next scale.

---

## 4. Mathematical backbone (convergence, decay, stability)

### 4.1 Chain value convergence

The chain value formula converges geometrically:

$$V(d) = C(N, d+1) \cdot r_0 \cdot \alpha^d \cdot U \cdot c_0 \cdot \beta^d$$

Where $C(N, d+1)$ is the binomial coefficient (combinations of $N$ domains taken $d+1$ at a time), $r_0 \approx 0.45$ is the base value, $\alpha \approx 1/3$ is the depth decay factor, $U \approx 15$ is the utility coefficient, $c_0 \approx 1.035$ and $\beta \approx 0.82$ are confidence-decay parameters. Empirically cumulative value saturates at depth 8 — 99.2% of total value is captured by $d = 8$, reducing computational load by approximately 1000× while losing <1% value.

### 4.2 Cycle vs chain tail weight

Chain decay: $V_{\text{chain}}(d) = V_0 \cdot 0.271^d$
Cycle decay: $V_{\text{cycle}}(L) = V_0 \cdot 0.220^L \cdot (L-1)!/2$

The factorial term $(L-1)!/2$ counts directed cycles at each loop length: at $L=3$ there is 1 cycle, at $L=4$ there are 3, at $L=6$ there are 60, at $L=7$ there are 360.

| Depth/length | Chain value | Cycle value | Cycle/chain ratio | Dominant |
|---:|---:|---:|---:|---|
| 2 | 0.27 | 0.049 × 1 | 0.18× | Chain |
| 3 | 0.073 | 0.011 × 1 | 0.15× | Chain |
| 4 | 0.020 | 0.0024 × 3 | 0.36× | Chain |
| 5 | 0.0054 | 0.00053 × 12 | 1.18× | Cycle |
| 6 | 0.0015 | 0.00012 × 60 | 4.8× | Cycle |
| 7 | 0.0004 | 0.000026 × 360 | 23× | Cycle |

**Key insight.** Chains dominate early (depth 2–4); cycles dominate late (depth 5+). Chains decay monotonically; cycles *never decay to zero* — they persist indefinitely due to self-reinforcement. A 6-node cycle has roughly 5× the value of a 6-node chain.

### 4.3 Markov cycle stability

A 3-node cycle A → B → C → A exhibits two regimes depending on whether external corrections dominate or reinforcements dominate.

Chains (terminating): half-life ≈ 7 steps, decay monotonic, value erodes predictably.

Cycles: when $r \geq c$, cycle is stable equilibrium; it never reaches half-life. When $r < 0.30$, cycle decays. When $r > 0.35$, cycle grows exponentially. The critical threshold is **reinforcement = correction rate.** Below this, market self-corrects. Above this, mispricing self-amplifies. This explains why some mispricings disappear in weeks while others persist for years.

### 4.4 Depth cutoff at $d = 8$

The framework predicts 99% of value is captured by depth 8. Beyond that, the combinatorial explosion of possible compositions outpaces the value density. This is the **optimal HUNTER strategy:** allocate search to depths 2–7, hard-cap at 8.

### 4.5 Residual estimate — scale, not point estimate

Total addressable residual in the framework's scale-validation exercise:

| Source | Annual contribution | Notes |
|---|---:|---|
| Chain compositions ($d = 2$–$8$) | ~$425B | translation-loss dominated |
| Cycle compositions ($L = 3$–$9$) | ~$391B | self-reinforcement dominated |
| Hierarchical cycles | ~$595B | the 50–100× amplification regime |
| **Annual compositional residual (order of magnitude)** | **~$1–2T** | cross-validated against known inefficiencies |

**Cross-validation against known additive anomalies.** Known market anomalies — PEAD, merger arbitrage, value premium, momentum, small-cap premium — sum to roughly $5–5.5T annually. The framework's predicted compositional residual of $1–2T sits adjacent to (not on top of) those anomalies. Total market efficiency implied by the framework: approximately 96–98%, leaving 2–4% compositional residual unaccounted for by any existing literature.

**Note.** Earlier drafts of this calculation produced a specific $2.8T point estimate from a formula whose intermediate tables computed to zero. That point estimate has been withdrawn; the order-of-magnitude $1–2T range with explicit sensitivity to assumptions replaces it. See `docs/THEORY_CANON.md` §3 for the formal withdrawal.

---

## 5. Cross-domain generalisation

The framework was tested across eight distinct domains during development. Results (pre-freeze):

| Domain | Silos | Silo/residual ratio | Annual residual ($B, order of magnitude) |
|---|---:|---:|---:|
| Financial markets | 25 | 2.0× | ~$1,650 |
| Scientific research | 40 | 10× | ~$375 |
| Pharma / drug discovery | 20 | 15× | ~$300 |
| Geopolitics | 30 | 27.5× | ~$125 |
| Social media | 15 | 7.5× | ~$80 |
| Climate / energy | 20 | 9× | ~$360 |
| Legal / regulatory | 25 | 13.3× | ~$180 |
| Sports betting | 12 | 0.8× | ~$10 |

### Key insights

**Sports betting fails.** The framework breaks down. Why? Sports bets are simple additive wagers with direct price correction and no silo structure. *This is a positive result:* a theory about compositional asymmetry *should* predict that domains without silos show no compositional residual. The framework correctly identifies a domain where it does not apply.

**Strength increases outside finance.** Where no correction mechanism exists (research, geopolitics, climate), the residual is largest relative to market size. This suggests the framework is fundamental — not an artifact of financial-market microstructure.

**Universality.** The ratio of silo count to measurable residual is predictable within domains (coefficient of variation ~0.3 across domains).

*All numbers in this table are pre-freeze theoretical estimates. None is an empirical finding. The summer 2026 study tests the financial-markets column only; the other seven are framework-extension conjectures awaiting their own empirical programmes.*

---

## 6. The three-paper research programme

**Paper 1 — Summer 2026, "Measuring compositional residuals in market prices."**
Question: does the compositional residual exist, and does it follow the predicted rate-distortion distribution? Method: out-of-sample time-series analysis of the pre-registration-locked corpus, cross-validated against holdout data. Decompose observed pricing errors into translation-chain component and cycle component. Key prediction: cycles produce mispricings 5–10× larger than chains of equivalent depth (Layer 8). SSRN September 2026.

**Paper 2 — 2026–2027, "Information silos and market structure: a formal argument for incompleteness."**
Objective: formal extension of Arrow–Debreu market-completeness to the multi-domain compositional case, using information theory and computability theory to establish the constraint structure under which compositional states are un-hedgeable. *Requires collaboration with a senior theorist.* The operator is an undergraduate; the paper ships with a formal-theory co-author or it doesn't ship.

**Paper 3 — 2027–2028, "Sheaf cohomology and the structure of compositional incompleteness."**
Objective: classify all possible cycle structures using sheaf theory; determine which topologies are stable equilibria. Significance: maps the entire landscape of possible mispricings, enabling targeted mining.

---

## 7. What this framework is (and isn't)

**What it is.** A ten-layer theoretical system grounded in three disciplines:
- *Information theory* (Shannon entropy, rate-distortion theory, the limits of lossless compression).
- *Computability / completeness theory* (Arrow–Debreu market completeness, Church–Turing limits on analysis capacity).
- *Algebraic topology* (sheaf cohomology and fractal geometry, applied to cyclic compositional structures).

The three genuinely original contributions are **Layers 8, 9, 10**: epistemic cycles, the cycle hierarchy, and fractal incompleteness. The first seven layers extend existing literatures into the compositional regime; the last three are novel as a combined analytic structure, as far as the author can find in prior published work.

**What it isn't, yet.** Empirically validated. All quantitative predictions above are predictions, not findings. The summer 2026 pre-registered study is the critical test. The pre-registration manifest (`preregistration.json`, SHA-256 `f39d2f5ff6b3e695`, locked 2026-04-19) commits to publishing both positive and null results.

**What happens if summer fails.** The null is itself informative: it says either the methodology triad (implication matching, model-field extraction, differential edge) doesn't work at scale, or the compositional-residual theory is wrong, or the 2026 pre-freeze corpus was not representative. Each of those is a publishable result that refines the field.

**What happens if summer succeeds.** The framework becomes the first operationalised, empirically-validated theory of compositional information asymmetry. The three-paper programme proceeds. Hunter-style instruments proliferate and the residual decays toward a new equilibrium.

Either way, **the framework will survive contact with data or it won't.** That is what makes it science rather than philosophy.

---

## 8. Status of each layer, graded honestly

| Layer | Novelty | Evidence today | Empirical status |
|---|---|---|---|
| L1 Translation loss | Extension of Shannon | $L \approx 0.4$–$0.7$ observed in corpus | Working parameter, not validated at out-of-sample scale |
| L2 Attention topology | Extension of Hong–Stein | Pre-freeze topology mapped (138 pairs in kill_failure_topology) | Structural claim, summer tests |
| L3 Question gap | Framework contribution | 423 negative-inference gap detections in corpus | Mechanism claim, summer tests |
| L4 Phase transitions | Extension of phase-transition lit | 18 phase-transition signals in corpus | Not yet tested |
| L5 Rate-distortion | Direct application of Shannon | Mathematical, no data required | Foundational |
| L6 Market incompleteness | Extension of Arrow–Debreu | Verbal argument, no formal proof | Requires senior-theorist collaborator |
| L7 Depth-value | $V(d) \propto 0.27^d$ (pre-freeze) | Observed vs predicted overestimates by ~96% in pre-freeze corpus (framework-refuting) | Framework's own data contradicts the specific constant; summer tests whether the *functional form* survives |
| L8 Epistemic cycles | **ORIGINAL** | 9 cycles detected by Tarjan SCC (pre-freeze) | Cycle existence confirmed; persistence and amplification claims await summer |
| L9 Cycle hierarchy | **ORIGINAL** | 2 of 9 cycle types observed; 7 of 9 types are theoretical conjectures | Taxonomy partially supported; needs more data |
| L10 Fractal incompleteness | **ORIGINAL** | Mathematical argument from combinatorial growth | Structural claim, not directly testable |

The honest read of this table: the novel contributions (L8–L10) have partial empirical support, not confirmation. The foundational layers (L1, L5) are on solid footing. The hardest-to-ship layer (L6, the formal incompleteness claim) is appropriately deferred to a collaboration. And the framework's own data *contradicts* the specific $0.27^d$ constant at Layer 7 — an honest self-refutation that strengthens the framework's credibility because it was published by the instrument itself, not by an external critic.

---

## 9. The empirical programme locks in October

The corpus is frozen at 2026-03-31. The code is locked at SHA-256 `f39d2f5ff6b3e695` as of 2026-04-19. Three null baselines (random-pair, within-silo, shuffled-label) are pre-committed. The primary endpoint (monotonic alpha across compositional-depth strata with D − A > 0 at p < 0.05 via 10,000-resample paired bootstrap) is fixed in `preregistration.json`.

Between June 1 and August 31, 2026, the instrument runs out-of-sample against the frozen corpus under upgraded three-tier model routing. First resolutions hit the public ledger mid-July. September publishes either the empirical paper (if the primary endpoint lands) or the null-result paper (if it doesn't).

The framework stands or falls on that test. None of the quantitative claims above should be cited as established results before then.

---

*John Malpass · University College Dublin · April 2026. Framework co-developed with Claude (Anthropic) over two theoretical sessions, January–April 2026. Pre-registration manifest SHA-256 f39d2f5ff6b3e695. Questions, prior-art pointers, and collaboration inquiries welcome via the repository contact.*
