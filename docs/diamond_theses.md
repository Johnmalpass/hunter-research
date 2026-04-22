# HUNTER — Diamond theses (pre-freeze)

*Fifteen of the 324 pre-freeze hypotheses scored diamond-tier (≥ 75 on the 100-point adversarial review) across the two pipeline iterations. This document catalogues them by theme, with hypothesis ID, score, and the silo composition that makes each one compositional rather than additive. **None of these is a finding.** Each is a pre-freeze candidate that the Summer 2026 pre-registered study either validates out-of-sample or discards. Not investment advice.*

**Corpus of record:** `hunter_corpus_v1.sqlite` on Zenodo ([10.5281/zenodo.19667567](https://doi.org/10.5281/zenodo.19667567)). Every ID below resolves to a row in either `hypotheses` (main pipeline, April 1–4, upgraded mechanism-kill) or `hypotheses_archive` (earlier pipeline, March 28 – April 3). Score is the internal adversarial-review diamond grade.

---

## How to read this

A thesis appears here if it satisfies three conditions: (1) it scored ≥ 70 on internal adversarial review, (2) it combines at least three silos in a load-bearing way (removing any one silo collapses the thesis), and (3) the first round of fact-check and competitor-kill did not find a disconfirming external source.

Every diamond below is a **structural mechanism**, not a trade. A mechanism like *"statutory filing dates create snapshot-to-consequence timing mismatches that bridge CMBS distress into insurance regulatory capital"* can recur across many specific filings and many specific carriers. Pointing at the mechanism is not the same as identifying a specific asset. The specific assets sit inside the corpus; they are not surfaced in this document.

What makes each one compositional: each thesis combines ≥ 3 silos where removing any one silo breaks the claim. That is the operational test the framework is built to impose, and the test any single-silo analyst can't perform because they can't see across the silo boundary.

---

## Tier 1 — diamond ≥ 90 (the hardest to kill pre-freeze)

### 1. NYC development municipal bond pre-shock underwriting

**Pipeline:** archive · **ID:** 318 · **Score:** 100

NYC development-backed municipal bonds issued 2022–2025 were underwritten against pre-March 2026 pro formas that don't account for mandatory NYCECC compliance costs and transmission-infrastructure cost pass-throughs. The underwriting silo (financial disclosure) and the regulatory silo (NYC Energy Conservation Code + FERC transmission) operate on different refresh cycles, so the compliance-cost delta has not been priced into the bond's credit-spread curve.

**Silos:** municipal underwriting · NYC building code · FERC transmission · pension fund allocators · bond rating methodology.

**Why single-silo analysts can't see it.** The pro forma is in a disclosure filing; the code change is in a municipal regulation; the transmission-cost pass-through is in a FERC tariff; the pension holder sees only the bond's cusip-level data. No analyst reads all four for the same bond.

### 2. Life insurance CRE credit-risk underpricing

**Pipeline:** main · **ID:** 328 · **Score:** 97

Life insurers are reserving against CRE credit risk at roughly 0.43% default experience while CMBS office delinquency is running 12.34% in the same loan population. The reserve silo (actuarial) and the CMBS silo (servicer-reported performance) are nominally connected through the NAIC schedule D and statutory filings, but reserve models refresh on slower timetables than market delinquency and have not reflected the current regime. The underpricing compounds across carriers because every carrier uses the same AA-corporate-yield construction input.

**Silos:** life insurance statutory filings · NAIC RBC methodology · CMBS servicer reports · corporate bond universe construction · rating agency models.

### 3. Appraisal-database contamination via forced bank liquidations

**Pipeline:** main · **ID:** 261 · **Score:** 95

Forced bank CRE liquidations during 2025–2026 are feeding into the comparable-sales databases (CoStar, Real Capital Analytics) that every commercial appraiser is legally required to use under USPAP. Because distressed sales are entering the comp set above the 5–15% outlier threshold appraisers typically exclude, the databases themselves are drifting. This propagates through every downstream appraisal, which propagates through lender loan-to-value calculations, which propagates into servicer reserve adequacy and regulatory capital.

**Silos:** bank regulatory filings · CoStar/RCA comparable-sales databases · USPAP appraisal methodology · servicer loss-severity models · NAIC RBC formulas.

### 4. Independent transmission-operator refinancing into CMBS

**Pipeline:** archive · **ID:** 326 · **Score:** 95

Independent transmission operators face a scheduled FERC debt-authority expiry on June 30, 2026, forcing them into refinancing markets where their novel concentration profile triggers SASB concentration penalties they were not structured to absorb. The refinancing silo (debt capital markets) and the regulatory silo (FERC rate-structure rulings) are time-coupled through the expiry date and the CMBS concentration-threshold calculations.

**Silos:** ITO debt-maturity schedules · FERC Order 2222 debt authority · CMBS SASB concentration rules · institutional private credit allocators · rating methodology.

### 5. Corporate pension ASC 715/IAS 19 discount-rate construction

**Pipeline:** archive · **ID:** 306 · **Score:** 94

Pension liability valuations are systematically understated by 8–12% because the ASC 715 / IAS 19 discount-rate models exclude CMBS securities from the AA yield curve construction due to office-sector stress, and the EU Taxonomy regime forces additional CMBS exclusions on European-issuer curves. The exclusion is mechanical and the resulting curve drift feeds directly into reported liability — but the actuarial applications silo doesn't monitor index-construction methodology in real time.

**Silos:** ASC 715 / IAS 19 actuarial methodology · AA corporate bond universe construction · CMBS office distress · EU Taxonomy classification · pension sponsor disclosures.

### 6. Q1 2026 statutory filings reveal latent CMBS distress

**Pipeline:** main · **ID:** 268 · **Score:** 93

Office CMBS delinquency hit 11.76% and special servicing reached 17.30% in October 2025; statutory filings due May 15, 2026 are the point where insurers must reflect that distress on the balance sheet. The market has priced the CMBS distress but has not specifically priced the May-15 inflection where statutory capital ratios step-change. Filing-calendar mechanics create a discrete repricing moment on a predictable date.

**Silos:** CMBS servicer reports · insurance statutory filing calendar · NAIC RBC formulas · rating agency trigger rules · bond portfolio managers.

### 7. PJM forward curves and the oil-peaker break-point

**Pipeline:** archive · **ID:** 314 · **Score:** 93

PJM forward power curves assume oil-fired peaker plants remain economically marginal at $113/barrel oil. They are already uneconomic. The grid-planner silo has not refitted the marginal-generator assumption because 8-year interconnection queues prevent new capacity from substituting on the forward curve. The joint mispricing is in the gap between the marginal-cost assumption in the forward curve and the actual clearing economics.

**Silos:** PJM forward curve publications · crude oil commodity markets · EPA peaker compliance cost · grid-interconnection queues · power-market analysts.

### 8. Hotel CMBS default-probability models with stale refinancing spreads

**Pipeline:** archive · **ID:** 277 · **Score:** 92

Hotel CMBS tranches maturing in 2025–2026 were underwritten with 200–300 bps refinancing spread assumptions. Current market spreads are 6.25–7.0%. The default-probability models in circulation carry the stale spread; the refresh cycle for the underlying servicer models is slow. No revision has been pushed to the pricing layer, so the market is pricing to the old curve.

**Silos:** CMBS servicer default models · hotel property performance · fixed-income analytics (current spread data) · rating agency recovery curves · lodging-sector equity research.

---

## Tier 2 — diamond 80–89

### 9. Office-to-data-center CMBS BB/BBB arbitrage

**Pipeline:** main · **ID:** 272 · **Score:** 89

Office-to-data-center cap rate spreads of 580+ bps (versus historical norms of 200–400 bps) create a pricing dislocation in CMBS BB/BBB tranches where the rating methodology weights office property-type exposure without distinguishing conversion-ready assets from stranded assets. The conversion-option value is not in the ratings model but is in the underlying collateral.

**Silos:** CMBS tranche ratings · data-center cap rate trends · conversion-feasibility engineering · office-sector NOI trends · Moody's MILAN / S&P LEVELS loss models.

### 10. CMBS → LDI portfolio rebalancing cascade

**Pipeline:** archive · **ID:** 284 · **Score:** 88 *(duplicate: ID 323 at 88)*

CMBS downgrades trigger mark-to-market losses in LDI portfolios, which force rebalancing to maintain duration matching, which creates temporary selling pressure in liquid assets (equities, treasuries). The pressure is concentrated around specific rebalancing dates and is detectable if you read the LDI methodology alongside the CMBS surveillance reports; single-silo managers on either side do not.

**Silos:** CMBS surveillance · LDI portfolio construction · duration-matching mechanics · equity-market liquidity · treasury futures markets.

### 11. European pharma DCF demographic assumptions

**Pipeline:** archive · **ID:** 302 · **Score:** 85

European sell-side pharma DCF models embed demographic assumptions (population aging, treatment adherence, reimbursement environments) that haven't kept pace with 2024–2026 policy changes in specific jurisdictions. The models are nominally sensitive to demographics but the embedded assumptions are stale on different refresh cycles than the policy changes that break them.

**Silos:** Eurostat demographic projections · national reimbursement policies · sell-side DCF inputs · pharma product launch schedules · academic outcomes literature.

### 12. Insurance regulatory capital timing squeeze

**Pipeline:** archive · **ID:** 334 · **Score:** 85

Insurance company Q1 2026 statutory filings (due May 15) will reflect deteriorated CMBS valuations from ongoing office distress. Between the filing date and the rating/regulatory response, a timing window opens where the statutory position has been filed but the consequence has not yet been priced.

**Silos:** NAIC statutory filing calendar · CMBS valuation marks · rating agency action timing · capital-markets response latency · bond portfolio managers.

### 13. Pension fund ALM stale cap-rate mispricing

**Pipeline:** archive · **ID:** 288 · **Score:** 80

Pension funds using ALM models with stale cap-rate assumptions face a 175+ bps mispricing between distressed office assets (11%+ caps) and stabilised data centres (6.5% caps). The spread is mechanical if you read current cap-rate surveys, but the ALM refresh cycle is slower than the rate-regime change.

**Silos:** pension ALM methodology · CRE cap-rate surveys · property-type-specific NOI · actuarial assumption-setting · data-centre valuation.

---

## Tier 3 — the remaining Chapter 4 canonical themes (diamond 70–79)

Four additional structural themes are explicitly named in the April 12 theory drafts as canonical HUNTER outputs even though their internal scores sit below the diamond-≥ 80 tier. They are listed here because their composition is exemplary of what the framework is designed to find, and because they are cross-referenced in `docs/HUNTER_THEORY.md`'s Layer-8 cycle example and in `docs/research_themes.md`.

### 14. ARGUS Enterprise DCF legacy-platform disconnect

**Pipeline:** main · **ID:** 271 · **Score:** 75

Recent evolution in ARGUS Enterprise DCF valuation capabilities (the 2024 Intelligence platform launch) may create systematic disconnects in CMBS pricing models that still rely on legacy valuation assumptions. ARGUS Enterprise DCF cap-rate assumptions are the single highest-degree node in HUNTER's causal graph (degree 9), so any systematic shift in its defaults propagates through the entire valuation stack.

**Silos:** ARGUS software evolution · legacy CMBS pricing models · cap-rate methodology · servicer loss-severity models · commercial appraisal practice.

### 15. Steel EAF silica-compliance real-option value

**Pipeline:** archive / main · **ID:** 192 (archive) / 327 (main) · **Score:** 73

Steel companies with flexible production capacity (40–70% EAF ratios) are undervalued because equity models don't capture the real-option value of production shifting under new OSHA crystalline silica regulations. Compliance cost differentials of $15–25 per ton between integrated and EAF operations create an optionality no equity analyst is modelling.

**Silos:** OSHA silica rule · steel production economics · EAF-vs-BOF operational flexibility · equity research models · commodity-cost analytics.

### 16. CMBS grid-connection arbitrage

**Pipeline:** main · **ID:** 224 · **Score:** 71

Distressed office buildings from CMBS defaults carry embedded electrical infrastructure (grid connections, substation access) worth 3–5× the building value on the renewable-energy development market due to 90% interconnection-queue failure rates. The CMBS silo values the office; the interconnection silo values the grid connection. The joint value lives in the composition and is priced by neither.

**Silos:** CMBS distressed-asset pricing · interconnection queue economics · renewable project development · commercial appraisal · utility planning.

### 17. D&O insurance post-Rule-10D-1 actuarial gap

**Pipeline:** archive / main · **ID:** 169 (archive) / 329 (main) · **Score:** 70

D&O insurance carriers are underpricing executive liability because actuarial loss tables are calibrated to a pre-2024 regime where mandatory no-fault clawback enforcement under SEC Rule 10D-1 didn't exist. The actuarial silo and the securities-regulation silo are nominally connected through claim-experience data, but the regime change happened faster than the calibration refresh.

**Silos:** D&O actuarial methodology · SEC Rule 10D-1 enforcement · XBRL tagging of disclosures · corporate governance filings · executive-compensation disclosures.

### 18. Healthcare REIT SNF RADV audit cascade

**Pipeline:** main · **ID:** 331 · **Score:** 70

Healthcare REITs with concentrated skilled-nursing-facility (SNF) portfolios are mispriced because REIT valuation models use trailing rent-coverage ratios from a pre-RADV-expansion revenue environment. The CMS 50-fold RADV audit-capacity increase creates simultaneous pressure on operator revenue that the REIT NAV models have not yet reflected.

**Silos:** REIT NAV construction · SNF operator financials · CMS RADV audit methodology · CMS RAC programme · healthcare equity analyst coverage.

---

## Thematic roll-up

Across the eighteen theses above, eight structural themes recur. They align with `docs/research_themes.md` and with HUNTER's Layer-2 attention topology (Federal Register and NAIC filings being the two highest-centrality silo-connectors):

1. **Commercial real-estate credit cascades** — CMBS → insurance reserve → pension liability → corporate bond (§1, §2, §6, §9, §10, §12, §13).
2. **Regulatory-effective-date timing** — compliance costs on bonds priced before the rule (§1, §15, §17, §18).
3. **Benchmark-universe construction lag** — AA yield curves, appraisal comps, LDI benchmarks (§3, §5, §13).
4. **Filing-calendar snapshot-to-consequence mismatch** — Q1 statutory, 10-K timing, FERC expiry (§4, §6, §12).
5. **Forward-curve marginal-generator assumptions** — oil peakers, interconnection queues (§7, §16).
6. **Refinancing-spread regime shift** — stale spread assumptions in CMBS servicer models (§8, §9).
7. **Hub-and-spoke methodology propagation** — ARGUS DCF assumptions as the single highest-degree node (§14).
8. **Cross-border demographic / reimbursement lag** — Europe-specific (§11).

---

## What these theses are and aren't

**Pre-freeze, not validated.** Each thesis above scored ≥ 70 on the pre-freeze adversarial reviewer. That is not the same as surviving a pre-registered out-of-sample test. The summer 2026 study is the proper test. Until it reports, treat this document as *a catalogue of candidate mechanisms*, not findings.

**Compositional, not additive.** Every thesis requires reading two or more professional silos simultaneously. Each cluster could be dismissed by a specialist in any single one of its silos as "already priced in within my domain" — and each of those specialists would be correct about their own silo. The claim is that the joint implication is not priced because no single participant holds all the silos.

**Structural, not temporal.** These are mechanisms, not trades. The mechanisms recur; the trades implementing them do not necessarily recur. Pointing at a mechanism is not the same as identifying an asset to buy or sell. The specific assets sit in the corpus but are not surfaced here.

**What the pre-registered test actually examines.** `preregistration.json` (SHA-256 `f39d2f5ff6b3e695`, locked 2026-04-19) commits the summer 2026 run to testing whether the compositional-depth gradient A ≤ B ≤ C ≤ D holds across four strata, whether detected cycles persist uncorrected for ≥ 14 days, and whether cross-silo collisions (domain distance ≥ 0.60) score materially higher than within-silo pairs. If the summer produces null results, the null paper ships and the eighteen theses above remain candidates, not confirmations.

---

*John Malpass · University College Dublin · April 2026. Every thesis above resolves to a row in `hunter_corpus_v1.sqlite` (Zenodo [10.5281/zenodo.19667567](https://doi.org/10.5281/zenodo.19667567)). IDs prefixed "main" map to `hypotheses`; those prefixed "archive" map to `hypotheses_archive`. Not investment advice. Not a solicitation. Pre-freeze candidates awaiting the summer's out-of-sample test.*
