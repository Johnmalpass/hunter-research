# HUNTER: Known Limitations

*As of launch. This file is public on purpose. Publishing my own known weaknesses is a stronger move than letting someone find them in review.*

## L1. Single operator, single instrument

One person built and runs HUNTER. Independent replication by a second instrument operated by different researchers is required before any claim is generalised. The public release of corpus, code, and manifest is intended to make that replication possible.

## L2. Small pre-freeze sample

61 hypotheses with completed adversarial review is small. The pre-registered summer 2026 out-of-sample study targets n ≥ 300 and is the primary evidence base for every published empirical claim.

## L3. LLM substrate contamination

The hypothesis generator and the adversarial reviewer are both language models. Shared blind-spot risk cannot be fully excluded. Three mechanisms attempt to bound it:
- Fresh-context review (the reviewer has no memory of the generator's reasoning).
- Four calibrated anchor scores in the reviewer prompt.
- Three pre-registered null baselines: random-pair, within-silo, shuffled-label.

None of these proves independence; together they bound how much of the signal is genuine versus shared-model artifact.

## L4. Corpus concentration

~60% of pre-freeze high-scoring hypotheses concentrate in CMBS, insurance, and regulatory-transition finance. Results, if they hold, should be read as applying most robustly to that regime. Generalisation to other domains awaits further corpora.

## L5. v3 Golden / Stratum B > D inversion

An earlier retrospective pilot (the `V3_GOLDEN_*` run in `config.py`) produced Stratum D < Stratum B, directly contradicting the pre-registered primary hypothesis. That pilot ran with `RETROSPECTIVE_DISABLE_WEB_SEARCH = True` — the kill phase could not check mechanisms against live web evidence, which is the specific channel H1 is about. The summer 2026 study runs prospectively with web-searched kill rounds. If the summer also produces D ≤ B or violates monotonicity, the manifest's decision rule rejects H1 and the null paper ships. The framework then needs structural revision, not recalibration.

## L6. Pipeline tier change

Pre-freeze empirical patterns (the mechanism/audience kill asymmetry, the hub-and-spoke methodology graph around ARGUS Enterprise DCF, the bimodal diamond-score distribution, the narrative/survival r = −0.49, the nine Tarjan cycles) were produced on an earlier pipeline tier. Summer 2026 re-runs under upgraded three-tier routing (Opus 4.7 for critical reasoning, Sonnet 4.5 for extraction, Haiku 4.5 for ingestion). All pre-freeze numbers should be read as *hypotheses to be tested*, not as findings.

## L7. Framework iteration in `findings` table

The `findings` table in the Zenodo corpus contains outputs from multiple HUNTER framework iterations during development, including entries from early exploratory phases that predate the current cross-silo financial inference framing. Researchers interpreting specific findings should filter by creation date and/or domain to isolate the current-framing subset.

## L8. Temporal concentration in v1

v1 facts concentrate in the 2024 calendar year of active ingestion. Temporal dynamics cannot be tested within v1 alone. Multi-year temporal analysis awaits v2.

## L9. Formal theoretical claims await senior collaboration

The formal proofs sketched in the working papers (particularly the non-zero compositional residual bound) are informal. Rigorous measure-theoretic treatment is pending collaboration with a formal theorist. The operator is a second-year undergraduate; the papers are framed as conjectures inviting collaboration, not finished results.

## L10. Zero track record at launch

The prediction board is empty on launch day by design. Nothing has resolved. Track record begins accumulating at mid-July 2026 when the first summer hypotheses resolve. Until then, every empirical claim in every artifact (Substack, X, README, papers) is provisional.

---

*If a limitation is missing from this list, email me. Honest criticism is worth more than private reassurance, and a limitation I don't know about is worse than one I've published.*

*John Malpass · University College Dublin · April 2026*
