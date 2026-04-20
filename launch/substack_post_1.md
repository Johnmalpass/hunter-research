# What I built, and why every prediction goes on a public ledger

*The HUNTER Ledger · Issue #1 · April 28, 2026*

---

Here is a thing about markets that stuck with me last November.

Walk through a big financial firm and you'll pass specialists who never talk to each other. The patent lawyer is reading USPTO filings, very good at it, paid a lot to be good at it. He doesn't read 10-Ks. His clients don't care.

The insurance actuary two floors up is reading NAIC reserve filings and mortality tables. She doesn't read patent filings. Her models don't need them.

The CMBS guy down the hall reads Trepp and special-servicer reports. He doesn't read OSHA enforcement. His book doesn't include steel mills.

Everyone good at their jobs. Nobody talking.

And somewhere in the gap between them — the gap between the patent office and the insurance reserve and the CMBS servicer and the OSHA investigator — there are facts that, put together, mean a specific asset is mispriced and you can tell you exactly when it gets corrected.

Nobody walks through all those rooms. So the facts are public and the *implication* is private, and only the price ever shows it, and the price is wrong.

That gap is what I spent the last six months building a machine to read.

The machine is called HUNTER. It goes public today. Here's what it does, what I'm not claiming, and what you can watch happen over the next year.

## The thing, in one paragraph

HUNTER is a Python program that runs all day. It reads 18 kinds of financial filings that don't normally get read together — patents, SEC disclosures, NAIC insurance reserves, OSHA actions, CMBS delinquency reports, Federal Register rule changes, commodity inventories, analyst targets, academic preprints, pharma approvals, distressed credit, healthcare REIT stuff, energy infrastructure, specialty real estate, government contracts, earnings transcripts, job listings, and app rankings. It tears each fact into pieces: who's involved, what it implies for people in other fields, which models or assumptions it breaks, and what causes what. Then it looks for pairs of facts from different fields that, together, imply something neither fact implies alone. When it finds one, it doesn't try to prove it right. It tries to destroy it. Four rounds of trying to destroy it: fact-check, competitor-check, barrier-check, and the hard one — the mechanism check, which demands that every causal arrow name the actual database or filing or workflow through which one silo's output becomes another silo's input. If a claim survives all four rounds, a different reviewer scores it in a fresh context against four calibration anchors. Anything scoring 65 or higher goes on a public board with a name, a direction, and a resolution date. Win or lose, both stay on the ledger. Forever.

That's HUNTER. A machine that reads across 18 professional silos at once and keeps a public, timestamped, falsifiable record of what the integration reveals.

## What I'll actually say

I want to be specific about what's real and what isn't, because the whole point of the thing is that it's pre-registered and honest.

**The instrument is real.** About 32,000 lines of Python across 60+ modules and 43 database tables, with 26 different language-model prompts doing different jobs. The frozen corpus has 12,030 facts from 18 silos, touching 77 countries, and resolves down to 11,835 distinct entities across 30,967 normalised mentions. The causal graph has 171 directed edges, each with a specific named transmission pathway — not "X causes Y" but "X's output flows through this filing into that database which feeds that portfolio manager's model." The corpus includes 6,670 model-field extractions (which methodology each fact could break), 474 collisions between facts from different silos, and 52 multi-link chains. The pre-registration manifest is SHA-256 locked at hash `f39d2f5ff6b3e695`. Corpus frozen 2024-12-31. Code locked 2026-04-19. The prediction board is live at a public URL and it is empty on purpose. A 12-week out-of-sample study runs June through August. Code is MIT. Corpus is CC-BY-4.0 on Zenodo with a DOI.

**I do not have a track record.** Zero predictions have resolved. Some empirical patterns showed up when I ran analysis on the pre-freeze corpus — a very sharp asymmetry between how easy it is to kill a claim by attacking its causal mechanism versus attacking it on "someone else already found this" grounds, a hub-and-spoke shape in the methodology graph where one node (ARGUS Enterprise DCF cap-rate assumptions) does most of the connecting, a weirdly bimodal distribution of quality scores, and a negative correlation between how cleanly a hypothesis tells a story and how often it survives the kill phase (r = −0.49, which is the opposite of what I predicted). I'm not calling any of these findings. The pipeline has been upgraded since then. I'm holding them as hypotheses the summer will test.

**Things I'm not going to say.** Nothing about revolutionary anything. No market-opportunity numbers. No "HUNTER has found $X in alpha." If you see that phrase with my name on it, assume it was rewritten by someone who didn't read the original.

## Why this is possible now and wasn't in 2019

Reading 18 professional silos in one afternoon isn't an information problem. The filings are free. What's expensive is holding the vocabulary, databases, and methodological defaults of multiple professional worlds in one head long enough to notice the implication. No human does this at scale. The specialist gets rewarded for depth; the breadth-seeker gets called a dilettante. The incentive structure actively selects against the person who would find the compositions.

Around 2022–2023 language models got good enough to simulate multiple specialist perspectives at once. Good enough to extract methodology metadata from a fact, name the causal arrows, and adversarially review cross-silo claims. That's the capability HUNTER uses. The new thing isn't the instrument; it's that one person can now read every specialist's output in parallel. Whether that actually produces measurable cross-silo alpha, and for how long, is an empirical question. The ledger is where I answer it.

## What I think is new

Three pieces, as far as I can find, aren't in the prior financial-NLP or knowledge-graph literature.

*Implication matching.* Every fact HUNTER pulls in carries an explicit note about who in other fields should care about it and why. Collisions get searched across that implication field, not across shared keywords. Two facts can collide even if they have no vocabulary in common, as long as their implications overlap in a specific named way.

*Model-field extraction.* Every fact gets tagged with methodology, assumption, practitioner community, calibration, and disruption channel. That turns the graph into a methodology graph instead of an entity graph. Nodes are things like "ARGUS Enterprise DCF cap-rate assumption" or "NAIC RBC C-1 calculation," not companies.

*Differential edge.* A causal arrow only enters the graph if the kill phase verifies a specific, named transmission pathway. No named pathway, no arrow. The graph that survives is the one where every edge corresponds to a real filing or database or workflow.

I'll write more about each of these in later posts. If any of it is prior art I missed, email me. I'd rather find out now than in peer review.

## The summer study

From June 1 through August 31, HUNTER runs out-of-sample on the frozen corpus. The pre-registration locks the primary endpoint: median realised alpha over SPY total return, ordered across four strata by how many silos each hypothesis combines (A ≤ B ≤ C ≤ D, with D − A > 0 at p < 0.05 under a 10,000-resample paired bootstrap). Three null baselines sit in the manifest: randomly paired facts, same-silo pairs, shuffled-silo labels. Decision rules are fixed. A script auto-detects any drift in code or data during the study and reports it in the final paper regardless of outcome.

Worth flagging upfront: an earlier retrospective pilot (the "v3 Golden" run) gave Stratum D < Stratum B, which is the opposite of what the primary says. That pilot disabled web search in the kill phase, which is the whole channel through which cross-silo mechanism verification is supposed to do anything. Summer runs with web-searched kills. That's a different test. If summer also gives D ≤ B, the manifest's decision rule kicks in, I publish the null, and the framework needs structural revision. That outcome is on the table.

If the endpoint lands, there's probably a real structural source of cross-silo alpha worth measuring carefully. If it doesn't, the null says automated cross-silo tools don't beat standard single-silo arbitrage. Either way I publish.

## What this Substack will be

Methodology posts now. Resolutions from mid-July. I'll walk through how the model-field extraction actually works, what the kill-failure topology looks like across the 138 domain pairs where adversarial review consistently fails, and why strong narratives die more often than weak ones in the kill rounds (that was one of several things the data surfaced that contradicted my own prior). When predictions start resolving I walk through each one in public, with evidence and URL, win or lose. September gets a full summer report.

The Substack exists because the board exists. Every resolution is a post. That's the rhythm.

## Caveats, in advance

I'm in my second year of economics at UCD. One operator, one instrument. The corpus is small — 12,030 facts is a research prototype, not a production feed. About 60% of the pre-freeze high-scoring stuff concentrates in CMBS, insurance, and regulatory-transition finance, which will shape how the results should be read. The generator and the reviewer are both language models. The decorrelation effort (fresh context, web search, four calibrated anchors, three null baselines) reduces shared blind spots but does not eliminate them. Every empirical claim on this Substack is provisional until the summer is done.

None of that changes what I'm doing next.

## Come watch

HUNTER is live. The repo is public. The corpus is on Zenodo with a DOI. The methods paper is going up on SSRN. The methodology brief is a free 2-page PDF. The board accepts resolution dates and is empty by design. It fills starting June 1 as summer hypotheses clear the upgraded pipeline.

If you think this is interesting, subscribe. Methodology posts weekly until June, live resolutions from mid-July, full summer report in September.

If you think it's nonsense the ledger is public and you can watch it fail in real time. That's the whole point of putting it there.

If you see a design flaw, a prior-art pointer, or a reason one of the pre-freeze patterns is misread, email. Honest criticism is worth more than anything else right now.

The fun part starts now.

---

*John Malpass · University College Dublin · April 2026*

*Repo · Zenodo · SSRN · Prediction board · Methodology brief — linked on the About page.*
