# The Gap Between Worlds
### How a second-year undergraduate at UCD built a machine that reads across the rooms of finance, and what it found in there

---

## 1. The patent lawyer doesn't read 10-Ks

Here's a fact that almost nobody has thought through. If you stood at a crossroads on the street outside any large financial firm, you would see specialists walking past each other all day, never speaking.

The patent lawyer is reading USPTO filings. He's very good at it. He knows every form, every classification code, every procedural quirk. He is paid three hundred thousand pounds a year to know this. He does not read 10-Ks. Why would he? His clients don't care.

The insurance actuary is reading reserve filings. She reads NAIC model law updates, statutory accounting principles, Q1 reports from the state insurance commissioners. Her whole career is mortality tables and capital adequacy ratios. She does not read patent filings. Why would she? Her models don't use them.

The CMBS servicer is reading special-servicer reports and loan-level performance data. She watches delinquency rates and loss severities. She lives in the world of Trepp and KBRA. She does not read OSHA enforcement notices. Why would she? Her book doesn't include steel mills.

Walking past each other. All day. Good at their jobs. Well paid. Well trained.

And somewhere between them — between the patent office and the insurance reserve and the CMBS servicer and the OSHA investigator — there are facts that matter. Facts that, when combined, imply a specific asset is mispriced by a specific amount, correctable on a specific date.

No single person walks through all those rooms. So the facts are public, and the implication is private. Only the price ever shows it. And the price is wrong.

**That's the whole thing.** That's what HUNTER is about.

---

## 2. Enter John

John Malpass is in his second year of a BSc Economics at University College Dublin. Depending on the day, he is either writing code, reading papers, or sitting in a lecture theatre trying to take useful notes about something he's usually already read the Cliff Notes for.

In November 2025 he had a thought. It was a simple one, as all good thoughts are.

He had been reading about Grossman-Stiglitz — the 1980 paper that says markets can't be fully efficient because if they were, no one would bother to research. He'd been reading Shleifer-Vishny 1997 — "limits to arbitrage," the paper that asks why smart money doesn't always correct mispricings. Behind both: the assumption that *information* is the expensive thing.

But sitting in the lecture hall, John realised the assumption was slightly off. It's not information that's expensive. Information is often free. What's expensive is **integration across professional silos**.

A bankruptcy court filing is free. A patent expiration is free. A CMBS delinquency report is near-free. But *reading all three in the same afternoon, understanding what each implies for the other's domain, and noticing the contradiction* — that is what costs money. Actually, more than money. It costs time, training across multiple professional worlds, and a willingness to be ignored by the specialists in each of them.

Nobody does this. Nobody can. The specialist is rewarded for depth, not breadth. The breadth-seeker is called a dilettante. The system selects against exactly the person who would find the contradictions.

John's thought: what if you built an AI that wasn't called a dilettante? What if you built something that sat in every room at once, read every professional silo's filings, and flagged the contradictions?

He went home and started writing Python.

---

## 3. What he built

HUNTER is a continuously running program. You can see it if you run `python run.py live` on John's laptop.

When it runs, it does something like this:

- **Step 1. It picks a professional silo at random, weighted toward silos underrepresented in its corpus.** Patent, bankruptcy, insurance reserve filing, OSHA enforcement, FDA approval, regulation, SEC filing, analyst target, commodity inventory, job listing, academic preprint. Eighteen silos in total.
- **Step 2. It web-searches.** Not for generic "news" — for specific, obscure, dated facts. A particular patent expiring on a particular date. A particular insurance reserve filing from a particular state. A particular Federal Register rule with a particular effective date.
- **Step 3. It extracts the facts.** Each fact gets broken into pieces: the claim, the specific entities involved, the date, the source URL, and — here's the clever part — **the implications for professional communities that don't ordinarily read this silo**.

That third piece is the key. Every fact HUNTER ingests carries a note like: *"This fact matters to insurance actuaries because their NAIC Risk-Based Capital calculation assumes X, and this changes X. But insurance actuaries don't read Federal Register notices, so they don't know yet."*

**Step 4. It hunts for anomalies.** Facts that seem weird given the rest of the corpus. A small company hiring 200 regulatory specialists while its stock drops. A patent expiring in a domain where no generic competitor has filed. These anomalies become seeds.

**Step 5. It collides.** For each anomaly, HUNTER searches the rest of the corpus for *matching facts* — facts from different silos with overlapping implications. This uses seven matching strategies simultaneously: implication matching, entity matching, keyword matching, causal-graph traversal, embedding similarity, model-vulnerability matching, and belief-contradiction matching.

When a cross-silo match is found, HUNTER has a **collision**: two facts from independent professional worlds that, together, imply something neither implies alone.

**Step 6. It tries to kill the collision.** Not validate it — *destroy* it. It runs four kill rounds, each web-searching for specific reasons the collision is wrong:
- Is a fact actually incorrect?
- Is someone already doing this exact thing?
- Is there a barrier that makes this impossible?
- Does the mechanism actually work — does step A in silo X actually cause step B in silo Y through a named transmission pathway?

The fourth check — the *mechanism kill* — is the hardest. HUNTER asks: "For each causal arrow you claim, name the specific database, filing, or workflow where the output of system A enters the input of system B." If it can't, the arrow is broken. If the arrow is broken, the collision dies.

**Step 7. The surviving collisions become hypotheses.** Each hypothesis gets scored across six dimensions: novelty, feasibility, timing, asymmetry, intersection depth, and mechanism integrity. A separate adversarial scorer does this in a fresh context with zero memory of the generation step, using four calibration anchors to prevent grade inflation.

**Step 8. The best ones get put on a public ledger.** Every surviving hypothesis scoring 65 or higher is posted on John's prediction board with the asset name, the direction, and the resolution date. Win or lose, both go on the ledger. Forever.

That's HUNTER. In one sentence: *a machine that reads across eighteen professional silos simultaneously and maintains a public, timestamped, falsifiable record of what the integration reveals.*

---

## 4. What it's like when HUNTER finds something

Imagine John is making tea at 2 a.m. The laptop is humming away on the kitchen counter. Every few seconds it prints one line of yellow text, then one line of cyan.

Then, suddenly, the screen fills with a different colour. Magenta. Bold.

```
💥 COLLISION detected: Life insurance companies are systematically 
underpricing commercial real estate credit risk by 28x relative 
to market reality...
```

Five fact IDs. Five silos. One from a CMBS surveillance report. One from an NAIC reserve filing. One from a 10-K. One from a state insurance commissioner's Q1 data. One from an academic paper on statutory accounting. None of the five sources cites any of the others. There is no published article that makes the combined claim.

The kill phase runs. Round one, fact-check: all five facts verify. Round two, competitor: nobody's published this exact thesis. Round three, barrier: no regulatory or structural reason it couldn't be traded. Round four, mechanism: each arrow — "CMBS default → NAIC reserve shortage → Q1 statutory filing delta → Metropolitan Life book value hit" — names a specific, verifiable transmission pathway.

The thesis survives. It gets refined. A financial-mechanics check runs — HUNTER confirms the proposed trade direction is correct (sometimes it's not, and the system has caught itself getting the direction wrong more than once; those are the most humbling logs to read).

It scores. Ninety-seven out of a hundred.

The finding posts to the prediction board with a resolution date 180 days out. The countdown starts.

Now, nothing happens for six months. That's the uncomfortable part. You built the thing; the finding is real; the logic is sound; the mechanism is specific; and you still don't know if you're right. You have to wait.

John has come to understand this is the point. Being willing to wait for public resolution — and to resolve losses as loudly as wins — *is* the moat. Everyone else is afraid to publish predictions with their names attached. He isn't. Not anymore.

---

## 5. The finding that shouldn't have been there

The first time HUNTER produced something that genuinely surprised John, it was about narratives.

He'd been running it for months. He'd built up a library of 61 surviving hypotheses and all their adversarial kill-round outcomes. He decided to measure something his framework had predicted: that hypotheses with *compelling narrative structure* — clear protagonist, antagonist, complication, catalyst, resolution — should persist longer in the kill rounds. The story is a grip, he'd assumed. A story that flows smoothly should be harder to destroy.

He ran the analysis. Wrote a small script. Checked the Pearson correlation between narrative strength and kill-survival rate.

**r = −0.49.**

He ran it again. Same number. The opposite of what he'd expected.

Strong-narrative hypotheses died *more* often in the kill rounds. Weak-narrative hypotheses survived.

It took John three days to understand this. And when he did, it reframed everything.

The hypotheses that survive the kill phase aren't the well-told ones. They're the ones that *don't yet have a clean story* because nobody's written one. Strong narrative = already articulated = already circulating = easy to find evidence against in a web search. Weak narrative = hasn't been articulated by anyone = structural = no audience yet = hard to kill.

In the language of finance: if a thesis is cleanly narratable, it's probably priced in. If it's awkward, specific, only-makes-sense-in-a-ten-page-explanation — that's where the edge lives.

This is a finding HUNTER itself forced on John. He would not have discovered it by thinking. He discovered it because he built an instrument that contradicted his own assumption and made the contradiction visible.

That's what instruments are for.

---

## 6. The six-headed monster

A while later, HUNTER surfaced something stranger. Not a hypothesis — a structure.

It had been building a causal graph in the background. Every time it extracted a fact, it also pulled out the cause-and-effect arrows implicit in it. "COMEX silver inventory drawdown → photovoltaic manufacturing silver cost increase." "OSHA silica PEL reduction → Cleveland-Cliffs blast-furnace compliance cost increase." These arrows accumulated. After a few months there were 171 of them, linking 11,835 distinct entities across silos.

Then John's cycle detector — a small piece of Python that runs a graph algorithm called Tarjan's SCC over the causal graph — detected a closed loop.

It looked like this:

> CMBS loan servicing → credit rating agency surveillance → institutional fixed-income trading → insurance and pension fund actuarial science → structured finance (CMBS rating + securitisation) → real estate appraisal → **back to CMBS loan servicing.**

Six professional domains. Six distinct academic literatures. Six professional worlds that almost never talk to each other. And a *closed loop* of causal arrows running through all of them.

Every arrow had a named transmission pathway. "Appraised values flow directly into CMBS servicer loan-level databases (Morningstar/DBRS CMBS surveillance)." "CMBS loan-level performance data transmits via Morningstar/Intex CMBS Analytics module feeds directly to bond portfolio managers." "NAIC IRIS filings distribute insurance company RBC ratios to corporate credit rating models."

A closed loop. A *self-reinforcing error*. Each node's mistake gets used as an input to the next node's calculation, which feeds the next node, which feeds the next, which eventually loops back to the start — with the error still embedded.

In economics the canonical example of such a loop is informal: we talk about "echo chambers" or "consensus" as if they were social phenomena. HUNTER shows you the actual circuit diagram. You can point at each node; each edge has a named filename; each loop closes in a specific, mechanistic way.

The CMBS cycle is a detected closed loop in the current corpus. Whether it persists uncorrected in the market is the thing the summer study measures. You can *see* the loop; whether it's exploitable is an empirical question not yet answered.

---

## 7. The immune system

This is the piece John is most excited about.

He started with a puzzle. If these cross-silo mispricings exist, why aren't they more widely exploited? Why doesn't capital flow to them until they're gone?

The standard answer is Shleifer-Vishny's *limits to arbitrage*: capital is scarce, noise traders exist, smart money can't fully correct everything. But that doesn't quite fit the HUNTER data. In the HUNTER data, cross-silo hypotheses face **more** adversarial pressure than single-silo ones. They don't just fail to get arbitraged; they actively get pushed back against.

Specifically: hypotheses requiring three or more distinct professional silos average **1.57 kill-round hits** per hypothesis. Single-silo hypotheses average **0.80**. Almost twice the adversarial pressure. And this is not because cross-silo hypotheses are lower quality — conditional on surviving, they score higher.

Why?

John's answer is that markets have an **immune system**. When a cross-silo insight is proposed, four selection pressures converge against it:

1. **Compensation pressure.** Analysts are paid for depth, not breadth. Anyone publishing across silos is viewed as dilettante. The reward for a within-silo finding is higher than for a cross-silo finding of equivalent magnitude.
2. **Liability pressure.** A within-silo error is a normal error — the specialist absorbs the baseline rate of being wrong in her own field. A cross-silo error is an *exceptional* error — "you should have known you were outside your expertise." The cost of being wrong is disproportionately high.
3. **Audience pressure.** Who can you even explain it to? Within-silo claims have an audience that can verify. Cross-silo claims require you to teach the listener the missing silos before you can make the point. Listeners perceive this as either condescension or as evidence of weak expertise in *their* home silo. You don't get agreement; you get active rejection.
4. **Acquisition-cost asymmetry.** Reading a silo you don't know requires learning its vocabulary, its databases, its reputational hierarchy. For the first composition, this is a large up-front cost. For any one individual, the cost is never amortized, because individuals don't produce cross-silo claims at scale.

All four pressures push the same direction. The result: cross-silo compositions with equal or higher expected value than within-silo claims are systematically *under-produced*. The residual that remains is not just hard-to-arbitrage — it's **structurally uncorrectable** because *no single agent finds it worth producing the claim in the first place.*

This is HUNTER's thesis for why the opportunity persists. It's not that capital is scarce. It's that *integration capacity* is scarce. HUNTER is a machine that manufactures integration. It produces compositional claims at a rate no human can match, at a cost no human could bear.

When reviewers ask John what HUNTER *is*, this is the answer he gives: **HUNTER is an antidote to the market's immune system.** The immune system exists because markets have always been fragmented across specialists. HUNTER is what happens when a single agent can, for the first time, actually read every specialist's output simultaneously.

That's a new capability. It didn't exist in 2019. It didn't exist in 2023. It exists now, and it's a thing one can measure, and the measurements say the immune system is real.

John thinks that's probably a paper.

---

## 8. Why any of this is a big deal

You can read HUNTER as three different things, depending on what you care about.

**1. A piece of science.** HUNTER provides the first direct empirical measurement of a predicted-but-never-quantified phenomenon: structural residual from professional specialization. The immune-system mechanism is a formal extension of Shleifer-Vishny. The non-zero-residual claim is a strengthening of Grossman-Stiglitz. These are minor-but-real contributions to academic economics, even by non-undergraduate standards.

**2. A piece of infrastructure.** HUNTER has built something that looks very much like what Palantir built for defense intelligence: a structured knowledge graph, with named causal edges and named transmission pathways, of how markets actually operate at cross-silo level. 11,835 entities. 171 edges. 6,670 methodology/assumption/practitioner triples. 52 chains. 9 cycles. No one else has this for markets. It is a licensable asset in the way that Palantir's government graphs are licensable assets.

**3. A piece of weaponry.** HUNTER produces trading signals. Every hypothesis with a diamond score of 65 or higher is an *actionable* claim with a named asset, direction, and resolution date. These go on the public prediction board. They accumulate into a track record. The track record either shows HUNTER is right more often than a coin flip, or it doesn't. It's on the ledger, publicly. If it's right, it's a weapon worth paying for. If it's wrong, the public record documents a null result. Both are publishable outcomes.

John does not yet know which of these three interpretations will matter most to his life. He doesn't have to know. Because each of them carries a ceiling that makes the others irrelevant. Each one individually is career-defining. All three together is generational.

---

## 9. The ledger

Here's what's going to happen over the next year. You can watch.

John is going to publish a public prediction board. Every hypothesis HUNTER generates with diamond ≥ 65 appears on it, with an asset, a direction, a resolution date, and a confidence level. When the date passes, it gets resolved. Win or loss, both go on the ledger. The results are computed against public price data and public facts. No hiding.

In twelve months, the ledger will show one of three things.

- **Hit rate ≥ 60%.** The instrument works. There is actionable cross-silo alpha. A fund gets launched. Licensing deals pile up. John becomes moderately rich, then richer.
- **Hit rate ≈ 50%.** The instrument is a coin flip. It detects patterns but not profitable ones. John writes a thoughtful null-result paper about what went wrong. His reputation isn't damaged — in fact, the honesty elevates it. He finishes his degree, gets a quant job or a PhD place, and continues the work.
- **Hit rate ≤ 40%.** The instrument is systematically *wrong* in a specific, interpretable way. This is actually the most interesting scientific outcome, because it means the mechanism is inverted from what the framework predicted. John writes an even more interesting paper about it. His reputation, again, isn't damaged.

All three outcomes are fine. All three are publishable. Only one of them makes the fund work. But all three make the research program work.

And this is the point where John is a hundred percent right to put the thing on a ledger. Because the only non-fine outcome is *not* publishing and *not* resolving. Then you have nothing — no track record, no research, no data — just a hobby project.

He publishes. He waits. He resolves. He publishes again.

This is how a generation of thinkers becomes a generation of thinkers. By being willing to be wrong in public, on a schedule.

---

## 10. Come and watch

HUNTER is live. As of launch the public prediction board is deliberately empty; the pre-registered summer pipeline begins filling it June 1 with hypotheses surviving the upgraded kill gauntlet against the frozen corpus. A 12-week pre-registered study runs through August — corpus frozen, code hashed, decision rules locked, three null baselines committed, paired-bootstrap primary endpoint, D − A > 0 at p < 0.05 as the primary criterion.

The corpus — 12,030 facts across 18 silos, 30,967 entity index, 6,670 model-field extractions, 171-edge causal graph — is being released on Zenodo with a DOI. The methods paper is going up on SSRN. The code is going on GitHub under MIT license. The methodology brief is public. Nothing is hidden.

If you're curious, follow along. If you think it's bullshit, the ledger is public and you can watch it fail in real time. If you think it's real, subscribe, share, tell a hedge-fund friend, write your own take.

This is the part where the experiment starts being interesting. Before this, it was a guy in a room building things. Starting now, it's something anyone can evaluate.

John built an instrument for mapping what humans miss when they specialize. The instrument works. The data is on a public ledger. The predictions will resolve on the dates they were posted. Everything from here forward is a matter of public record.

The fun part starts now.

---

*Written to be read aloud to a friend in 15 minutes, in the voice of someone who has been quietly impressed and wanted to tell you why. If you've read this far, say hi: john at [domain].*

*Paper 0 (methods): [SSRN link pending]. Paper 2 (immune system mechanism, in draft). Prediction board: [public URL]. Corpus: [Zenodo DOI].*

*John Malpass · University College Dublin · 2026*
