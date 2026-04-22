# HUNTER Ledger, Launch Post

---

## Title options

**A. Good Stories Die First** ← *recommended*
**B. The First Thing HUNTER Told Me**
**C. What Happens When You Read All of Finance at Once**

**Why A.** A cold reader needs the punch in the title. "Good Stories Die First" is a complete sentence that does work. It's a finding. It's counter-intuitive. It names the whole post in five words. B is more literary but makes you read the subtitle to know what the post is about. C is the safe explainer and won't bang.

## Subtitle options

**A.** ← *recommended*. I built a program that reads across eighteen corners of finance nobody reads together. The first thing it told me was the opposite of what I'd predicted. Here's what that means, and why I'm about to let you watch it succeed or fail in public.
**B.** Field notes from month six of an autonomous cross-silo research instrument. One surprise finding and a pre-registered twelve-week public scoreboard starting June 1.
**C.** Launching *The HUNTER Ledger*. One operator, one instrument, one public scoreboard.

**Why A.** A subtitle for a cold reader should explain the project in one breath and promise stakes in the next. A does both. B assumes the reader already respects "pre-registered" and "out-of-sample"; most don't. C belongs on the About page.

---

# Good Stories Die First

*I built a program that reads across eighteen corners of finance nobody reads together. The first thing it told me was the opposite of what I'd predicted. Here's what that means, and why I'm about to let you watch it succeed or fail in public.*

---

## Two specialists who never meet

Let me start with a picture.

The first specialist is a patent lawyer. He works out of an office in Midtown Manhattan. He reads USPTO filings eight hours a day. He's extraordinary at it. He knows every classification code, every procedural quirk, every examiner's individual voice. He is paid very well to know these things. He does not read 10-K filings. His clients have no reason to need him to.

The second specialist is an insurance actuary in Hartford. She reads reserve filings. She knows the NAIC model law by heart. She tracks statutory accounting principles across all fifty states. She can read a quarterly solvency filing and tell you, in thirty seconds, which life insurer is under-reserved. She does not read patent filings. Her models don't use them.

These two people walk past each other every morning, metaphorically speaking. Maybe they live in the same city. Maybe they sit next to each other on a train. They are both excellent at what they do. They never talk about work.

And somewhere between them, between the patent office and the insurance reserve, between the Federal Register and the commercial real-estate delinquency report, between the safety regulator and the semiconductor supply chain, there are facts that, when combined, imply something specific about a specific price.

No single person reads both rooms. So the facts are public, and the implication is private. Only the price ever shows it. And the price is wrong.

**That is the whole thing.** That is what this newsletter is about.

---

## What I built

Six months ago I opened a blank Python file and started writing a program called HUNTER. The name is the job description. It sits between the rooms. It reads across eighteen corners of finance: patents, regulatory filings, insurance reserves, CMBS servicing reports, commodity inventories, federal rule-making, job listings, app rankings, academic preprints, drug approvals, distressed credit, healthcare property trusts, and so on.

Every fact HUNTER reads gets broken down into its components. The specific claim. The entities involved. The date. The source. And the part that matters most: its implications for professional communities that don't read this room. Every fact carries a note that says, in effect, "this matters to insurance actuaries because their capital calculation assumes X, and this changes X. But they don't read Federal Register notices, so they don't know yet."

Then HUNTER looks for collisions. Two facts from two different rooms that, when combined, imply a third thing neither implies alone. When it finds one, it doesn't celebrate. It tries to destroy it. It runs four rounds of adversarial checks, each one web-searching for reasons the claim is wrong. Maybe one of the underlying facts is incorrect. Maybe a competitor already published the thesis. Maybe a regulatory barrier makes it un-tradeable. Maybe the causal mechanism doesn't actually connect the way you claim.

If the claim survives all four rounds, HUNTER writes it down as a *hypothesis*. A specific, time-bounded, testable statement about a specific asset, with a resolution date. Each hypothesis gets scored on six dimensions by a separate adversarial reviewer, a fresh context with no memory of how the claim was generated, against four calibrated reference points. Only the strongest claims cross the threshold.

After six months of running, HUNTER had produced three hundred and twenty-four of these surviving hypotheses across two pipeline iterations. Each one carried a mechanism, a direction, and a date. I was ready to stop feeding it and start asking it questions about its own work.

---

## The Thursday afternoon

It was a Thursday afternoon in April. I was at the kitchen table, laptop open, running on tea.

The question I had was small. Months earlier, when I'd been designing the scoring system, I'd added a measurement called *narrative strength*. Every hypothesis got a score from zero to one on whether it read like a clean story, with a protagonist, a villain, a catalyst, and a resolution, or like a tangle of technical notes.

I'd added this because I had a theory. Hypotheses with clean stories ought to survive adversarial review longer. A good narrative is a grip. It's what lets a claim stay memorable under attack. I expected a positive correlation between narrative strength and kill-round survival. Maybe a strong one.

I opened a new terminal. Pulled the numbers. Asked Python for the Pearson correlation across all three hundred and twenty-four hypotheses.

**r = −0.27.**

Negative.

On the sixty-one most recent hypotheses, the ones from the upgraded pipeline that adds an extra adversarial round, the number was **r = −0.49**. Roughly twice as strong, and same sign.

Strong-narrative hypotheses died *more* often under adversarial review. Not less. The ones with the cleanest arcs died almost every time. The ones that looked like a tangle of technical notes, awkward, full of jargon, impossible to tell at a dinner party, survived most often.

![Narrative strength versus kill-round survival, n = 324](launch_img/narrative_survival.png)

The permutation test over ten thousand shuffles gives *p* < 0.00001 on the combined sample. The sign of the correlation is not an accident. What the correlation *means* took me three days to understand.

---

## Why

Here is what I think the data is telling me.

The adversarial review rounds work by web-searching. Each round tries to find counter-evidence. Papers that already make the argument. Blog posts. Rebuttals. Competitor filings. Analyst notes. A real adversarial reviewer with access to Google and a reason to destroy the claim.

And a clean narrative is exactly what a web search can find.

If a hypothesis has a protagonist and a villain and a clean arc, somebody has probably already written that story. Several somebodies, usually. Which means the adversarial round finds the counter-article, and the hypothesis dies. **The story is a fingerprint.** It tells you the thesis is already in circulation, which means the market has already reacted to it, which means the edge is already gone.

The hypotheses that survive are the ones nobody has written yet. They take ten pages to explain because four of those pages are teaching the reader the vocabulary of the four rooms the thesis draws on. They are awkward because they live in a space no single professional community has colonised. Nobody has written the clean version because no single person has had access to all the rooms at once. So the stories stay unwritten, the corrections never get published, and the price stays wrong.

Weak narrative, in other words, is evidence of structural opacity. The market's correction machinery runs through analyst reports, business press, academic literature, sell-side research. That is the narrative apparatus. If a story hasn't been written, there is nothing for that machinery to read, and nothing drives the price to correct.

There is a second fact in the data that tightens this interpretation. When I split the correlation by pipeline tier, the effect roughly doubles, from −0.23 in the older batch to −0.49 in the newer one. The difference between the two batches is one specific adversarial round. The newer pipeline demands that every causal arrow in the hypothesis name the specific filing, database, or software product through which one room's output becomes another room's input. If you can't name the pathway, the arrow is rejected and the hypothesis dies.

That is the round that involves web search. That is the round that destroys clean stories. And that is precisely where the anti-narrative effect is concentrated.

I would not have figured this out by sitting in a chair and thinking. I figured it out because I'd built an instrument that disagreed with my prediction, logged the disagreement, and, when I extended the analysis to a larger sample, told me *which part of itself* was doing the disagreeing.

That, I think, is what instruments are for.

---

## One honest note on the chart

The high-narrative bin, the bar on the right marked 14.3%, is only seven hypotheses. If two of those seven had gone the other way, the headline would soften considerably. The real workhorse of the correlation is the drop from 42.6% to 19.9% between the two larger bins, with 176 and 141 hypotheses respectively. That part would survive even if the high-narrative tail were removed entirely.

I mention this because it is the kind of thing I would catch in somebody else's chart within four seconds, and I'd rather catch it first in my own.

---

## The part where I put it on the line

Everything above is a pattern in a frozen dataset. I've measured things. I've published the measurements. The data is on Zenodo, the code is on GitHub. That is useful. It is not proof.

What would be proof is if HUNTER, running forward in time, could identify mispricings in advance, post them publicly with resolution dates, and be right more often than chance.

Starting June 1, that is what this Substack is going to be a record of.

I have locked the rules in advance. On April 19 I wrote a file called `preregistration.json`, committed it to GitHub, and hashed the code state. The hash is `f39d2f5ff6b3e695`. That hash is immutable. It means I cannot retroactively change what HUNTER looks for, how it scores, or what counts as a win. The corpus is frozen at March 31. Three null-baseline tests are committed in advance. The primary test is one specific statistical claim about how compositional depth across rooms affects realised returns. I have committed, in writing, to publishing the null result if HUNTER fails.

Between June 1 and August 31, HUNTER runs prospectively. Hypotheses that clear the threshold get posted on a public prediction board with asset name, direction, and resolution date. First resolutions land around mid-July. By late August, the board either shows the instrument can do what it claims or shows that it can't. In September, one of two papers goes up on SSRN. There is no third option.

I am going to be honest about something else. An earlier internal pilot of HUNTER produced the *opposite* of what the summer test predicts. The deepest cross-room hypotheses underperformed the simpler ones. That pilot ran with the web-searching adversarial rounds disabled, which happens to be the specific channel the summer test is about. The summer runs with web search turned on. If the summer also inverts, the framework needs structural revision rather than recalibration, and I've committed in advance to saying so.

I'm one person. I built this alone at University College Dublin. Those are reasons to be sceptical. What I've tried to do, instead of waiting to be caught, is publish my own weaknesses first. HUNTER's own data contradicts its most specific quantitative prediction. The framework said value should decay across composition depth at a rate of roughly 0.27. The data says 0.94, much shallower. The refutation replicates across both pipeline tiers independently, which rules out a pipeline artefact. I published that in the repository on purpose. I would rather be the person who finds the hole than the person who gets caught not seeing it.

---

## What this Substack is going to be

Between now and June 1, I will post weekly. Methodology notes. Field reports. The engineering story of why the current version of HUNTER finds *fewer* hypotheses than its predecessor and why that is the right direction of travel. The eight recurring structural themes the top-scoring output keeps returning to. The single closed-source piece of commercial-real-estate valuation software that, according to HUNTER's causal graph, sits at the centre of nine different cross-silo pathways simultaneously, and what it means for market structure that a corporate software default is the highest-centrality node in financial methodology.

After June 1, the content changes. The prediction board fills. I post. Dates pass. Predictions resolve. Some will be wins. Some will be losses. I don't get to hide either.

If the ledger works, what exists at the end of summer is a structured map of cross-silo financial reality with a dated public track record attached. If the ledger doesn't work, what exists at the end of summer is a pre-registered null result, which is also a finding, and also publishable.

Both outcomes are fine. The only outcome that isn't fine is not publishing.

If you've read this far and you're curious, subscribe. If you think it's nonsense, subscribe anyway. The ledger is public, and you can watch it fail in real time. If you think it's real, tell one person.

The fun part starts now.

John

*John Malpass · University College Dublin · April 2026. Code: [github.com/Johnmalpass/hunter-research](https://github.com/Johnmalpass/hunter-research). Corpus: [10.5281/zenodo.19667567](https://doi.org/10.5281/zenodo.19667567). Public prediction board: [johnmalpass.github.io/hunter-research](https://johnmalpass.github.io/hunter-research/).*
