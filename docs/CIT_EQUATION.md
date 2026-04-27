# What HUNTER's edge actually is, in one line

**John Malpass · April 2026 · working draft**

---

I built HUNTER over six months. The whole time I couldn't tell you in one
line what its edge actually was. I had a framework with ten layers and
seven collision strategies and 18 silos. None of it was an answer. It
was scaffolding around an answer I hadn't found yet.

A few weeks ago I think I found it. It's one equation. It's testable.
This short paper is just me writing it down.

## The equation

```
realised alpha = synergy × conversion × lead time × decay − publication cost
```

Five things multiplied, then a subtraction at the end. If any one of the
factors is zero, the whole thing is zero. If they're all moderate, the
realised alpha is moderate. If they're all big, the realised alpha is big.

That's it. Now let me say what each one is, in plain English.

## The five factors

### Synergy

Two facts in two different rooms. A patent for a bismuth-based photovoltaic
substitute. A COMEX silver inventory at decade lows. Each fact is normal in
its own room. Together they say something specific that neither alone says.
That "extra" thing the joint says is **synergy**.

It's measured in bits, using a thirty-year-old idea from neuroscience called
partial information decomposition. The HUNTER `SynergyEstimator` produces a
number for any pair of facts and any target outcome. Some compositions have
high synergy. Most have low synergy. The high ones are interesting.

### Conversion

A bit of compositional information has different dollar value in different
markets. A bit about CRE pricing converts to dollars differently than a
bit about a Brazilian sovereign bond. There's a constant for each market
that says: *one bit of compositional information here is worth roughly X
dollars per unit time*. It depends on volatility, market depth, and how
exposed the asset is to the specific compositional channel. We calibrate
it empirically.

### Lead time

How many days does HUNTER articulate the thesis before anyone else does?
If HUNTER posts to the public prediction board on June 1, and the first
sell-side note appears June 15, lead time is 14 days. If the market beat
HUNTER, lead time is zero. The longer the lead, the more time HUNTER has
to act before consensus catches up. The `ArticulationLeadTracker` measures
this from GDELT global news data.

### Decay

How many other agents could replicate HUNTER's discovery? If only HUNTER
reads all 18 silos, the decay factor is one — full alpha intact. If a
hundred quant funds catch up next month, the alpha decays exponentially.
Mathematically, e^(−λN), where N is competitor count. The harder it is to
replicate (because the corpus is hard to build, the methodology is novel,
the integration takes work), the smaller λ, the bigger the surviving
alpha.

### Publication cost

When HUNTER publishes a thesis publicly, that publication itself causes
the market to move toward HUNTER's view. That movement eats some of the
alpha I would have realised in the counterfactual where I traded silently.
The `StrangeLoopAssessment` estimates this cost from history. I subtract
it.

## Why this matters

Most quant strategies optimise for one factor at a time.

- Renaissance optimises for synergy in microsecond-scale data.
- Most CTAs optimise for conversion in liquid futures.
- Soros optimised for lead time on macro narratives.
- Coopers optimised for decay (proprietary models nobody else could copy).

HUNTER is the first system I'm aware of that's designed to optimise for
the **whole product**. Every thesis is a five-factor optimisation. A
thesis with moderate synergy but huge lead time can beat a thesis with
huge synergy but no lead time. Every factor matters.

Each factor corresponds to a specific module in the codebase:

- **Synergy** — `quant/research/synergy.py`
- **Conversion** — calibrated empirically from realised alpha; module to come after summer
- **Lead time** — `quant/research/articulation_lead.py`
- **Decay** — proxy via GDELT mention growth and sell-side coverage frequency
- **Publication cost** — `quant/research/strange_loop.py`

The full implementation isn't in any single file. It's in the **multiplication**.

## The bet

The summer 2026 study is the empirical test. If the four-factor product
(with publication cost subtracted) correlates with realised alpha at
Pearson r ≥ 0.4 across the surviving theses, the equation holds and HUNTER
has its first foundational law. If it doesn't, I learn which factor was
wrong and I revise.

Either outcome is good.

A clean confirmation makes this the first quantitative theory of
compositional alpha — there isn't one yet. A clean refutation tells me
what to build next. The bet is asymmetric in the right direction.

## What this means for me, the operator

If the equation is right, the daily workflow changes:

1. **Don't just hunt for high-synergy theses.** Hunt for theses where ALL
   five factors are favourable simultaneously.
2. **Time publication carefully.** A thesis with high synergy and high
   lead time should be traded *before* it's published, then published
   for the residual influence. The CIT equation tells me which order.
3. **Pick markets where conversion is high.** Some asset classes have
   higher dollars-per-bit than others.
4. **Build moats that slow decay.** Open-source the methods (priority of
   discovery is the academic moat). Keep the live mechanism configs and
   corpus calibrations private (the operational moat). The framework is
   replicable; the specific HUNTER calibration is not.
5. **Maintain a publication cadence that maximises (lead time accumulated)
   minus (publication cost).**

This is a different way to operate. Not a different product.

## What I don't know

A few honest gaps:

- The conversion factor κ has not been calibrated. Until I have realised
  alpha data from summer, I'm guessing.
- The decay constant λ depends on how aggressively other agents copy.
  I can't observe their copying directly.
- The publication cost δ might be larger than I'm estimating. Strange
  loops are subtle.
- The five factors might not multiply cleanly. Maybe two of them have a
  hidden correlation.

The summer study addresses all four. If the test passes, I write up the
paper. If it fails in interesting ways, I revise the equation. If it fails
uninterestingly, I learn the framework was wrong.

## What's next

Three things, in order:

1. **Run the summer study.** Already pre-registered. Code locked at SHA
   `f39d2f5ff6b3e695`. Corpus frozen 2026-03-31.
2. **Compute the four factors for every surviving thesis.** Plug into the
   equation. See if it predicts realised alpha.
3. **Either** publish the equation as a foundational law of compositional
   alpha, **or** publish the null result with the specific factor that
   broke. Both go on SSRN.

That's the whole bet. One equation. Five factors. One test.

---

*Working draft. Comments welcome.*
*John Malpass · University College Dublin · April 2026.*
