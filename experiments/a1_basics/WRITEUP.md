# Assignment 1 — Basics of Empirical ML: full writeup

This is the one document to read. For each problem: the question in plain words, what we ran,
the numbers, how to think about them, and the answer to the assignment's example quiz question.
Data: 116 finished runs on W&B (`aleyang-stanford-university/assignments`, tag `a1`), plus 21
earlier P1 grid points; 34.1 of 48 GPU-hours. Tables-only version: `RESULTS.md`. Plots: `figures/`.

## How to read the numbers

Everything below is **final validation loss** in nats for the standard d8 model (34.6M params,
8 layers, 614M training tokens, lr 3e-3, batch 64, 1% warmup, linear decay, weight decay 0.1)
unless a row says otherwise. The baseline is **2.9277**.

Before comparing anything, know the noise floor. Problem 3 tells us that rerunning the same
recipe with different seeds moves the final loss by about **0.002** (standard deviation). So:

- two runs within **0.005** of each other are the same run;
- a gap of **0.01** is real but small;
- a gap of **0.1** is a different regime.

Keep that scale in your head; every conclusion below is stated relative to it.

## The five things to remember

1. Two knobs matter at the 0.1–0.25 level: **peak learning rate** (too low is worse than too high)
   and **whether the learning rate decays** (no decay costs 0.25). Everything else is ≤ 0.06.
2. Hyperparameters that are "linked" are linked through a product or a count: LR × weight decay
   (the product `lr·wd` sets how fast weights forget), LR × batch (batch sets step count at fixed
   tokens), LR × warmup (warmup buys LR headroom).
3. Scaling laws are recipe-dependent. Same model family, same data, four recipes → exponents
   0.042–0.067. Small-model fits over-predict d9 by 10–40× the seed noise. Optimizer choice,
   instability, and repeated data break the law outright.
4. Randomness is small and it saturates. Seeds, GPU nondeterminism, or one corrupted token move
   the endpoint by ≤ 0.005. A perturbation changes the trajectory forever but the gap plateaus at
   the nondeterminism floor instead of growing.
5. Diagnostics beat the loss curve. Train-loss jitter is only batch size. Activation and gradient
   norms flag a broken run 500 steps in, when the loss still looks fine.

---

## Problem 1 — Hyperparameter sweeps

### The question
Vary warmup, learning rate, batch size and weight decay one at a time (a), then in pairs (b),
then together with the LR schedule and the Adam betas (c). Which matter, which are linked, how?

### What we ran
Log-spaced sweeps around the d8 defaults (factor 3), three 3×3 grids for the pairs, five
schedules, β₁/β₂ and gradient clipping on/off. 42 runs plus 21 reused from an earlier grid.

### (a) One at a time

| learning rate | 3e-4 | 1e-3 | **3e-3** | 9e-3 | 2.7e-2 |
|---|---|---|---|---|---|
| final val | 3.1475 | 2.9768 | **2.9277** | 2.9537 | 3.0175 |

| batch size | 16 | 32 | **64** | 128 | 256 |
|---|---|---|---|---|---|
| final val | 2.9284 | **2.9182** | 2.9277 | 2.9517 | 3.0123 |

| warmup (% of steps) | 0 | 0.3 | **1** | 3 | 10 |
|---|---|---|---|---|---|
| final val | 3.0343 | 2.9405 | 2.9277 | 2.9240 | **2.9200** |

| weight decay | 0 | 0.01 | 0.03 | **0.1** | 0.3 | 1.0 |
|---|---|---|---|---|---|---|
| final val | 2.9503 | 2.9474 | 2.9408 | 2.9277 | **2.9180** | 2.9624 |

**How to think about it**

1. Look at the *range* of each row first. LR spans 0.22, warmup 0.11, batch 0.09, weight decay
   0.045. That is the importance ranking, and it is what a grid search budget should follow.
2. Look at the *shape*. LR is an asymmetric basin: 10× too small costs 0.22, 9× too large costs
   0.09. Undershooting is the expensive mistake, because a small LR simply doesn't finish
   training in the token budget, while a too-large LR is rescued by the decay phase.
3. Warmup is "steep first dollar": going from 0 to 0.3% (28 steps) recovers 0.09 of the 0.11
   penalty. The damage from no warmup happens in the first few hundred steps at full LR and is
   never repaired. Beyond 0.3%, more warmup is a slow monotone 0.02.
4. Batch at *fixed tokens* is really a step-count knob. Batch 256 has 4× fewer optimizer steps
   than batch 64. Below ~64 the curve is flat (we are under the critical batch size — extra steps
   aren't buying anything), above it each doubling costs more (0.024, then 0.06).
5. Weight decay has a broad optimum near 0.3 and only over-regularises at 1.0. It is a
   fine-tuning knob, not make-or-break — but see (b), because its right value depends on LR.

### (b) Pairs

| LR \ batch size | 32 | 64 | 128 |
|---|---|---|---|
| 0.001 | 2.9423 | 2.9768 | 3.0251 |
| 0.003 | **2.9182** | 2.9277 | 2.9517 |
| 0.009 | 2.9512 | 2.9537 | 2.9898 |

| LR \ weight decay | 0.03 | 0.1 | 0.3 | 1.0 |
|---|---|---|---|---|
| 0.001 | 2.9904 | 2.9768 | **2.9508** | — |
| 0.003 | 2.9408 | 2.9277 | **2.9180** | 2.9624 |
| 0.009 | 2.9649 | **2.9537** | 2.9780 | — |

| LR \ warmup | 0% | 1% | 3% | 10% |
|---|---|---|---|---|
| 0.001 | — | 2.9768 | 2.9734 | 2.9680 |
| 0.003 | 3.0343 | 2.9277 | 2.9240 | 2.9200 |
| 0.009 | — | 2.9537 | 2.9196 | **2.9138** |

(— = trimmed to save GPU time.)

**How to think about it** — for each grid, ask "does the best column move as I go down the rows?"

1. **LR × batch.** Read the lr 1e-3 row: its penalty versus the best LR grows from +0.024 at
   batch 32 to +0.073 at batch 128. Bigger batch = fewer steps = each step must do more = you need
   a bigger LR. The link is through the step count. (Note the best LR did *not* move to 9e-3 at
   batch 128 in our grid — at this scale the effect is "low LR gets punished more", not "high LR
   becomes optimal".)
2. **LR × weight decay.** The best weight decay slides down as LR goes up: 0.3 at lr 1e-3, 0.3 at
   3e-3, 0.1 at 9e-3. Multiply them: the winners all have `lr·wd` between 3e-4 and 9e-4; every
   cell with `lr·wd ≥ 2.7e-3` is bad. AdamW's decay step is `w ← w − lr·wd·w`, so the product is
   the fraction of the weights forgotten per step. Tune the product, not the two numbers.
3. **LR × warmup.** Going 1% → 10% warmup gains 0.009 at lr 1e-3 but 0.040 at lr 9e-3, and it
   *flips the best LR* from 3e-3 to 9e-3. Warmup buys headroom to use a higher LR. Whenever you
   raise LR, re-tune warmup.

### (c) Schedules, betas, clipping (all at lr 3e-3)

| LR schedule | linear | cosine | WSD 20% | WSD 50% | constant |
|---|---|---|---|---|---|
| final val | 2.9277 | 2.9347 | 2.9371 | **2.9190** | **3.1784** |

| β₁ | 0.8 | **0.9** | 0.95 | | β₂ | 0.9 | **0.95** | 0.99 | | grad clip | 1.0 | none |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| final val | 2.9298 | 2.9277 | 2.9343 | | final val | 2.9321 | 2.9277 | 2.9225 | | final val | 2.9277 | 2.9320 |

**How to think about it**

1. Constant LR is +0.25 — the single largest effect in P1, bigger than any LR mistake. Any
   decay recovers it: linear, cosine, WSD all land within 0.02 of each other. The *fact* of
   decaying matters; the *shape* is a 0.01 detail.
2. WSD with a 50% decay beats a 20% decay by 0.018 — so the key extra hyperparameter a schedule
   introduces is the **decay fraction**. Too short a decay phase means you are still noisy at the
   end; that's also why WSD "looks bad" mid-run (P5).
3. β₁, β₂ and clipping move the loss by ≤ 0.007 — noise. They matter only near instability,
   which we will find in P6/P7, not in a healthy recipe.

### Example question (Easy): rank A (lr 3e-3, bs 64), B (lr 1e-3, bs 128), C (lr 9e-3, bs 128)

**A < C < B.** Measured 2.9277, 2.9898, 3.0251. Doubling the batch halves the steps, so the LR
that pairs with it is the higher one: C recovers part of the loss, B loses on both counts. Neither
beats A because bs 64 → 128 costs ~0.024 even at the best LR, and A is already at the LR optimum.

### Example question (Medium): lr 9e-3, wd 1.0, rank A (wsd0.2, warmup 0), B (wsd0.2, warmup 20%), C (linear, warmup 10%)

**C < B < A** (predicted — these exact runs were trimmed).
- `lr·wd` = 9e-3, thirty times the baseline's 3e-4: the weights forget everything within ~110
  steps while LR is at peak. The schedule that spends the least time at peak wins → linear (C).
  WSD holds peak for 60–80% of training.
- No warmup at 3× the baseline LR is the worst thing on the board: no warmup cost +0.11 even at
  lr 3e-3, and warmup mattered 4× more at lr 9e-3 than at 1e-3. A is last.
- B vs C is close (~0.01): 20% warmup is barely better than 10% (3% → 10% only gained 0.006),
  and B has the worse schedule.

---

## Problem 2 — Are scaling laws reliable?

### The question
Fit power laws on small models (d4–d6), pre-register predictions for d8/d9, then check. Which
interventions bend the slope (b), and which break the law entirely (c)?

### What we ran
Six depths (d4 5.9M → d9 48M params) under four recipes; model-size and data-size ladders with
LR and batch changes; four "breaker" interventions at d4/d6/d8. 56 runs.

### (a) Four recipes

Pre-registered: baseline exponent ≈ 0.06; constant LR flatter; dropout parallel to baseline;
lr 3e-2's gap to baseline shrinks with size.

| recipe | d4 | d5 | d6 | d7 | d8 | d9 | exponent | d4–d6 fit → d9 | actual d9 | fit → d20 |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 3.2903 | 3.1541 | 3.0562 | 2.9840 | 2.9277 | 2.8818 | 0.063 | 2.814 | 2.882 | 2.52 |
| constant LR | 3.4283 | 3.3302 | 3.2429 | 3.2197 | 3.1784 | 3.1242 | 0.042 | 3.051 | 3.124 | 2.87 |
| dropout 0.2 | 3.4518 | 3.3206 | 3.2157 | 3.1289 | 3.0594 | 3.0034 | 0.067 | 2.972 | 3.003 | 2.61 |
| lr 3e-2 | 3.3165 | 3.2134 | 3.1104 | 3.0702 | 3.0257 | 2.9849 | 0.050 | 2.901 | 2.985 | 2.68 |

**How to think about it**

1. Three of four pre-registrations held; the lr 3e-2 one was wrong — its gap to baseline *grew*
   (0.03 at d4 → 0.10 at d9). Wider models have larger attention logits, so a too-high LR hurts
   more, not less, as you scale. Instability gets worse with width.
2. The exponents span 0.042–0.067 on the *same* architecture and data. "The" scaling exponent
   of a model family doesn't exist; it belongs to a recipe.
3. Three-point fits over-predicted d9 by 0.03–0.08. Recall the noise floor is 0.002: these
   forecasts are off by 15–40 standard deviations. Extrapolating a 3-point fit even 3 depths is
   already unreliable; the d20 predictions (2.52–2.87) differ by more than the entire d4 → d9 gain.
4. Where the fits fail is informative: the d4–d6 slope is always steeper than the d7–d9 slope
   (returns diminish faster than a pure power law). At fixed tokens the bigger models are
   increasingly under-trained, which bends the curve.

### (b) Slope benders

| model-size axis | d4 | d5 | d6 | d7 | d8 | exponent |
|---|---|---|---|---|---|---|
| baseline | 3.2903 | 3.1541 | 3.0562 | 2.9840 | 2.9277 | 0.063 |
| lr 1e-3 | 3.3528 | 3.2185 | 3.1173 | 3.0362 | 2.9768 | 0.067 |
| bs 256 | 3.3615 | 3.2274 | 3.1324 | — | 3.0123 | 0.062 |

| data axis (tokens) | 77M | 154M | 307M | 614M | 1.23B | exponent |
|---|---|---|---|---|---|---|
| d8, lr 3e-3 | 3.5352 | 3.2303 | 3.0582 | 2.9277 | 2.8372 | 0.078 |
| d8, lr 1e-3 | 3.5768 | 3.2879 | 3.1088 | 2.9768 | — | 0.088 |
| d6, lr 3e-3 | 3.5730 | 3.3314 | 3.1659 | 3.0562 | — | 0.075 |
| d6, lr 1e-3 | 3.7936 | 3.4369 | 3.2495 | 3.1173 | — | 0.093 |

**How to think about it.** Ordinary hyperparameter choices move the exponent by 5–20% without
breaking linearity. The interesting one: lr 1e-3 shows a *steeper* data law (0.088 vs 0.078).
That is not "more efficient" — it is a small LR being badly under-trained at 77M tokens (gap 0.22
at d6) and catching up by 614M (gap 0.06). A steep slope can mean "the small points are
artificially bad", which is exactly the situation where you should distrust the extrapolation.

### (c) Law breakers

| intervention | d4 | d6 | d8 | what the curve does |
|---|---|---|---|---|
| baseline | 3.2903 | 3.0562 | 2.9277 | power law, exponent 0.063 |
| SGD (same LR/schedule) | 6.5861 | 6.5989 | 6.6535 | flat / slightly inverted |
| no QK-norm, lr 3e-2 | 4.2168 | 4.9412 | 5.1513 | **inverted — bigger is worse** |
| lr 3e-2, no warmup, no clip | 3.3235 | — | 3.0343 | survives, gap widens |
| 8 epochs × 75k seqs (same tokens) | 3.3252 | 3.1446 | 3.0900 | flattens, exponent ≈ 0.04 |

**How to think about it.** A scaling law says "capacity is the bottleneck". It breaks whenever
something else is:
1. the optimizer (SGD at Adam's LR barely moves — all sizes stuck at ~6.6, unigram level);
2. stability that worsens with width (no QK-norm: attention logits scale with width, so bigger
   models blow up harder — the law *inverts*);
3. data (repeat 75k sequences 8 times and the penalty grows with size: +0.035 → +0.09 → +0.16 —
   bigger models memorise more). This happens on the model-size axis in the over-trained,
   small-data regime.

### Example question (Hard): fixed vs depth-scaled context, 15 epochs of 39M tokens, d10 gap?

**0.03 < Δ ≤ 0.15 — and don't trust the fit.** The pilot fit says +0.055, which lands in this box
for the wrong reason; the realistic miss is upward.
- Why the pilots converge: the scaled recipe is catastrophic at d4 (4.75 vs 3.45) and nearly
  caught up by d6. Small depths train on contexts *shorter* than the 1024-token validation
  sequences, so they're evaluated at positions they never saw. That effect vanishes once context
  ≥ 1024 — the fit is extrapolating a transient, not a law. (Our own clean d4–d6 fits still
  missed d9 by 0.03–0.08.)
- Why the gap stops shrinking: 15 epochs of repeated data. Our P2(c) repetition run shows the
  penalty grows with size (+0.035 → +0.16 from d4 to d8), and longer contexts mean fewer
  independent sequences per epoch. The d6 curves already show it: scaled context bottoms at 3.50
  around epoch 8 and drifts *up* to 3.57 while fixed context is still descending. At d10 that
  overfitting is larger. If repetition dominates as in P2(c), Δ > 0.15.

---

## Problem 3 — Run-to-run variation

### The question
How much does a run vary if you change nothing but the seeds (a)? How much comes from each source —
initialization, data order, hardware (b)? Do hyperparameters or model size change the variability (c)?

### What we ran
3–4 replicas per group. Seeds varied together, then one at a time; same seeds with GPU
nondeterminism on; deterministic mode across two H100s and an A100; then seed groups for
batch 16, lr 9e-3, d4, and 2× tokens. 26 runs.

| what varies | n | mean | std |
|---|---|---|---|
| nothing (same seeds, nondeterministic kernels) | 3 | 2.9269 | 0.0007 |
| nothing, deterministic mode, 2× H100 + 1× A100 | 3 | 2.9273 | 0.0003 |
| data seed only | 4 | 2.9275 | 0.0012 |
| model (init) seed only | 4 | 2.9278 | 0.0036 |
| both seeds | 4 | 2.9289 | 0.0016 |

| setting (both seeds varied) | n | mean | std | × baseline std |
|---|---|---|---|---|
| d8 baseline | 4 | 2.9289 | 0.0016 | 1 |
| batch size 16 | 4 | 2.9305 | 0.0024 | 1.5 |
| 2× tokens (1.23B) | 3 | 2.8378 | 0.0017 | 1 |
| lr 9e-3 | 4 | 2.9528 | 0.0047 | **3** |
| d4 | 4 | 3.2838 | 0.0087 | **5** |

**How to think about it**

1. (a) Total variation is ~0.002, tight and symmetric, no outliers. Compare with P1: the
   smallest hyperparameter effect we called real (0.01) is 5 standard deviations. Nearly every
   recipe decision is far above noise; the exceptions (betas, clipping, cosine vs linear) are
   exactly the ones we called "noise" above.
2. (b) Initialization ≈ 3× data order > hardware. Hardware nondeterminism alone is 0.0007, and
   deterministic mode cuts it to 0.0003 *even across GPU types* — so "same seed, same result" is
   achievable if you pay for it. Caveat: with n = 3–4 each std has ~±40% uncertainty, so read
   "init 0.0036 vs data 0.0012" as "init is bigger", not as a precise ratio.
3. (c) Train-loss curves from different seeds separate early (~0.008 per step at step 500) and
   then run in parallel (~0.002 late) — they diverge, then don't re-converge or keep spreading.
4. (c) Only two things raise variability: pushing toward instability (lr 9e-3, 3×) or shrinking
   capacity (d4, 5×). Batch size and training length leave it at 0.002. More capable models are
   *less* noisy: bigger cuts std 5×; training longer lowers the mean by 0.09 with std flat, so
   noise relative to the improvements you care about shrinks with capability.
5. Intuition for why: a high LR takes big steps, so a small difference in the starting point is
   carried further each step. A small model has fewer near-equivalent solutions to fall into, so
   which one it lands in depends more on luck. Seed variance is a cheap "how close to the edge
   am I" meter.

### Example question (Medium): 10 seeds; does batch 16 or lr 9e-3 more than double the std?

**2 — only the higher learning rate.** Batch 16: 1.5× (within the ±40% error of n = 4: "same").
LR 9e-3: 2.9×. Batch at fixed tokens makes each step 2× noisier but gives 4× more steps to
average over, so the endpoint isn't noisier. A higher LR does not average out — each step is 3×
larger and carries seed differences further.

---

## Problem 4 — Amplification of randomness

### The question
Fix every source of randomness. Change one token in one training sequence. How much does the
final loss change (a)? How do the size of the perturbation and when it happens matter (b)?

### What we ran
Deterministic twins: an unperturbed control and a run where k tokens of one sequence are
overwritten at step T. k ∈ {1, 10, 100, 1024}, T ∈ {0, 25%, 50%, 90%}. 11 runs.

| perturbed at | tokens | \|Δ train loss\| first 100 steps | \|Δ\| plateau | Δ final val |
|---|---|---|---|---|
| (control) | 0 | 0.0000 | 0.0000 | 0.0000 |
| step 0 | 1 | 0.010 | 0.0027 | −0.0025 |
| step 0 | 10 | 0.009 | 0.0024 | −0.0011 |
| step 0 | 100 | 0.011 | 0.0024 | −0.0012 |
| step 0 | 1024 | 0.016 | 0.0026 | −0.0004 |
| 25% | 1 / 1024 | 0.0001 / 0.003 | 0.0003 / 0.0005 | +0.0001 / +0.0002 |
| 50% | 1 / 1024 | 0.0001 / 0.003 | 0.0001 / 0.0001 | 0.0000 |
| 90% | 1 / 1024 | 0.0001 / 0.002 | 0.0001 / 0.0001 | 0.0000 |

**How to think about it**

1. First check the control: the two deterministic runs are bit-identical (max |Δ| = 0.0) before
   the perturbation. Without that, none of the rest means anything. (Getting this to work on
   Modal required setting `CUBLAS_WORKSPACE_CONFIG` before CUDA initialised — see `APPROACH.md`.)
2. (a) One token out of 614 million changes the trajectory *permanently*: the per-step
   train-loss difference never returns to zero. But the final validation loss moves by ≤ 0.0025 —
   inside the seed noise. So: yes it matters, no you can't see it at the end.
3. (b) Size saturates. 1 token and 1024 tokens start 1.6× apart but reach the same plateau,
   ~0.0025 per step — which is the hardware-nondeterminism floor from P3. Once two runs are
   "different", they are as different as any two runs with different rounding; there is no
   "more different". Chaos in training is bounded by the attractor, not exponential forever.
4. Timing dominates. The same perturbation at 25/50/90% of training produces a 10–30× smaller
   divergence: the LR has decayed (small steps amplify less) and there are fewer steps left to
   compound. A perturbation's effect ≈ (LR at T) × (steps after T), and both shrink together.
5. Combined with P3: differences ≤ 0.003 between two d8 runs are not evidence of anything, even
   under a fixed seed.

---

## Problem 5 — Loss-curve augury

### The question
What shapes do loss curves take (a)? What sets the small-scale wiggle (b)? Which of our past
runs changed the shape (c)?

### What we ran
No new runs; all curves come from P1/P2/P7.

**Macro shape (a)**

| factor | what changes in the curve |
|---|---|
| peak LR | curves separate by step ~500 and hold their order until decay. lr 2.7e-2 sits *above* lr 3e-4 for 80% of training and finishes 0.13 *below* it |
| schedule | WSD looks worse than linear until its decay phase, then drops 0.2 in the last 20% of steps; constant LR never gets that drop |
| β₁ 0.8 / 0.9 / 0.95 | indistinguishable at every step |
| batch size | a level shift (bs 256 higher throughout, at fixed tokens) |

**Micro smoothness (b)** — std of train loss around its 51-step mean, second half of training:

| bs 16 | bs 32 | bs 64 | bs 128 | bs 256 | any LR (3e-4…2.7e-2) | any β | no clip | constant / WSD | wd 1.0 |
|---|---|---|---|---|---|---|---|---|---|
| 0.079 | 0.057 | 0.040 | 0.028 | 0.020 | 0.039–0.040 | 0.040 | 0.040 | 0.040 | 0.039 |

**How to think about it**

1. Curves have a predictable anatomy: a fast drop (first ~500 steps), a slow log-linear
   middle, and a final dip whose depth is set by the decay phase. The middle section's *height*
   is set by the current LR: higher LR = higher loss mid-run, lower at the end. So **never rank
   runs before their LRs have decayed to the same value** — mid-run rankings invert.
2. The wiggle is minibatch sampling noise and nothing else: 0.079 / 0.020 = 3.95 ≈ √16 for a
   16× batch change, and LR, betas, clipping, schedule, weight decay leave it at 0.040. A jagged
   curve tells you the batch size. It does not tell you the LR (the quiz below relies on this).
3. The optimizer's signal is in *spikes*, not jitter — and healthy runs here have none. Where
   spikes do appear (no QK-norm at high LR, P7) they precede a plateau at loss 4–5.
4. (c) Past runs that changed the shape: constant LR (no final drop), no-QK-norm at lr ≥ 9e-3
   (spike then plateau), SGD (flat at 6.6), weight decay 1.0 (bends upward late as decay
   overwhelms learning).

### Example question (Easy): babysitting a d8 curve

**(a) D, learning rate 0.009. (b) a, at most +0.05** (measured +0.028).

How to get there without guessing:
1. Count steps. "First quarter" spans ~2,340 steps, so the run has ~9,400 steps = 614M tokens
   at batch 64. Batch 256 would have 2,343 steps *in total*. B is out.
2. Read the level. From the plot: ~5.2 at step 100, ~3.9 at 500, ~3.65 at 800, ~3.35 at 2,000.
   Our curves at those steps — lr 9e-3: 5.18 / 3.90 / 3.64 / 3.37 (match); standard:
   4.98 / 3.71 / 3.50 / 3.26 (0.1–0.2 too low); lr 3e-4: 5.51 / 4.26 / 4.02 / 3.51 (too high).
   A higher LR is slightly *behind* early because it exits warmup with bigger, noisier steps; it
   overtakes only during decay.
3. Ignore the jitter: it's the batch-64 amount for every LR, so it can't distinguish A/C/D.
4. Terminal loss: lr 9e-3 is in the flat part of the LR basin (+0.026). 3× too high is mild;
   10× too low (+0.22) would have been the worry. With QK-norm on, nothing here diverges.

---

## Problem 6 — Activations and gradient norms

### The question
What do parameter, gradient and activation norms look like across training and across depth in a
healthy run (a)? How do interventions change them (b)? Can you steer them (c)?

### What we ran
RMS per module logged every 100 steps in all runs; a residual-stream probe after layers 1/3/6 on
8 fixed validation sequences; one extra broken run (lr 3e-2, no QK-norm, no clipping). 4 runs.

**Healthy d8 across training (a)**

| statistic | init | step 100 | mid | end |
|---|---|---|---|---|
| parameter RMS | 0.0422 | 0.0436 | 0.0977 | 0.0892 |
| gradient RMS | 2.7e-3 | 6.7e-5 | 1.9e-5 | 2.2e-5 |
| activation RMS | 0.94 | 1.56 | 3.16 | 2.59 |
| residual RMS after layer 6 | — | 12.1 | 23.1 | 14.1 |

**Interventions (b)/(c)**

| run | param RMS mid / end | grad RMS late | act RMS mid / end | residual L6 start → end | final val |
|---|---|---|---|---|---|
| healthy baseline | 0.098 / 0.089 | 2e-5 | 3.2 / 2.6 | 12 → 14 | 2.9277 |
| lr 3e-2, no QK-norm, no clip | 0.33 / 0.31 | 6e-4, spiky | **8800 / 970** | **5400 → 2200** (peak 1e5) | 5.1513 |
| wd 1.0 | 0.049 / 0.049 | 8e-5 | 1.4 / 1.3 | 9 → 0.5 | 2.9624 |
| wd 0 | 0.15 / 0.16 | 1.3e-5 | 7.0 / 7.2 | 13 → 67 | 2.9503 |
| lr 3e-4 | 0.043 / 0.042 | 7.6e-5 | 1.5 / 1.5 | | 3.1475 |
| lr 9e-3 | 0.19 / 0.16 | 1.6e-5 | 6.4 / 3.9 | | 2.9537 |
| constant LR | 0.12 / 0.13 | 1.8e-5 | 5.4 / 4.8 | | 3.1784 |
| bs 256 | 0.066 / 0.067 | 1.5e-5 | 2.4 / 2.3 | | 3.0123 |
| no clip, lr 3e-3 | 0.097 / 0.089 | 2.3e-5 | 3.2 / 2.6 | | 2.9320 |

**How to think about it**

1. Healthy anatomy: parameter RMS *rises* 2.3× then *falls* as LR decays (weight decay wins once
   the steps get small). Gradient RMS drops 100× in the first 100 steps and then sits flat.
   Activations grow with depth — the residual stream accumulates (~2 after layer 2, ~15 after
   layer 8) — and the layers are ordered and separated.
2. Parameter norm is an equilibrium set by `lr·wd` (the same product as P1(b)): wd 1.0 pins it at
   its init value, wd 0 lets it double and keep growing, lr 3e-4 never leaves init, constant LR
   keeps growing because the decay phase never arrives.
3. Activations follow parameters; gradients run *inversely*. RMSNorm makes the network
   scale-invariant, so ‖grad‖ ∝ 1/‖weights‖: wd 1.0 has 4× the gradient RMS of wd 0. A "big
   gradient" is not necessarily a problem — check what the weights are doing.
4. (c) To shrink everything uniformly: raise weight decay. To shrink only the end: use a
   decaying schedule (decay phase is when weight decay wins). To grow the end: wd 0 (residual
   goes 13 → 67).
5. The broken run is unmistakable and *early*: activations 10³–10⁴× normal, residual RMS peaking
   at 1e5, gradients 30× and spiky — all within ~500 steps, before the loss curve looks odd.
   Without QK-norm the attention logits scale with the activations, softmax saturates, and the
   run never recovers (final 5.15).

### Example question (Easy): residual-stream RMS traces at layers 1/3/6 — healthy or broken?

**Alternative (broken).** Three tells: the scale (10⁵–10⁶, versus ~10–25 in a healthy d8);
the shape (all three layers collapse onto one line — the stream is dominated by one enormous
component written in the first steps, versus ordered, separated layers in health); the timing
(step 0 at ~1, matching our init 0.94, then a vertical jump — the first warmup steps at an LR the
attention logits can't take). The slow downward slope afterwards is weight decay eroding the blob,
which is why the run doesn't NaN but never recovers. It matches our lr 3e-2 / no QK-norm / no clip
run: 5,400 → peak ~1e5 → 2,200.

---

## Problem 7 — Our own prediction problem: QK-norm and tied embeddings

### The question we posed
The d8 recipe normalises queries and keys before attention (`qk_norm=True`).
(i) Without QK-norm, at which peak LR does the run first finish more than 0.5 above baseline?
(ii) Tying input and output embeddings removes 1.6M parameters (4.5%). Does it change the final
loss by more than 0.01? Code: `p7_own_prediction_problem.py` (6 runs).

Hypothesis before running: (i) low LRs unaffected, a cliff somewhere between 9e-3 and 3e-2;
(ii) tying is free.

| peak LR | QK-norm on | QK-norm off | Δ |
|---|---|---|---|
| 1e-3 | 2.9768 | 2.9857 | +0.009 |
| 3e-3 | 2.9277 | 2.9479 | +0.020 |
| 9e-3 | 2.9537 | **4.0429** | +1.09 |
| 3e-2 | 3.0257 | **5.1513** | +2.13 |

| tied embeddings (QK-norm on) | lr 3e-3 | lr 9e-3 |
|---|---|---|
| untied | 2.9277 | 2.9537 |
| tied | 2.9391 | 2.9495 |
| Δ | +0.011 | −0.004 |

**How to think about it**

1. (i) The cliff is between 3e-3 and 9e-3 — one factor-of-3 step. With QK-norm the loss varies by
   0.1 across a 30× LR range; without it, a 3× LR increase is fatal and the run plateaus at 4.0
   (roughly bigram level) forever. QK-norm isn't a small tweak; it is what makes "LR is forgiving
   on the high side" (P1) true at all.
2. Even in the stable regime QK-norm is worth 0.01–0.02, and P2(c) showed the no-QK-norm failure
   gets worse with width. Removing it is a bad trade at every LR.
3. The mechanism is P6's: attention logits ∝ ‖q‖‖k‖ ∝ activation scale; a high LR grows the
   activations, softmax saturates, gradients spike, the residual stream fills with one component.
4. (ii) Tying is neutral (+0.011 / −0.004, straddling the noise floor). 4.5% of parameters in the
   embedding table are not doing 4.5% of the work at this scale.

---

## How the experiments themselves went (the operational lesson)

1. Cost estimates were off twice: first 2× too pessimistic (old runs took 25 min; ours took 11),
   then a placeholder W&B entity hung a wave in `wandb.init` for ~5 GPU-hours. Calibrate on a real
   run before planning a budget.
2. Modal reuses one Python process across jobs. That leaked (a) `CUBLAS_WORKSPACE_CONFIG`, which
   is read at CUDA init, so deterministic runs crashed unless they were first in the container;
   (b) poisoned Torch state after a crash; (c) an open `wandb.run` after an OOM, so the next job
   logged into the wrong W&B record. Two records were quarantined (`a1-corrupted`) and rerun.
   Lesson: after any failure, drain the container and finish the W&B run; verify each W&B
   record's config against its name before analysing.
3. We trimmed 28 low-value runs (two-factor P1 interactions beyond the three grids, P2 extras) to
   land at 34.1 GPU-hours. Every dropped config is in `trim_dropped.csv`.
