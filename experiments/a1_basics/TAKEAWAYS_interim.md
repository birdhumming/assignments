# A1 interim takeaways (73 / 141 runs finished, 2026-09-29 05:40 UTC)

Everything here is from the W&B project `aleyang-stanford-university/assignments`
(tag `a1`). P3 and P4 are essentially complete; P2 is ~half done; P1/P5/P6/P7
are still queued. Numbers will be refreshed when the suite finishes.

Plots: `p3_variation.png`, `p4_divergence.png`, `p2_scaling.png` (attached).

---

## The one-sentence version

The d8 model is *remarkably* reproducible — every source of randomness we can
name (init seed, data order, GPU nondeterminism, a corrupted token) changes the
final val loss by ≤ 0.005 nats — which is *smaller* than every recipe decision
we tried (0.03–0.25 nats), so recipe conclusions from a single run are
trustworthy at this scale; but P2 shows those recipe gaps do **not** stay
constant with model size, so a recipe ranking made at d4 can be wrong at d9.

---

## P3 — how much do runs vary? (24/26 done)

Final val loss of the d8 baseline (lr 3e-3, bs 64, 614M tokens), std across 3
runs, by *what* was varied:

| what varies                         | mean  | std      | range   |
|-------------------------------------|-------|----------|---------|
| nothing (same seeds, `deterministic=False`, 3 reps) | 2.927 | 0.0007 | 0.0012 |
| data seed only (same init)          | 2.927 | 0.0015   | 0.0030  |
| model seed only (same data order)   | 2.928 | 0.0044   | 0.0084  |
| both seeds                          | 2.929 | 0.0017   | 0.0032  |
| both seeds, batch size 16           | 2.931 | 0.0024   | 0.0046  |
| both seeds, 2× tokens (1.23B)       | 2.838 | 0.0017   | 0.0033  |
| both seeds, lr 9e-3 (2 of 3 done)   | 2.953 | 0.0080   | 0.0114  |
| both seeds, **d4** model            | 3.282 | 0.0093   | 0.0172  |

Intuition:

* **Init seed matters more than data order.** With only 3 samples the std
  estimates are rough (±~40 %), but model-seed-only (0.0044) is consistently
  the widest d8 group and data-seed-only (0.0015) the narrowest. Once you have
  614M tokens, *which* order they arrive in washes out; *where you started* in
  weight space leaves a slightly larger fingerprint.
* **GPU nondeterminism is not free but is tiny**: identical seeds without
  `deterministic=True` give runs that differ by ~0.001 at the end. On the
  per-step train loss the two "identical" runs differ by ~0.008 in the first
  100 steps and ~0.002 late — so nondeterministic kernels perturb the
  trajectory immediately, but the perturbation does not grow.
* **Smaller models are noisier**: d4 std is ~5× d8 std (0.009 vs 0.002). This
  is the standard picture — fewer parameters, fewer directions to average over,
  more sensitivity to init. Consequence for P2: the *small* end of a scaling
  curve is the least reliable end, exactly the end you would like to fit on.
* **Higher LR is noisier**: lr 9e-3 has std 0.008 vs 0.002 at 3e-3. Loss
  variance is a cheap proxy for "how close to the edge of stability am I".
* **Batch size 16 barely changes the mean (+0.002) or the noise.** With
  the same token budget, 4× more optimizer steps at 4× smaller batch lands in
  the same place — we are in the regime where bs 64 is still below the
  critical batch size, so steps and tokens are interchangeable.
* **Seed noise over training**: seed-std of val loss is 0.005 at 10 % of
  training, 0.001–0.002 from 25 % onward. Early loss curves are the noisy part;
  by a quarter of the way in, runs have "found their lane".

Practical rule from this: at d8, a difference of **< 0.005** between two single
runs is noise; **> 0.01** is real. That threshold is used for everything below.

---

## P4 — does a one-token corruption amplify? (10/11 done)

Setup: `deterministic=True` runs, identical except that at a chosen step,
1 / 10 / 100 / 1024 tokens of one training sequence are overwritten with token
17. Because everything else is bit-identical, we compare *train loss at every
step* to an unperturbed twin (`|Δ train loss|`, `p4_divergence.png`).

First, the determinism fix works: two runs that differ only in a perturbation
at step 8437 are **bit-identical (max |Δ| = 0.0) for all 8437 preceding steps**.
That's the control that makes the rest meaningful.

What happens after the perturbation:

| perturb at | tokens | |Δ| first 100 steps | |Δ| steady state | Δ final val |
|-----------|--------|------------------|----------------|-------------|
| step 0    | 1      | 0.010            | 0.0027         | −0.0025     |
| step 0    | 10     | 0.009            | 0.0024         | −0.0011     |
| step 0    | 100    | 0.011            | 0.0024         | −0.0012     |
| step 0    | 1024   | 0.016            | 0.0026         | −0.0004     |
| 25 %      | 1      | 0.0001           | 0.0003–0.0004  | +0.0001     |
| 25 %      | 1024   | 0.003            | 0.0004–0.0005  | +0.0002     |
| 50 %      | 1      | 0.0001           | 0.0001         | 0.0000      |
| 50 %      | 1024   | 0.003            | 0.0001         | 0.0000      |
| 90 %      | 1      | 0.0001           | 0.0001         | 0.0000      |
| 90 %      | 1024   | 0.002            | 0.0001         | 0.0000      |

Reference ceilings: two runs with the *same data order but different init
seeds* differ by ~0.008 per step late in training; two *nondeterministic* runs
with the same seeds differ by ~0.002.

Intuition:

* **Yes, a single token amplifies — but it saturates, it doesn't explode.** A
  1-token change at step 0 produces trajectories that differ by ~0.0027 per
  step for the rest of training, i.e. about the same as GPU nondeterminism and
  ~⅓ of the "different init seed" ceiling. Training is chaotic in the weak
  sense (tiny cause → macroscopic, permanent divergence) but the divergence is
  bounded by the noise floor of SGD itself; it does not keep growing.
* **Size of the perturbation almost doesn't matter** once it's nonzero: 1 vs
  1024 tokens at step 0 land on the same ~0.0025 plateau. The first ~100 steps
  see a bigger kick from 1024 tokens (0.016 vs 0.010), but the dynamics forget
  the *magnitude* and remember only *that* they were kicked. This is the
  signature of a chaotic system with a finite attractor width.
* **Timing matters a lot, and monotonically.** The same 1-token change gives a
  steady-state |Δ| of 0.0027 at step 0, 0.0003 at 25 %, 0.0001 at 50 % and 90 %.
  Late training has a decaying learning rate (linear to 0), so the same
  gradient nudge moves the weights less *and* there are fewer steps left for
  the difference to compound. Early in training with a large LR, the system is
  most sensitive.
* **Δ final val is essentially zero everywhere** (|Δ| ≤ 0.0025, i.e. inside
  P3's seed noise). So "amplification" is about the trajectory, not the
  destination: the perturbed model ends up equally good, just a different
  model. If you only look at final eval you would conclude nothing happened.
* The 1024-token perturbations at 25 / 50 / 90 % show a visible ~0.003 spike
  right at the perturbed step and a second bump ~700 steps later (visible in
  the plot for 25 % and 50 %). The first is the corrupted batch itself; the
  second I don't have a confirmed explanation for — my hypothesis is that
  it is the Adam second-moment estimate (EMA with β2 = 0.95–0.999, i.e. a
  memory of hundreds of steps) finishing its response to the corrupted
  gradient. Worth checking against the optimizer's β2 once P1's β2 runs land.

---

## P2 — are scaling laws recipe-independent? (29/56 done)

Final val loss vs depth (hidden = 64·depth, 614M tokens for all):

| recipe            | d4    | d5    | d6    | d7    | d8    | d9    | fitted exponent |
|-------------------|-------|-------|-------|-------|-------|-------|-----------------|
| baseline lr 3e-3  | 3.290 | 3.154 | 3.056 | 2.984 | 2.928 | 2.882 | 0.054 |
| lr 1e-3           | 3.353 | 3.219 | 3.117 |  –    |  –    |  –    | 0.060 |
| lr 3e-2           | 3.316 | 3.213 | 3.110 | 3.070 | 3.026 | 2.985 | 0.043 |
| constant LR       | 3.428 | 3.330 | 3.243 | 3.220 | 3.178 | 3.124 | 0.037 |
| dropout 0.2       | 3.452 | 3.321 | 3.216 | 3.129 | 3.059 | 3.003 | 0.058 |

(non-embedding params: d4 3.8M · d5 7.4M · d6 12.8M · d7 20.4M · d8 30.4M · d9 43.3M)

Intuition:

* **The exponent is recipe-dependent** (0.037–0.060 across recipes). The
  slope is the thing people extrapolate with, and it moves by ±30 % depending
  on choices that have nothing to do with architecture or data.
* **Gap to baseline is not constant** (right panel of `p2_scaling.png`):
  * constant LR: gap **grows** 0.14 → 0.24 from d4 to d9. Bigger models
    benefit more from LR decay (they have more to "anneal" into at the end).
  * lr 3e-2: gap **grows** 0.03 → 0.10. A too-high LR is nearly harmless for a
    tiny model and increasingly costly as width grows — consistent with the
    optimal LR shrinking with width (µP intuition).
  * dropout 0.2: gap **shrinks** 0.16 → 0.12. Regularisation costs less as
    the model gets bigger; extrapolate far enough and it may cross zero.
  So the *ranking* of recipes is stable here, but the *magnitudes* trend in
  different directions, and rankings can flip further out.
* **Extrapolation is bad even within one recipe.** Fitting a power law on
  d4–d6 only and predicting d9: baseline predicts 2.837 (actual 2.882),
  constant predicts 3.070 (actual 3.124), lr 3e-2 predicts 2.922 (actual
  2.985). Three-point fits are optimistic by 0.05–0.06 nats — an order of
  magnitude more than the seed noise. Dropout is the exception (pred 2.996,
  actual 3.003) because it is the one recipe whose curve is genuinely straight
  in log-log.
* **Why the baseline curve bends**: it is a *fixed-token* curve, not a
  compute-optimal one. 614M tokens is ~20 tokens/param for d8 (~Chinchilla)
  but ~160 tokens/param for d4 — the small models are heavily over-trained and
  d9 is starting to be under-trained. So the "scaling law" you measure at
  fixed tokens is really a slice through a 2-D (N, D) surface, and its
  curvature is an artefact of where the slice goes. Pending runs
  (`tok76.8M / 154M / 307M` at d6 and d8) are there to show the data-size axis.
* Small-model noise (P3: d4 std 0.009) is larger than some of the residuals of
  the d4–d9 fits (≈0.01), so with one run per point you cannot distinguish
  "the curve bends" from "d4 got a lucky seed" at the small end.

Pending in P2: SGD, no-warmup+no-clip and 8-epoch "law breakers", bs 256
family, lr 1e-3 for d7, data-size axis.

---

## P6 (1/4 done), P1 / P5 / P7 (queued)

The healthy baseline with activation probes logged fine (residual RMS at
layers 1/3/6, per-module gradient/parameter RMS); comparison against the
stressed (lr 3e-2, no clip, no QK-norm) and weight-decay-extreme runs is
pending. One early P2 data point relevant to P7: at d4, removing QK-norm at
lr 3e-2 was catastrophic (val 4.22 vs 3.32 with QK-norm — a 0.9-nat hit, the
largest single effect in the suite so far). That is the "high LR without
QK-norm blows up" branch of the P7 prediction; the d8 branch and
lower LRs are still queued.

---

## Operational lessons (short; full postmortems in APPROACH.md)

* Deterministic runs need `CUBLAS_WORKSPACE_CONFIG` *before* CUDA init —
  setting it in `train()` is too late when Modal reuses the process.
* Measured runtimes: d8 ≈ 11–12 min on H100, d9 ≈ 35 min, and d4 anywhere
  from 3.6 to 8.3 min for the same step count — small models are dominated by
  per-step overhead and compile/cache state rather than FLOPs, so my
  depth-cost table (d4 = 0.3× d8) was wrong at both ends. Dropout runs
  are ~2× slower (27 vs 12 min at d8); I have not confirmed why — likely the
  dropout RNG ops interacting badly with `torch.compile`.
