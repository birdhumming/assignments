# A1 takeaways — final (P1–P7)

Data: W&B project `aleyang-stanford-university/assignments`, tags `a1` (116
finished runs: all 113 kept after the trim, plus 3 that finished before it) + `a1-basics-p1b-v3` (21 earlier P1 grid points
reused by P1/P5). 28 low-priority runs were trimmed (`trim_dropped.csv`), two
W&B records were quarantined as `a1-corrupted` (see "Operational lessons"),
and their four configs were rerun cleanly (`repair_manifest.csv`,
`repair2_manifest.csv`). Analysis code: `analysis/pull_wandb.py` (download
histories) → `analysis/analyze.py` (all tables + plots); the plots referenced
below are in `figures/`.

Every number is *final validation loss* (nats, 1000 held-out sequences) unless
stated. P3 gives the noise floor for the whole document: **two single d8 runs
that differ by < 0.005 are indistinguishable; > 0.01 is real.**

---

## The five things I'd remember

1. **Learning rate and its schedule are the only "macro" knobs.** 10× too-small
   LR costs 0.22 nats, no decay costs 0.25; every other single hyperparameter
   (batch, warmup, weight decay, betas, clipping) moves the result by ≤ 0.09 and
   most by ≤ 0.03.
2. **Hyperparameters interact through products and timescales, not
   independently.** Optimal weight decay falls as LR rises (AdamW's decay per
   step is `lr·wd`; the sweet spot here is `lr·wd ≈ 1e-3`, i.e. a ~1000-step
   memory); higher LR wants longer warmup; a too-low LR hurts more as the batch
   grows (fewer steps).
3. **Scaling-law exponents and extrapolations are recipe-dependent.** Six
   recipes give exponents 0.042–0.076 on the *same* model family and data; a
   d4–d6 fit over-predicts d9 by 0.03–0.08 nats; at d20 the recipes disagree by
   0.35 nats. Some interventions (SGD, no QK-norm at high LR, data repetition)
   don't just shift the law — they flatten or *invert* it.
4. **Training is reproducible to ~0.002 nats and chaotic at the same time.** All
   randomness sources (seeds, GPU nondeterminism, a single corrupted token)
   change the final loss by ≤ 0.005 but permanently change the trajectory; the
   divergence saturates at the SGD noise floor rather than exploding.
5. **You can read the batch size off a loss curve but nothing else from its
   wiggle**; you can read the LR/schedule off its *shape*; and the earliest
   warning of a broken run is activation RMS, not loss.

---

## P1 — hyperparameter sensitivity (d8, 614M tokens, `p1a_sweeps.png`, `p1b_pairs.png`, `p1c_schedules.png`)

### (a) One at a time, around the baseline (lr 3e-3, bs 64, warmup 1 %, wd 0.1, linear decay)

| knob | values → final val | reading |
|---|---|---|
| LR | 3e-4 **3.147** · 1e-3 2.977 · **3e-3 2.928** · 9e-3 2.954 · 2.7e-2 3.017 | Asymmetric basin: 10× too small costs 0.22, 9× too large costs 0.09. Undershooting LR is the expensive mistake. |
| batch (fixed tokens) | 16 2.928 · 32 **2.918** · 64 2.928 · 128 2.952 · 256 3.012 | Flat up to 64, then −0.024/−0.06 per doubling. At fixed tokens, bigger batch = fewer steps; past the critical batch size the extra steps were doing the work. (LR was *not* re-tuned per batch — see (b).) |
| warmup | 0 **3.034** · 0.3 % 2.940 · 1 % 2.928 · 3 % 2.924 · 10 % **2.920** | No warmup at all costs **+0.11** (the biggest P1 effect after LR and schedule) — the first few hundred steps at full LR do lasting damage; but 0.3 % (28 steps) already recovers 0.09 of it, and 0.3 % → 10 % is a slow, monotone 0.02 more. Warmup is cheap insurance with a steep first dollar. |
| weight decay | 0 2.950 · 0.01 2.947 · 0.03 2.941 · 0.1 2.928 · 0.3 **2.918** · 1.0 2.962 | Broad optimum at 0.3; 1.0 over-regularises (+0.04). Even wd 0 → 0.3 is only 0.03 — wd is a fine-tuning knob, not a make-or-break one. |

### (b) Pairs — do they co-vary? Yes, in three specific ways

* **LR × batch.** Penalty for *too-low* LR grows with batch: at bs 32, lr 1e-3
  costs +0.024 vs the best; at bs 128 it costs +0.073. Penalty for too-high LR
  is ~constant (+0.03–0.04). Intuition: large batch → few steps → a small LR
  simply cannot travel far enough; the "optimal LR grows with batch" rule is
  really "the minimum acceptable LR grows with batch".
* **LR × weight decay.** Best wd is 0.3 at lr 1e-3 (2.951 vs 2.977 at 0.1),
  0.3 at lr 3e-3 (2.918), but 0.1 at lr 9e-3 (2.954; wd 0.3 → 2.978). The
  products `lr·wd` of the winners are 3e-4, 9e-4, 9e-4 and the losers (lr
  9e-3 × 0.3 = 2.7e-3, lr 3e-3 × 1.0 = 3e-3) are all ≥ 2.7e-3. AdamW multiplies
  weights by `(1 − lr·wd)` each step, so `1/(lr·wd)` is the memory of the
  weights in steps: ~1000 steps (≈10 % of training) is good, ~400 is too
  forgetful. Tune the *product*, not wd alone.
* **LR × warmup.** Warmup 1 % → 10 % gains 0.009 at lr 1e-3, 0.008 at 3e-3
  and **0.040 at 9e-3**. Higher LR needs more warmup, and with 10 % warmup the
  best LR shifts up (9e-3 → 2.914 beats 3e-3 → 2.920). Warmup is what buys
  you access to a higher LR.

(The remaining two-factor cells — lr 1e-3/9e-3 × bs 16/256, × wd 0.01/1.0, ×
warmup 0, bs × wd 1.0, lr 2.7e-2 × warmup 10 % — were trimmed to save ~4 h.)

### (c) Schedules, betas, clipping (lr 3e-3)

| | final val | |
|---|---|---|
| linear-to-zero (default) | 2.928 | |
| cosine | 2.935 | ≈ linear (within 0.007) |
| constant (no decay) | **3.178** | +0.25 — by far the biggest single effect in P1 |
| WSD, 20 % decay | 2.937 | flat line, then a 20 %-long plunge that lands within 0.01 of linear |
| WSD, 50 % decay | **2.919** | best schedule tested |
| β₁ 0.8 / 0.9 / 0.95 | 2.930 / 2.928 / 2.934 | noise |
| β₂ 0.9 / 0.95 / 0.99 | 2.932 / 2.928 / 2.923 | 0.99 slightly better, borderline (0.005) |
| no grad clipping | 2.932 | +0.004: clipping is inactive at lr 3e-3 (P6 confirms gradient norms are far below 1) |

Intuition for the schedule result: `p1c_schedules.png` right panel shows the
WSD runs *above* linear for 80 % of training and then dropping ~0.2 nats in the
decay phase. The loss you see mid-training with a high LR is an "iterate-noise"
floor set by LR × gradient variance; decay removes that noise and reveals where
the weights actually are. Constant LR never removes it, hence +0.25. The
other Adam knobs matter only near instability (see P7).

---

## P2 — are scaling laws reliable? (`p2a_scaling.png`, `p2b_benders.png`, `p2c_breakers.png`)

Model family d4…d9 (hidden = 64·depth; total params 5.9M · 10.0M · 16.0M ·
24.0M · 34.6M · 48.0M incl. embeddings), 614M tokens for every point.

### (a) Same family, four recipes

| recipe | d4 | d5 | d6 | d7 | d8 | d9 | exponent (6 pts) | exponent (d4–d6) | d4–d6 fit → d9 (actual) | fit → d20 (370M params) |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 3.290 | 3.154 | 3.056 | 2.984 | 2.928 | 2.882 | 0.063 | 0.074 | 2.814 (2.882) | 2.52 |
| constant LR | 3.428 | 3.330 | 3.243 | 3.220 | 3.178 | 3.124 | 0.042 | 0.056 | 3.051 (3.124) | 2.87 |
| dropout 0.2 | 3.452 | 3.321 | 3.216 | 3.129 | 3.059 | 3.003 | 0.067 | 0.071 | 2.972 (3.003) | 2.61 |
| lr 3e-2 | 3.316 | 3.213 | 3.110 | 3.070 | 3.026 | 2.985 | 0.050 | 0.064 | 2.901 (2.985) | 2.68 |

* The exponent depends on the recipe (0.042–0.067, ±25 % around baseline) —
  and the exponent is the thing you extrapolate with. Gaps to baseline grow
  with size for constant-LR (0.14 → 0.24) and lr 3e-2 (0.03 → 0.10) but shrink
  for dropout (0.16 → 0.12): the recipe *ranking* is stable to d9 but the
  extrapolated d20 predictions span 2.52–2.87.
* Three-point fits are optimistic everywhere: predicting d9 from d4–d6 misses
  by 0.03 (dropout), 0.07 (baseline, constant) and 0.08 (lr 3e-2) — 10–40×
  the seed noise. The fixed-token slice is *not* a straight power law: d4 is
  trained on ~100 tokens/param, d9 on ~13, so the small end is over-trained
  and the large end under-trained, which bends the curve downward in log-log.
  Dropout is straightest because regularisation partly cancels the
  over-training at the small end.
* Seed noise at d4 (P3: std 0.009) is comparable to the residuals of these
  fits, so with one run per point you cannot separate "the law bends" from
  "d4 drew a lucky seed" at the small end — the end you'd most like to fit on.

### (b) Slope benders — mild, "reasonable" changes

* **Model-size axis.** lr 1e-3: 3.353 · 3.219 · 3.117 · 3.036 · 2.977 (d4–d8),
  exponent 0.067 vs 0.063; the gap to baseline shrinks 0.063 → 0.049 with size
  (bigger models tolerate a lower LR — the optimum LR falls with width).
  bs 256: 3.362 · 3.227 · 3.132 · – · 3.012 (d4–d6, d8), exponent ~0.06; the gap
  to baseline is a steady 0.07–0.08. Neither change is exotic, yet each moves the exponent
  by ~5–10 %.
* **Data axis** (76.8M → 1.23B tokens): d8 at lr 3e-3 follows a clean
  exponent of 0.078 over 16× data (3.535 → 2.837); d6 gives 0.075. At lr 1e-3
  the *same* models give 0.086 (d8) and 0.093 (d6) — a lower LR looks like a
  steeper data law because it is under-trained at short horizons and catches
  up later (d6 gap 0.22 at 77M tokens → 0.06 at 614M). Two people using two
  sane LRs would publish two different data-scaling exponents.

### (c) Law breakers (d4 / d6 / d8; baseline 3.290 / 3.056 / 2.928)

| intervention | d4 | d6 | d8 | what happened to the law |
|---|---|---|---|---|
| SGD (same LR/schedule) | 6.586 | 6.599 | 6.654 | **flat/inverted**: at unigram-level loss, model size is irrelevant — the optimizer, not capacity, is the bottleneck |
| no QK-norm, lr 3e-2 | 4.217 | 4.941 | 5.151 | **inverted**: bigger is *worse*. Instability grows with width (attention logits scale with hidden size) |
| lr 3e-2, no warmup, no clip | 3.324 | – | 3.034 | law survives (+0.03 / +0.11); gap grows with size, same story as lr 3e-2 above |
| 8 epochs × 75k sequences (same tokens) | 3.325 | 3.145 | 3.090 | **flattened**: gap 0.035 → 0.09 → 0.16, exponent ~0.04. Bigger models memorise the repeated 77M tokens instead of generalising — repeated data is worth less per token the larger the model |

Take-home: a scaling law is a statement about a *recipe family*, and any
knob that changes with scale (stability, memorisation, optimizer efficiency)
changes the law. Extrapolate only within the recipe you measured, from the
size range you measured, and add the seed noise of the smallest points to
your error bars.

---

## P3 — run-to-run variation (`p3_variation.png`)

Std of final val across replicas (3–4 runs each; the seed-42 baseline is a member of every seed group), d8 unless stated:

| what varies | mean | std |
|---|---|---|
| nothing (same seeds, nondeterministic kernels) | 2.927 | 0.0007 |
| nothing (same seeds, `deterministic=True`, 2× H100 + A100) | 2.927 | 0.0003 |
| data seed only | 2.928 | 0.0012 |
| model (init) seed only | 2.928 | 0.0036 |
| both seeds | 2.929 | 0.0016 |
| both seeds, bs 16 | 2.931 | 0.0024 |
| both seeds, 2× tokens (1.23B) | 2.838 | 0.0017 |
| both seeds, lr 9e-3 | 2.953 | 0.0047 |
| both seeds, **d4** | 3.284 | 0.0087 |
| hardware: same seeds, `deterministic=True`, H100 vs A100 | Δ = 0.0001 | (two deterministic H100 refs in different containers: Δ = 0.0006) |

* Init seed leaves a ~3× bigger fingerprint than data order; once 614M tokens
  have gone by, their order washes out but the starting point does not.
* Hardware barely matters (H100 vs A100, deterministic mode: 0.0001) — but two `deterministic=True` runs in *different* containers still differ by 0.0006, so "deterministic" is only bit-exact within one process/compile cache. GPU nondeterminism alone is ~0.001 — real, tiny, and does not grow (per-step
  |Δ train loss| between two "identical" runs is 0.008 in the first 100 steps
  and 0.002 late).
* **(c) Do hyperparameters change variability?** Only the ones that push
  toward instability or small capacity: lr 9e-3 is 3× noisier (0.0047 vs
  0.0016; it looked 4–5× with only 2 seeds — n=3–4 stds carry ±35–40 % error),
  d4 is 5× noisier. Batch size (16 vs 64) and training length (1.23B vs 614M) leave the
  std unchanged. So *more capable* models are less noisy: bigger cuts the std
  ~5×, longer keeps it flat while lowering the mean, so the *relative* noise
  shrinks with capability. Loss variance across seeds is a cheap "how close to
  the edge am I" signal.
* Seed std of val loss is 0.005 at 10 % of training and 0.001–0.002 from 25 %
  on: runs "find their lane" early.

---

## P4 — amplification of a single corrupted token (`p4_divergence.png`)

Deterministic twins that differ only in 1/10/100/1024 tokens of one sequence
overwritten at step 0, 25 %, 50 % or 90 %; compare train loss at every step.
Control: twins are bit-identical (max |Δ| = 0.0) for all steps before the
perturbation — the determinism fix works.

| perturbed at | tokens | |Δ| first 100 steps | |Δ| steady state | Δ final val |
|---|---|---|---|---|
| step 0 | 1 / 10 / 100 / 1024 | 0.010 / 0.009 / 0.011 / 0.016 | 0.0027 / 0.0024 / 0.0024 / 0.0026 | −0.0025 / −0.0011 / −0.0012 / −0.0004 |
| 25 % | 1 / 1024 | 0.0001 / 0.003 | 0.0003 / 0.0005 | +0.0001 / +0.0002 |
| 50 % | 1 / 1024 | 0.0001 / 0.003 | 0.0001 / 0.0001 | 0.0000 |
| 90 % | 1 / 1024 | 0.0001 / 0.002 | 0.0001 / 0.0001 | 0.0000 |

* Yes, one token amplifies — permanently — but it **saturates**: the
  steady-state divergence (0.0027/step) equals the GPU-nondeterminism floor
  and ~⅓ of the different-init-seed ceiling (0.008). Chaotic in the weak
  sense: tiny cause, macroscopic permanent effect, bounded by SGD's own noise.
* Magnitude is forgotten, timing is remembered: 1 vs 1024 tokens land on the
  same plateau, but step 0 vs 25 % vs 50 % differ 10–30× (decaying LR moves
  the weights less and leaves fewer steps to compound).
* Final val is unchanged (|Δ| ≤ 0.0025, inside seed noise): amplification is
  about the *trajectory*, not the destination. The models end up equally good
  and different.
* The ~700-step "echo" bump after late perturbations is consistent with the
  Adam second-moment EMA (β₂ = 0.95 has a ~20-step memory, so this is *not*
  the β₂ memory; more likely the corrupted sequence's row being re-sampled by
  the evaluation or a slow mode of the LR-scaled dynamics — unresolved).

---

## P5 — loss-curve augury (`p5a_macro.png`, `p5b_micro.png`)

**Macro (what the shape tells you).** Peak LR separates the curves by step
~500 and the ordering is stable *until the decay phase*: lr 2.7e-2 sits above
lr 3e-4 for 80 % of training and finishes 0.13 below it. WSD curves look worse
than linear-decay until their decay starts, then drop 0.2 nats in 20 % of the
steps. β₁ 0.8/0.9/0.95 curves are indistinguishable at every step. Rule: a
high-LR or un-decayed curve can look bad until annealing; never rank runs
before their LR has decayed to the same level.

**Micro (what the wiggle tells you).** Jitter = std of train loss around its
51-step running mean, second half of training:

| bs 16 | bs 32 | bs 64 | bs 128 | bs 256 | everything else (LR ×90, β₁, β₂, clip, schedule, wd) |
|---|---|---|---|---|---|
| 0.079 | 0.057 | 0.040 | 0.028 | 0.020 | 0.039–0.040 |

Jitter is exactly ∝ 1/√batch (0.079/0.020 = 3.95 ≈ √16) and *nothing else
moves it*. The train-loss wiggle is the sampling noise of a minibatch loss
estimate, not optimizer noise — so it diagnoses batch size and nothing about
the optimizer. Spikes are a different story (see P6/P7).

---

## P6 — activation & gradient statistics (`p6a_layers.png`, `p6c_interventions.png`)

Per-module RMS of parameters / gradients / activations every 100 steps (all
runs) plus a residual-stream probe after layers 1/3/6 on 8 fixed val
sequences (`-actprobe` runs).

**(a) Healthy d8.** Parameter RMS grows from init 0.042 to 0.098 by
mid-training then *falls* to 0.089 as the LR decays (weight decay wins once
the gradient step shrinks). Gradient RMS collapses 2.7e-3 → 7e-5 in the first
100 steps and drifts to 2e-5. Activation RMS 0.94 → 3.2 → 2.6. By layer:
attention `o_proj` weights and all activations grow with depth (the residual
stream accumulates: `mlp.down_proj` output RMS goes from ~2 at layer 2 to ~15
at layer 8); at step 100 the first layer has 10× the gradient of the others,
and by mid-training gradients are flat across layers. Residual RMS after
layer 6: 12 → 23 → 14.

**(b, c) Interventions.**

| run | param RMS (mid/end) | grad RMS (late) | act RMS (mid/end) | residual RMS L6 (start→end) | final val |
|---|---|---|---|---|---|
| healthy | 0.098 / 0.089 | 2e-5 | 3.2 / 2.6 | 12 → 14 | 2.928 |
| lr 3e-2, no QK-norm, no clip | 0.33 / 0.31 | 6e-4, spiky (10–100×) | 8800 / 970 | 5400 → 2200 (peak 1e5) | 5.151 |
| wd 1.0 | 0.049 / 0.049 | 8e-5 | 1.4 / 1.3 | 9 → 0.5 | 2.962 |
| wd 0.0 | 0.15 / 0.16 | 1.3e-5 | 7.0 / 7.2 | 13 → 67 | 2.950 |
| lr 3e-4 | 0.043 / 0.042 | 7.6e-5 | 1.5 / 1.5 | | 3.147 |
| lr 9e-3 | 0.19 / 0.16 | 1.6e-5 | 6.4 / 3.9 | | 2.954 |
| constant LR | 0.12 / 0.13 | 1.8e-5 | 5.4 / 4.8 (58 at step 100) | | 3.178 |
| no clip (lr 3e-3) | = healthy | = healthy | = healthy | | 2.932 |

* **Parameter norm is an equilibrium set by `lr·wd`**: wd 1.0 pins weights at
  init scale, wd 0 lets them double, lr 3e-4 never moves them off init (the
  model is under-trained, matching its 3.147 loss), constant LR keeps growing
  them because the decay phase that normally shrinks them never comes. This is
  the same `lr·wd` timescale P1(b) found empirically.
* **Activation scale tracks parameter scale, and gradient RMS runs inversely**
  (wd 1.0: small weights, 4× larger gradients; wd 0: large weights, 40 %
  smaller gradients). With RMSNorm everywhere the network is scale-invariant,
  so `‖grad‖ ∝ 1/‖w‖`; the *effective* step size `lr·‖g‖/‖w‖` is what
  weight decay actually controls.
* **The unhealthy signature is unmistakable within 500 steps**: activations
  10³–10⁴× baseline, residual stream at 10⁵, and gradient RMS that is both
  30× larger and spiky. Loss (5.15) only tells you *that* it failed;
  activation RMS tells you *where* (attention logits without QK-norm) and
  *when*.
* Gradient clipping at 1.0 is a no-op for the healthy recipe (curves overlap
  exactly) — it is insurance for the spiky regime, not a regulariser.

---

## P7 — my prediction problem: QK-norm and tied embeddings (`p7_qknorm.png`, `p7_curves.png`)

**Question.** The recipe uses QK-norm (RMSNorm on queries and keys). Predict:
(i) without QK-norm, the *stable* LR range shrinks — no loss at low LR, a
cliff at high LR; (ii) tying input/output embeddings (−1.6M params, 4.5 % of
d8) is roughly free at this scale.

| peak LR | QK-norm on | QK-norm off | Δ |
|---|---|---|---|
| 1e-3 | 2.977 | 2.986 | +0.009 |
| 3e-3 | 2.928 | 2.948 | +0.020 |
| 9e-3 | 2.954 | **4.043** | +1.09 |
| 3e-2 | 3.026 | **5.151** | +2.13 |

Tied embeddings (QK-norm on): lr 3e-3 → 2.939 (+0.011), lr 9e-3 → 2.950
(−0.004).

* (i) confirmed and sharper than predicted: with QK-norm the loss varies by
  only 0.1 across a **30× LR range**; without it there is a cliff between
  3e-3 and 9e-3 (3×). The `p7_curves.png` no-QK lr 9e-3 run never recovers —
  it is a permanent-damage failure, not a slow one, and P6 shows the
  mechanism (attention-logit blow-up → residual RMS 10⁵). QK-norm's job is
  to buy LR headroom, and the P2(c) inverse scaling (d4 4.22 → d8 5.15) says
  that headroom matters *more* as models grow. Small surprise: even in the
  stable regime it costs 0.01–0.02 — QK-norm is a (mild) optimization aid,
  not only a stabiliser.
* (ii) confirmed: tying is neutral at both LRs (within the 0.01 noise band),
  so 4.5 % of the parameters can go. This also means "parameter count" on the
  P2 x-axis is fuzzy by ~5 % depending on this one bit.

---

## Operational lessons (details in APPROACH.md)

* Modal reuses a container's Python process across sequential jobs. Three
  things leaked between jobs: `CUBLAS_WORKSPACE_CONFIG` (must be set in the
  image env, not in `train()`), torch/inductor/CUDA-graph state after a crash
  (containers now drain after any failure), and — found at the very end —
  **W&B run identity + GPU memory**: three bs-256 jobs OOM'd in a reused
  container and the *next* job in that process attached to the orphaned
  `wandb.run`, producing two mislabeled and one merged history. Fixed by
  `wandb.finish(exit_code=1)` on failure, `reinit=True`, and an explicit
  `torch._dynamo.reset()` + `empty_cache()` between jobs; the two bad records
  are tagged `a1-corrupted`, two more were relabeled from checkpoint
  `run_state.json`, and all four configs were rerun cleanly. Lesson: after any
  crash, verify the W&B record's *config*, not just its name.
* Cost model: the first estimate (from 25-min older d8 runs) was 2× too high,
  the second (from this suite's 11-min runs) about right; d9 is 35 min and d4
  3.6–8.3 min for the same steps (small models are overhead-bound).
* GPU ledger (W&B `_runtime`, all runs in the project): 10.2 h before this
  assignment + 23.9 h for this suite (incl. 0.5 h of corrupted + 0.6 h of
  rerun work) = **34.1 h of the 48 h allowance**.
