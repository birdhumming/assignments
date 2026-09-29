# Assignment 1: Basics — results tables

All numbers are **final validation loss** (nats) of a d8 model on 614M tokens unless a
row says otherwise. Baseline (lr 3e-3, bs 64, warmup 1%, wd 0.1, linear decay) = **2.9277**.
Noise floor from Problem 3: two runs within **0.005** are the same run; **> 0.01** is real.
"—" = cell trimmed from the plan to save GPU time. Long-form reasoning: `TAKEAWAYS.md`; plots: `figures/`.

---

## Problem 1 — hyperparameter sweeps

### (a) One at a time

Hypothesis: learning rate is the only knob with a big, asymmetric effect; the rest move loss by < 0.05.

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

Result: LR 10× too small costs +0.22, 9× too large +0.09 (undershooting is the expensive mistake).
No warmup costs +0.11 but 0.3% warmup recovers most of it. Batch is flat to 64 then −0.03/−0.06
per doubling (fixed tokens → fewer steps). Weight decay spans only 0.045 across 0 → 1.0.
Ranking: LR ≫ warmup=0 > batch > weight decay.

### (b) Co-varying pairs

Hypothesis: LR×batch and LR×weight-decay are linked (through steps and through `lr·wd`); LR×warmup mildly.

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

Result:
1. LR × batch: too-low LR hurts more as batch grows (+0.024 at bs 32 → +0.073 at bs 128 for lr 1e-3). Fewer steps need a bigger step.
2. LR × weight decay: best wd falls as LR rises (0.3 → 0.3 → 0.1). Winners have `lr·wd` ≈ 3e-4–9e-4; losers ≥ 2.7e-3. Tune the product.
3. LR × warmup: 1% → 10% warmup gains 0.009 at lr 1e-3 but **0.040 at lr 9e-3**, and flips the best LR from 3e-3 to 9e-3. Warmup buys LR headroom.

### (c) Schedules, betas, clipping (all at lr 3e-3)

Hypothesis: any decay ≈ linear; constant LR is much worse; betas and clipping are noise at this LR.

| LR schedule | linear | cosine | WSD 20% | WSD 50% | constant |
|---|---|---|---|---|---|
| final val | 2.9277 | 2.9347 | 2.9371 | **2.9190** | **3.1784** |

| β₁ | 0.8 | **0.9** | 0.95 |  | β₂ | 0.9 | **0.95** | 0.99 |
|---|---|---|---|---|---|---|---|---|
| final val | 2.9298 | 2.9277 | 2.9343 |  | final val | 2.9321 | 2.9277 | 2.9225 |

| grad clip | 1.0 (default) | none |
|---|---|---|
| final val | 2.9277 | 2.9320 |

Result: no decay costs +0.25 — the largest single effect in P1. Every decaying schedule lands
within 0.02 of linear (WSD 50% best by 0.009). β₁/β₂/clipping: ≤ 0.007, i.e. noise. Key extra
hyperparameter = decay fraction; clipping only matters near instability (P6/P7).

---

## Problem 2 — reliability of scaling laws

Params: d4 5.9M · d5 10.0M · d6 16.0M · d7 24.0M · d8 34.6M · d9 48.0M (incl. embeddings); 614M tokens each.

### (a) Four recipes, d4–d9

Pre-registered (before d8/d9): baseline exponent ≈ 0.06; constant LR flatter; dropout parallel; lr 3e-2 gap shrinks with size.

| recipe | d4 | d5 | d6 | d7 | d8 | d9 | exponent (6 pts) | d4–d6 fit → d9 | actual d9 | fit → d20 (370M) |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 3.2903 | 3.1541 | 3.0562 | 2.9840 | 2.9277 | 2.8818 | 0.063 | 2.814 | 2.882 | 2.52 |
| constant LR | 3.4283 | 3.3302 | 3.2429 | 3.2197 | 3.1784 | 3.1242 | 0.042 | 3.051 | 3.124 | 2.87 |
| dropout 0.2 | 3.4518 | 3.3206 | 3.2157 | 3.1289 | 3.0594 | 3.0034 | 0.067 | 2.972 | 3.003 | 2.61 |
| lr 3e-2 | 3.3165 | 3.2134 | 3.1104 | 3.0702 | 3.0257 | 2.9849 | 0.050 | 2.901 | 2.985 | 2.68 |

Result: exponents differ by recipe (0.042–0.067) on the same family and data; d4–d6 fits
over-predict d9 by 0.03–0.08 (10–40× seed noise); d20 predictions span 2.52–2.87. The lr 3e-2
prediction was wrong: its gap to baseline *grows* with size (0.03 → 0.10).

### (b) Slope benders

| model-size axis | d4 | d5 | d6 | d7 | d8 | exponent |
|---|---|---|---|---|---|---|
| baseline | 3.2903 | 3.1541 | 3.0562 | 2.9840 | 2.9277 | 0.063 |
| lr 1e-3 | 3.3528 | 3.2185 | 3.1173 | 3.0362 | 2.9768 | 0.067 |
| bs 256 | 3.3615 | 3.2274 | 3.1324 | — | 3.0123 | 0.062 |

| data axis (tokens) | 76.8M | 154M | 307M | 614M | 1.23B | exponent |
|---|---|---|---|---|---|---|
| d8, lr 3e-3 | 3.5352 | 3.2303 | 3.0582 | 2.9277 | 2.8372 | 0.078 |
| d8, lr 1e-3 | 3.5768 | 3.2879 | 3.1088 | 2.9768 | — | 0.088 |
| d6, lr 3e-3 | 3.5730 | 3.3314 | 3.1659 | 3.0562 | — | 0.075 |
| d6, lr 1e-3 | 3.7936 | 3.4369 | 3.2495 | 3.1173 | — | 0.093 |

Result: ordinary choices move the exponent 5–20%. Lower LR looks like a *steeper* data law
because it is under-trained at short horizons and catches up (d6 gap 0.22 → 0.06 from 77M to 614M tokens).

### (c) Law breakers

| intervention | d4 | d6 | d8 | law |
|---|---|---|---|---|
| baseline | 3.2903 | 3.0562 | 2.9277 | power law, exp 0.063 |
| SGD (same LR/schedule) | 6.5861 | 6.5989 | 6.6535 | flat / inverted |
| no QK-norm, lr 3e-2 | 4.2168 | 4.9412 | 5.1513 | **inverted** — bigger is worse |
| lr 3e-2, no warmup, no clip | 3.3235 | — | 3.0343 | survives, gap grows |
| 8 epochs × 75k seqs (same tokens) | 3.3252 | 3.1446 | 3.0900 | flattened, exp ≈ 0.04 |

Result: scaling breaks when the bottleneck is not capacity — optimizer (SGD), stability that
worsens with width (no QK-norm), or repeated data that bigger models memorize. It breaks on the
model-size axis, in the over-trained/small-data regime for repetition and at high LR for stability.

---

## Problem 3 — run-to-run variation

d8 unless stated; 3–4 replicas per row; std = standard deviation of final val.

### (a)/(b) Sources of variation

| what varies | n | mean | std |
|---|---|---|---|
| nothing (same seeds, nondeterministic GPU kernels) | 3 | 2.9269 | 0.0007 |
| nothing, `deterministic=True`, 2× H100 + 1× A100 | 3 | 2.9273 | 0.0003 |
| data seed only | 4 | 2.9275 | 0.0012 |
| model (init) seed only | 4 | 2.9278 | 0.0036 |
| both seeds | 4 | 2.9289 | 0.0016 |

Result: total variation ≈ 0.002 (distribution tight, no outliers). Init seed ≈ 3× data order;
hardware ≈ 0.001 and deterministic mode removes it even across GPU types. Train-loss curves
differ by ~0.008/step early, ~0.002/step late — they diverge, then run in parallel.

### (c) Hyperparameters vs variability

| setting (both seeds varied) | n | mean | std | vs baseline std |
|---|---|---|---|---|
| d8 baseline | 4 | 2.9289 | 0.0016 | 1× |
| batch size 16 | 4 | 2.9305 | 0.0024 | ~1× |
| 2× tokens (1.23B) | 3 | 2.8378 | 0.0017 | 1× |
| lr 9e-3 | 4 | 2.9528 | 0.0047 | **3×** |
| d4 | 4 | 3.2838 | 0.0087 | **5×** |

Result: only settings that push toward instability (high LR) or small capacity (d4) raise
variability. More capable models are *less* noisy: bigger cuts std 5×; longer training keeps std
flat while lowering the mean. Caveat: n = 3–4 → each std is ±40%; trust only the 3–5× gaps.

---

## Problem 4 — amplification of randomness

Deterministic twins; one differs by overwriting k tokens of one sequence at time T. Control: bit-identical before T (max |Δ| = 0.0).

| perturbed at | tokens | |Δ train loss| first 100 steps | |Δ| steady state | Δ final val |
|---|---|---|---|---|
| step 0 | 1 | 0.010 | 0.0027 | −0.0025 |
| step 0 | 10 | 0.009 | 0.0024 | −0.0011 |
| step 0 | 100 | 0.011 | 0.0024 | −0.0012 |
| step 0 | 1024 | 0.016 | 0.0026 | −0.0004 |
| 25% | 1 / 1024 | 0.0001 / 0.003 | 0.0003 / 0.0005 | +0.0001 / +0.0002 |
| 50% | 1 / 1024 | 0.0001 / 0.003 | 0.0001 / 0.0001 | 0.0000 |
| 90% | 1 / 1024 | 0.0001 / 0.002 | 0.0001 / 0.0001 | 0.0000 |

Result: (a) one token changes terminal *train* loss by ~0.003 permanently, terminal *val* by
≤ 0.0025 (inside seed noise). (b) Magnitude saturates — 1 and 1024 tokens reach the same
plateau, equal to the GPU-nondeterminism floor. Timing dominates: the same perturbation at
25/50/90% is 10–30× smaller (decayed LR, fewer steps to compound).

---

## Problem 5 — loss-curve augury

### (a) Macro shape

| factor | what changes in the curve |
|---|---|
| peak LR | curves separate by step ~500; ordering holds until decay. lr 2.7e-2 sits *above* lr 3e-4 for 80% of training and finishes 0.13 *below* it |
| schedule | WSD looks worse than linear until its decay phase, then drops 0.2 in 20% of the steps; constant LR never drops |
| β₁ 0.8/0.9/0.95 | indistinguishable at every step |
| batch size | level shift only (bs 256 higher throughout at fixed tokens) |

Rule: never rank runs before their LRs have decayed to the same level.

### (b) Micro smoothness (std of train loss around a 51-step mean, second half)

| bs 16 | bs 32 | bs 64 | bs 128 | bs 256 | LR 3e-4 … 2.7e-2 | β₁, β₂ | no clip | constant / WSD | wd 1.0 |
|---|---|---|---|---|---|---|---|---|---|
| 0.079 | 0.057 | 0.040 | 0.028 | 0.020 | 0.039–0.040 | 0.040 | 0.040 | 0.040 | 0.039 |

Result: jitter ∝ 1/√batch (0.079/0.020 = 3.95 ≈ √16) and nothing else moves it. The wiggle is
minibatch sampling noise, not optimizer noise; spikes (P6/P7) are the optimizer signal.

### (c) Past runs & the sample quiz

| | answer | evidence |
|---|---|---|
| past runs that changed shape | constant LR (no final drop), no-QK-norm at lr ≥ 9e-3 (spike then plateau at 4–5), SGD (flat at 6.6), wd 1.0 (curve bends up late) | P1(c), P2(c), P7 curves |
| quiz (a): "high learning-rate" curve | **D, lr 0.009** — steeper early descent, higher early jitter is *not* present (jitter is a batch effect), so not B; A/C are the smooth low curves | P5(b) table |
| quiz (b): terminal gap vs 2.926 | **a, at most +0.05** (lr 9e-3 → 2.954, gap +0.026) | P1(a) |

---

## Problem 6 — activations and gradient norms

RMS per module every 100 steps; residual-stream probe after layers 1/3/6 on 8 fixed val sequences.

### (a) Healthy d8 across training

| statistic | init | step 100 | mid | end |
|---|---|---|---|---|
| parameter RMS | 0.0422 | 0.0436 | 0.0977 | 0.0892 |
| gradient RMS | 2.7e-3 | 6.7e-5 | 1.9e-5 | 2.2e-5 |
| activation RMS | 0.94 | 1.56 | 3.16 | 2.59 |
| residual RMS after layer 6 | — | 12.1 | 23.1 | 14.1 |

Across depth: activations and `o_proj`/`down_proj` outputs grow with layer (residual stream
accumulates, ~2 at layer 2 → ~15 at layer 8); layer 1 has 10× the gradient of the others at step
100 and gradients flatten across layers by mid-training. Parameter RMS peaks then *falls* as LR decays.

### (b)/(c) Interventions

| run | param RMS mid / end | grad RMS late | act RMS mid / end | residual L6 start → end | final val |
|---|---|---|---|---|---|
| healthy baseline | 0.098 / 0.089 | 2e-5 | 3.2 / 2.6 | 12 → 14 | 2.9277 |
| lr 3e-2, no QK-norm, no clip | 0.33 / 0.31 | 6e-4, spiky | **8800 / 970** | **5400 → 2200** (peak 1e5) | 5.1513 |
| wd 1.0 | 0.049 / 0.049 | 8e-5 | 1.4 / 1.3 | 9 → 0.5 | 2.9624 |
| wd 0.0 | 0.15 / 0.16 | 1.3e-5 | 7.0 / 7.2 | 13 → 67 | 2.9503 |
| lr 3e-4 | 0.043 / 0.042 | 7.6e-5 | 1.5 / 1.5 | | 3.1475 |
| lr 9e-3 | 0.19 / 0.16 | 1.6e-5 | 6.4 / 3.9 | | 2.9537 |
| constant LR | 0.12 / 0.13 | 1.8e-5 | 5.4 / 4.8 (58 at step 100) | | 3.1784 |
| bs 256 | 0.066 / 0.067 | 1.5e-5 | 2.4 / 2.3 | | 3.0123 |
| no clip (lr 3e-3) | 0.097 / 0.089 | 2.3e-5 | 3.2 / 2.6 | | 2.9320 |

Result: parameter norm is an equilibrium set by `lr·wd` — wd 1.0 pins it at init, wd 0 doubles it,
lr 3e-4 never leaves init, constant LR keeps growing. Activation scale follows parameter scale;
gradient RMS runs inversely (RMSNorm makes the net scale-invariant, so ‖g‖ ∝ 1/‖w‖). To shrink
everything uniformly: raise wd. To shrink only the end: use a decaying schedule (the decay phase
is when wd wins). The unhealthy signature (activations 10³–10⁴×, spiky gradients) appears within
500 steps — earlier than the loss.

---

## Problem 7 — own prediction problem: QK-norm and tied embeddings

Question: the d8 recipe normalizes queries and keys (`qk_norm=True`). (i) Without QK-norm, at
which peak LR does the run first exceed baseline + 0.5? (ii) Does tying input/output
embeddings (−1.6M params, 4.5%) change final val by more than 0.01? Code diff: `p7_own_prediction_problem.py`.

Hypothesis: (i) low LRs unaffected, cliff somewhere at 9e-3–3e-2; (ii) tying is free.

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

Result: (i) cliff between 3e-3 and 9e-3 — answer 9e-3. With QK-norm the loss varies 0.1 over a
30× LR range; without it, a 3× step is fatal and never recovers (mechanism in P6: attention-logit
blow-up, residual RMS 10⁵). QK-norm also costs 0.01–0.02 to remove even in the stable regime,
and P2(c) shows the failure worsens with width. (ii) tying is neutral (within 0.01).

---

## Run ledger

| | runs | GPU-h |
|---|---|---|
| before this suite (earlier P1 grid, killed runs) | 27 | 10.2 |
| this suite, finished & used | 116 | 22.8 |
| trimmed from plan (`trim_dropped.csv`) | 28 | 0 |
| corrupted W&B records (quarantined, `a1-corrupted`) + reruns | 2 + 4 | 1.1 |
| **total** | | **34.1 / 48** |
