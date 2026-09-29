# Answers to the six example quiz questions

Each answer cites the run(s) in `RESULTS.md` that back it. Baseline d8 = 2.9277 final val;
noise floor 0.005 (P3).

---

## P1 (Easy) — batch ×2 with LR tweaks

Rank A (lr 3e-3, bs 64), B (lr 1e-3, bs 128), C (lr 9e-3, bs 128).

**Answer: A < C < B.** Measured: A 2.9277, C 2.9898, B 3.0251 (P1(b) LR×batch grid).

Why: doubling the batch at fixed tokens halves the number of optimizer steps. Fewer steps
need a bigger step, so the LR that pairs with bs 128 is the *higher* one — C recovers part of
the loss, B makes it worse on both counts (too few steps, each too small). In our grid the
penalty of lr 1e-3 grows from +0.024 at bs 32 to +0.073 at bs 128; the penalty of lr 9e-3
shrinks from +0.033 at bs 32 to +0.038 → it stays roughly flat while the low-LR penalty balloons.
Neither beats A: bs 64 → 128 costs ~0.024 even at the best LR we tried, and A is already near the
LR optimum.

## P1 (Medium) — lr 9e-3, wd 1.0, three schedule/warmup combos

A (wsd0.2, warmup 0) · B (wsd0.2, warmup 20%) · C (linear, warmup 10%). These exact runs were
trimmed, so this is a prediction from neighbours.

**Answer: C < B < A** (C best). Confidence: A worst is solid; C vs B is a ~0.01 call.

Why:
1. `lr·wd` sets how fast weights forget. At lr 9e-3, wd 1.0 the product is 9e-3 — 30× the
   baseline's 3e-4 — so the network can only hold ~110 steps of memory while the LR is at peak.
   P6 shows wd 1.0 already pins parameter RMS at its init value (0.049 vs 0.098) at lr 3e-3.
   Whatever schedule spends the least time at peak `lr·wd` wins → linear decay (C) beats WSD,
   whose constant phase keeps the product maxed for 60–80% of training. At lr 3e-3 alone,
   wsd0.2 was 0.009 worse than linear (P1(c)).
2. Warmup matters more the higher the LR. No warmup cost +0.11 at lr 3e-3 (P1(a)); at lr 9e-3
   going 1% → 10% warmup gained 0.040 versus 0.009 at lr 1e-3 (P1(b) LR×warmup). A, with 0%
   warmup at 3× the baseline LR and maximal decay, is the clear loser.
3. B vs C: B's 20% warmup is not much better than 10% (3% → 10% only bought 0.006 at 9e-3),
   and its schedule is the worse one, so C edges it.

## P2 (Hard) — fixed vs depth-scaled context, 15 epochs, predict the d10 gap

**Answer: 0.03 < Δ ≤ 0.15, and no, don't trust the pilot fit** — its +0.055 is the right box for
the wrong reason, and the risk is on the upside (Δ > 0.15 is the plausible miss).

Why the pilots converge: the scaled recipe is catastrophic at d4 (4.75 vs 3.45) and nearly
caught up by d6 (3.57 vs 3.42). The likely cause is that small depths train on contexts shorter
than the 1024-token validation sequences, and a model never sees positions it is then evaluated
on. That effect saturates once context ≥ 1024, so the fitted curve — which has only d4–d6 to
learn from — sees a fast-shrinking gap and extrapolates it to almost nothing. A power law fitted
through a transient is not a scaling law; in our own P2(a), d4–d6 fits on *clean* recipes still
over-predicted d9 by 0.03–0.08.

Why the gap should stop shrinking: both recipes repeat 39M tokens 15 times. Our P2(c) repeated-data
run (8 epochs) shows the penalty for repetition *grows* with model size: +0.035 at d4, +0.088 at
d6, +0.162 at d8, flattening the exponent from 0.063 to ~0.04. Longer contexts make this worse —
the same tokens become fewer, longer sequences, so there is less independent data per epoch and
more for a big model to memorise. The d6 validation curves already show it: the scaled-context
run bottoms out at 3.50 around epoch 7–8 and drifts *up* to 3.57, while fixed context is still
descending. At d10, with more capacity and even longer sequences, that overfitting is larger,
not smaller. Net: the positional-mismatch term goes to zero, the memorisation term grows, and
0.05–0.15 is where they land; if repetition dominates the way it did in P2(c), > 0.15.

## P3 (Medium) — which change more than doubles the seed-to-seed std?

**Answer: 2 — only the higher learning rate.**

Measured (P3(c), 4 seeds each): standard std 0.0016; batch 16 std 0.0024 (1.5×); lr 9e-3 std
0.0047 (2.9×). With n = 4 each std has ~±40% error, so 1.5× is "same" and 2.9× is real; with the
quiz's 10 seeds the ratios would be sharper but the ordering wouldn't move.

Why: batch size at fixed tokens changes how noisy each *step* is (train-loss jitter at bs 16 is
2× bs 64, P5(b)) but the optimizer averages 4× more steps, so the endpoint is no noisier — the
per-step noise and the step count cancel. A higher LR does not average out: each step is 3×
larger, so the same seed difference is carried 3× further and the run sits closer to the edge
of stability (lr 9e-3 is already +0.026 worse than 3e-3, and without QK-norm it's where training
fails, P7). Seed variance is a cheap "distance to instability" meter.

## P5 (Easy) — babysitting a d8 run

(a) **D, learning rate 0.009.** (b) **a, at most +0.05** (measured gap +0.028: 2.9537 vs 2.926).

Why D: two independent reads agree.
1. Step count. "First quarter" spans ~2,340 steps, so the whole run is ~9,400 steps = 614M tokens
   at bs 64. Batch 256 would have 2,343 steps *total* (first quarter ≈ 585) → not B.
2. Level. Read off the plot: ~5.2 at step 100, ~3.9 at 500, ~3.65 at 800, ~3.35 at 2,000.
   Our train-loss curves at those steps: lr 9e-3 = 5.18 / 3.90 / 3.64 / 3.37 (match);
   standard = 4.98 / 3.71 / 3.50 / 3.26 (0.1–0.2 too low); lr 3e-4 = 5.51 / 4.26 / 4.02 / 3.51
   (0.2–0.4 too high). Higher LR is slightly *behind* the baseline early because warmup ends at a
   bigger, noisier step and the run is still paying for it (P5(a) curves cross only during decay).
3. What it isn't: the jitter is normal. Micro-smoothness is set by batch size alone
   (0.040 at bs 64 for every LR we tried, P5(b)), so a jagged curve can't identify LR — you have
   to use the level and the step count.

Why a: lr 9e-3 is inside the flat part of the LR basin (P1(a): 3e-3 → 9e-3 costs +0.026;
even 2.7e-2 only costs +0.09). 3× too high is a mild mistake; 10× too low (3e-4, +0.22) would
have been the one to worry about. With QK-norm on, nothing in this recipe diverges below
lr ~3e-2.

## P6 (Easy) — residual-stream RMS traces at layers 1/3/6

**Answer: Alternative** (a broken run; matches our lr 3e-2 / no QK-norm / no clipping run).

Why:
1. Scale. Healthy d8 residual RMS after layer 6 is ~12 at step 100, ~23 mid-run, ~14 at the end
   (P6(a)); the plot shows 10⁵–10⁶ within the first few hundred steps, decaying to ~10³. Our
   unhealthy run did exactly this: 5,400 at the first probe, peak ~1e5, 2,200 at the end.
2. Shape. In a healthy run the three layers are ordered and separated — the residual stream
   accumulates with depth (layer 1 ≈ 2, layer 3 ≈ 6, layer 6 ≈ 15). Here all three collapse onto
   one line, i.e. the stream is dominated by a single enormous component written early and never
   removed. Weight decay then slowly shrinks it (the downward slope), which is why the run doesn't
   NaN but also never recovers (final val 5.15 vs 2.93).
3. Timing. Step 0 is ~1 for all layers (same as our init, 0.94), then a vertical jump: this is
   the first warmup steps at an LR the attention logits can't take, the P7 failure mode.
   Gradient RMS in that run is 30× baseline and spiky — the diagnostics flag the run 500 steps
   in, long before the loss curve looks unusual.
