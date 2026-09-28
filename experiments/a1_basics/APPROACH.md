# A1 Basics: how the experiment set was designed

This is the thinking behind `launch_all.py` and the `p1..p7` files: what the
constraints were, how each problem was turned into runs, what I expect to see,
and how to read the results. Written to build intuition, not just to record
choices.

## 1. Start from the budget, not from the questions

The handout suggests ~150 runs (P1 50, P2 50, P3 25, P4 15, P5 5, P6 0-5,
P7 5-10). The hard constraints are 48 GPU-hours and 2 concurrent GPUs.

The first thing to do was **measure the unit cost**, not guess it. Your 21
finished P1 runs in W&B took 24.6-27.6 min each (mean 25.2 min) for the
standard d8 recipe (600k sequences x 1024 tokens = 614M tokens). So:

* one d8 run ~= 0.42 GPU-h, ~115 d8-runs fit in the whole budget;
* 150 runs at d8 cost would be ~63 h -- the handout's plan only fits if a
  large share of runs are shallow (P2's d4-d7) and runs are shared across
  problems;
* my first estimator assumed 12.5 min/run and was off by 2x. Always calibrate
  against a real run before committing a budget.

Ledger at launch time: ~11.0 h already spent (27 runs, 6 killed early), so
~37 h remained. I targeted **~31 h of new runs** to leave ~6 h for retries,
container start-up overhead (~1-2 min/run) and estimator error.

Depth cost model (relative to d8; only d8 is measured, the rest is a guess to
verify from the first d4/d9 runs): d4 0.30, d5 0.40, d6 0.50, d7 0.75, d8 1.0,
d9 1.4. Compute scales like depth^3 (N ~ depth * width^2, width = 64 * depth)
but small models under-utilise the GPU, so wall clock falls much more slowly
than FLOPs. Token count scales cost linearly.

## 2. Three levers that make the budget work

1. **Share runs across problems.** Every config is named by its diff from the
   default (`training_run_name`), and `launch_all.py` dedupes by name. The d8
   baseline is simultaneously P1's centre point, P2's d8 point, P3's
   nondeterministic reference and P5's "normal" loss curve. `TrainConfig`
   loggers are not part of the name, so P6 runs get a `run_name_suffix` so
   they don't collapse into the baseline.
2. **Don't repeat what exists.** The 21 finished runs already cover lr in
   {0.001, 0.003, 0.009} x {default, bs32, bs128, wd0.03, wd0.3, warmup0.03,
   warmup0.1}. P1 here only fills gaps: sweep endpoints (lr 3e-4/2.7e-2, bs
   16/256, wd 1.0), the extreme corners of the paired grids, schedules, betas,
   clipping. When analysing P1, merge the old runs (tag `a1-basics-p1b-v3`)
   with the new ones (tag `a1-p1`).
3. **Priority order + a hard cap.** `--budget-hours` drops runs from the end
   of the list. Order is P3, P4, P2, P1, P5, P6, P7: the noise floor (P3/P4)
   is needed to interpret *every* other comparison, the scaling ladder is the
   most expensive-to-redo, P1 is already half done, P6/P7 are cheapest to
   defer.

Final manifest: 90 runs, ~30.6 GPU-h estimated (`run_manifest.csv`).

## 3. Problem by problem: design, hypotheses, what to look at

### P1 -- hyperparameters (18 new runs, 21 existing)

*Design.* Base-3 log sweeps around the default. Pairs chosen from mechanism:
* **lr x batch size** -- bigger batches give lower-variance gradients, so the
  optimum lr should shift up (roughly sqrt(B) to B in the small-batch regime).
  Corners run: (1e-3, bs16), (9e-3, bs256).
* **lr x weight decay** -- AdamW's decay per step is `lr * wd`; if what
  matters is total decay, high lr should want *lower* wd. Run: (9e-3, wd 1.0),
  plus the handout medium example (lr 9e-3, wd 1.0 with three schedules).
* **lr x warmup** -- warmup exists to protect early training from a large lr
  while Adam's second-moment estimate is still noisy. Run: (2.7e-2, warmup
  0.1) asks "can warmup rescue a too-high lr", (9e-3, warmup 0) asks "does a
  moderately high lr need it at all".
* **Schedules** -- constant vs linear vs WSD. Prediction: decaying schedules
  win at fixed tokens (the loss drop during decay is large, ~0.05-0.1 nats);
  WSD with a 20% decay should be within noise of linear; constant lr should be
  clearly worse and its loss curve flatter at the end.
* beta1 0.8 / beta2 0.99 / no clipping are there for P5 (curve shape) as much
  as for P1.

*Read.* Plot final val loss vs each hyperparameter on a log x-axis. Expect a
U-shape with a flat bottom for lr (default should be near the optimum), a
mild monotone trend for batch size at fixed tokens (smaller batch = more steps
= usually lower loss at this scale, up to a point), a very weak effect for
warmup unless lr is high, and a weak effect for wd until it is ~1.0.

### P2 -- scaling-law reliability (39 runs, mostly shallow)

*Design.* Four recipes x d4..d8; d9 only for `baseline` and `constant`
(d9 costs 1.4 d8-runs each). **Pre-register**: fit `L = a*C^-alpha + e` on
d4-d7 for each recipe *before* looking at d8/d9, write down the prediction.
Part (b) bends slopes gently (lr 1e-3 and bs256 at d4/d6, shorter data
horizons at d6/d8). Part (c) tries to break the law at d4 and d8: SGD, no
QK-norm at lr 0.03, lr 0.03 with no warmup and no clipping, and 8 epochs over
a 75k-sequence subset (same tokens, repeated data).

*Hypotheses.*
* Baseline and constant will both look like clean power laws; constant is
  offset upward with a similar slope (schedule ~ constant-factor effect).
* Dropout 0.2 hurts most at small scale (under-fitting regime) and its
  slope will differ -- a "slope bender" you get for free.
* lr 0.03 is the interesting one: small models tolerate it, larger ones may
  diverge or plateau. If d8 at lr 0.03 is fine but d9 isn't, that is a
  scale-dependent break, exactly the pattern the hard quiz question is about.
* Repeated data should track the baseline at d4 (small model can't memorise
  75k sequences x 8) and fall off the law at d8 (it can).
* SGD will be far worse and probably not a power law at all -- it is a
  different optimiser regime, not a bent one.

*Read.* Loss vs compute (6ND) on log-log axes, one line per recipe. A
"reliable" law is one whose d4-d7 fit predicts d8/d9 within the P3 noise
floor.

### P3 -- measuring variation (19 runs + 1 A100)

*Design.* Vary everything (3 paired seeds), then isolate: model seed only
(2), data seed only (2), hardware only (two identical nondeterministic runs:
the baseline itself and `nondeterministic-rep1`), and two deterministic
references to establish the true floor. One deterministic reference is
repeated on an A100 for cross-hardware nondeterminism. Part (c) adds seeds
1,2 at lr 0.009 and bs16 (seed 42 comes from the P1 runs) and d4 x 3 seeds.

*Hypotheses.* Total seed-to-seed SD of final val loss at d8 is O(0.005-0.01)
nats. Data order should dominate model init at this data scale. Hardware
nondeterminism alone should be smaller but *not* zero after 9k steps
(chaotic amplification, see P4). Deterministic pairs should match to
~1e-6. Expect higher variance at bs16 and lr 0.009 (the handout's medium
quiz is exactly this), and the smaller d4 to be *noisier* in absolute loss
terms.

*Read.* Compute SD per group; the "significance threshold" for the whole
assignment is roughly 2x the all-sources SD.

### P4 -- amplification of randomness (7 runs, all deterministic)

*Design.* Deterministic reference (shared with P3), then flip 1, 10 or 1024
tokens of row 0 to token 17. Then keep the perturbation at 1 token but move it
to the row seen at 25%, 50%, 90% of training (`perturb_row = step * 64`),
plus 1024 tokens at 50%. `train.py` gained `perturb_row`/`perturb_num_tokens`
to allow this.

*Hypotheses.* A single token changes the loss trajectory by a floating-point
epsilon at first; the gap grows roughly exponentially for a while and then
saturates at the P3 hardware-noise level. Perturbation size matters less than
you'd think (chaos amplifies any nonzero difference); *time remaining after
the perturbation* matters more -- the 90% perturbation should end closest to
the reference.

### P5 -- loss-curve augury (0 new runs)

Every config the handout asks about is already in P1: baseline, bs256, bs16,
lr 3e-4, lr 9e-3 (existing), constant, WSD, beta1 0.8, beta2 0.99, no
clipping. Read curves, not final numbers: large lr = fast early drop, then
plateau/spikes; small lr = smooth but slow; small batch = noisy microstructure;
constant schedule = no late-stage drop; WSD = sharp drop in the last 20%;
beta1 low = noisier; beta2 0.99 = slower adaptation, possible instability;
no clipping = occasional spikes at high lr.

### P6 -- activations and gradient norms (3 runs)

Module RMS statistics are logged by default (`log_module_rms`). The extra
`log_activation_rms` logger (handout example) hooks layers 1/3/6 and logs
residual-stream RMS and abs-max on 8 fixed validation sequences after each
eval. Three probed runs: healthy d8; "crazy" (lr 0.03, no QK-norm, no
clipping); wd 1.0. Hypothesis: residual RMS grows with depth and with
training; the crazy run's RMS at layer 6 blows up by orders of magnitude
(that is the shape in the quiz figure); strong weight decay shrinks parameter
RMS and, with it, activation RMS late in training.

### P7 -- own prediction problem (4 runs)

Question: "Disable QK-norm on d8. At which peak lr does it first lose to the
default recipe by more than the noise floor, and does it ever diverge?" Runs:
no QK-norm at lr 1e-3, 3e-3, 9e-3, 3e-2 (the last is shared with P2c), plus
tied embeddings at default lr as a second architectural probe. It is
checkable (final val loss), compact (a `TrainConfig` flag), and tests the
idea from P2/P6 that normalisation buys lr tolerance rather than raw quality.

## 4. Operational notes

* All runs are in one detached Modal app with `max_parallel_runs=2`; the A100
  reference is a separate app with `max_parallel_runs=1`.
* Re-launching is safe: `launch_training_jobs` skips any config whose final
  model already exists in the user volume.
* Estimator sanity check: the first H100 jobs reported an ETA of ~52 min at
  step 282; early ETAs are inflated by `torch.compile` warm-up and data
  materialisation, so judge against the wall clock of finished runs in W&B and
  update `D8_MINUTES` if it drifts from 25 min.
