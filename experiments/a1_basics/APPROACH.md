# A1 Basics: how the experiment set was designed

This is the thinking behind `launch_all.py` and the `p1..p7` files: what the
constraints were, how each problem was turned into runs, what I expect to see,
and how to read the results. Written to build intuition, not just to record
choices.

## 1. Start from the budget, not from the questions

The handout suggests ~150 runs (P1 50, P2 50, P3 25, P4 15, P5 5, P6 0-5,
P7 5-10). The hard constraints are 48 GPU-hours and 2 concurrent GPUs.

The first thing to do was **measure the unit cost**, not guess it -- and it
took two rounds to get right, which is itself the lesson:

* My first estimator guessed 12.5 min per d8 run (600k sequences x 1024
  tokens = 614M tokens).
* Your 21 older P1 runs in W&B took 24.6-27.6 min each (mean 25.2), so I
  recalibrated to 25 min and trimmed the suite from 162 to 90 runs to fit the
  ~37 h left (~11 h were already spent, incl. 6 killed runs).
* The first 7 d8 runs of *this* suite then finished in 10.3-12.6 min (mean
  11.0). The older runs were slower for reasons outside the model (heavier
  eval/logging settings, or a shared GPU); the throughput I actually get is
  ~2.3x better than the number I planned against.

So the anchor is now `D8_MINUTES = 11.0`: one d8 run ~= 0.18 GPU-h. That
freed ~17 h, and I launched the trimmed runs as a **second wave** (51 runs,
~8.3 h, `--exclude-manifest wave1_manifest.csv` so nothing already queued is
resubmitted). Ledger: ~11 h prior + ~13.4 h wave 1 + ~8.3 h wave 2 ~= 33 h of
48, leaving ~15 h of headroom for retries and container start-up (~1-2 min
per run, which matters more now that runs are short).

Takeaway: calibrate against runs *from the same code and settings you are
about to launch*, not against whatever happens to be in the project already,
and re-check after the first few finish.

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
   16/256, wd 0/0.01/1.0, warmup 0/0.003), the corners of the paired grids,
   schedules, betas, clipping. When analysing P1, merge the old runs (tag `a1-basics-p1b-v3`)
   with the new ones (tag `a1-p1`).
3. **Priority order + a hard cap.** `--budget-hours` drops runs from the end
   of the list. Order is P3, P4, P2, P1, P5, P6, P7: the noise floor (P3/P4)
   is needed to interpret *every* other comparison, the scaling ladder is the
   most expensive-to-redo, P1 is already half done, P6/P7 are cheapest to
   defer.

Final manifest: 141 runs in two waves -- `wave1_manifest.csv` (90 runs, the
budget-trimmed core) + `run_manifest.csv` (51 runs restored once the unit cost
was re-measured), ~21.7 GPU-h total at the measured rate.

## 3. Problem by problem: design, hypotheses, what to look at

### P1 -- hyperparameters (42 new runs, 21 existing)

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

### P2 -- scaling-law reliability (56 runs, mostly shallow)

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

### P3 -- measuring variation (25 runs + 1 A100)

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

### P4 -- amplification of randomness (11 runs, all deterministic)

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

### P6 -- activations and gradient norms (4 runs)

Module RMS statistics are logged by default (`log_module_rms`). The extra
`log_activation_rms` logger (handout example) hooks layers 1/3/6 and logs
residual-stream RMS and abs-max on 8 fixed validation sequences after each
eval. Three probed runs: healthy d8; "crazy" (lr 0.03, no QK-norm, no
clipping); wd 1.0. Hypothesis: residual RMS grows with depth and with
training; the crazy run's RMS at layer 6 blows up by orders of magnitude
(that is the shape in the quiz figure); strong weight decay shrinks parameter
RMS and, with it, activation RMS late in training.

### P7 -- own prediction problem (6 runs)

Question: "Disable QK-norm on d8. At which peak lr does it first lose to the
default recipe by more than the noise floor, and does it ever diverge?" Runs:
no QK-norm at lr 1e-3, 3e-3, 9e-3, 3e-2 (the last is shared with P2c), plus
tied embeddings at default lr as a second architectural probe. It is
checkable (final val loss), compact (a `TrainConfig` flag), and tests the
idea from P2/P6 that normalisation buys lr tolerance rather than raw quality.

## 4. Operational notes

* Runs are in two detached Modal H100 apps (wave 1, wave 2), each with
  `max_parallel_runs=2`; the A100 reference is a third app with
  `max_parallel_runs=1`. Modal enforces the course's 2-GPU cap across apps, so
  wave 2 and the A100 job simply queue behind wave 1 (verified: 2 containers
  total after the second launch).
* Re-launching is safe: `launch_training_jobs` skips any config whose final
  model already exists in the user volume.
* Estimator sanity check: the first H100 jobs reported an ETA of ~52 min at
  step 282; early ETAs are inflated by `torch.compile` warm-up and data
  materialisation, so judge against the wall clock of finished runs in W&B and
  update `D8_MINUTES` if it drifts from 11 min.


## Postscript: what actually broke on the first launch, and the fixes

Two things failed on the first submission, and both are worth understanding
because they are the kind of infrastructure bug that silently corrupts an
empirical study if you don't check your runs.

1. **Every `deterministic=True` run crashed on H100** (P3 deterministic
   references, all of P4) with `Deterministic behavior was enabled ... you
   must set CUBLAS_WORKSPACE_CONFIG`. The code *did* set that env var, in
   `configure_deterministic_training`, but cuBLAS reads it once, when the
   CUDA context is first created. Modal reuses one Python process for many
   jobs, so any deterministic job that ran *after* a nondeterministic one in
   the same container saw a cuBLAS that had already been initialised
   without it. The A100 reference worked only because it was the first job
   in its container. Fix: `CUBLAS_WORKSPACE_CONFIG=:4096:8` is now set in
   the Modal image env (`modal_utils.py`), i.e. before Python starts. It is
   harmless for nondeterministic runs.

2. **A crashed job poisons the container.** After (1) raised,
   `torch.use_deterministic_algorithms(True)` and half-built cudagraph
   trees were left behind in the process, and the *next* job in that
   container died inside TorchInductor with an `AssertionError`. Modal then
   retried the same inputs into the same broken process, burning the retry
   budget. Fix: `_run_training` now calls
   `ContainerIOManager.stop_fetching_inputs()` on any exception, so the
   container drains and the retry lands in a fresh process.

3. Wave 2 was submitted before `utils.py` had the real W&B entity, so every
   run hung for 90 s in `wandb.init` against `YOUR_WANDB_USERNAME_OR_TEAM`,
   then retried. Fixed by setting `CONFIG_WANDB_ENTITY`/`CONFIG_MODAL_ENVIRONMENT`.
   Cost: roughly 5 GPU-hours of two containers spinning on retries.

Relaunch: waves 1+2 were stopped, the 13 runs that had finished (and the
A100 reference) were excluded via `--exclude-manifest`, and the remaining
128 runs were submitted as one app
(https://modal.com/apps/cs312-f26/cs312-aleyang/ap-w3Dcf9TXJkNiGFu8stlGpZ),
~19 GPU-hours estimated. Ledger: ~11 h prior + ~3 h finished + ~5 h wasted
+ ~19 h remaining ≈ 38 h of 48.

### Postscript 2 — the bug that only shows up in the data

After every run had "finished" with 0 failures, reconciling W&B configs
against run names found a fourth container-reuse leak:

4. **An OOM'd job leaves its W&B run open, and the next job inherits it.**
   Three batch-size-256 jobs hit CUDA OOM (the previous job's compiled graphs
   and allocator cache were still resident). Modal marked them failed and
   retried them elsewhere, but the *next* job scheduled into the same process
   called `wandb.init()` while `wandb.run` was still the dead job's run — so
   it logged into that record. Result: two W&B runs whose name said
   `bs256` but whose history (and `config`) belonged to a different config,
   and one record containing two concatenated histories. Fixes in
   `modal_train.py`/`train.py`: `wandb.finish(exit_code=1)` on any exception,
   `reinit=True` in `wandb.init`, and `torch._dynamo.reset()` +
   `gc.collect()` + `torch.cuda.empty_cache()` after every job. The two
   unrecoverable records are tagged `a1-corrupted` (kept for the audit
   trail, excluded from analysis), two were relabeled from their checkpoint
   `run_state.json`, and the four affected configs were rerun
   (`repair_manifest.csv`, `repair2_manifest.csv`, ~0.6 GPU-h). Lesson: a
   green dashboard is not a verified dataset — check that each record's
   logged `config` matches its name before analysing.

Final ledger (W&B `_runtime`, whole project): 10.2 h prior + 23.9 h for this
suite = 34.1 h of the 48 h allowance.
