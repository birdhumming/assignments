# A3 run plan (max 2 H100s; d8 614M run ~11.5 min; Hessian hvp ~2 s)

Calibration: d8 614M run with 4 EMAs = 16.5 min on H100 (val every ~94 steps); P3 minibatch run ~ 4 min train + ~32 sharpness probes x (10-16 hvps).

Status log (UTC): 2026-10-15 09:35 P2 wave 1 launched (15 runs, app ap-oemKixNr9ofTTvEDz3MP10). 10:08 first 4 schedule runs finished (lr 3e-3): linear 2.927, cos 2.934, wsd0.2 2.936, constant 3.099. Scope decision pending: Full ~28 h / Trimmed ~23 h / Lean ~18 h.

Order: P2 wave1 (running) -> P4 a (13 ckpts + 4 densities, pass max_parallel=2) -> P3 (a: 3 full-batch, b: 1 minibatch) -> P1 starter grid (prefix -> branches -> measure) -> P3 c exploration -> P4 b-rescale, b-train, c -> P2 wave2 (dip depth/duration + control, budget-tuned baselines) -> P1 d knobs.

## P2 wave 1 findings (15/15 finished 11:44)
- Final val by schedule x peak LR (1.5e-3 / 3e-3 / 6e-3): linear 2.9475 / 2.9269 / 2.9395; cos 2.9553 / 2.9339 / 2.9448; wsd0.2 2.9372 / 2.9362 / 2.9679; constant 3.0609 / 3.0987 / 3.1839. Reference = linear 3e-3 (2.9269).
- EMA: decay 0.999 is best for constant-LR runs and recovers 0.10-0.16 of the gap (constant 3e-3: 3.0987 -> 2.9735, still 0.047 behind linear). Decayed schedules gain nothing (<0.001): the schedule already did the averaging. 0.9999 is far too slow (4.0+).
- Dip (wsd0.2, LR x0.1 over 40-50%): val drops 0.12 below control at the bottom (3.066 vs 3.189 at step 4229), then after rewarming sits 0.008-0.011 ABOVE control for the rest of the plateau and finishes 0.003 worse (2.9394 vs 2.9362). The gain does not survive; EMA0.99 shows the same crossover.
- Horizon-free at 3e-3: invsqrt 3.0243 final (EMA0.999 2.9900), sgdr2000 3.0394 (EMA0.999 2.9925) vs linear 2.9269; at 25%/50% invsqrt is close to or better than cos (3.2538/3.1270 vs 3.2693/3.1029).

## P2 wave 2 (after wave 1 is read)
- Reference = best final val among the 12 schedule x LR runs. Regret curves vs it.
- Dips on wsd0.2 at the best LR: (0.4,0.45,0.5,0.1) [wave1], (0.4,0.45,0.5,0.3), (0.4,0.5,0.6,0.1), (0.2,0.25,0.3,0.1), (0.6,0.65,0.7,0.1). Control = plain wsd0.2 (same cooldown).
- Horizon-free: invsqrt and sgdr2000 are fixed; read their val loss at 25%/50%/100% of training; budget-tuned baselines = cos runs with num_epochs 0.25 and 0.5 (2 runs).

## P3 (p3_edge_of_stability.py)
- (a) full_batch(32,1500,lr=0.01) + lr_schedule="step750x0.5" and "step750x2".
- (b) minibatch().
- (c) exploration, 18 (Trimmed keeps the first 10): minibatch(learning_rate=1e-3), (learning_rate=1e-2), (batch_size=16), (batch_size=256), (beta1=0.0), (beta1=0.98), (weight_decay=0.0), (warmup_percent=0.125), (lr_schedule="linear"), (embedding_init_scale=16.0); then (grad_norm=None), (qk_norm=False), (weight_decay=0.3), (learning_rate=6e-3), full_batch(128,1500,lr=0.01), full_batch(32,1500,lr=0.02), full_batch(32,1500,lr=0.005), full_batch(32,1500,optimizer_name="adamw",learning_rate=1e-3).
- Plot: eta_t * lambda(_pre) vs tokens on log axis, dashed line at 2 (SGD) or 2(1+b1)/(1-b1) (Adam, 38 at b1=0.9).

## P4 (p4_inside_the_hessian.py)
- (a) LOOK_JOBS as shipped; add a random-partition baseline for block energy (same block sizes, shuffled parameter assignment) via H.hvp on sign vectors.
- (b) rescale (loss-invariant): RMSNorm gain g -> g/c and the linear it feeds W -> cW gives identical logits; Hessian of the W block scales by 1/c^2 and of the gain block by c^2. Jobs: c=4 on input_layernorm + q/k/v_proj (all layers); c=4 on post_attention_layernorm + gate/up_proj; c=4 on model.norm + lm_head; q_proj x4 alone (QK-norm makes q scale irrelevant). Check loss unchanged to 1e-4.
- (b) training: candidates that keep val 2.925-2.931 but move final lambda_max >=2x: wsd0.0 constant LR (ends at high LR -> flatter, lambda ~38/eta); lower final LR (min_lr_ratio or cos) -> sharper; weight_decay=0 (norm gains grow -> linear-block curvature shrinks); a code change adding `norm_init_scale` (init gains at 1/c and compensate linears, same function at init). Measure finals with b-measure.
- (c) SUBSPACE_JOBS as shipped (full/top/removed, k=10, refresh 25, 300 steps from step 1526) + variants: k=50, refresh=5, lr_scale=0.3, start step 305, reset_momentum=False (Trimmed: k=50, lr_scale=0.3).

## P1 (p1_mode_connectivity.py)
- Starter grid x in {0,50,100,250,500,1000}, y in {25,...,500}, two batch orders, constant LR after warmup. prefix -> branches -> measure (connectivity.measure_pairs_on_modal). Part (c) interpolation alpha in {0,.25,.5,.75,1} for representative cells (small x / large y, large x / small y, diagonal).
- (d) knobs, each repeats the grid with a new BASE: learning_rate=6e-3; batch_size=16; lr_schedule="linear" (decaying). Trimmed: first two.

## P4(a) calibration (first result, 12:10 UTC Oct 15)
- Plain measurement job: 13.5 min on H100 (810 s compute + load). Density steps (0, 153, 1526, 9375) add ~13 min.
  13 jobs at 2 parallel -> ~1.8 h wall; app ap-Z3EjcjkWE4KX3ksqUiWlFj, launched 11:44.
- Step 3 (2M tokens): lambda_max 1572, lambda_min -602 (ratio 0.38), top eigenvector 98% embedding;
  family lambda_max: embedding 1554, attention 11.0, mlp 10.4, head 1.39, norm 0.022, qk_norm 0.008.
  Early curvature is all in the embedding table.

## P3 cost correction (12:30 UTC Oct 15)
The 10.5 GPU-h P3 estimate assumed 614M-token runs. Actual configs: full-batch runs are 1,500 updates
of 32 sequences (49M tokens) and the minibatch runs are 0.32 epochs (196M tokens). Manifest estimates:
core 0.1 h, explore_trimmed 0.7 h, explore_rest 0.4 h, before sharpness-measurement overhead
(Lanczos on the probe every `every` updates). Plan: run all 22 P3 runs; calibrate overhead on p3_core.
Revised total if everything else holds: Full ~20 GPU-h.
A3 tests after P3/P4 edits: 34 OK (1 skipped). Ruff: own analysis files clean; 18 remaining findings
are in course-shipped files (unchanged).
