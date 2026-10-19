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

## P4(a) results (12/13 at 13:40 UTC Oct 15; step 9375 pending)
- lambda_max: 1431 (step 0) -> 1572 (3) -> 53 (15) -> 18 (61) -> 6.5 (153) -> 4.3 (305) -> 3.0 (763) -> 2.2 (1526, minimum)
  -> 3.2 (3052) -> 3.7 -> 4.2 -> 5.3 (7630). Sharpness falls 650x in the first 1.5% of training, then RISES 2.4x
  during the linear-decay phase as the learning rate shrinks (the step the optimizer can tolerate grows as eta falls).
- lambda_min: -1410 at init (|min|/max 0.985: init is a saddle, as negative as positive) -> -602 (3) -> -17 (15)
  -> below -0.1 from step 305 on; |min|/max < 0.03 after step 153. Negative curvature is gone early.
- Top eigenvector: 99% embedding at steps 0-3, 71-88% embedding at 15-61, then 93-97% head from step 153 on.
  Embedding block lambda_max 1414 -> 0.016 (collapses by 10^5); head block 0.1 -> 17 (step 15) -> 2-5 and is the
  top block from step 153 to the end. Attention/MLP blocks grow 0.3 -> 1.0 / 0.1 -> 0.5 over the decay phase.
- Top-5/top-1 ratio 0.82 at init -> 0.40-0.52 later: one outlier direction dominates late.
- Gradient overlap with top eigenvector: +2.19 at step 3, |overlap| < 0.03 after step 153 (gradient lives off the top direction).
- Family-pair energy (step 7630): off-diagonal share 0.256; largest couplings attention-mlp (0.63) and embedding-attention/mlp (~0.39);
  head couples weakly to everything (<0.18).
- Runtime: 13.5 min per checkpoint, 27 min with density; 13 jobs = ~3.3 GPU-h.
- Step 9375 (final, 13/13 at 13:50 UTC): lambda_max 5.97 (2.7x the step-1526 minimum), lambda_min -0.08, head 94% of top eigenvector,
  grad overlap -0.002. P4(a) total 3.3 GPU-h. Trace stage (Hutchinson) still to run.

## P4(b) rescaling check (13:58 UTC Oct 15)
Verified on a tiny model in float64: gain / 4 with the fed linears * 4 (attention input norm -> q,k,v;
MLP input norm -> gate,up; final norm -> lm_head; and all three together) changes no logit (max |delta| = 0.0).
So any lambda_max change from b-rescale jobs is purely a reparameterization at identical loss.

## Launch log
- 13:37 p3_core (4 runs, parallel 1): full-batch run ~14 min each (sharpness every 25 updates, ~0.9 s/update), eta*lambda = 2.36 at update 1349.
- 13:51 P1 prefix (1000 steps, done 13:55); 13:56 P1 branches (12 runs, parallel 1).
- Queue: P1 measure -> P4 trace -> P4 b-rescale -> P4 b-train -> P3 explore -> P4 c -> P1(d) variants -> P2 wave 2.

## P3 core results (14:25 UTC Oct 15; 3 full-batch runs done, minibatch running)
- All three full-batch runs (lr 0.01, 32 sequences) start at eta*lambda = 0.22, climb (progressive sharpening) and cross 2 at
  update 375, then hover at 2.2-2.7 (plateau medians 2.56 constant / 2.24 halved / 2.70 doubled). Loss rises on 0% of updates
  before the crossing and on 24-41% after: classic edge of stability.
- Halve lr at 750: eta*lambda drops to 1.3 at once (eta halves, lambda does not), then lambda re-sharpens and is back at 2 by ~950.
- Double lr at 750: eta*lambda jumps to ~6 (above 2 -> unstable), loss spikes briefly, lambda falls over ~600 updates until
  eta*lambda is back near 2.2 at 1400 and dips below 2 at the end (loss 0.33, model is memorizing the 32 sequences).
- Final losses 1.23 / 2.25 / 0.33: doubling the lr trains faster once the curvature has adapted; the sharpness self-regulates
  to lambda ~ 2/eta in both directions.

## P4(a) trace results (14:40 UTC Oct 15; 13 Hutchinson jobs, 16 probes each, ~35 s per checkpoint)
- tr(H)/d is negative at init (-4.2e-4: more negative than positive curvature on average, saddle), jumps to +3.8e-3 at step 3
  (tr(H) = 132k), then falls to a minimum 1.2e-5 (tr(H) = 422) at step 1526 and rises to 2.7e-5 (947) at the end,
  the same U shape as lambda_max.
- lambda_max / (tr(H)/d) sits at 1.8e5-2.5e5 from step 61 on: the top eigenvalue is ~2e5 times the mean eigenvalue and that
  ratio barely moves, so the whole spectrum shrinks and regrows together.
- Participation ratio tr(H)^2 / (d tr(H^2)) = 1.1e-4 to 1.4e-4 after step 300, i.e. ~4,000-5,000 of the 34.6M directions
  carry the curvature. At init it is 2e-8 (a handful of directions in the embedding).

## P3(b) minibatch AdamW reference (14:45 UTC Oct 15; lr 3e-3, bs 64, 3000 updates, warmup 293)
- eta*lambda_pre: 1847 at update 16 (top eigvec 100% embedding), 98 at 32, then 35-48 from update 48 to 208 (oscillating around
  the threshold 38; top eigvec now ~50% MLP / 40% attention), then drifts down: 33.7 @500, 27.3 @1000, median 24.2 over
  100-180M tokens, 20.4 at the last probe (update 2700). Touches the edge during warmup, then sits below it. Final train loss 3.24.
- Cache bug fixed: p3_tables cached a running run's partial history; now only finished runs are cached.

## P4(b) loss-invariant rescaling (final AdamW checkpoint, 16:30 UTC Oct 15)
- Probe loss is 2.9474 in all 6 jobs (exactly invariant). lambda_max: reference 5.97, attn-norm 5.88, mlp-norm 6.46,
  head-norm 5.51, all-norms 7.82, qproj-only 5.97.
- Per-tensor: lm_head block 5.64 -> 0.35 (/16 as predicted) under head-norm; final norm gain 0.30 -> 4.79 (x16).
  The curvature moves to the other side of the reparameterization; the top eigenvector flips from 94% head to 89% norm.
  Scale 4 overshoots. Balanced scale s=2 should give head 1.41, norm 1.20 -> lambda_max ~1.4-1.5 (4x cut, same loss).
  Prediction locked before the b-rescale-head-norm-s2 job runs.
- Gradient norm also changes (0.130 -> 0.106 head, 0.084 all): the gradient is not invariant, only the loss is.
- s2 result: loss 2.9474, lambda_max 2.35 (predicted 1.4). Blocks as predicted (head 1.41, norm 1.20); the cross term
  between them lifts the top eigenvalue; top eigvec 44% norm / 29% head / 19% MLP. Net: 2.5x cut at identical loss.

## P3(c) exploration (10 runs, 17:35 UTC Oct 15; median eta*lambda_pre over 100-180M tokens / final)
- lr 1e-3: 19.4/17.1, 3e-3: 24.2/20.4, 1e-2: 30.5/48.6 (loss 3.39, worst). Higher lr -> closer to / over the edge.
- bs16: 8.6/8.8 (far below), bs256: 47.7/59.9 (above 38 the whole run; loss 3.34). Noise keeps small batches off the edge.
- beta1=0: 10.5/11.5 (threshold for RMSProp ~2, so still 5x over it; loss 3.37); beta1=0.98: 34.8/31.6.
- wd 0: 18.4/17.1 (loss 3.20, best of constant-lr runs); warmup 12.5%: 22.6/23.5 (same as 3%).
- cosine decay: 10.8/5.4 -> eta*lambda_pre FALLS with eta, i.e. lambda_pre does not grow to compensate (prediction wrong).
- emb16: 22.1/21.6 (same as reference). Final losses are training losses at 3000 updates.

## P4(b) retrained variants (final-checkpoint Hessian, 21:50 UTC Oct 19; probe loss / lambda_max)
- reference linear 2.9474 / 5.97; cos 2.9532 / 5.62; wsd0.2 2.9592 / 4.39; lr1.5e-3 2.9674 / 5.56; wd0 2.9751 / 2.83 (top eigvec 75% norm);
  constant lr 3.1231 / 3.30. Predictions (constant lr and wd0 lowest lambda_max) held. Training knobs move lambda_max <=2x and always
  move the loss; the s=2 rescaling moved it 2.5x at zero loss change.

## P4(c) projected updates (SGDM step 1526, 300 steps, k=10, refresh 25; 21:50 UTC Oct 19)
- full -0.112 val, removed -0.112, top-only +3.12 (diverges, lambda_max 2.65 -> 16.8, crosses heavy-ball threshold 3.8/eta ~ 4.5 near step 1700).
- top fraction of grad energy: full 0.09, removed 0.03, k=50: 0.15 / 0.05. removed never worse than full (k50, lr0.3, step305 all the same).
- removed with refresh 5: loss unchanged (3.584) but lambda_max runs to 69 (eta*lambda 56): curvature is regulated only by stepping in it.

## P1(d) recipe variants: prediction locked 22:15 UTC Oct 19 (branch runs training, nothing measured yet)
- Same 6x5 grid as default (lr 3e-3, bs 64, constant lr). Variants: lr6e-3, lr1.5e-3, bs256, linear decay.
- lr6e-3: more failing cells (expect x=100,y=250 and x=250,y=500 to join the four default failures); larger negative barrier where it works (bigger bounce).
- lr1.5e-3: fewer failures (only the x=0 column), smaller averaging gain (~half the default -0.1).
- bs256: fewer failures (data-order noise /4 at the same step count); smaller averaging gain. Higher endpoint loss at equal steps is not the point; the grid compares cells within a variant.
- linear decay: identical to default to within noise. The schedule has decayed <16% of the way by step 1500 of 9375, so the branches see nearly the same lr as the constant run.

## P1(d) results (22:52 UTC Oct 19; lr1.5e-3 still running). pass = cells with B<=0.01 of 30; gain = median averaging gain where it works
- default 26/30, fails (0,250)(0,500)(50,500)(100,500), worst B 0.59, gain 0.085, L(1000,500) 3.384
- linear  26/30, same four failing cells, worst 0.61, gain 0.080, L 3.369 -> identical to default (prediction held)
- lr6e-3  24/30, adds (0,100)=+0.040 and (50,250)=+0.073; worst (0,500)=1.20 (2x default), gain 0.091, L 3.448
- bs256   24/30, adds (0,50)=+0.022 and (0,100)=+0.214; worst (0,500)=1.31; gain 0.058 (smaller bounce, as predicted); L 3.164
  Prediction WRONG on failures: less data noise did not help; bs256 branches travel further per step (4x tokens), so a 500-step
  branch is a longer branch in progress units. x=250 is the first fully-connected prefix in every variant.
- Pattern: the knob that matters is how far the branches move relative to the prefix (lr x steps, or tokens), not gradient noise.
  Further tests: match branches by tokens (bs256 y=125 vs bs64 y=500) or by loss reached; vary the data-order seeds for the prefix.
- lr1.5e-3 (23:12 UTC) 27/30, fails (0,250)(0,500)(50,500); worst (0,500)=0.31; gain 0.076; L 3.388. First fully connected prefix x=100.
  B(0,500) vs lr: 0.31 / 0.59 / 1.20 at 1.5e-3 / 3e-3 / 6e-3: barrier roughly proportional to the learning rate.
