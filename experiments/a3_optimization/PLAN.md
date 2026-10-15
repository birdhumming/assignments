# A3 run plan (max 2 H100s; d8 614M run ~11.5 min; Hessian hvp ~2 s)

Calibration: d8 614M run with 4 EMAs = 16.5 min on H100 (val every ~94 steps); P3 minibatch run ~ 4 min train + ~32 sharpness probes x (10-16 hvps).

Status log (UTC): 2026-10-15 09:35 P2 wave 1 launched (15 runs, app ap-oemKixNr9ofTTvEDz3MP10). 10:08 first 4 schedule runs finished (lr 3e-3): linear 2.927, cos 2.934, wsd0.2 2.936, constant 3.099. Scope decision pending: Full ~28 h / Trimmed ~23 h / Lean ~18 h.

Order: P2 wave1 (running) -> P4 a (13 ckpts + 4 densities, pass max_parallel=2) -> P3 (a: 3 full-batch, b: 1 minibatch) -> P1 starter grid (prefix -> branches -> measure) -> P3 c exploration -> P4 b-rescale, b-train, c -> P2 wave2 (dip depth/duration + control, budget-tuned baselines) -> P1 d knobs.

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
