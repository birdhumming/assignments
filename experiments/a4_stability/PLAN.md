# A4 plan and ledger (branch a4-experiments; due Oct 29 1 PM PT)

Supplied: 105 P1/P2 outcomes (manifest.json, copied to results/). P3/P4 trained here. W&B host moved to forge.coreweave.com;
the public hashimoto-group project is not readable with our key, so P1/P2 losses come from the manifest.

## Supplied best validation loss (manifest)
P1 d8 2.9265@3e-3 | gated-residual 2.9096@3e-3 | gated-attention 2.9174@6e-3 | gated-norm 2.9385@2.25e-3 | no-qknorm 2.9511@3e-3 (diverges >=6e-3)
   no-final-norm 2.9561@3e-3 | no-rope 3.0654@3e-3 | no-prenorm 3.0685@1e-3 (nonfinite >=2e-3) | postnorm 3.1449@1e-3 (6.875 at 4e-3) | no-residual 6.875 (all 5 lr)
P2 rnn1 4.0326@3e-3 | rnn4 4.0809@1e-3 (6.875 at >=2e-3) | modern rnn 3.5160@2e-3 | no-prenorm 3.4822@2e-3 | gated-residual 3.4956@1e-3
   gated-norm 3.5059@1e-3 | gated-output 3.5125@1e-3 | postnorm 3.5124@1e-3 (diverges >=2e-3) | final-norm 3.5213@1e-3 | rope 3.5664@2e-3 | no-residual 3.5813@1e-3

## Predictions locked 10:40 PT Oct 28, before any P3/P4 run
- P3 fixed mixing: EMA beta=0.1 (carried stream decays 10x per step) behaves like no-residual: transformer >> 3.2, RNN > 3.6.
  beta=0.9: small loss (+0.02 to +0.05 d8, +0.03 RNN). Half-and-half ~ additive (+-0.02). Uniform: worse than additive for d8 (+0.05, late
  updates diluted by 1/j), about the same for the 4-layer RNN (only 8 depth steps).
- P3 full attention residuals: d8 within +-0.02 of additive (2.93), RNN improves slightly (-0.02 to -0.04 vs 3.516). Blocked B=2/B=4 land between
  full and additive; B=4 ~ additive. Routing weights will favour the embedding and the most recent 1-2 sources.
- P4 GRU: beats tanh modern RNN by >=0.10 at its best lr (expect ~3.40 or better), best lr >= 3e-3 (gates stabilise); update gate matters more than reset.

## Ledger
- 10:37 UTC Oct 28: calibration runs launched (W&B init working again): gru lr3e-3 app ap-0d7dmFRXmslDYuyTe2EK8m, rnn-attention lr1e-3 app ap-nF1BbwuZdyfYlyMgVwiadB.
- Student variants in student.py (registered in sweeps.VARIANTS): d8-hybrid / rnn-hybrid (P3d: additive stream + attention-routed branch input),
  gru-no-reset, gru-reset-only, gru-input-gate (P4a ablations), normgate (P4b: no reset, RMSNorm(h) in candidate, update bias -1), normgate-b0.
  CPU smoke test + 15 a4 tests pass. Launch with sweeps.launch((name,), rates=(...)); max_parallel_runs=min(2, n) per call.
- analysis/p12_tables.py -> P1/P2 best-loss table + figures/p12_lr_curves.png (from manifest). measure_supplied.py -> one Modal H100 job
  measuring init/early/final for the 20 best-LR supplied runs (d8-no-residual has no checkpoints) into results/a4_stability/<run>/<stage>.json
  (results/ is gitignored; keep JSON locally, regenerate figures with analysis scripts).
- 10:36-10:55 UTC: both calibration apps sat at 0 containers (H100 capacity wait).
- 10:56 UTC: no H100 container in 19 min; stopped both H100 apps and relaunched on H200: gru 3e-3 ap-tRYCpvAMQ6IYuW69STCO7T, rnn-attention 1e-3 ap-JzDXYMDUdergprAVxzq5zd (launch_training_jobs(..., gpu="H200")).
- Calibration (H200): GRU 3e-3 runs ~94 steps/min -> ~1.7 h per GRU-type run. Queue = launch_plan.py (watcher nohup, log /tmp/a4_lanes.log):
  lane A1 d8 P3 (16 runs: 7 schemes x {3e-3, 6e-3} + block2 {4.5e-3, 9e-3}), lane B P4 (8 runs: gru 1.5e-3/6e-3, normgate 3e-3/1.5e-3,
  no-reset/reset-only/input-gate 3e-3, normgate-b0 3e-3), lane A2 RNN P3 (16 runs: 8 schemes x {1e-3, 2e-3}). Each lane = one app, 1 container.
  Next lane launches when < 2 ephemeral apps. Then: measure_supplied (P1/P2 checkpoints) and the same driver on our P3/P4 checkpoints (user volume).
- 11:05 UTC lane A1 app ap-cSqdhMO9UzbXII7Jf03d2l (16 d8 P3 runs). Watcher in 'watch' mode: measure -> B-p4 -> A2-rnn-p3 (log /tmp/a4_lanes_watch.log). Restart: setsid nohup uv run python -m experiments.a4_stability.launch_plan watch &
- 11:18 UTC: lane A1 on H200 got no container in 12 min (second H200 never schedules); stopped it and relaunched lane A1 on H100: app ap-FAcMZFxxNgMz8FlaHVIiBi.
- 11:28 UTC: H100 lane A1 also got no container in 11 min; stopped it, relaunched lane A1 on A100-80GB: app ap-2JNlaZ9MVx9vIXHPBrNqxn. GRU 3e-3 at 28%, ETA 12:45 UTC.
- 11:32 UTC: second container never schedules on H200, H100 or A100-80GB while the GRU app holds one GPU -> environment concurrency
  limit looks like 1 right now (README promises 2). Lane A1 (A100 app ap-2JNlaZ9MVx9vIXHPBrNqxn) is pending; definitive test = does it
  start when GRU finishes (~12:45 UTC). If limit stays 1: trim P3 to 1 LR per scheme (+2nd LR for attention/hybrid only), P4 to gru x2,
  no-reset, reset-only, normgate x1. Merge check: a1 modal_train fixes (orphaned W&B close, GPU memory release) survived the a4 merge.
- 11:42 UTC: lane A1 got its A100 container after 13 min -> 2 concurrent GPUs OK; it was capacity, not a limit. GRU 3e-3 at 43%, val 3.431 at step 4042.
- 11:55 UTC: d8 runs take ~1 h on A100 (vs 11 min H100). Reordered every lane to run all center-LR variants first (stop early = still a full grid); stopped A1 and relaunched reordered on A100; watcher restarted. GRU 3e-3 52%, val 3.387.
- 11:52 UTC: lane A1 (reordered, A100) app ap-ISYOEYaTCZC0nFbMVhno5I. Watcher restarted (measure -> B-p4 -> A2-rnn-p3).
- 12:39 UTC RESULT gru lr3e-3: final val 3.1874 (tanh modern RNN 3.5160 -> -0.33; prediction >=0.10 held, far exceeded). Lane A1 run 2 (d8-half 3e-3) started; d8 runs ~40 min on A100.
- 12:40 UTC RESULT d8-attention lr3e-3: final val 2.9036 (d8 additive 2.9265 -> -0.023; prediction 'within +-0.02' slightly beaten). Next: d8-half 3e-3.
- 12:41 UTC: watcher launched the measure lane (measure_supplied, app ap-jWMvntVWK4zxJfDjsPNwY3, H100; blocking subprocess in the watcher,
  writes results/a4_stability/<run>/{init,early,final}.json for 20 supplied best-LR runs). After it: lane B (P4, 8 runs) then A2 (RNN P3, 16 runs).
  Lane A1 (A100, ap-ISYOEYaTCZC0nFbMVhno5I) running d8-half 3e-3 (run 2/16, ~40 min each -> lane ends ~22:00 UTC).
  Our run checkpoints live on the user volume at /root/data/ckpts/<run_name> (final) with step_100.pt inside; measure them with
  measure_runs(jobs, root=Path('/root/data/ckpts')) once the early-checkpoint layout is confirmed (the supplied runs use <run>/early/).
- 12:50 UTC: measure lane H100 container never scheduled (9 min); stopped it, rerunning measure_supplied on A100-80GB (nohup, log /tmp/a4_measure.log). Watcher restarted with explicit lanes B-p4 A2-rnn-p3.
- 12:50 UTC: measure_supplied running on A100 (scheduled immediately), ~1 min per run; watcher (lanes B-p4, A2-rnn-p3) waits for <2 ephemeral apps.
- 12:58 UTC: measure_supplied DONE: results/a4_stability/<run>/{init,early,final}.json for 20 runs (gitignored; local only). Watcher then launches lane B (P4) on H200.
- 12:59 UTC RESULT d8-half lr3e-3: final val 2.9355 (+0.009 vs additive 2.9265; prediction 'half ~ additive +-0.02' held). Lane B app ap-JSGDYvghlcemV8GGfkDxov (H200) launched by watcher, waiting for container. Next in A1: d8-ema09 3e-3.
- 13:00 UTC JSON layout (results/a4_stability/<run>/<stage>.json): keys full/last (loss, depth[17 rows for d8 / 9 for rnn: activation_rms,
  gradient_rms, raw_branch_input_ratio, adjacent_cosine], hidden_rms[layer][t], hidden_gradient_rms[layer][t], fraction_abs_gt_095[layer][t]),
  attention{layer: query_positions, first_mass, self_mass, uniform_first_self, mean_lag}, routing, temporal_cosine[layer]{lag: cos},
  context_loss_increase{'<keep>_first_<bool>': mean dLoss}, full_last_loss, num_params.
  d8 final: act rms grows 0.15 -> 24 across depth; grad rms shrinks 1.3e-5 -> 3e-7 (full L). Context: keep 16 costs +0.97 (+0.53 with BOS kept);
  keep 256 ~ 0 -> d8 uses ~64-256 tokens. Modern RNN: last-token hidden grad falls 1e-19 (t=0) .. 6e-5 (t=896): strong vanishing; tanh saturation 0.14-0.67.
  NEXT: analysis/p12_diagnostics.py -> figures (depth act/grad per variant, attention first/self mass vs uniform, RNN time curves, context curve),
  then report. Lane B (P4) app ap-JSGDYvghlcemV8GGfkDxov waiting for H200 container; lane A1 at run 3/16 (d8-ema09).
- 13:05 UTC analysis/p12_diagnostics.py runs (figures p1_depth.png, p2_time.png, p12_context.png). Key supplied-run findings (final ckpts):
  d8 act RMS x11.7 across depth, grad ratio bottom/top 4.4; postnorm x1.4 growth but grad ratio 0.3 (gradients shrink toward input) and
  worst loss; no-prenorm grad ratio 11, first-token attention mass 0.44 (sink); d8 first mass 0.17, self 0.075, uniform baseline 0.5.
  RNN: modern rnn sat 0.15-0.68, no-prenorm/rope layers 3-4 saturate ~0.99 (rms 1.0); no-residual adjacent cos ~0. Context: d8 uses
  ~64-256 tokens (K16 +0.53 with BOS); RNNs flat beyond K16-64 (K16 +0.1-0.2). BOS-dropped K1 for rnn/rnn-rope explodes (+29..+680):
  state without BOS reset is out of distribution -> report as the 'preserve first token' control.
  TODO: temporal_summary reach metric is wrong (always 1023; should find first t where grad < 1e-3*final scanning backwards, use np.where).
- 13:03 UTC lane B (P4, ap-JSGDYvghlcemV8GGfkDxov) got its H200 container after ~8 min; both GPUs busy (A1 d8 on A100, B on H200).
  Watcher (pid 100527) will launch A2-rnn-p3 when one lane finishes. ETA: A1 ~4 h (16 d8 runs x ~15 min), B ~14 h (8 recurrent runs x ~1.7 h).
