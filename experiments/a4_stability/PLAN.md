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
