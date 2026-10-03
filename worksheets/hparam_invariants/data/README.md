# Provided A2 source measurements

`provided_sweeps.csv` contains 78 unique completed runs. A run may serve
multiple parts; `parts` separates these labels with `|`. Values are measured
final validation losses. Fitting, plotting, and prediction remain student work.

All source runs use the default width-512, depth-8 model with unscaled RoPE,
QK normalization, batch 64, one pass through a prefix of the globally shuffled
source dataset (data seed 42), and model seed 42. They use moment coefficients
(.9, .95), 1% warmup, and the listed optimizer and schedule. AdamW uses
epsilon 1e-8 and fused execution on CUDA. Validation uses the same 1,000 sequences.

| Part | Supplied runs | Optimizer / schedule | Token budgets |
| --- | ---: | --- | --- |
| P1a | 9 | AdamW / linear | 153.6M, 307.2M, 614.4M |
| P1b | 9 | AdamW / linear | 1.2288B, 1.8432B, 2.4576B |
| P1d | 24 | Hyperball (`adamh`) / linear | 153.6M, 307.2M, 614.4M |
| P1e | 9 | AdamW / cosine | 153.6M, 307.2M, 614.4M |
| P2a | 36 | AdamW / linear | Four supplied budgets; overlaps P1 |

P1e uses LR {.0015, .003, .006} and WD .1. P1d uses LR
{.0003, .001, .0015, .003, .006, .01, .015, .03} and WD 0.
The listed Hyperball LR applies to linear-layer matrices. Its ordinary-Adam
parameters (embeddings, normalization gains, and biases) use LR
`(0.000656 / 0.00630) * listed_lr`, with the same schedule multiplier and epsilon 1e-8.
Reuse P1a as the linear AdamW control. No constant-LR source sweep is supplied.
No 4.9152B run or target prediction is included. Fit the three Hyperball source
budgets and test a new 1.2288B target as specified in P1(d).

Run `uv run python -m experiments.a2.provided_sweeps --part P1d` to read
measurements offline. Replace `P1d` with another part; add `--links` for configurations
and direct W&B URLs. All 78 runs are in the
[public course project](https://wandb.ai/hashimoto-group/public-alchemy/overview).
The CSV identifies every measured run by its URL and run ID and also works offline.

`reference_diagnostics.json` contains probes from the same width-512,
153.6M-token reference run at LR .003 and WD .1:
[jysl6iqh](https://wandb.ai/hashimoto-group/public-alchemy/runs/jysl6iqh). Use it for P4.2.

- `diagnostics`: fixed-input records (`kind == "fixed_batch"`) containing
  `features` (each normalized feature's `rms` and `movement`), `residual_rms`,
  `branch_rms`, probe `loss`, and `logit_rms`.
- `gradients`: pre-clipping norm, embedding fraction of squared gradient norm,
  and clipping coefficient, measured on training batches.
- `alignment`: each matrix's actual-update alignment, actual fan-in, input step,
  and status. Omit records with undefined denominators.
- `readout_alignment`: `current` and `movement` readout alignment on eight fixed
  validation sequences. Use `movement` for $S(V_0,h_t-h_0)$ and its `omega`
  for the requested movement exponent. Omit zero-movement records.

Feature and readout checkpoints include initialization, updates 1–5, and
approximately 1%, 10%, 50%, and 100% of training (final update 2,344).
Actual-update alignment uses updates 1–5 plus logarithmically spaced checkpoints.

`example_measurements.csv` records the 54 worked-example measurements across
initialization seeds 42, 43, and 44, with data order fixed at seed 42: nine
delayed-weight-decay runs, 18 batch-switch/control runs, and 27 copied-width
trajectories. It includes per-seed final losses, copied-model logit discrepancies,
and direct run URLs. Use these per-seed values to compare means and variability.
