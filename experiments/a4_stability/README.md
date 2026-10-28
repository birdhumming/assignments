# Assignment 4: depth and time

Start with the repository [setup guide](../../README.md). Commands run from the
repository root and use your usual data, W&B settings, and Modal environment.

| Problem | Model definitions |
|---|---|
| 1: transformer depth | [p1_transformer_depth.py](p1_transformer_depth.py) |
| 2: RNN depth | [p2_rnn_depth.py](p2_rnn_depth.py) |
| 3: residual mixing | [p3_attention_residuals.py](p3_attention_residuals.py) |
| 4: recurrent updates | [p4_recurrent_updates.py](p4_recurrent_updates.py) |

Each architecture is a short class that overrides the relevant component or
update equation. Read its parent for the shared algorithm.

P3's EMA variants use `x' = beta*x + F(N(x))`: only the carried stream decays.
Half-and-half uses `x_next = .5*x + .5*x_previous + F(N(x))`, mixing the two
preceding stream states. Its first depth step is the usual `x + F(N(x))`.
Each recurrent/attention branch and each MLP counts as one depth step.

## Supplied training runs

The course reference sweep covers **P1 and P2 only: 21 variants × 5 learning rates = 105 outcomes**.
It is published in the public
[course W&B project](https://wandb.ai/hashimoto-group/public-alchemy-a4/overview),
tagged `a4-reference` and `a4-release-v1`. The five rates are
`(.25, .5, 1, 2, 4) × center`; [sweeps.py](sweeps.py) specifies the centers.
Baseline aliases across problems reuse the same runs.

Use these runs for the architecture comparisons in Problems 1 and 2; you do
not need to repeat those LR sweeps. `sweeps.PROVIDED_VARIANTS` lists their names,
and `sweeps.grid()` returns only these reference configs. Analyze the results
and checkpoints, and measure diagnostics. Add follow-up runs when useful.

No P3 or P4 training runs or checkpoints are supplied. For P3, train the
fixed/full/blocked residual variants and your own proposal; the implementations
are provided. For P4, train the base GRU, choose and implement your own ablations,
and develop a new update. The starter includes the GRU implementation without
predefined component ablations. Reuse P1/P2's additive and tanh baselines.

P1/P2 checkpoints are in `cs312-shared-data`, volume `hard-dl-dclm-v1`, mounted
read-only at `/root/shared_data/a4_stability`. Its `manifest.json` lists
all outcomes, public run links, model definitions, and available checkpoints.
Finite completed runs have final weights and an `early/` checkpoint from
step 100. A divergent run has an outcome and W&B link, without final weights.
The manifest contains only P1/P2 outcomes.

Names are identical in W&B, on Modal, and locally, for example:
`a4-rnn-no-residual-lr0.001`.

## Measure an existing checkpoint

[measure_default.py](measure_default.py) is a complete example: load the
released default d8, take a held-out input, compute depth/attention gradients
and paired context-removal losses, and save the returned JSON locally.

```bash
uv run python -m experiments.a4_stability.measure_default
```

Edit its `RUN` constant to inspect another supplied model. For local copies,
call `measure(run, root, device)` directly. The reusable measurement functions
are `diagnostics.profile(model, tokens)` and `diagnostics.context_loss(...)`.
They use FP32 probes and retain true recurrent-state nodes before stacking.
GRU `z` writes new information; reset precedes the recurrent candidate map.

## Run a new experiment

Edit the single call at the bottom of a problem file, then run that module:

```python
from experiments.a4_stability.sweeps import launch_one, launch
launch_one('rnn', .001)                    # One run on Modal.
launch(('rnn',))                          # Its five-rate sweep on Modal.
launch(('rnn',), dry_run=True)            # Print names/configs only.
launch_one('rnn', .001, local=True)        # The same run on a local GPU.
```

The module entrypoints have editable Python calls, without argument parsers.
`sweeps.VARIANTS` maps each available implementation to its model class,
problem, and LR center; only `PROVIDED_VARIANTS` have released sweeps.
Students' new runs use their own configured W&B project. Staff pass
public-project settings to these same configs.

For a new architecture, subclass a component in the relevant problem file,
then add your model to `sweeps.VARIANTS`. For example, select a new `cell_type`
on a `ModernRNN` subclass. Carry only `h` across time for Problem 4.
Use a new variant/run name when changing the architecture, seed, or recipe.

The recipe is width 512, vocabulary 4096, context 1024, batch 64, 600,000
sequences, AdamW, linear LR decay, weight decay .1, 1% warmup, clipping
at 1, and seed 42, inherited from `TrainConfig`. The default RMS logger
records module activations, parameters, and parameter gradients every 100
steps, with 100 validation evaluations per run. Detailed depth/time/attention
diagnostics run separately on checkpoints using `measure_default.py`.
The compile mode is `default` to avoid nested CUDA replay in the recurrent backend.

The small [_runtime.py](_runtime.py) backend supplies cuDNN tanh recurrence,
compiled 16-token editable updates, and CUDA replay. States stay connected
across chunks. The problem files contain the algorithm equations.

Synthetic CPU checks require no dataset download or cloud connection:

```bash
uv run python -m unittest discover -s tests -p 'test_a4_*.py' -v
```
