# Assignment 2 student starter

Use the shared trainer to implement the experiments in the
[handout](../../worksheets/hparam_invariants/README.md).
This starter supplies the language-model baseline, data preparation, and
measurement hooks. Source sweeps for P1(a,b,d,e) and P2(a) are supplied. New target experiments,
parameterization recipes, and analysis are student work.

## Setup and launch

Use the same one-time setup as [the course README](../../README.md): install
with `uv sync`, run `uv run modal setup`, and set the three student-facing
fields near the top of `utils.py`:

```python
CONFIG_MODAL_ENVIRONMENT = "YOUR_ASSIGNED_ENVIRONMENT"
CONFIG_WANDB_ENTITY = "YOUR_WANDB_USERNAME_OR_TEAM"
CONFIG_WANDB_PROJECT = "assignments"
```

Reuse your A1 settings and W&B secret if you already configured them. Set the
Modal CLI environment to the same assigned environment with
`uv run modal config set-environment YOUR_ASSIGNED_ENVIRONMENT`.
The course README describes creating the `dl-alchemy-wandb` secret if needed.

## Analyze the supplied sweeps first

P1(a,b,d,e) and P2(a) use 78 supplied source runs. Do not launch those sweeps again.
The [W&B project](https://wandb.ai/hashimoto-group/public-alchemy/overview)
contains all 78 supplied runs, including the cosine AdamW and linear Hyperball
source sweeps, and is publicly readable without logging in. The bundled CSV
also works offline.
List the configurations and direct run links for a part:

```sh
uv run python -m experiments.a2.provided_sweeps --part P1a --links
```

Use `P1b` only after recording your predictions for its larger budgets, and
`P2a` for the joint grids. Use `P1e` for cosine AdamW and `P1d` for linear
Hyperball at the three P1(a) budgets; reuse `P1a` as the control.
Omit `--links` to print the measurements as JSON.
Neither command launches training or requires W&B credentials. In Python:

```python
from experiments.a2.provided_sweeps import load, reference_diagnostics
rows = load('P1a')  # tokens, learning_rate, weight_decay, final_val_loss, ...
# TODO: fit the loss curves and make your own plots.
```

The [data README](../../worksheets/hparam_invariants/data/README.md) describes
the columns and width-512 diagnostic logs for P4.2. Fit the scaling laws and
make target predictions from these measurements.

P1(c)'s 4.9152B target sweep and both predicted-LR runs are **not provided**.
After fitting your two laws, pass your predicted LRs to the target launcher:

```sh
uv run python -m experiments.a2.p1_learning_rate --predicted-lrs LR_ALL_SIX LR_LARGER_THREE
# After checking the preview, repeat with --execute to launch.
```

Replace the two LR arguments with numbers from your own fits. The launcher
combines them with the three target-grid LRs and deduplicates exact matches.
Use the config factory below to construct P2(c)'s target experiments.

The launcher uses your configured Modal environment, course data mount,
private output volume, and W&B settings. It first prepares
each requested training prefix on a CPU worker, then launches independent
training runs concurrently through the shared launcher. Existing compatible
prefixes are reused. No data paths need to be filled in.

Training data is globally permuted with seed 42 before selecting the first N
sequences. Different budgets share the same training prefix.
The raw source is mounted at
`/root/shared_data/datasets/dclm_9p6m_ctx1024/train`; prepared prefixes are stored
under `/root/data/datasets/a2-global-prefixes` in your writable volume. Validation uses
the shared unshuffled validation split. These container paths are supplied by
the launcher, not paths that need to exist on your laptop.

The baseline uses 1% warmup followed by **linear LR decay**, AdamW coefficients
(.9, .95), QK normalization, unscaled RoPE, and the batching settings below.

The supplied measurements use these model and data settings. See the
[measurement guide](../../worksheets/hparam_invariants/data/README.md)
for run configurations, diagnostics, and worked-example measurements.
The baseline uses AdamW (epsilon `1e-8`, fused on CUDA). It caps
microbatch size at 64 sequences:
`microbatch_size = min(batch_size, 64)` and
`num_micro_batches = batch_size // microbatch_size`. Batches above 64 must be
multiples of 64. Thus batches 8/16/32/64 use one microbatch, 128 uses two, and
256 uses four. Model size also affects peak memory.

Incomplete final accumulation groups are skipped,
and included microbatch mean losses have equal weight. At 614.4M requested
tokens, batch 128 skips 64 sequences and batch 256 skips 192 (about 0.011%
and 0.032%). Without accumulation, the final partial batch is kept. Accumulated
runs skip compilation and can differ numerically
from full-batch execution. Record explicit overrides and actual processed
budgets when comparing runs.
For warmup followed by constant LR, use `lr_schedule='wsd0'`;
the optional cosine comparison uses `'cos'`.

For a custom sweep, use the same factory without path arguments:

```python
from experiments.a2.modal_launcher import config, launch_training_jobs

RUNS = [config(tokens=153_600_000, batch=128, learning_rate=.0015,
               weight_decay=.1, run_name_suffix='a2-my-comparison')]

if __name__ == '__main__':
    launch_training_jobs(RUNS)
```

For local CUDA runs, `baseline.config(train_path, val_path, ...)` and
`data.stage_prefix` remain available. Follow [the local GPU guide](../../gpu/README.md)
to download data, prepare each prefix once, and use the same prepared prefix
across comparisons. The Modal commands above do not require this local workflow.

## Hyperball and Muon

Both implementations are included; no additional package is required. The A2
Modal config factory selects their optimizer builder automatically.

For P1(d), pass your predicted or swept LR to:

```python
from experiments.a2.modal_launcher import config, launch_training_jobs

def hyperball_run(lr):
    return config(tokens=1_228_800_000, optimizer_name='adamh',
                  learning_rate=lr, weight_decay=0.,
                  run_name_suffix='a2-hyperball-target')
```

Launch your chosen configurations with `launch_training_jobs(RUNS)`.
[`hyperball.py`](hyperball.py) supplies AdamH for linear-layer weights,
including the readout. Embeddings, normalization parameters, and biases use
ordinary Adam at `(0.000656 / 0.00630) * learning_rate`, matching the supplied
Hyperball runs. Both groups follow the same scheduler; epsilon is `1e-8`.
Hyperball requires zero weight decay. Keep the fallback LR ratio fixed for P1(d).

For the optional Muon exploration:

```python
muon_run = config(
    optimizer_name='muon', learning_rate=.02, weight_decay=.1,
    optimizer_kwargs={'adam_learning_rate': 3e-4, 'momentum': .95},
    run_name_suffix='a2-muon-exploration',
)
```

These are starting settings, not tuned assignment results. Tune Muon's LR and
the auxiliary AdamW LR separately. [`muon.py`](muon.py) uses five BF16
Newton–Schulz steps, Nesterov momentum, and the upstream rectangular-matrix
factor `sqrt(max(1, rows / columns))`. Hidden linear weights use Muon;
embedding, readout, normalization, and bias parameters use AdamW. Weight decay
applies to hidden and readout weights; embedding, normalization, and bias
parameters are exempt. The auxiliary AdamW uses the config's betas and epsilon
`1e-8`. Both optimizers' groups follow the configured LR schedule.

The Muon implementation is adapted from
[KellerJordan/Muon](https://github.com/KellerJordan/Muon/tree/f98f1cacc0263b04290753e32be8d498c1efc806)
under its [MIT license](licenses/Muon.txt). It supports one device without
initializing a distributed process group. Missing gradients are skipped, and
optimizer steps preserve the recorded gradients.

For local training, set
`optimizer_builder='experiments.a2.optimizers:build_optimizer'` alongside the
optimizer name in `TrainConfig`; pass optional settings in `optimizer_kwargs`.
Muon routing expects an output module named `lm_head` or `head`.
For custom parameter groups, use `AdamH` or `SingleDeviceMuonWithAuxAdam`
directly in your optimizer factory. Width/depth initialization, readout
multipliers, and parameter-group scaling remain student work.

## Implement the experiments

P1 and P2 analyze supplied source sweeps and train new target configurations. P4.1 derives and
tests parameterization rules in a five-step Transformer; P4.2 returns to the
course model for longer training. P3.1 is an independent noisy-quadratic
simulation, and P3.2 tests its predictions in the language model.

The extension interfaces are:

- **Model:** `model_builder='your_module:YourModel'` and `model_builder_kwargs`.
  The builder receives the model config, `dtype`, `qk_norm`,
  `tie_word_embeddings`, and `dropout`. It may implement
  `initialize_parameters()` and `post_initialize()`. Forward behavior should
  be established in its constructor so it survives checkpoint loading.
- **Optimizer:** `optimizer_builder='your_module:build_optimizer'` and
  `optimizer_kwargs`. The factory receives the model, optimizer name, LR,
  WD, moment coefficients, and these extra arguments. The A2 baseline leaves
  `optimizer_builder=None` and `optimizer_kwargs={}`, using the
  shared optimizer (epsilon `1e-8`, fused AdamW on CUDA). Implement your own
  builder for parameter-group scaling rules. For a zero-momentum AdamW
  ablation, pass `beta1=0.0`, as required by the shared optimizer's float betas.
- **Run names:** Give custom architectures descriptive `model_config.name`
  values. Use distinct `run_name_suffix` values for custom optimizer/model
  settings and accumulation choices, so experiments save to separate directories.
- **Measurements:** `metric_loggers` accepts `MetricLogger(event, fn)` from
  `metric_logging.py`. Events include `AFTER_BACKWARD`, `AFTER_TRAIN_STEP`, and
  `AFTER_EVAL`. Loggers may implement `setup(context)` and `close()` and work
  when `wandb_online=True` (the default). These hooks write the local diagnostic
  files used in P4.2. Keep logging enabled when collecting those measurements.


Use FP32 for the width stress test and BF16 autocast with FP32 residual additions
for the depth stress test. The stress model uses the course gated-SiLU architecture
and context length 1024, with unscaled RoPE and untied embedding/readout.
Its five-step, constant-LR protocol differs from the longer-training baseline.
For alignment, use the handout's definition and each matrix's actual fan-in.
The before/after-update hooks can bracket each optimizer update.

For P4.2, implement the readout scaling derived in the handout in your custom
model builder. Pass its settings through `model_builder_kwargs` and establish
forward behavior in the constructor so saved models and checkpoints reconstruct
it. The P4.1 stress scaffold separately provides `model.output_multiplier`.


## Reuse measurements across problems

Reuse the supplied P1/P2 configurations wherever later questions request them.
For P4.2's width-512 source, `reference_diagnostics()` returns feature and
alignment records; select fixed-input feature probes with
`record['kind'] == 'fixed_batch'`. Do not retrain this reference just to obtain
its diagnostics. The measurement guide describes the fields and checkpoints.
Other widths and target experiments still require training.

Baseline feature and actual-update alignment logs are enabled by default on new
runs. Keep them from the beginning of training. Each run writes `features.jsonl`
and `alignment.jsonl` beneath its output directory.
W&B shows per-matrix curves under `logging/alignment`, readout alignment under
`logging/readout_alignment`, and combined plots under `alignment_charts`.
Alignment curves use completed optimizer updates on the x-axis; undefined
ratios are omitted. The JSON logs are also available in the run's Files tab.

## P4.1: supplied Transformer scaffold

The architecture and five-update experiment driver are in
[`stress.py`](stress.py). Complete **only the two policy functions** in
[`p31_student.py`](p31_student.py) to start an experiment:

1. `initialize(model)`: initialize every parameter and set any forward multipliers
   from your derivation. Parameters initially contain NaNs so omitted weights
   cannot silently use PyTorch's default initialization.
2. `parameter_groups(model, base_lr)`: return Adam groups containing `params`,
   `lr`, and `eps`. Include every parameter exactly once. The runner supplies
   coefficients (.9, .95), zero weight decay, and constant group LRs.

The model has untied embedding/readout,
pre-RMSNorm, causal attention, QK normalization, unscaled RoPE, and gated-SiLU
feedforward layers with inner width 3.5 times hidden width. Context length is
1024 and head dimension is 64. There are no biases, dropout, gradient clipping,
or warmup. For width comparisons, use `--precision fp32`: all computation is FP32,
attention uses the PyTorch math backend, and TF32 is disabled. For depth
comparisons, use `--precision mp`: BF16 autocast with FP32 parameters, gradients,
Adam states, and residual additions. This driver is separate from the
longer-training P1 baseline.

After implementing the two functions, launch one configuration from your laptop:

```sh
uv run python -m experiments.a2.modal_stress \
  --width 640 --depth 2 --head-dim 64 --precision fp32 --lr .001 \
  --output my-policy-width640-lr001.json
```

This uses your existing course Modal environment and saves the JSON in your
private volume under `/a2-stress/`. It also logs losses, alignment curves, and
plots to your configured W&B project using the `dl-alchemy-wandb` secret.
Use `--no-wandb` for a local-output-only run. Download the JSON with `uv run modal volume get
VOLUME_NAME /a2-stress/my-policy-width640-lr001.json ./`; use the volume name
printed in your setup (`volume-dl_alchemy` by default) and your assigned Modal
environment. Use a unique filename for every run.

If you already have a CUDA worker with the course data mounted at
`/root/shared_data`, the equivalent direct command is:

```sh
uv run python -m experiments.a2.p31_student \
  --train-path /root/shared_data/datasets/dclm_9p6m_ctx1024/train \
  --val-path /root/shared_data/datasets/dclm_9p6m_ctx1024/val \
  --width 640 --depth 2 --head-dim 64 --precision fp32 --lr .001 \
  --output output/p31/my-policy-width640-lr001.json
```

For width comparisons, use widths 640, 2560, and 5120.
For depth comparisons, use width 64, head dimension 64, and `--precision mp`,
and change `--depth` among 2, 100, and 1000.
Implement each requested depth policy in the same two functions. Use distinct
output names for different policies and LRs; existing files are never overwritten.
You can call `stress.run` from your own sweep or existing remote launcher.
Nothing launches when either module is imported.

`load_tokens` reads a fixed contiguous token prefix from each prepared DCLM
split and rechunks it into length-1024 sequences. Reuse the same train and
validation tensors across all policies and LRs. The CLI reads 320 training
sequences (five batches of 64) and 64 validation sequences. It does not download
data or stage a new Modal volume. Microbatches accumulate to the full batch
before an update; changing `--microbatch` changes memory use, not the number
of optimizer updates. Segment checkpointing reduces activation memory at depth.

Parameter names are `embed.weight`, `head.weight`, `norm.weight`, and
`blocks.<i>.{q,k,v,o,gate,up,down,norm1,norm2,qnorm,knorm}.weight`.
`model.output_multiplier` multiplies the final normalized input to the readout.
Each `block.residual_multiplier` multiplies its attention and feedforward branch
before residual addition. Set both multipliers according to your prescription.

The JSON records loss and fixed-input diagnostics at steps 0 through 5:

- `history`: validation loss, logit RMS, normalized-feature RMS/movement,
  residual RMS before normalization, and branch RMS before its multiplier.
  Movement is the RMS change from initialization, pooled over probe sequences,
  token positions, and hidden coordinates; relative movement divides by initial
  feature RMS. Feature probes use the first, midpoint, and last block's first RMSNorm output,
  plus the final RMSNorm output. Block indices are zero-based and included.
- `alignment`: actual single-update measurements for every block's seven matrices
  and the readout. Inputs come from the preceding probe, before that update;
  each record includes the actual matrix fan-in and `alpha`, the alignment
  exponent defined in the handout (the logarithm of the RMS ratio with base
  equal to fan-in). Undefined
  alignment exponents are `null`, with the reason recorded in `status`.
- Configuration, parameter-group LRs/stabilizers, forward multipliers, and token
  hashes identify the run. Select/fill your own policy name in the output filename.

Validation loss uses all supplied validation inputs; feature/alignment probes
use the first eight. Alignment snapshots live on CPU and are transferred one
matrix at a time for measurement. `--no-alignment` omits those snapshots when
only the depth feature diagnostics are needed. Sweep selection, curve fitting,
and plotting remain student work.

## P3 settings

P3.1 fixes the NQM second-moment coefficient at beta2=.95. P3.2 uses
614.4M tokens; pass `tokens=614_400_000` explicitly to the config factory.
The config factory defaults to 153.6M tokens.

The batch-switching example also uses 614.4M tokens, switching from total
batch 64 to 128 at exactly 307.2M tokens. It compares initialization seeds
42, 43, and 44 at fixed token order, peak LR .0015, and initial WD 1.6.
Implement the switching policies and preserve Adam moments; the LR schedule
continues by tokens processed. The starter does not implement the switching policies. The handout includes
the example-question answer key for checking your interpretation.

## Readout--feature-change alignment probes

Use `readout_alignment.movement` for the requested $S(V_0,h_t-h_0)$,
where $V_0$ is the initial readout and $h_t$ is the final RMSNorm output
before the readout multiplier. The record contains `ratio`, `omega`
(the base-width logarithm), RMS components, and `status`. Zero feature
movement has `omega: null`. The action is measured in FP32 even during BF16
training, with output tiles to bound workspace. Initial weights and features
are retained for the movement probe; P4.2 saves them in `initial_features.pt`
for checkpoint resume.

In P4.2 these fields are in `features.jsonl`. In the P4.1 student scaffold they
are in each `history` row.
The readout probes remain enabled when per-matrix update alignment is disabled.

## Interpreting finite-width effects

The baseline embedding scale decreases as `1/width`, so RMSNorm epsilon can dominate
normalization; clipping can also vary with width. Compare the feature RMS logs
and `gradients.jsonl` (pre-clipping norm, embedding fraction of squared gradient
norm, and clipping coefficient) before attributing a difference solely to muP.
Matched ablations are optional, with no extra required GPU runs.

See [runtime details](RUNTIME.md) for checkpoints, batching, and diagnostics.
With positive warmup, the first update uses zero LR, and the scheduler reaches
zero again after the last update. W&B records the LR applied to each update.
