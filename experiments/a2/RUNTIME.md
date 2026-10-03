# A2 runtime details

Use the complete assignment checkout and configure your Modal environment and
W&B credentials as described in the [runtime guide](README.md).

## Optimizers and model builders

The baseline uses shared AdamW with epsilon `1e-8` and fused execution on CUDA.
Set `optimizer_builder` and `optimizer_kwargs` to supply an optimizer factory
for custom parameter-group rules. The A2 Modal config factory also supports
`optimizer_name="adamh"` (Hyperball) and `optimizer_name="muon"`; see the
[optimizer guide](README.md#hyperball-and-muon) for grouping and LR settings.
Set `model_builder` and
`model_builder_kwargs` for a custom model.

## Run names and checkpoints

Run names include the model name, standard hyperparameters, and
`run_name_suffix`. Give each custom architecture a descriptive
`model_config.name` and use a distinct suffix for changes not represented in
the standard name, such as parameterization, builder arguments, data contents,
or accumulation settings. For example: `run_name_suffix="mup-w256-d8-mb64"`.
Reuse a name only when resuming the same experiment.

Use the same configuration to resume a training checkpoint. Set
`init_checkpoint_path` to start a new optimizer trajectory from saved model
weights. Keep model-builder code available when loading its checkpoints.

## Batching and schedules

The baseline uses `microbatch_size = min(batch_size, 64)` and
`num_micro_batches = batch_size // microbatch_size`. Batches above 64 must be
multiples of 64. For N selected sequences, G microbatches per update, and
microbatch size M, one epoch performs `ceil(N / M) // G` updates. Incomplete
accumulation groups are skipped; included microbatch mean losses have equal
weight. With G=1, the final partial batch is included. Record actual processed
token counts when comparing runs.

Accumulated runs skip compilation and can differ numerically from full-batch
execution. Model size also affects peak memory. The P4.1 scaffold has its own
configurable microbatch protocol.

With positive warmup, the first update uses LR zero; the scheduler reaches
zero again after the final update. W&B records the LR actually applied. Use
that value for weight-decay products. The batch-switch example specifies a
schedule indexed by tokens consumed before each update.

## Diagnostics and readout scaling

The baseline writes feature, alignment, and pre-clipping gradient diagnostics
when `wandb_online=True` (the default). Keep logging enabled when collecting
P4.2 measurements. Use these logs to interpret finite-width normalization and
clipping effects.
W&B receives per-matrix update-alignment exponents and current/movement
readout-alignment exponents and ratios at each probe checkpoint. The
`alignment_charts` section contains combined plots; `features.jsonl` and
`alignment.jsonl` are uploaded to the run's Files tab. Undefined ratios remain
null in JSON and are omitted from plots. P4.1 launchers also publish these
diagnostics by default; use `--no-wandb` to save only the local JSON.

For P4.2, implement readout scaling in your custom model builder and pass the
settings through `model_builder_kwargs`. Establish forward behavior in the
constructor; the loader reconstructs the same builder and arguments for final
models and training checkpoints. If calling `save_model` directly, supply
`model_builder` and `model_builder_kwargs` in its metadata. The trainer saves
these automatically. The P4.1 scaffold exposes its own `output_multiplier`.
