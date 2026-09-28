"""Problem 6: activation and gradient statistics.

Parts (a)/(b) use the per-module RMS statistics logged by default. Part (c)
adds a residual-stream RMS probe (forward hooks on layers 1/3/6, eight fixed
validation sequences, evaluated after every eval) to a healthy d8 and to
recipes chosen to push the statistics around.
"""

import torch

from experiments.a1_basics.common import dedupe, run
from metric_logging import AFTER_BACKWARD, AFTER_EVAL, MetricLogger
from modal_train import launch_training_jobs
from module_rms_logging import log_module_rms
from train import TrainConfig


PROBE_LAYERS = (0, 2, 5)


def log_activation_rms(ctx):
    input_ids = next(iter(ctx.val_batches["val"]))[:8]
    input_ids = input_ids.to(next(ctx.model.parameters()).device)
    stats, handles = {}, []
    for layer_idx in PROBE_LAYERS:

        def hook(module, inputs, output, layer_idx=layer_idx):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden.detach().float()
            stats[f"layer_{layer_idx + 1}_residual_rms"] = hidden.square().mean().sqrt().item()
            stats[f"layer_{layer_idx + 1}_residual_absmax"] = hidden.abs().max().item()

        handles.append(ctx.model.model.layers[layer_idx].register_forward_hook(hook))
    was_training = ctx.model.training
    ctx.model.eval()
    try:
        with torch.no_grad():
            ctx.model(input_ids=input_ids)
    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            ctx.model.train()
    return stats


PROBED_LOGGERS = (
    MetricLogger(event=AFTER_BACKWARD, fn=log_module_rms),
    MetricLogger(event=AFTER_EVAL, fn=log_activation_rms),
)


def probed(*tags, **changes):
    return run(*tags, metric_loggers=PROBED_LOGGERS, run_name_suffix="actprobe", **changes)


RUNS = dedupe(
    [
        probed("a1-p6", "a1-p6c-healthy"),
        # Unhealthy: high LR, no QK-norm, no gradient clipping.
        probed("a1-p6", "a1-p6c-unhealthy", learning_rate=0.03, qk_norm=False, grad_norm=None),
        # Shrink parameters / activations via strong decay; grow them by disabling it.
        probed("a1-p6", "a1-p6c-wd", weight_decay=1.0),
    ]
)


def main():
    launch_training_jobs(RUNS, max_parallel_runs=2)


if __name__ == "__main__":
    main()
