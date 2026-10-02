"""P4.2 actual-update alignment through the shared metric logger interface."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments.a2.probe_math import measure_delta, training_steps
from experiments.a2.wandb_diagnostics import configure, log_plots, update_metrics
from train import training_run_name
from utils import autocast_context


class AlignmentLogger:
    def setup(self, ctx):
        import wandb
        configure(wandb)
        self.model = ctx.model
        self.batch = torch.cat([ctx.val_batches['val'][i]
            for i in range(min(8, len(ctx.val_batches['val'])))])[:8].to(next(ctx.model.parameters()).device)
        total = training_steps(ctx.config)
        self.steps = set(range(1, min(total, 5) + 1)) | {total} | {
            int(t) for t in np.rint(np.geomspace(6, max(6, total), 24)) if t <= total}
        self.maps = {n + '.weight': m for n, m in self.model.named_modules()
                     if isinstance(m, torch.nn.Linear)}
        self.path = Path(ctx.config.model_dir) / training_run_name(ctx.config) / 'alignment.jsonl'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.pending = None

    @torch.no_grad()
    def before(self, ctx):
        self.pending = None
        if ctx.step + 1 not in self.steps:
            return {}
        inputs, handles = {}, []
        for name, module in self.maps.items():
            def capture(_module, args, name=name):
                inputs[name] = args[0].detach().float().clone()
            handles.append(module.register_forward_pre_hook(capture))
        devices = [self.batch.device.index or 0] if self.batch.is_cuda else []
        try:
            with torch.random.fork_rng(devices=devices), autocast_context(ctx.config.precision, device=self.batch.device):
                self.model(input_ids=self.batch)
        finally:
            for h in handles:
                h.remove()
        weights = {n: m.weight.detach().clone() for n, m in self.maps.items()}
        if any(w.dtype != torch.float32 for w in weights.values()):
            raise ValueError('Alignment logger expects FP32 optimizer parameters')
        self.pending = (ctx.step, inputs, weights)
        return {}

    @torch.no_grad()
    def after(self, ctx):
        if self.pending is None:
            return {}
        step, inputs, weights = self.pending
        if step != ctx.step:
            raise RuntimeError('Alignment measurements must surround the same update')
        rows = []
        with self.path.open('a') as f:
            for name, module in self.maps.items():
                row = dict(step=step + 1, input_step=step, parameter=name,
                           **measure_delta(weights[name], module.weight, inputs[name]))
                f.write(json.dumps(row, allow_nan=False) + '\n')
                rows.append(row)
        self.pending = None
        return update_metrics(rows, step + 1)

    def close(self):
        if self.path.exists():
            import wandb
            rows = [json.loads(s) for s in self.path.read_text().splitlines()]
            log_plots(wandb, alignments=rows)
            wandb.save(str(self.path), base_path=str(self.path.parent), policy='now')


_logger = AlignmentLogger()

def before_update(ctx):
    return _logger.before(ctx)


def after_update(ctx):
    return _logger.after(ctx)


before_update.setup = _logger.setup
before_update.close = _logger.close
