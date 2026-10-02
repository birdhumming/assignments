"""P4.2 fixed-input probes via the shared metric-logger interface."""
import json
from pathlib import Path

import torch

from experiments.a2.probe_math import training_steps
from experiments.a2.readout import readout_alignment
from experiments.a2.wandb_diagnostics import configure, log_plots, readout_metrics
from train import causal_lm_loss, training_run_name
from utils import autocast_context


class FeatureLogger:
    def setup(self, ctx):
        import wandb
        configure(wandb)
        self.model = ctx.model
        self.batch = torch.cat([ctx.val_batches['val'][i] for i in range(min(8, len(ctx.val_batches['val'])))])[:8].to(next(ctx.model.parameters()).device)
        self.directory = Path(ctx.config.model_dir) / training_run_name(ctx.config)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'features.jsonl'
        self.initial_path = self.directory / 'initial_features.pt'
        self.total = training_steps(ctx.config)
        self.steps = set(range(6)) | {self.total} | {round(self.total*f) for f in (.01,.1,.5)}
        self.initial = {}
        if ctx.step:
            self.initial = torch.load(self.initial_path, map_location='cpu', weights_only=True)
        if ctx.step == 0:
            self.initial['readout_weight'] = self.model.lm_head.weight.detach().float().cpu().clone()
        elif 'readout_weight' not in self.initial:
            raise ValueError('Readout probes require initial weights saved at step zero; start a new run.')
        stats = self.sample(ctx, ctx.step)
        if stats:
            wandb.log({'logging/' + key: value for key, value in stats.items()})
        if ctx.step == 0:
            torch.save(self.initial, self.initial_path)

    @torch.no_grad()
    def sample(self, ctx, step):
        if step not in self.steps:
            return {}
        row = {'step': step, 'features': {}, 'residual_rms': {}, 'branch_rms': {}}
        handles = []
        def feature(name):
            def hook(module, inputs, output):
                h = output.detach().float().cpu()
                if step == 0:
                    self.initial[name] = h.clone()
                if name == 'model.norm':
                    row['readout_alignment'] = readout_alignment(
                        self.model.lm_head.weight, output,
                        self.initial['readout_weight'], self.initial[name])
                row['features'][name] = {'rms': h.square().mean().sqrt().item(),
                    'movement': (h-self.initial[name]).square().mean().sqrt().item()}
                row['residual_rms'][name] = inputs[0].float().square().mean().sqrt().item()
            return hook
        def branch(name):
            def hook(module, inputs, output):
                row['branch_rms'][name] = output.detach().float().square().mean().sqrt().item()
            return hook
        for name, module in self.model.named_modules():
            if name.endswith(('input_layernorm', 'post_attention_layernorm')) or name == 'model.norm':
                handles.append(module.register_forward_hook(feature(name)))
            elif name.endswith(('o_proj', 'down_proj')):
                handles.append(module.register_forward_hook(branch(name), prepend=True))
        was_training = self.model.training
        devices = [self.batch.device.index or 0] if self.batch.is_cuda else []
        try:
            self.model.eval()
            with torch.random.fork_rng(devices=devices), autocast_context(ctx.config.precision, device=self.batch.device):
                logits = self.model(input_ids=self.batch)
                row['loss'] = causal_lm_loss(logits, self.batch).item()
                row['logit_rms'] = logits.float().square().mean().sqrt().item()
            with self.path.open('a') as f:
                f.write(json.dumps(row, allow_nan=False)+'\n')
        finally:
            self.model.train(was_training)
            for handle in handles:
                handle.remove()
        return {'probe/logit_rms': row['logit_rms'],
                'probe/final_movement': row['features']['model.norm']['movement'],
                **readout_metrics(row)}

    def close(self):
        if self.path.exists():
            import wandb
            rows = [json.loads(s) for s in self.path.read_text().splitlines()]
            log_plots(wandb, features=rows)
            wandb.save(str(self.path), base_path=str(self.path.parent), policy='now')

    def __call__(self, ctx):
        return self.sample(ctx, ctx.step+1)


features = FeatureLogger()


class GradientLogger:
    """Sample pre-clipping norms without changing gradients or optimizer state."""
    def setup(self, ctx):
        self.path = Path(ctx.config.model_dir) / training_run_name(ctx.config) / 'gradients.jsonl'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        total = training_steps(ctx.config)
        self.steps = set(range(1, 6)) | {total} | {round(total * f) for f in (.01, .1, .5)}

    def __call__(self, ctx):
        if ctx.step + 1 not in self.steps:
            return {}
        squares = {}
        for name, p in ctx.model.named_parameters():
            if p.grad is not None:
                squares[name] = p.grad.detach().double().square().sum().item()
        total = sum(squares.values())
        norm = total ** .5
        embedding = sum(v for name, v in squares.items() if 'embed_tokens' in name)
        coefficient = 1. if ctx.config.grad_norm is None else min(1., ctx.config.grad_norm / (norm + 1e-6))
        row = {'step': ctx.step + 1, 'pre_clip_norm': norm,
                   'embedding_fraction_squared_norm': embedding / total if total else 0.,
                   'clip_coefficient': coefficient}
        with self.path.open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        return {'gradient/pre_clip_norm': norm, 'gradient/clip_coefficient': coefficient}


gradients = GradientLogger()
