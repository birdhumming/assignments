"""Shared helpers for the A2 experiment modules: tagging, cost estimates, manifests."""
import csv
from dataclasses import replace
from pathlib import Path

from experiments.a2.modal_launcher import config as a2_config
from train import checked_train_config, training_run_name

BASE_TOKENS = 614_400_000
# Measured on this account's A1 runs (H100): d8 at 614.4M tokens, batch 64, ~12 min
# without the A2 diagnostics; the handout quotes ~15 min with them.
D8_BASE_MINUTES = 15.0
# Small batches do more optimizer steps per token (A1 batch-16 took 1.57x batch-64).
BATCH_FACTOR = {8: 2.4, 16: 1.6, 32: 1.2, 64: 1.0, 128: 0.95, 256: 0.95}


def run(*tags, tokens=BASE_TOKENS, batch=64, suffix, **overrides):
    tags = tuple(dict.fromkeys(('a2', *tags)))
    return a2_config(tokens=tokens, batch=batch, run_name_suffix=suffix, wandb_tags=tags, **overrides)


def width_factor(config):
    hidden = config.model_config.hidden_size if config.model_config is not None else 512
    layers = config.model_config.num_hidden_layers if config.model_config is not None else 8
    return (hidden / 512) ** 2 * (layers / 8)


def estimated_minutes(config):
    tokens = config.num_train_sequences * 1024
    return D8_BASE_MINUTES * tokens / BASE_TOKENS * BATCH_FACTOR[config.batch_size] * width_factor(config)


def dedupe(configs):
    seen, unique = set(), []
    for c in configs:
        name = training_run_name(c)
        if name in seen:
            continue
        seen.add(name)
        unique.append(c)
    return unique


def manifest_rows(configs):
    rows = []
    for c in configs:
        checked_train_config(c)
        rows.append(dict(run_name=training_run_name(c), tags=','.join(c.wandb_tags),
                         tokens=c.num_train_sequences * 1024, batch=c.batch_size,
                         lr=c.learning_rate, wd=c.weight_decay, beta1=c.beta1, optimizer=c.optimizer_name,
                         est_minutes=f'{estimated_minutes(c):.1f}'))
    return rows


def write_manifest(path, rows):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def preview(configs, manifest_path):
    rows = manifest_rows(configs)
    write_manifest(manifest_path, rows)
    total = sum(float(r['est_minutes']) for r in rows)
    for r in rows:
        print(f"  {r['est_minutes']:>6}m  {r['run_name']}")
    print(f'{len(rows)} runs, ~{total / 60:.1f} GPU-hours estimated -> {manifest_path}')
    return rows
