"""Pull our a5-* runs (summary + sampled history) into results/a5_capacity/runs.json.

    uv run python -m experiments.a5_capacity.analysis.pull_wandb
"""
import json
from pathlib import Path
import wandb

PROJECT = 'aleyang-stanford-university/assignments'
OUT = Path('results/a5_capacity/runs.json')


def main():
    api = wandb.Api(timeout=120)
    out = {}
    for r in api.runs(PROJECT, filters={'display_name': {'$regex': '^a5-'}}):
        s = dict(r.summary)
        # keys= needs every key in the same row, and val_loss rows carry no train_loss, so pull the two series separately
        hist = r.history(samples=400, keys=['optimizer_step', 'val_loss'], pandas=False)
        hist += r.history(samples=400, keys=['optimizer_step', 'train_loss'], pandas=False)
        out[r.name] = dict(id=r.id, state=r.state, url=r.url, created=str(r.created_at),
                           val_loss=s.get('val_loss'), train_loss=s.get('train_loss'), step=s.get('optimizer_step', s.get('step')),
                           runtime=s.get('_runtime'), switch_before=s.get('switch/before_val_loss'),
                           switch_after=s.get('switch/after_val_loss'), history=hist,
                           lr=r.config.get('learning_rate'), kwargs=r.config.get('model_builder_kwargs'))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    for name, v in sorted(out.items()):
        print(f"{name:55s} {v['state']:9s} {v['val_loss']!s:>8s} step {v['step']}")


if __name__ == '__main__':
    main()
