"""Per-tag run status: uv run python -m experiments.a3_optimization.analysis.status a3-river-valley"""
import sys
from collections import Counter

import wandb

tag = sys.argv[1] if len(sys.argv) > 1 else "a3-river-valley"
runs = list(wandb.Api().runs("aleyang-stanford-university/assignments", filters={"tags": tag}))
print(Counter(r.state for r in runs))
for r in sorted(runs, key=lambda r: r.name):
    s = r.summary
    print(f"{r.state:9} {r.name[:70]:70} step {s.get('step', s.get('_step'))!s:>6} "
          f"val {s['val_loss']:.4f}" if 'val_loss' in s else f"{r.state:9} {r.name[:70]}")
