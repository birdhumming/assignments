"""Pull a3-tagged runs (config, summary, trimmed history) into $A3_DATA_DIR/runs/<id>.json.

    uv run python -m experiments.a3_optimization.analysis.pull_wandb [tag ...]
Tags default to all four a3 problems; per-tensor logging/rms columns are dropped."""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import wandb

TAGS = ("a3-mode-connectivity", "a3-river-valley", "a3-edge-of-stability", "a3-inside-the-hessian")
DATA_DIR = os.environ.get("A3_DATA_DIR", os.path.expanduser("~/a3/data"))
os.makedirs(os.path.join(DATA_DIR, "runs"), exist_ok=True)
api = wandb.Api(timeout=300)
tags = sys.argv[1:] or TAGS
runs = [r for tag in tags
        for r in api.runs("aleyang-stanford-university/assignments", filters={"tags": tag}, per_page=500)
        if "a3-corrupted" not in r.tags and r.state == "finished"]
print(len(runs), "runs", flush=True)


def _plain(o):
    if hasattr(o, "items"):
        return dict(o.items())
    if hasattr(o, "__iter__"):
        return list(o)
    return str(o)


def _keep(key):
    return not key.startswith("logging/rms/")


def pull(r):
    path = os.path.join(DATA_DIR, "runs", f"{r.id}.json")
    if os.path.exists(path):
        return r.name, "cached"
    t = time.time()
    rows = [{k: v for k, v in row.items() if _keep(k)} for row in r.scan_history(page_size=20000)]
    record = {"name": r.name, "id": r.id, "state": r.state, "tags": r.tags, "config": dict(r.config),
              "summary": dict(r.summary.items()), "history": rows, "created_at": r.created_at}
    with open(path, "w") as fh:
        json.dump(record, fh, default=_plain)
    return r.name, f"{len(rows)} rows {time.time() - t:.0f}s"


with ThreadPoolExecutor(4) as ex:
    for name, msg in ex.map(pull, runs):
        print(name, msg, flush=True)
print("DONE")
