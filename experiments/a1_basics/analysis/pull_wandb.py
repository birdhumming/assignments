"""Pull every finished `a1` / `a1-basics-p1b-v3` run (full history) into $A1_DATA_DIR/runs/<id>.json."""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import wandb

DATA_DIR = os.environ.get("A1_DATA_DIR", os.path.expanduser("~/a1"))
os.makedirs(os.path.join(DATA_DIR, "runs"), exist_ok=True)
api = wandb.Api(api_key=os.environ["WANDB_API_KEY"], timeout=300)
runs = [
    r
    for r in api.runs("aleyang-stanford-university/assignments", filters={"tags": {"$in": ["a1", "a1-basics-p1b-v3"]}})
    if "a1-corrupted" not in r.tags and r.state == "finished"
]
print(len(runs), "runs", flush=True)


def pull(r):
    path = os.path.join(DATA_DIR, "runs", f"{r.id}.json")
    if os.path.exists(path):
        return r.name, "cached"
    t = time.time()
    rows = list(r.scan_history(page_size=20000))
    record = {"name": r.name, "id": r.id, "state": r.state, "tags": r.tags, "config": dict(r.config),
              "summary": dict(r.summary.items()), "history": rows, "created_at": r.created_at}
    with open(path, "w") as fh:
        json.dump(record, fh)
    return r.name, f"{len(rows)} rows {time.time() - t:.0f}s"


with ThreadPoolExecutor(8) as ex:
    for name, msg in ex.map(pull, runs):
        print(name, msg, flush=True)
print("DONE")
