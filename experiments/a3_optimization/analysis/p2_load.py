"""Load pulled a3-river-valley runs into tidy per-run records (val rows aligned to optimizer steps)."""
import glob
import json
import os

DATA_DIR = os.environ.get("A3_DATA_DIR", os.path.expanduser("~/a3/data"))
EMAS = ("0.9", "0.99", "0.999", "0.9999")


def load_runs(tag="a3-river-valley"):
    out = []
    for path in glob.glob(os.path.join(DATA_DIR, "runs", "*.json")):
        d = json.load(open(path))
        if tag not in d["tags"]:
            continue
        step, lr, train, val = 0, None, [], []
        for row in d["history"]:
            if row.get("step") is not None:
                step, lr = row["step"], row.get("learning_rate")
                train.append((step, lr, row["train_loss"]))
            if row.get("val_loss") is not None:
                val.append({"step": step, "lr": lr, "val": row["val_loss"],
                            **{f"ema{e}": row.get(f"val_ema{e}_loss") for e in EMAS}})
        c = d["config"]
        out.append({"name": d["name"], "config": c, "schedule": c["lr_schedule"], "peak_lr": c["learning_rate"],
                    "dip": c.get("lr_dip"), "train": train, "val": val, "final": val[-1]})
    return sorted(out, key=lambda r: (r["schedule"], r["peak_lr"]))
