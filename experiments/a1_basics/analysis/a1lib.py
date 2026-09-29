"""Shared loaders for the A1 analysis: run lookup by config, curves, fits."""
import glob
import json
import math

import numpy as np

BASE = {"learning_rate": 0.003, "batch_size": 64, "warmup_percent": 0.01, "weight_decay": 0.1, "lr_schedule": "linear",
            "beta1": 0.9, "beta2": 0.95, "grad_norm": 1.0, "model_name": "d8", "num_train_sequences": 600000, "qk_norm": True,
            "tie_word_embeddings": False, "optimizer_name": "adamw", "num_epochs": 1.0, "dropout": 0.0, "model_seed": 42,
            "data_seed": 42, "deterministic": False, "perturb_one_token": False, "run_name_suffix": None}
BASE_SUFFIXES = {None, "a1-basics-p1b-v3"}

import os

DATA_DIR = os.environ.get("A1_DATA_DIR", os.path.expanduser("~/a1"))
COMPACT = os.path.join(DATA_DIR, "compact")
os.makedirs(COMPACT, exist_ok=True)

def _compact(raw):
    hist = raw["history"]
    cols = {"optimizer_step": [], "train_loss": [], "val_loss": [], "learning_rate": []}
    extra = {}
    for r in hist:
        st = r.get("optimizer_step")
        if st is None: continue
        for k, col in cols.items(): col.append(r.get(k))
        for k, v in r.items():
            if v is None: continue
            if k.startswith("logging/rms/") or "residual_" in k:
                extra.setdefault(k, ([], []))
                extra[k][0].append(st); extra[k][1].append(v)
    return {"name": raw["name"], "id": raw["id"], "state": raw["state"], "tags": raw["tags"], "config": raw["config"],
            "created_at": raw["created_at"], "summary": {k: v for k, v in raw["summary"].items() if not k.startswith("logging")},
            "cols": cols, "extra": extra}

RUNS = {}
for f in glob.glob(os.path.join(DATA_DIR, "runs", "*.json")):
    cpath = os.path.join(COMPACT, os.path.basename(f))
    if not os.path.exists(cpath) or os.path.getmtime(cpath) < os.path.getmtime(f):
        with open(f) as src, open(cpath, "w") as dst:
            json.dump(_compact(json.load(src)), dst)
    with open(cpath) as fh:
        d = json.load(fh)
    RUNS[d["id"]] = d

def cfg_matches(c, spec):
    for k, v in spec.items():
        cv = c.get(k)
        if k == "run_name_suffix":
            if v is None:
                if cv not in BASE_SUFFIXES: return False
            elif cv != v: return False
        elif isinstance(v, float) and cv is not None:
            if not math.isclose(float(cv), v, rel_tol=1e-6): return False
        elif cv != v: return False
    return True

def find_all(**spec):
    full = dict(BASE); full.update(spec)
    out = [d for d in RUNS.values() if cfg_matches(d["config"], full)]
    return out

def find(**spec):
    out = find_all(**spec)
    if not out: return None
    out.sort(key=lambda d: (0 if "a1" in d["tags"] else 1, d["created_at"]))
    return out[0]

def curve(d, key="train_loss"):
    st = d["cols"]["optimizer_step"]; col = d["cols"][key]
    pairs = [(s, y) for s, y in zip(st, col) if y is not None]
    return np.array([p[0] for p in pairs], float), np.array([p[1] for p in pairs], float)

def final_val(d):
    if d is None: return float("nan")
    _xs, ys = curve(d, "val_loss")
    return float(ys[-1]) if len(ys) else float(d["summary"].get("val_loss", float("nan")))

def series(d, key):
    xs, ys = d["extra"].get(key, ([], []))
    return np.array(xs, float), np.array(ys, float)

def fmt(x): return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"

def powerlaw_fit(sizes, losses):
    # L = a * N^-b  (log-log linear fit)
    x = np.log(np.array(sizes, float)); y = np.log(np.array(losses, float))
    b, loga = np.polyfit(x, y, 1)
    return math.exp(loga), -b

def smoothness(d, frac=(0.5, 1.0), win=51):
    _xs, ys = curve(d, "train_loss")
    n = len(ys); lo, hi = int(frac[0]*n), int(frac[1]*n)
    ys = ys[lo:hi]
    k = np.ones(win)/win
    sm = np.convolve(ys, k, mode="valid")
    resid = ys[win//2: win//2+len(sm)] - sm
    return float(resid.std())
