"""Pull our A4 P3/P4 runs from W&B and emit LaTeX loss tables (latex/p3_losses.tex, latex/p4_losses.tex)."""
import json
from pathlib import Path

import wandb

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "aleyang-stanford-university/assignments"
UNIFORM = 8.0  # ln(4096) ~ 8.32; anything above 6.5 is "learned nothing"
BASE = {"d8": 2.9265, "rnn": 3.5160}  # supplied additive baselines (manifest, best LR)
MANIFEST = Path("experiments/a4_stability/results/manifest.json")


def supplied_losses():
    """{family: {lr: validation_loss}} for the supplied additive baselines (d8 transformer, modern RNN)."""
    rows = json.loads(MANIFEST.read_text())
    rows = rows if isinstance(rows, list) else list(rows.get("runs", rows.values()))
    out = {"d8": {}, "rnn": {}}
    for r in rows:
        if r.get("variant") in out and r.get("status") == "complete":
            out[r["variant"]][float(r["lr"])] = float(r["validation_loss"])
    return out


SUPPLIED = supplied_losses()
P3_ROWS = [
    ("additive (supplied)", None), ("half-and-half", "half"), ("depth EMA $\\beta=0.1$", "ema01"),
    ("depth EMA $\\beta=0.9$", "ema09"), ("uniform routing", "uniform"), ("full attention", "attention"),
    ("blocked $B=2$", "block2"), ("blocked $B=4$", "block4"), ("hybrid (ours)", "hybrid"),
]
P4_ROWS = [
    ("modern tanh RNN (supplied)", None), ("GRU", "gru"), ("GRU, no reset gate", "gru-no-reset"),
    ("GRU, reset gate only", "gru-reset-only"), ("GRU, input-only update gate", "gru-input-gate"),
    ("NormGate (ours)", "normgate"), ("NormGate, zero update bias", "normgate-b0"),
]


def pull():
    api = wandb.Api(timeout=120)
    out = {}
    for r in api.runs(PROJECT, filters={"display_name": {"$regex": "^a4-"}}):
        name, lr = r.name[3:].rsplit("-lr", 1)
        out[(name, float(lr))] = (r.state, r.summary.get("val_loss"), r.summary.get("step"))
    return out


def cell(runs, name, lr):
    rec = runs.get((name, lr))
    if rec is None:
        return "--"
    state, loss, _step = rec
    if state != "finished" or loss is None:
        return f"\\textit{{{loss:.3f}}}$^{{*}}$" if loss else "--"
    return "div." if loss > UNIFORM - 1.5 else f"{loss:.4f}"


def table(runs, rows, family, rates, caption, label):
    lines = ["\\begin{table}[H]\\centering\\small", f"\\caption{{{caption}}}\\label{{{label}}}",
             "\\begin{tabular}{l" + "r" * len(rates) + "r}\\toprule",
             "variant & " + " & ".join(f"lr {lr:g}" for lr in rates) + " & best $-$ additive \\\\\\midrule"]
    for label_, key in rows:
        if key is None:
            cells = [f"{SUPPLIED[family][r]:.4f}" if r in SUPPLIED[family] else "--" for r in rates]
            tail = "--" if any(r in SUPPLIED[family] for r in rates) else f"{BASE[family]:.4f}"
            lines.append(f"{label_} & " + " & ".join(cells) + f" & {tail} \\\\")
            continue
        cells = [cell(runs, f"{family}-{key}" if family == "d8" or not key.startswith("gru") and not key.startswith("normgate") else key, lr) for lr in rates]
        finite = [float(c) for c in cells if c.replace(".", "").isdigit()]
        delta = f"{min(finite) - BASE[family]:+.4f}" if finite else "--"
        lines.append(f"{label_} & " + " & ".join(cells) + f" & {delta} \\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{table}"]
    return "\n".join(lines)


if __name__ == "__main__":
    runs = pull()
    (ROOT / "results" / "own_runs.json").write_text(json.dumps({f"{k[0]}@{k[1]}": v for k, v in runs.items()}, indent=1))
    p3 = table(runs, P3_ROWS, "d8", (.003, .006), "P3, d8 transformer: final validation loss. ``div.'' = stuck at the uniform-prediction loss ($\\approx 6.87$); italics$^*$ = still training when the table was built.", "tab:p3d8")
    p3 += "\n" + table(runs, P3_ROWS, "rnn", (.001, .002), "P3, modern 4-layer RNN: final validation loss.", "tab:p3rnn")
    (ROOT / "latex" / "p3_losses.tex").write_text(p3)
    (ROOT / "latex" / "p4_losses.tex").write_text(table(runs, P4_ROWS, "rnn", (.0015, .003, .006), "P4, recurrent cells (4 layers, same depth residuals, pre-norm and MLPs as the modern RNN): final validation loss.", "tab:p4"))
    for k in sorted(runs):
        print(k, runs[k])
