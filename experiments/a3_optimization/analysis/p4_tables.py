"""P4(a) tables + figures from fetched a-adamw-<step> measurements.
usage: MODAL_ENVIRONMENT=cs312-aleyang uv run python -m experiments.a3_optimization.analysis.p4_tables [--no-fetch]"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.a3_optimization import hessian_jobs
from experiments.a3_optimization.p4_inside_the_hessian import STAGES

MEASURE_JOBS = STAGES["a"]

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)
FAMILIES = ("embedding", "attention", "qk_norm", "mlp", "norm", "head")

names = [j["name"] for j in MEASURE_JOBS]
if "--no-fetch" not in sys.argv:
    hessian_jobs.fetch(names)
res = {}
for n in names:
    try:
        res[n] = hessian_jobs.load_result(n)
    except FileNotFoundError:
        pass
rows = sorted(res.values(), key=lambda r: r["step"])
print(f"{len(rows)}/{len(names)} measurements")

print("\nstep | tokens | probe loss | grad norm | lambda_max | lambda_min | |min|/max | top-5 / top-1 | top eigvec family | grad overlap | min")
for r in rows:
    ff = r["top"]["family_fractions"][0]
    dom = max(ff, key=ff.get)
    t5 = r["top"]["values"]
    print(f"{r['step']:5d} | {r['tokens']/1e6:6.1f}M | {r['probe_loss']:.3f} | {r['gradient_norm']:.3f} | "
          f"{r['lambda_max']:9.2f} | {r['lambda_min']:9.2f} | {abs(r['lambda_min'])/r['lambda_max']:.3f} | "
          f"{t5[-1]/t5[0]:.2f} | {dom} {ff[dom]:.2f} | {r['top']['gradient_overlap'][0]:+.3f} | {r['seconds']/60:.1f}")

print("\nfamily block lambda_max by step")
print("step  | " + " | ".join(f"{f:>10}" for f in FAMILIES))
for r in rows:
    print(f"{r['step']:5d} | " + " | ".join(f"{r['families'][f]['lambda_max']:10.3f}" for f in FAMILIES))

print("\nhutchinson (if measured): step | tr(H)/d | tr(H^2)/d | tr(H) | normalized sharpness lambda_max*d/tr(H)")
for r in rows:
    h = r.get("hutchinson")
    if h:
        d = r["size"]
        print(f"{r['step']:5d} | {h}")

print("\nfamily-pair energy ||H_cb||_F^2 (mean over probes), final available step, row-normalized by diagonal")
r = rows[-1]
if "energy" in r:
    blocks = r["energy"]["family"]["blocks"]
    E = np.mean(np.array(r["energy"]["family"]["per_probe"]), axis=0)
    print("      " + " ".join(f"{b[:6]:>8}" for b in blocks))
    for i, b in enumerate(blocks):
        print(f"{b[:6]:>6}" + " ".join(f"{E[i, j]/np.sqrt(E[i, i]*E[j, j]):8.3f}" for j in range(len(blocks))))
    off = sum(E[i, j] for i in range(len(blocks)) for j in range(len(blocks)) if i != j)
    print(f"off-diagonal share of total energy: {off/E.sum():.3f}")

# figures
if rows:
    steps = [r["step"] + 0.5 for r in rows]  # step 0 on log axis
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for f in FAMILIES:
        ax[0].plot(steps, [r["families"][f]["lambda_max"] for r in rows], marker="o", label=f)
    ax[0].plot(steps, [r["lambda_max"] for r in rows], "k--", label="full lambda_max")
    ax[0].set(xscale="log", yscale="log", xlabel="optimizer step (+0.5)", ylabel="top eigenvalue",
              title="family block curvature along AdamW training")
    ax[0].legend(fontsize=7)
    ax[1].plot(steps, [r["lambda_max"] for r in rows], "k-o", label="lambda_max")
    ax[1].plot(steps, [-r["lambda_min"] for r in rows], "r-o", label="-lambda_min")
    ax[1].plot(steps, [r["top"]["values"][4] for r in rows], "k:", label="5th eigenvalue")
    ax[1].set(xscale="log", yscale="log", xlabel="optimizer step (+0.5)", title="extreme eigenvalues")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "p4a_curvature.png"), dpi=130)
    # top eigenvector family composition
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bottom = np.zeros(len(rows))
    for f in FAMILIES:
        v = np.array([r["top"]["family_fractions"][0][f] for r in rows])
        ax.bar(range(len(rows)), v, bottom=bottom, label=f)
        bottom += v
    ax.set(xticks=range(len(rows)), xticklabels=[r["step"] for r in rows], xlabel="optimizer step",
           ylabel="share of top eigenvector norm", title="which family the top Hessian direction lives in")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "p4a_top_eigvec.png"), dpi=130)
    print("figures written")
