"""P2 summary tables: uv run python -m experiments.a3_optimization.analysis.p2_tables"""
from experiments.a3_optimization.analysis.p2_load import EMAS, load_runs

runs = load_runs()
print(f"{'run':62} {'final':>7} " + " ".join(f"{'ema'+e:>9}" for e in EMAS) + "  best-ema  gain")
for r in runs:
    f = r["final"]
    emas = [f[f"ema{e}"] for e in EMAS]
    best = min(x for x in emas if x is not None)
    print(f"{r['name'].replace('-ema0.9-0.99-0.999-0.9999',''):62} {f['val']:7.4f} "
          + " ".join(f"{x:9.4f}" for x in emas) + f"  {best:8.4f} {f['val']-best:+6.4f}")
ref = min(runs, key=lambda r: r["final"]["val"])
print("\nreference (best final val):", ref["name"], round(ref["final"]["val"], 4))
# val loss at 25/50/75/100% of training for every run (horizon-free comparison)
n = len(ref["val"])
print(f"\n{'run':62} " + " ".join(f"{p:>7}" for p in ("25%", "50%", "75%", "100%")))
for r in runs:
    v = r["val"]
    print(f"{r['name'].replace('-ema0.9-0.99-0.999-0.9999',''):62} " + " ".join(f"{v[int(q*(n-1))]['val']:7.4f}" for q in (0.25, 0.5, 0.75, 1.0)))
