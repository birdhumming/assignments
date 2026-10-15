"""P4(a) trace table: Hutchinson tr(H)/d and tr(H^2)/d per checkpoint, with lambda_max
from the matching a-adamw-<step>.json, giving normalized sharpness lambda_max * d / tr(H)
and the participation ratio tr(H)^2 / (d * tr(H^2)) (fraction of directions carrying curvature)."""

import json
import statistics
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results" / "a3-inside-the-hessian"


def main():
    rows = []
    for path in sorted(RESULTS.glob("a-trace-*.json"), key=lambda p: int(p.stem.split("-")[-1])):
        t = json.loads(path.read_text())
        step = t["step"]
        look = RESULTS / f"a-adamw-{step}.json"
        lam = json.loads(look.read_text())["lambda_max"] if look.exists() else None
        d = t["size"]
        tr = t["hutchinson"]["trace_per_param"]
        sq = t["hutchinson"]["squared_per_param"]
        m_tr, m_sq = statistics.mean(tr), statistics.mean(sq)
        sem = statistics.stdev(tr) / len(tr) ** 0.5
        rows.append({
            "step": step, "tokens": t["tokens"], "tr_per_param": m_tr, "tr_sem": sem, "trace": m_tr * d,
            "sq_per_param": m_sq, "lambda_max": lam,
            "normalized": lam * d / (m_tr * d) if lam else None,
            "participation": (m_tr * d) ** 2 / (d * m_sq * d) if m_sq > 0 else None,
            "d": d,
        })
    print(f"{'step':>5} {'tokens':>7} {'tr(H)/d':>10} {'+-':>8} {'tr(H)':>10} {'tr(H^2)/d':>10} {'lmax':>8} {'lmax/(trH/d)':>12} {'tr^2/(d tr H^2)':>15}")
    for r in rows:
        print(f"{r['step']:>5} {r['tokens']/1e6:>6.1f}M {r['tr_per_param']:>10.3e} {r['tr_sem']:>8.1e} {r['trace']:>10.1f} "
              f"{r['sq_per_param']:>10.3e} {r['lambda_max'] or float('nan'):>8.2f} {r['normalized'] or float('nan'):>12.3g} {r['participation'] or float('nan'):>15.3g}")
    out = RESULTS.parent / "p4a_trace_table.json"
    out.write_text(json.dumps(rows, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
