"""Aggregate pattern-budget experiments (results/pattern_budget/*.jsonl).

Reports, per (method, n): number of faults, repeats R, coverage pooled
over repeats with a Wilson 95 percent interval, the mean per-fault
detection rate across repeats (pattern-drawing variance), and compute
(row-equivalents and estimated FLOPs, median and max per fault-repeat).

Coverage here treats every (fault, repeat) pair as one Bernoulli trial;
repeats of the same fault are correlated, so the Wilson interval is a
lower bound on the true uncertainty. The per-repeat coverage values are
also printed so the spread across pattern draws is visible.
"""

import argparse
import glob
import json
import math
import os
from collections import defaultdict


def wilson(p, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def load(path):
    recs = []
    with open(path) as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    return recs


def detected(rec):
    if "detected" in rec and isinstance(rec["detected"], bool):
        return rec["detected"]          # hotflip record
    return any(rec["detected"])          # random-budget record (vector)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="results/pattern_budget")
    ap.add_argument("--out", default=None,
                    help="optional output path for a JSON summary")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.jsonl")))
    if not files:
        raise SystemExit(f"no jsonl files under {args.dir}")
    summary = {}
    for path in files:
        recs = load(path)
        by = defaultdict(list)
        for r in recs:
            n = r.get("n_patterns", r.get("hf_starts"))
            by[(r["method"], n)].append(r)
        print(f"== {path}")
        for (method, n), recs_mn in sorted(by.items()):
            scopes = sorted({r["scope"] for r in recs_mn})
            det = [detected(r) for r in recs_mn]
            n_pairs = len(det)
            hits = sum(det)
            p = hits / n_pairs
            lo, hi = wilson(p, n_pairs)
            faults = len({(r["scope"], r["seed"]) for r in recs_mn})
            R = len({r.get("repeat", 0) for r in recs_mn})
            per_rep = defaultdict(list)
            for r, d in zip(recs_mn, det):
                per_rep[r.get("repeat", 0)].append(d)
            rep_covs = [sum(v) / len(v) for _, v in sorted(per_rep.items())]
            per_fault = defaultdict(list)
            for r, d in zip(recs_mn, det):
                per_fault[(r["scope"], r["seed"])].append(d)
            rates = [sum(v) / len(v) for v in per_fault.values()]
            mean_rate = sum(rates) / len(rates)
            rows = [r["compute"]["row_equiv"] for r in recs_mn]
            flops = [r["compute"]["flops"] for r in recs_mn]
            rows.sort(); flops.sort()
            med_rows = rows[len(rows) // 2]
            med_flops = flops[len(flops) // 2]
            print(f"  {method:14s} n={n:3d}: faults={faults} R={R} "
                  f"scopes={','.join(scopes)}")
            print(f"    coverage pooled: {p:.1%} "
                  f"(Wilson 95%: {lo:.1%} to {hi:.1%}); "
                  f"per-repeat: {' | '.join(f'{c:.1%}' for c in rep_covs)}")
            print(f"    mean per-fault detection rate: {mean_rate:.1%} "
                  f"(min {min(rates):.0%}, max {max(rates):.0%})")
            print(f"    compute/fault-repeat: median {med_rows} row-equiv "
                  f"(~{med_flops / 1e12:.2f} TFLOP est.), "
                  f"max {rows[-1]} row-equiv")
            summary[f"{method}_n{n}"] = {
                "faults": faults, "repeats": R, "scopes": scopes,
                "coverage_pooled": p, "wilson_lo": lo, "wilson_hi": hi,
                "coverage_per_repeat": rep_covs,
                "mean_per_fault_rate": mean_rate,
                "median_row_equiv": med_rows,
                "median_flops_est": med_flops,
            }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"summary written to {args.out}")


if __name__ == "__main__":
    main()
