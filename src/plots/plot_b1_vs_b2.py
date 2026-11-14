"""Paired B1 (random probe baseline) vs B2 (HotFlip) comparison.

Reads the two baseline JSONLs (results/exp_random.jsonl and
results/exp_hotflip.jsonl), pairs faults by (scope, seed) on the shared
subset, and plots three panels:

- detection coverage per scope with Wilson 95% confidence intervals
- paired outcome decomposition (both / B1-only / B2-only / neither):
  the key structural result is whether B2 ever detects a fault that B1
  misses (B2-only > 0 would mean optimization extends coverage)
- compute per fault in row-equivalents (log scale), B1's flat 100-probe
  spend against B2's median / mean / max actual spend

Usage: python src/plots/plot_b1_vs_b2.py results/exp_random.jsonl \
           results/exp_hotflip.jsonl [--out results/b1_vs_b2.png]
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"mantissa": "tab:blue", "exponent": "tab:orange"}
METHOD_COLORS = {"B1 (random, 100 probes)": "tab:green",
                 "B2 (HotFlip, 8 starts)": "tab:red"}


def wilson_bounds(p, n, z=1.96):
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def load(paths):
    """-> {(scope, seed): record} for each input file."""
    maps = []
    for path in paths:
        m = {}
        for line in open(path):
            r = json.loads(line)
            m[(r["scope"], r["seed"])] = r
        maps.append(m)
    return maps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs=2,
                    help="exp_random.jsonl then exp_hotflip.jsonl")
    ap.add_argument("--out", default="results/b1_vs_b2.png")
    args = ap.parse_args()

    r1map, r2map = load(args.jsonl)
    shared = sorted(set(r1map) & set(r2map))
    scopes = sorted({s for s, _ in shared})
    print(f"paired faults: {len(shared)} over scopes {scopes}")

    per = {s: {"n": 0, "b1": 0, "b2": 0, "both": 0, "b1only": 0,
               "b2only": 0, "neither": 0} for s in scopes}
    for scope, seed in shared:
        a, b = r1map[(scope, seed)], r2map[(scope, seed)]
        d1, d2 = a["n_detected"] > 0, b["detected"]
        p = per[scope]
        p["n"] += 1
        p["b1"] += d1
        p["b2"] += d2
        if d1 and d2:
            p["both"] += 1
        elif d1:
            p["b1only"] += 1
        elif d2:
            p["b2only"] += 1
        else:
            p["neither"] += 1
    for scope, p in per.items():
        print(f"{scope:8s} n={p['n']} B1={p['b1'] / p['n']:.1%} "
              f"B2={p['b2'] / p['n']:.1%} both={p['both']} "
              f"B1-only={p['b1only']} B2-only={p['b2only']} "
              f"neither={p['neither']}")

    # compute: B1 flat, B2 distribution. NOTE: exp_random.jsonl (the first
    # B1 campaign) records the counter cumulatively; the verified per-fault
    # B1 spend is the constant 6,400 row-equiv/fault (762 decoded faults x
    # 6,400 = 4,876,800 = the file's final cumulative value). Later B1 runs
    # record per-fault deltas directly.
    b1_rows = 6400
    b2_rows = sorted(r2map[(s, sd)]["compute"]["row_equiv"]
                     for s, sd in shared)
    med = b2_rows[len(b2_rows) // 2]
    mean = sum(b2_rows) / len(b2_rows)
    print(f"B1 spend/fault: {b1_rows} row-equiv (flat); B2: median={med} "
          f"mean={mean:.0f} max={b2_rows[-1]}")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    x = list(range(len(scopes)))
    w = 0.36

    # Panel A: coverage with Wilson CIs
    ax = axes[0]
    for i, (meth, key) in enumerate([("B1 (random, 100 probes)", "b1"),
                                     ("B2 (HotFlip, 8 starts)", "b2")]):
        vals, lo, hi = [], [], []
        for s in scopes:
            p = per[s]
            v = p[key] / p["n"]
            l, h = wilson_bounds(v, p["n"])
            vals.append(v)
            lo.append(v - l)
            hi.append(h - v)
        off = (i - 0.5) * w
        bars = ax.bar([xi + off for xi in x], vals, w * 0.9,
                      yerr=[lo, hi], capsize=4,
                      color=METHOD_COLORS[meth], label=meth)
        for xi, v, p in zip(x, vals, [per[s] for s in scopes]):
            ax.text(xi + off, v + 0.03, f"{v:.0%}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(scopes)
    ax.set_ylim(0, 1)
    ax.set_ylabel("detection coverage")
    ax.set_title(f"coverage on shared subset (n={per[scopes[0]]['n']}/scope)\n"
                 "error bars: Wilson 95% CI")
    ax.legend(loc="lower right", fontsize=8)

    # Panel B: paired outcomes
    ax = axes[1]
    cats = [("both", "tab:purple"), ("b1only", "tab:green"),
            ("b2only", "tab:red"), ("neither", "0.75")]
    bottoms = [0] * len(scopes)
    for key, color in cats:
        vals = [per[s][key] for s in scopes]
        ax.bar(x, vals, 0.6, bottom=bottoms, color=color, label=key)
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v:
                ax.text(xi, b + v / 2, str(v), ha="center", va="center",
                        color="white", fontsize=9)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_xticks(x)
    ax.set_xticklabels(scopes)
    ax.set_ylabel("faults")
    ax.set_title("paired outcome decomposition\n"
                 "(B2-only = 0 means optimization never extends coverage)")
    ax.legend(loc="upper right", fontsize=8)

    # Panel C: compute per fault, log scale
    ax = axes[2]
    stats = [("B1 (flat)", [b1_rows] * 3, "0.5"),
             ("B2 median", [med] * 3, METHOD_COLORS["B2 (HotFlip, 8 starts)"]),
             ("B2 mean", [mean] * 3, METHOD_COLORS["B2 (HotFlip, 8 starts)"]),
             ("B2 max", [b2_rows[-1]] * 3, METHOD_COLORS["B2 (HotFlip, 8 starts)"])]
    xs = list(range(len(stats)))
    for xi, (label, v, color) in zip(xs, stats):
        ax.bar(xi, v[0], 0.6, color=color, alpha=0.5 if "mean" in label else 1)
        ax.text(xi, v[0] * 1.1, f"{v[0]:,.0f}", ha="center", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(50, 40000)
    ax.set_xticks(xs)
    ax.set_xticklabels([s[0] for s in stats], fontsize=9)
    ax.set_ylabel("row-equivalents per fault")
    ax.set_title("compute per fault (log scale)\n"
                 "B1/B2 median ratio: {:.0f}x".format(b1_rows / med))

    fig.suptitle("B1 vs B2 — paired detection coverage and compute "
                 "(Qwen3-0.6B bf16, k=1)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
