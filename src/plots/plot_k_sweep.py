"""Detection coverage vs flip multiplicity (fault-distribution axis).

Reads a random-baseline JSONL produced over the k-sweep manifest
(results/exp_random_k_sweep.jsonl, faults with k = 2..16 bit flips plus
the k=1 comparison set if present) and plots, per scope:

- single-prompt detection rate: mean fraction of the probe set that
  detects a fault (error bars: std over faults)
- set coverage: fraction of faults detected by at least one of the probes

This is the proposal's fault-distribution view: as faults accumulate more
bit flips they get easier to detect; the k=1 stratum is the hard core the
ATPG methods must crack.

Usage: python src/plots/plot_k_sweep.py [jsonl ...] [--out png]
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COLORS = {"mantissa": "tab:blue", "exponent": "tab:orange"}


def load_groups(paths):
    """-> {scope: {axis_value: [record, ...]}} where the axis value is the
    flip rate rho when the manifest recorded it, else the fixed flip count
    k. If a scope mixes both (k-stratum and rho-stratum files passed
    together), the rho strata win and the k strata are dropped."""
    groups = defaultdict(lambda: defaultdict(list))
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            f = r["fault"]
            axis = f["rho"] if f.get("rho") is not None else f["k"]
            groups[r["scope"]][axis].append(r)
    for scope, by_axis in groups.items():
        if any(isinstance(x, float) for x in by_axis):
            dropped = [x for x in by_axis if isinstance(x, int)]
            if dropped:
                print(f"note: {scope}: dropping k-strata {sorted(dropped)} "
                      "(rho strata present)")
            for x in dropped:
                del by_axis[x]
    return groups


def coverage_curves(groups):
    """-> {scope: (ks, rates_mean, rates_std, cov)}"""
    out = {}
    for scope, by_k in groups.items():
        ks = sorted(by_k)
        n_probes = len(next(iter(by_k.values()))[0]["detected"])
        rates, stds, covs = [], [], []
        for k in ks:
            recs = by_k[k]
            rates.append(sum(r["n_detected"] for r in recs)
                         / (len(recs) * n_probes))
            stds.append((sum((r["n_detected"] / n_probes - rates[-1]) ** 2
                             for r in recs) / len(recs)) ** 0.5)
            covs.append(sum(any(r["detected"]) for r in recs) / len(recs))
        out[scope] = (ks, rates, stds, covs, n_probes)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl", nargs="+",
                    default=["results/exp_random.jsonl",
                             "results/exp_random_k_sweep.jsonl"])
    ap.add_argument("--out", default="results/random_coverage_vs_k.png")
    args = ap.parse_args()
    paths = [p for p in args.jsonl if os.path.exists(p)]
    if not paths:
        raise SystemExit("no input files found")

    groups = load_groups(paths)
    curves = coverage_curves(groups)
    is_rho = any(isinstance(x, float) for s in curves.values() for x in s[0])

    fig, axes = plt.subplots(1, len(curves), figsize=(6.5 * len(curves), 4.8),
                             sharey=True)
    if len(curves) == 1:
        axes = [axes]
    for ax, (scope, (ks, rates, stds, covs, n_probes)) in \
            zip(axes, curves.items()):
        c = COLORS.get(scope, None)
        ax.errorbar(ks, rates, yerr=stds, marker="o", capsize=3,
                    label=f"single-prompt rate (± std)", color=c)
        ax.plot(ks, covs, marker="s", linestyle="--",
                label=f"coverage: detected by ≥1 of {n_probes} probes",
                color=c)
        if is_rho:
            ax.set_xscale("log")
            ax.set_xticks(ks)
            ax.set_xticklabels([f"{k:g}" for k in ks])
            ax.set_xlabel("bit-flip rate ρ")
            ax.set_title(f"{scope} scope — random baseline (f_bitflip(ρ))")
        else:
            ax.set_xscale("log", base=2)
            ax.set_xticks(ks)
            ax.set_xticklabels(ks)
            ax.set_xlabel("bit flips per fault (k)")
            ax.set_title(f"{scope} scope — random baseline")
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("detection rate / coverage")
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle("Random-baseline detection vs "
                 + ("fault rate ρ" if is_rho else "fault multiplicity"))
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")
    for scope, (ks, rates, stds, covs, _) in curves.items():
        for k, rate, std, cov in zip(ks, rates, stds, covs):
            label = f"ρ={k:g}" if is_rho else f"k={k}"
            print(f"  {scope:8s} {label:>9s}: rate {rate:5.1%} (±{std:.1%})"
                  f"  coverage {cov:5.1%}")


if __name__ == "__main__":
    main()
