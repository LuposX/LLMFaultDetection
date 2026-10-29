"""Plot probe fault-effect metrics vs. bit-flip fault magnitude from
trials.jsonl (the "sem" block written by run_degradation.py).

Four panels, one per metric, each with one line per prompt stratum plus
the overall aggregate: JS divergence (teacher-forced), token match,
embedding cosine, laya judge. x axis identical to plot_degradation.py:
expected flipped bits per fault draw, rho on each tick.

Trials without probe data (rho=0 baseline, zero-flip skips, --no-sem)
are skipped.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# one level below src/: add src/ to the import path for the lib imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import ROOT, flips_axis_setup, N_BITS  # noqa: E402

STRATA_COLORS = {"overall": "tab:blue", "wiki": "tab:green",
                 "qa": "tab:orange", "math": "tab:red"}
PANELS = [
    ("js", "Jensen-Shannon divergence (bound: log 2)", False),
    ("token_match", "token match (healthy vs faulty)", False),
    ("embed_cos", "embedding cosine (healthy vs faulty)", False),
    ("laya_same", "Laya judge P(same meaning)", False),
]


def plot_probes(trials, out):
    """One panel per probe metric, one line per stratum + overall."""
    # values[stratum][metric][rho] = [per-draw values]
    values = {s: {m: defaultdict(list) for m, _, _ in PANELS}
              for s in STRATA_COLORS}
    n_used = 0
    for t in trials:
        sem = t.get("sem")
        if not sem:
            continue
        n_used += 1
        for stratum in STRATA_COLORS:
            agg = sem.get(stratum)
            if not agg:
                continue
            for m, _, _ in PANELS:
                if agg.get(m) is not None:
                    values[stratum][m][t["rho"]].append(agg[m])
    if not n_used:
        print("no trials with probe data — nothing to plot")
        return
    print(f"plotting {n_used} trials with probe data")

    # rho=0 reference values per metric: the trivial ideal (0 or 1) except
    # laya_same, whose baseline is the judge's own P(same) on identical
    # pairs (measured at baseline time, recorded in the trial meta) — the
    # judge is imperfect, so its ceiling sits below 1
    laya_ceiling = next((t.get("laya_identical") for t in trials
                         if t.get("laya_identical") is not None), None)
    ref = {"js": 0.0, "token_match": 1.0, "embed_cos": 1.0,
           "laya_same": laya_ceiling}

    rhos = sorted({r for s in values for m in values[s]
                   for r in values[s][m] if values[s][m][r]})
    flips = [r * N_BITS for r in rhos]
    xlo, xhi = min(flips) / 3.0, max(flips) * 3.0

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), squeeze=False)
    for ax, (metric, ylabel, logy) in zip(axes.flat, PANELS):
        for stratum in ("wiki", "qa", "math", "overall"):
            means, pts = [], []
            for r in rhos:
                vals = values[stratum][metric].get(r, [])
                if not vals:
                    continue
                pts.extend((r * N_BITS, v) for v in vals)
                means.append((r * N_BITS, sum(vals) / len(vals)))
            if not means:
                continue
            style = dict(color=STRATA_COLORS[stratum],
                         lw=2.2 if stratum == "overall" else 1.2,
                         zorder=3 if stratum == "overall" else 2,
                         label=stratum)
            if stratum == "overall":
                ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                           s=14, alpha=0.35, color="tab:blue", zorder=1)
            ax.plot([p[0] for p in means], [p[1] for p in means],
                    "-o", markersize=4, **style)
        flips_axis_setup(ax, xlo, xhi, rhos)
        if logy:
            ax.set_yscale("log")
        if ref.get(metric) is not None:
            ax.axhline(ref[metric], ls="--", color="gray", zorder=1,
                       label="ρ=0 reference")
        ax.set_ylabel(ylabel)
        ax.set_title(metric)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(f"{trials[0]['model']} — fault-effect metrics vs. bit-flip "
                 f"fault magnitude (bf16, uniform over 9.6 Gbit)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", default=os.path.join(ROOT, "results/trials.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results/probes_vs_rho.png"))
    args = ap.parse_args()

    trials = [json.loads(line) for line in open(args.trials) if line.strip()]
    assert trials, f"no trials in {args.trials}"
    plot_probes(trials, args.out)


if __name__ == "__main__":
    main()
