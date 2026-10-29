"""Plot the SDoT soft-vs-hard logit difference vs. temperature from
results/sdot_tau_sweep.json (written by checks/check_sdot.py).

Two panels: worst-case max |Δlogit| (left) and mean |Δlogit| (right),
each prompt as a thin line plus the aggregate in bold. Symlog y (linear
below 1e-2, log above) so the exact zero and the small transition values
are readable distances. The green band marks the faithful region where
every prompt is still bitwise identical; the red dotted line marks the
first tau at which any prompt's argmax token differs — the optimization's
annealing schedule must end below it.
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# one level below src/: add src/ to the import path for the lib imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import ROOT  # noqa: E402


def plot_sweep(data, out):
    taus = data["taus"]
    rows = data["rows"]

    # faithful region: taus at which every prompt is still bitwise
    # identical (largest tau with zero diff anywhere)
    faithful_end = max(t for t in taus
                       if all(r["max"][taus.index(t)] == 0 for r in rows))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (key, agg_fn, agg_label) in zip(axes, [
            ("max", max, "worst over prompts"),
            ("mean", lambda v: sum(v) / len(v), "mean over prompts")]):
        for r in rows:
            ax.plot(taus, r[key], color="tab:blue", alpha=0.35, lw=1,
                    zorder=2)
        agg = [agg_fn([r[key][i] for r in rows]) for i in range(len(taus))]
        ax.plot(taus, agg, "-o", ms=4, color="tab:blue", lw=2, zorder=3,
                label=agg_label)
        # symlog: linear below linthresh so the exact zero and the small
        # transition values are readable distances, log above for the cliff
        ax.set_yscale("symlog", linthresh=1e-2)
        ax.set_ylim(-2e-3, 25)
        ax.axhline(0, ls="--", color="gray", lw=1, zorder=1)
        ax.axvspan(taus[-1], faithful_end, color="tab:green", alpha=0.1,
                   zorder=1, label=f"faithful: bitwise identical (τ ≤ {faithful_end:g})")
        ax.set_xlim(taus[-1], taus[0])
        ax.set_xticks([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
        ax.set_xlabel("temperature τ")
        ax.set_title({"max": "max |Δlogit|", "mean": "mean |Δlogit|"}[key])
        ax.grid(alpha=0.3)

    # faithfulness boundary: smallest tau with any argmax mismatch
    # (taus may be sorted descending, so scan for the minimum, not the first)
    bad = [t for t, bad in zip(
        taus, [any(r["mism"][i] > 0 for r in rows) for i in range(len(taus))])
        if bad]
    boundary = min(bad) if bad else None
    for ax in axes:
        if boundary is not None:
            ax.axvline(boundary, color="tab:red", lw=1.2, ls=":", zorder=1,
                       label=f"argmax mismatch first appears (τ ≈ {boundary:g})")
        # legend after the boundary line so its label is included
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"{data['model']} — SDoT soft path vs. hard token path "
                 f"(amplitude-{data['amplitude']:g} one-hot encodings)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", default=os.path.join(ROOT, "results/sdot_tau_sweep.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results/sdot_tau_sweep.png"))
    args = ap.parse_args()

    data = json.load(open(args.sweep))
    plot_sweep(data, args.out)


if __name__ == "__main__":
    main()
