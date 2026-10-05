"""Plot benchmark score vs flip probability rho from trials.jsonl.

One subplot per task: scatter of the per-seed draws plus a mean line.
rho=0 has no fault draws, so it is drawn as a dashed clean-baseline line.
"""

import argparse
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evalbench import PREFERRED_METRIC


def pick_metric(task, metrics):
    """Return (key, value) for the task's headline metric, with a fallback
    that tolerates lm-eval version changes in metric naming."""
    if task in PREFERRED_METRIC:
        pref = PREFERRED_METRIC[task]
        for k in metrics:
            if k.startswith(pref):
                return k, metrics[k]
    if task == "wikitext":
        for k in metrics:
            if "perplexity" in k:
                return k, metrics[k]
    # last resort: first numeric value
    for k, v in metrics.items():
        return k, v
    return None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", default="results/trials.jsonl")
    parser.add_argument("--out", default="results/benchmark_vs_rho.png")
    args = parser.parse_args()

    trials = [json.loads(line) for line in open(args.trials) if line.strip()]
    assert trials, f"no trials in {args.trials}"

    # points[task] = {"rho": [..], "values": [..], "baseline": v}
    points = defaultdict(lambda: {"rho": [], "values": []})
    baseline = {}
    metric_names = {}

    for t in trials:
        for task, metrics in t["results"].items():
            key, value = pick_metric(task, metrics)
            if key is None:
                continue
            metric_names[task] = key
            if t["rho"] == 0:
                baseline[task] = value
            else:
                points[task]["rho"].append(t["rho"])
                points[task]["values"].append(value)

    tasks = list(points)
    n_cols = 2
    n_rows = (len(tasks) + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows),
                             squeeze=False)

    for ax, task in zip(axes.flat, tasks):
        d = points[task]
        rhos = sorted(set(d["rho"]))
        means = []
        for rho in rhos:
            vals = [v for r, v in zip(d["rho"], d["values"]) if r == rho]
            ax.scatter([rho] * len(vals), vals, s=18, alpha=0.5,
                       label="seed draws" if rho == rhos[0] else None)
            means.append(sum(vals) / len(vals))
        ax.plot(rhos, means, "-o", color="tab:blue", label="mean")

        if task in baseline:
            ax.axhline(baseline[task], ls="--", color="gray",
                       label="clean baseline")

        ax.set_xscale("log")
        ax.set_xlabel("flip probability rho")
        ax.set_ylabel(metric_names[task])
        ax.set_title(task)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(f"{trials[0]['model']} — benchmark degradation vs. bit-flip probability "
                 f"(bf16, uniform over all parameters)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
