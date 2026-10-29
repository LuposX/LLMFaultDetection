"""Plot benchmark score vs. bit-flip fault magnitude from trials.jsonl.

One subplot per task: scatter of the per-seed draws plus a mean line.
rho=0 has no fault draws, so it is drawn as a dashed clean-baseline line.
The x axis is the tangible fault magnitude: expected number of flipped
bits per fault draw (rho * model bits, log scale), with the fault
magnitude rho annotated on each tick. The y axis goes log when the
metric spans more than two orders of magnitude (e.g. perplexity),
linear otherwise. The first subplot aggregates the accuracy-type tasks
(hellaswag, mmlu, gsm8k): each task's score is normalized to its clean
baseline before averaging, so the aggregate reads as relative
degradation. wikitext is excluded — perplexity is not an accuracy metric
and its direction is inverted (higher is worse).
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

# one level below src/: add src/ to the import path for the lib imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.benchmarks import PREFERRED_METRIC  # noqa: E402
from common import (  # noqa: E402
    ROOT, N_BITS, ACCURACY_TASKS, flips_axis_setup, pick_metric)


def plot_degradation(trials, out, subtasks=False):
    """One subplot per task: scatter of per-seed draws plus a mean line."""
    # points[task] = {"rho": [..], "values": [..]}
    points = defaultdict(lambda: {"rho": [], "values": []})
    # collapses[task][rho] = number of draws whose metric is undefined
    # (aggregation failed, e.g. perplexity overflow on a destroyed model)
    collapses = defaultdict(lambda: defaultdict(int))
    baseline = {}
    metric_names = {}

    for t in trials:
        for task, metrics in t["results"].items():
            key, value = pick_metric(task, metrics, PREFERRED_METRIC)
            if key is None:
                if "error" in metrics and t["rho"] > 0:
                    collapses[task][t["rho"]] += 1
                continue
            metric_names[task] = key
            if t["rho"] == 0:
                baseline[task] = value
            else:
                points[task]["rho"].append(t["rho"])
                points[task]["values"].append(value)

    # mmlu's result dict carries one entry per subtask (mmlu_philosophy,
    # ...); the headline metric is the group entry "mmlu" itself. Default
    # to the top-level tasks only, subtasks behind a flag.
    top_level = {"hellaswag", "wikitext", "mmlu", "gsm8k",
                 "winogrande", "arc_easy", "boolq"}
    tasks = [t for t in points if subtasks or t in top_level]
    aggregate = {"rho": [], "values": []}
    if sum(t in points and t in baseline for t in ACCURACY_TASKS) >= 2:
        for t in trials:
            if t["rho"] == 0:
                continue
            rel = [v / baseline[task]
                   for task, metrics in t["results"].items()
                   if task in ACCURACY_TASKS and task in points
                   and task in baseline
                   for k, v in [pick_metric(task, metrics, PREFERRED_METRIC)]
                   if k is not None]
            if any("error" in metrics
                   for task, metrics in t["results"].items()
                   if task in ACCURACY_TASKS):
                collapses["average (normalized)"][t["rho"]] += 1
            if rel:
                aggregate["rho"].append(t["rho"])
                aggregate["values"].append(sum(rel) / len(rel))
        if aggregate["rho"]:
            tasks = ["average (normalized)"] + tasks
    n_cols = 2
    n_rows = (len(tasks) + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows),
                             squeeze=False)

    for ax, task in zip(axes.flat, tasks):
        d = aggregate if task.startswith("average") else points[task]
        rhos = sorted(set(d["rho"]))
        means = []
        for rho in rhos:
            vals = [v for r, v in zip(d["rho"], d["values"]) if r == rho]
            flips = rho * N_BITS
            ax.scatter([flips] * len(vals), vals, s=18, alpha=0.5,
                       label="seed draws" if rho == rhos[0] else None)
            means.append(sum(vals) / len(vals))
        ax.plot([r * N_BITS for r in rhos], means, "-o", color="tab:blue",
                label="mean")

        # axis limits from the data (flips), with half-decade padding
        all_flips = [r * N_BITS for r in rhos]
        if task_coll := collapses.get(task):
            all_flips += [r * N_BITS for r in task_coll]
        xlo = min(all_flips) / 3.0
        xhi = max(all_flips) * 3.0

        # mark collapsed draws (metric undefined, excluded from the mean
        # above): red x at the bottom edge, one per collapsed trial
        task_coll = collapses.get(task)
        if task_coll:
            trans = blended_transform_factory(ax.transData, ax.transAxes)
            first = True
            for rho, count in sorted(task_coll.items()):
                for j in range(count):
                    # small horizontal jitter so stacked x's stay visible
                    x = rho * N_BITS * (1.0 + 0.05 * (j - (count - 1) / 2))
                    ax.scatter([x], [0.03], marker="x", s=55, color="tab:red",
                               linewidths=1.5, transform=trans, clip_on=False,
                               zorder=5,
                               label="collapsed draw (metric undefined)"
                                     if first else None)
                    first = False

        if task.startswith("average"):
            ax.axhline(1.0, ls="--", color="gray", label="clean baseline")
        elif task in baseline:
            ax.axhline(baseline[task], ls="--", color="gray",
                       label="clean baseline")

        flips_axis_setup(ax, xlo, xhi, rhos)
        # log y when the metric spans decades (perplexity), linear otherwise
        finite = [v for v in d["values"] + means if 0 < v < float("inf")]
        if finite and max(finite) / min(finite) > 100:
            ax.set_yscale("log")
        ax.set_ylabel("relative score (clean = 1.0)"
                      if task.startswith("average") else metric_names[task])
        ax.set_title(task)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(f"{trials[0]['model']} — degradation vs. bit-flip fault "
                 f"magnitude (bf16, uniform over 9.6 Gbit)", fontsize=11)
    # hide any unused subplot cells (odd task counts)
    for ax in axes.flat[len(tasks):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", default=os.path.join(ROOT, "results/trials.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results/degradation_vs_rho.png"))
    ap.add_argument("--subtasks", action="store_true",
                    help="also plot mmlu per-subtask panels (57 extra "
                         "subplots); default plots the top-level tasks")
    args = ap.parse_args()

    trials = [json.loads(line) for line in open(args.trials) if line.strip()]
    assert trials, f"no trials in {args.trials}"
    plot_degradation(trials, args.out, subtasks=args.subtasks)


if __name__ == "__main__":
    main()
