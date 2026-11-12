"""Detailed per-fault optimization analysis.

Creates a figure showing:
1. Per-fault success rate across different objectives/dtypes
2. Loss curves for each fault
3. Random vs optimized comparison per fault
4. Activation and random baseline scatter
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_optimization_records(results_dir):
    """Load all optimization_*.json files."""
    records = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.startswith("optimization_") and fname.endswith(".json"):
            path = os.path.join(results_dir, fname)
            with open(path) as f:
                record = json.load(f)
            records.append(record)
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--out", default=os.path.join(
        ROOT, "results/optimization_per_fault.png"))
    args = ap.parse_args()

    records = load_optimization_records(args.results_dir)
    if not records:
        print(f"No optimization records found in {args.results_dir}")
        return

    print(f"Loaded {len(records)} optimization runs")

    # Organize by fault
    by_fault = defaultdict(list)
    for record in records:
        fault_id = record["fault"]["id"]
        scope = record["fault"]["scope"]
        key = f"{fault_id}_{scope}"
        by_fault[key].append(record)

    # Get unique faults
    faults = sorted(by_fault.keys())
    print(f"Found {len(faults)} unique faults")

    # Create grid
    n_faults = len(faults)
    n_cols = 3
    n_rows = (n_faults + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)

    for idx, fault_key in enumerate(faults):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        runs = by_fault[fault_key]
        fault_id, scope = fault_key.rsplit("_", 1)

        # Plot loss curves for all runs of this fault
        colors_map = {"js": "tab:green", "margin": "tab:red", "unknown": "gray"}
        dtype_map = {"bf16": "-", "fp32": "--"}

        for run in runs:
            curve = run.get("curve", [])
            if not curve:
                continue

            steps = [c["step"] for c in curve]
            losses = [c.get("loss", c.get("js", 0)) for c in curve]
            objective = run["args"].get("objective", "unknown")
            dtype = run["args"].get("dtype", "unknown")
            detected = run.get("detected", False)

            color = colors_map.get(objective, "gray")
            linestyle = dtype_map.get(dtype, "-")
            label = f"{objective} ({dtype})" + (" ✓" if detected else "")
            marker = "o" if detected else "x"

            ax.plot(steps, losses, color=color, linestyle=linestyle, marker=marker,
                   label=label, linewidth=2, alpha=0.7)

        # Formatting
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title(f"{scope.upper()} fault {fault_id}\n"
                    f"({len(runs)} runs: {sum(1 for r in runs if r.get('detected'))} successful)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")

        # Add metadata
        activation = runs[0].get("activation", 0)
        random_det = runs[0].get("random_probe_detections", 0)
        ax.text(0.98, 0.05, f"Act: {activation:.2f}\nRandom: {random_det}/100",
               transform=ax.transAxes, ha="right", va="bottom",
               bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
               fontsize=8)

    # Hide unused subplots
    for idx in range(n_faults, len(axes.flat)):
        axes.flat[idx].set_visible(False)

    fig.suptitle("ATPG Optimization per Fault — Loss Curves and Success Status",
                fontsize=14, fontweight="bold")
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved detailed visualization to {args.out}")


if __name__ == "__main__":
    main()
