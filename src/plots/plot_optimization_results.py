"""Visualize ATPG optimization results across all runs.

Creates a multi-panel figure showing:
1. Success rate by scope (mantissa vs exponent)
2. Success rate by objective (margin vs js)
3. Success rate by dtype (bf16 vs fp32)
4. Optimization curves for individual faults
5. Activation vs detection outcome
6. Random baseline vs optimized verdict comparison
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
        ROOT, "results/optimization_summary.png"))
    args = ap.parse_args()

    records = load_optimization_records(args.results_dir)
    if not records:
        print(f"No optimization records found in {args.results_dir}")
        return

    print(f"Loaded {len(records)} optimization runs")

    # Aggregate by various dimensions
    by_scope = defaultdict(lambda: {"detected": 0, "total": 0})
    by_objective = defaultdict(lambda: {"detected": 0, "total": 0})
    by_dtype = defaultdict(lambda: {"detected": 0, "total": 0})
    by_fault_scope = defaultdict(list)
    by_fault_id = defaultdict(list)

    for record in records:
        fault = record["fault"]
        scope = fault["scope"]
        objective = record["args"].get("objective", "unknown")
        dtype = record["args"].get("dtype", "unknown")
        detected = record.get("detected", False)
        fault_id = fault["id"]
        activation = record.get("activation", 0)
        random_det = record.get("random_probe_detections", 0)

        # Aggregate by scope
        by_scope[scope]["total"] += 1
        if detected:
            by_scope[scope]["detected"] += 1

        # Aggregate by objective
        by_objective[objective]["total"] += 1
        if detected:
            by_objective[objective]["detected"] += 1

        # Aggregate by dtype
        by_dtype[dtype]["total"] += 1
        if detected:
            by_dtype[dtype]["detected"] += 1

        # Store per-fault info
        by_fault_scope[f"{fault_id}_{scope}"].append({
            "objective": objective,
            "dtype": dtype,
            "detected": detected,
            "activation": activation,
            "random_det": random_det,
            "curve": record.get("curve", [])
        })

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # 1. Success rate by scope
    ax1 = fig.add_subplot(gs[0, 0])
    scopes = sorted(by_scope.keys())
    detected_counts = [by_scope[s]["detected"] for s in scopes]
    total_counts = [by_scope[s]["total"] for s in scopes]
    success_rates = [d / t if t > 0 else 0 for d, t in zip(detected_counts, total_counts)]
    colors = ["tab:blue", "tab:orange"]
    bars1 = ax1.bar(scopes, success_rates, color=colors[:len(scopes)], alpha=0.7, edgecolor="black")
    ax1.set_ylabel("Success rate")
    ax1.set_title("Success Rate by Fault Scope")
    ax1.set_ylim(0, 1.05)
    ax1.axhline(0.5, ls="--", color="gray", alpha=0.5)
    for bar, (d, t) in zip(bars1, zip(detected_counts, total_counts)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                f"{d}/{t}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    # 2. Success rate by objective
    ax2 = fig.add_subplot(gs[0, 1])
    objectives = sorted(by_objective.keys())
    detected_counts = [by_objective[o]["detected"] for o in objectives]
    total_counts = [by_objective[o]["total"] for o in objectives]
    success_rates = [d / t if t > 0 else 0 for d, t in zip(detected_counts, total_counts)]
    colors = ["tab:green", "tab:red"]
    bars2 = ax2.bar(objectives, success_rates, color=colors[:len(objectives)], alpha=0.7, edgecolor="black")
    ax2.set_ylabel("Success rate")
    ax2.set_title("Success Rate by Objective")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(0.5, ls="--", color="gray", alpha=0.5)
    for bar, (d, t) in zip(bars2, zip(detected_counts, total_counts)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                f"{d}/{t}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    # 3. Success rate by dtype
    ax3 = fig.add_subplot(gs[0, 2])
    dtypes = sorted(by_dtype.keys())
    detected_counts = [by_dtype[d]["detected"] for d in dtypes]
    total_counts = [by_dtype[d]["total"] for d in dtypes]
    success_rates = [d / t if t > 0 else 0 for d, t in zip(detected_counts, total_counts)]
    colors = ["tab:purple", "tab:brown"]
    bars3 = ax3.bar(dtypes, success_rates, color=colors[:len(dtypes)], alpha=0.7, edgecolor="black")
    ax3.set_ylabel("Success rate")
    ax3.set_title("Success Rate by Data Type")
    ax3.set_ylim(0, 1.05)
    ax3.axhline(0.5, ls="--", color="gray", alpha=0.5)
    for bar, (d, t) in zip(bars3, zip(detected_counts, total_counts)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                f"{d}/{t}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax3.grid(axis="y", alpha=0.3)

    # 4. Activation vs Detection outcome (scatter)
    ax4 = fig.add_subplot(gs[1, 0])
    activations = []
    outcomes = []
    colors_scatter = []
    for record in records:
        activation = record.get("activation", 0)
        detected = record.get("detected", False)
        scope = record["fault"]["scope"]
        activations.append(activation)
        outcomes.append(1 if detected else 0)
        colors_scatter.append("green" if detected else "red")

    scatter = ax4.scatter(activations, outcomes, c=colors_scatter, s=100, alpha=0.6, edgecolor="black")
    ax4.set_xlabel("Fault activation (logit change)")
    ax4.set_ylabel("Detection success (0=fail, 1=success)")
    ax4.set_title("Activation vs Detection Outcome")
    ax4.set_ylim(-0.1, 1.1)
    ax4.grid(alpha=0.3)
    ax4.set_yticks([0, 1])

    # 5. Random baseline vs optimized verdict
    ax5 = fig.add_subplot(gs[1, 1])
    random_detected_all = sum(1 for r in records if r.get("random_probe_detections", 0) == 100)
    random_detected_some = sum(1 for r in records if 0 < r.get("random_probe_detections", 0) < 100)
    random_detected_none = sum(1 for r in records if r.get("random_probe_detections", 0) == 0)
    
    categories = ["Detected\nby all\nrandom\nprompts", "Detected\nby some\nrandom\nprompts", "Not detected\nby any\nrandom\nprompts"]
    counts = [random_detected_all, random_detected_some, random_detected_none]
    colors_baseline = ["tab:green", "#FFD700", "tab:red"]
    
    bars5 = ax5.bar(categories, counts, color=colors_baseline, alpha=0.7, edgecolor="black")
    ax5.set_ylabel("Number of faults")
    ax5.set_title("Random Probe Baseline Distribution")
    ax5.set_ylim(0, max(counts) + 1)
    for bar, count in zip(bars5, counts):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width() / 2, height + 0.1,
                f"{int(count)}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax5.grid(axis="y", alpha=0.3)

    # 6. Optimization curves (sample successful runs)
    ax6 = fig.add_subplot(gs[1, 2])
    curve_count = 0
    for record in records:
        if record.get("detected", False) and record.get("curve"):
            curve = record["curve"]
            steps = [c["step"] for c in curve]
            losses = [c["loss"] for c in curve]
            fault_id = record["fault"]["id"][:8]
            ax6.plot(steps, losses, marker="o", label=f"Success: {fault_id}", linewidth=2)
            curve_count += 1

    # Also show a sample failure
    for record in records:
        if not record.get("detected", False) and record.get("curve"):
            curve = record["curve"]
            steps = [c["step"] for c in curve]
            losses = [c["loss"] for c in curve]
            fault_id = record["fault"]["id"][:8]
            ax6.plot(steps, losses, marker="x", label=f"Failure: {fault_id}", linewidth=1.5, alpha=0.5, linestyle="--")
            break

    ax6.set_xlabel("Optimization step")
    ax6.set_ylabel("Loss")
    ax6.set_title("Optimization Curves (Sample)")
    ax6.legend(fontsize=8, loc="best")
    ax6.grid(alpha=0.3)

    # 7. Scope breakdown (stacked bar)
    ax7 = fig.add_subplot(gs[2, 0])
    scopes = ["mantissa", "exponent"]
    objectives_in_scope = defaultdict(lambda: {"mantissa": 0, "exponent": 0})
    objectives_detected = defaultdict(lambda: {"mantissa": 0, "exponent": 0})
    
    for record in records:
        scope = record["fault"]["scope"]
        objective = record["args"].get("objective", "unknown")
        objectives_in_scope[objective][scope] += 1
        if record.get("detected"):
            objectives_detected[objective][scope] += 1
    
    objectives = sorted(objectives_in_scope.keys())
    mantissa_total = [objectives_in_scope[o]["mantissa"] for o in objectives]
    exponent_total = [objectives_in_scope[o]["exponent"] for o in objectives]
    mantissa_det = [objectives_detected[o]["mantissa"] for o in objectives]
    exponent_det = [objectives_detected[o]["exponent"] for o in objectives]
    
    x_pos = np.arange(len(objectives))
    width = 0.35
    
    bars_m = ax7.bar(x_pos - width/2, mantissa_total, width, label="Mantissa total", alpha=0.5, color="tab:blue")
    bars_e = ax7.bar(x_pos + width/2, exponent_total, width, label="Exponent total", alpha=0.5, color="tab:orange")
    
    # Overlay success counts
    ax7_twin = ax7.twinx()
    bars_m_det = ax7_twin.bar(x_pos - width/2, mantissa_det, width/2, label="Mantissa detected", color="darkblue", alpha=0.7)
    bars_e_det = ax7_twin.bar(x_pos + width/2, exponent_det, width/2, label="Exponent detected", color="darkorange", alpha=0.7)
    
    ax7.set_ylabel("Total runs", fontsize=9)
    ax7_twin.set_ylabel("Detected runs", fontsize=9)
    ax7.set_xlabel("Objective")
    ax7.set_title("Scope Success Breakdown by Objective")
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(objectives)
    ax7.grid(axis="y", alpha=0.3)
    
    # 8. Dtype comparison per scope
    ax8 = fig.add_subplot(gs[2, 1])
    dtype_scope = defaultdict(lambda: defaultdict(lambda: {"total": 0, "detected": 0}))
    
    for record in records:
        scope = record["fault"]["scope"]
        dtype = record["args"].get("dtype", "unknown")
        dtype_scope[dtype][scope]["total"] += 1
        if record.get("detected"):
            dtype_scope[dtype][scope]["detected"] += 1
    
    dtypes_list = sorted(dtype_scope.keys())
    x_pos = np.arange(len(dtypes_list))
    width = 0.35
    
    mantissa_rates = [dtype_scope[d]["mantissa"]["detected"] / dtype_scope[d]["mantissa"]["total"]
                      if dtype_scope[d]["mantissa"]["total"] > 0 else 0
                      for d in dtypes_list]
    exponent_rates = [dtype_scope[d]["exponent"]["detected"] / dtype_scope[d]["exponent"]["total"]
                      if dtype_scope[d]["exponent"]["total"] > 0 else 0
                      for d in dtypes_list]
    
    ax8.bar(x_pos - width/2, mantissa_rates, width, label="Mantissa", color="tab:blue", alpha=0.7, edgecolor="black")
    ax8.bar(x_pos + width/2, exponent_rates, width, label="Exponent", color="tab:orange", alpha=0.7, edgecolor="black")
    
    ax8.set_ylabel("Success rate")
    ax8.set_xlabel("Data type")
    ax8.set_title("Success Rate: Scope × Data Type")
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels(dtypes_list)
    ax8.set_ylim(0, 1.05)
    ax8.legend()
    ax8.grid(axis="y", alpha=0.3)

    # 9. Summary statistics text
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis("off")
    
    total_runs = len(records)
    total_detected = sum(1 for r in records if r.get("detected", False))
    success_rate = total_detected / total_runs if total_runs > 0 else 0
    
    avg_activation = np.mean([r.get("activation", 0) for r in records])
    avg_random = np.mean([r.get("random_probe_detections", 0) for r in records])
    
    summary_text = f"""
OPTIMIZATION RESULTS SUMMARY

Total runs: {total_runs}
Successful detections: {total_detected}/{total_runs}
Success rate: {success_rate:.1%}

Average fault activation: {avg_activation:.2f}
Average random probe detections: {avg_random:.1f}/100

By scope:
  Mantissa: {by_scope['mantissa']['detected']}/{by_scope['mantissa']['total']} ({by_scope['mantissa']['detected']/by_scope['mantissa']['total']:.1%})
  Exponent: {by_scope['exponent']['detected']}/{by_scope['exponent']['total']} ({by_scope['exponent']['detected']/by_scope['exponent']['total']:.1%})

Key finding:
  Exponent faults are easier to optimize for,
  but mantissa faults (harder cases) remain
  largely undetectable even after optimization.
"""
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle("ATPG Optimization Results — Qwen3-0.6B Single-Bit Fault Detection",
                fontsize=14, fontweight="bold")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to {args.out}")


if __name__ == "__main__":
    main()
