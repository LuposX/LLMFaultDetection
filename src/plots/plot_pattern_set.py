"""Coverage-vs-K figure for the jointly optimized pattern set.

Per scope, two curves over the number of patterns K' = 1..K:

- optimized: fraction of TEST faults covered by the best K'-subset of the
  patterns, grown greedily (add the pattern that lifts test coverage most;
  greedy is near-optimal here and matches how the set would actually be
  deployed — pick the best K' patterns out of K).
- random: mean coverage of K' random probe prompts, drawn from the stored
  per-prompt detection vectors of the SAME test faults (the
  fault_coverage.jsonl baseline). Shaded band = std over random draws.

Reads results/pattern_set.json (one record per scope) and writes
results/pattern_set.png.
"""

import json
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "pattern_set.png")
FIG_W, FIG_H = 7.0, 4.2


def greedy_coverage_curve(mat, n_patterns):
    """Greedy subset growth on a detection matrix [n_faults x n_patterns]:
    returns list of coverage fractions after adding 1..n_patterns patterns."""
    covered = [False] * len(mat)
    curve = []
    remaining = list(range(n_patterns))
    for _ in range(n_patterns):
        best_p, best_gain = None, -1
        for p in remaining:
            gain = sum(1 for i, r in enumerate(mat)
                       if not covered[i] and r[p])
            if gain > best_gain:
                best_p, best_gain = p, gain
        for i, r in enumerate(mat):
            if r[best_p]:
                covered[i] = True
        remaining.remove(best_p)
        curve.append(sum(covered) / len(mat))
    return curve


def random_coverage_curves(vectors, ks, n_draws=200, rng=None):
    """Mean and std of coverage for |subset|=k random probe prompts."""
    rng = rng or random.Random(0)
    means, stds = [], []
    for k in ks:
        vals = []
        for _ in range(n_draws):
            sub = rng.sample(range(len(vectors[0])), min(k, len(vectors[0])))
            hits = sum(any(vec[i] for i in sub) for vec in vectors)
            vals.append(hits / len(vectors))
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
        means.append(mean)
        stds.append(var ** 0.5)
    return means, stds


def main():
    path = os.path.join(ROOT, "results", "pattern_set.json")
    records = [json.loads(line) for line in open(path) if line.strip()]
    n = len(records)
    fig, axes = plt.subplots(1, n, figsize=(FIG_W * n, FIG_H), squeeze=False)
    axes = axes[0]

    for ax, rec in zip(axes, records):
        K = len(rec["patterns"])
        ks = list(range(1, K + 1))
        opt_curve = greedy_coverage_curve(rec["det_matrix_test"], K)
        ax.plot(ks, [c * 100 for c in opt_curve], "o-", color="#c44e52",
                label="optimized set (test faults)")
        ds = rec.get("discrete_search")
        if ds:
            ref_curve = greedy_coverage_curve(
                ds["det_matrix_test_refined"], K)
            ax.plot(ks, [c * 100 for c in ref_curve], "o:", color="#dd8452",
                    label="+ discrete search (test faults)")
        vecs = [v for v in rec.get("random_vectors", {}).values()]
        if vecs:
            means, stds = random_coverage_curves(vecs, ks)
            ax.plot(ks, [m * 100 for m in means], "s--", color="#4c72b0",
                    label="random probes (test faults)")
            ax.fill_between(ks, [(m - s) * 100 for m, s in zip(means, stds)],
                            [(m + s) * 100 for m, s in zip(means, stds)],
                            color="#4c72b0", alpha=0.15)
        n_train = len(rec["faults_train"])
        n_test = len(rec["faults_test"])
        ax.set_title(f"{rec['scope']} (k={rec['args']['k']}, "
                     f"{n_train} train / {n_test} test faults)")
        ax.set_xlabel("number of patterns K'")
        ax.set_ylabel("test coverage [%]")
        ax.set_xticks(ks)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
