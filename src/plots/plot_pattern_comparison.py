"""Plot optimized-vs-random test pattern detection from
results/pattern_comparison.jsonl (written by run_pattern_comparison.py).

Per fault scope, two bars over the sampled faults:

  random (K=1)   : mean over faults of the random single-prompt detection
                   rate (per-fault fraction of the 100 probes that detect
                   the fault — the probability that ONE uniformly drawn
                   probe detects that fault)
  optimized (K=1): fraction of faults whose optimized pattern detects
                   (free-running, exact-match criterion)

If a discrete-search refinement ran, its bar is added (it only applies to
faults the soft prompt missed). Both quantities are per-single-pattern,
so the bars answer the experiment question directly.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default=os.path.join(
        ROOT, "results/pattern_comparison.jsonl"))
    ap.add_argument("--out", default=os.path.join(
        ROOT, "results/pattern_comparison.png"))
    args = ap.parse_args()
    records = [json.loads(line) for line in open(args.records) if line.strip()]
    assert records, f"no records in {args.records}"

    scopes = sorted({r["scope"] for r in records})
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    width = 0.8 / max(2, 2 + any(r.get("discrete_search") for r in records))

    for i, scope in enumerate(scopes):
        rows = [r for r in records if r["scope"] == scope]
        n = len(rows)
        rand = sum(r["random_single_rate"] for r in rows) / n
        opt = sum(r["soft_detected"] for r in rows) / n
        ds = [r["discrete_search"] for r in rows if r.get("discrete_search")]
        opt_ds = (sum(1 for r in rows
                      if r["soft_detected"]
                      or (r.get("discrete_search") or {}).get("detected"))
                  / n) if ds else None
        x = i
        ax.bar(x - width / 2, rand, width, color="tab:gray",
               label="random probe (K=1, mean rate)" if i == 0 else None)
        ax.bar(x + width / 2, opt, width, color="tab:blue",
               label="optimized pattern (K=1)" if i == 0 else None)
        if opt_ds is not None:
            ax.bar(x + 1.5 * width, opt_ds, width, color="tab:green",
                   label="optimized + discrete search" if i == 0 else None)
        ax.text(x - width / 2, rand + 0.02, f"{rand:.2f}",
                ha="center", fontsize=9)
        ax.text(x + width / 2, opt + 0.02, f"{opt:.2f}",
                ha="center", fontsize=9)
        if opt_ds is not None:
            ax.text(x + 1.5 * width, opt_ds + 0.02, f"{opt_ds:.2f}",
                    ha="center", fontsize=9)
        print(f"{scope}: n={n}  random K=1 mean rate {rand:.3f}  "
              f"optimized {opt:.3f}" + (f"  +search {opt_ds:.3f}" if ds else ""))

    labels = [f"{s}\n({sum(1 for r in records if r['scope'] == s)} faults)"
              for s in scopes]
    ax.set_xticks(range(len(scopes)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("detection rate (single test pattern)")
    ax.set_title("Optimized vs. random test pattern — deterministic "
                 "exact-match criterion\nQwen3-0.6B bf16, single-bit faults")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
