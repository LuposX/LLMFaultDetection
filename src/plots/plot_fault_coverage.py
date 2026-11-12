"""Plot deterministic fault coverage for mantissa and exponent faults.

The lines show cumulative probe coverage: for a fixed number of random probe
prompts, the fraction of sampled faults detected by at least one prompt. The
same ordered prompt prefixes are used for both scopes, so the comparison is
about the fault scope rather than a changing probe budget.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def plot(records, out):
    if not records:
        raise ValueError("no coverage records")
    n_prompts = records[0]["n_prompts"]
    x = list(range(1, n_prompts + 1))
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for scope, color in (("mantissa", "tab:blue"),
                         ("exponent", "tab:orange")):
        rows = [r for r in records if r["scope"] == scope]
        if not rows:
            continue
        # Each row is one sampled fault. A fault is covered once any of the
        # first k prompts detects it.
        coverage = []
        for k in x:
            covered = sum(any(r["detected"][:k]) for r in rows)
            coverage.append(covered / len(rows))
        ax.plot(x, coverage, color=color, lw=2.3, label=
                f"{scope} ({len(rows)} faults)")

    ax.set_xlim(1, n_prompts)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("number of random probe prompts")
    ax.set_ylabel("fault coverage\n(fraction of sampled faults detected)")
    ax.set_title("Deterministic fault coverage by bit scope\n"
                 "Qwen3-0.6B, bf16, single-bit faults")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")
    for scope in ("mantissa", "exponent"):
        rows = [r for r in records if r["scope"] == scope]
        if rows:
            print(f"  {scope}: {sum(r['n_detected'] > 0 for r in rows)} / "
                  f"{len(rows)} faults detected by the full probe set")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default=os.path.join(
        ROOT, "results/fault_coverage.jsonl"))
    ap.add_argument("--out", default=os.path.join(
        ROOT, "results/fault_coverage.png"))
    args = ap.parse_args()
    records = [json.loads(line) for line in open(args.records) if line.strip()]
    plot(records, args.out)


if __name__ == "__main__":
    main()
