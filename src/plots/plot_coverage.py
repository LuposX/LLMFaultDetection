"""Fault coverage vs. fault magnitude from trials.jsonl — one panel per
probe metric, the same four as plot_probes.py: token match, teacher-forced
Jensen-Shannon divergence, embedding cosine, laya judge.

Each panel turns its metric into a detection criterion and reports
draw-level coverage: the fraction of fault draws at that rho that the
criterion flags (Wilson 95% error bars over the draws).

- token match: the faulty model's greedy output differs from the healthy
  reference on >=1 probe prompt. Under deterministic decoding the
  false-flag rate on healthy models is exactly 0, so "any prompt" is a
  valid rule. The thin dotted line is the mean per-prompt divergence rate.
- JS divergence: mean JS over the probe set >= JS_THRESHOLD. Thin dotted
  line = the mean JS itself (right axis, log).
- embedding cosine: mean cosine <= COS_THRESHOLD. Thin dotted line = the
  mean cosine (right axis).
- laya: per-prompt flags (P(same) below the judge's identical-pair floor)
  above the 5% noise floor, one-sided binomial test at alpha = 0.05.
  "Any prompt flags" is invalid for this noisy criterion: the floor is
  ~5% by construction, so over 100 prompts it would flag healthy models
  ~99% of the time.

The JS/cos thresholds are PROVISIONAL. Under deterministic decoding the
healthy noise floor is degenerate (JS = 0, cos = 1 exactly on identical
outputs), so it cannot anchor a threshold; the proper calibration is the
tau-95 paraphrase procedure from the proposal, which is pending. Both
values sit at the geometric midpoint of the observed gap between the
shallow-drift draws (rho=1e-9: JS ~ 2e-4, cos ~ 0.85) and the collapsed
draws (rho=1e-8: JS ~ 0.6, cos ~ 0.03). Note the consequence visible in
the figure: mean-based JS/cos criteria are blind in the drift band where
token match and laya still detect — averaging over 100 prompts washes out
shallow per-prompt changes. Per-prompt thresholds require recording
per-prompt scores and calibrating them (future work).

Draws with zero flips are excluded (the model is unchanged — nothing to
detect); the rho=0 baseline is never counted. x axis in expected flipped
bits per draw with rho annotations, as in the other plots.

Plots read from trials.jsonl only; no re-evaluation happens here.
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# one level below src/: add src/ to the import path for the lib imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import ROOT, flips_axis_setup, N_BITS  # noqa: E402

# provisional detection thresholds (see module docstring for the basis)
JS_THRESHOLD = 0.01    # geometric midpoint of the observed 2e-4 / 0.56 gap
COS_THRESHOLD = 0.12   # geometric midpoint of the observed 0.87 / 0.02 gap

PANELS = [
    # key in sem["overall"], title, criterion description, twin-axis label
    ("frac_diverged", "token match",
     "greedy output differs from the healthy reference on >=1 prompt "
     "(deterministic: zero false flags)", None),
    ("js", "Jensen-Shannon divergence",
     f"mean JS over the prompt set >= {JS_THRESHOLD:g} (provisional, "
     "tau-95 calibration pending)", "mean JS"),
    ("embed_cos", "embedding cosine",
     f"mean cosine <= {COS_THRESHOLD:g} (provisional, tau-95 calibration "
     "pending)", "mean cosine"),
    ("laya_flag_frac", "semantic (laya, provisional)",
     "flag fraction above the 5% noise floor (binomial test)", None),
]


def binom_ge(k, n, p):
    """P(X >= k) for X ~ Binomial(n, p), exact via math.comb."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j)
               for j in range(k, n + 1))


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def detected(m, v, n_prompts):
    """Draw-level detection verdict for metric m with aggregate value v."""
    if v is None:
        return None
    if m == "frac_diverged":
        return v > 0
    if m == "js":
        return v >= JS_THRESHOLD
    if m == "embed_cos":
        return v <= COS_THRESHOLD
    if m == "laya_flag_frac":
        # noisy criterion: binomial test against the per-prompt
        # false-flag floor p0 = 0.05 (threshold is the P5 of
        # identical-pair scores), significance 0.05
        k = round(v * n_prompts)
        return binom_ge(k, n_prompts, 0.05) < 0.05
    raise ValueError(m)


def plot_coverage(trials, out):
    # per metric per rho: (detected draws, total draws) plus the metric
    # value per draw (the thin reference line)
    counts = {m: defaultdict(lambda: [0, 0]) for m, _, _, _ in PANELS}
    vals = {m: defaultdict(list) for m, _, _, _ in PANELS}
    threshold = None
    for t in trials:
        if t["rho"] == 0 or t.get("n_flips", 0) == 0:
            continue  # unchanged model: nothing to detect
        sem = t.get("sem")
        if not sem or not sem.get("overall"):
            continue
        if t.get("laya_threshold") is not None:
            threshold = t["laya_threshold"]
        n_prompts = sem["overall"].get("n") or 100
        for m, _, _, _ in PANELS:
            v = sem["overall"].get(m)
            if v is None:
                continue
            c = counts[m][t["rho"]]
            c[1] += 1
            if detected(m, v, n_prompts):
                c[0] += 1
            vals[m][t["rho"]].append(v)
    if not any(c[1] for m, _, _, _ in PANELS
               for c in counts[m].values()):
        print("no trials with fault-effect data — nothing to plot")
        return

    rhos = sorted({r for m, _, _, _ in PANELS for r in counts[m]})
    flips = [r * N_BITS for r in rhos]
    xlo, xhi = min(flips) / 3.0, max(flips) * 3.0

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), squeeze=False)
    for ax, (m, title, desc, twin_label) in zip(axes.flat, PANELS):
        xs, ys, lo_err, hi_err = [], [], [], []
        for r in rhos:
            k, n = counts[m][r]
            if not n:
                continue
            xs.append(r * N_BITS)
            ys.append(k / n)
            lo, hi = wilson(k, n)
            lo_err.append(k / n - lo)
            hi_err.append(hi - k / n)
        if xs:
            ax.errorbar(xs, ys, yerr=[lo_err, hi_err], fmt="o-",
                        color="tab:blue", lw=2.2, markersize=5, capsize=3,
                        zorder=3, label="draw-level coverage")
        ax.axhline(1.0, ls="--", color="gray", alpha=0.5, zorder=1)
        if m == "laya_flag_frac":
            ax.axhline(0.05, ls="-", color="tab:red", alpha=0.6, lw=1,
                       zorder=1, label="per-prompt noise floor (5%)")

        # thin reference line: per-prompt rate for the per-prompt
        # criteria, the metric mean itself (twin axis) for JS/cos
        rx = [r * N_BITS for r in rhos if vals[m].get(r)]
        ry = [sum(vals[m][r]) / len(vals[m][r])
              for r in rhos if vals[m].get(r)]
        if rx:
            if twin_label:
                ax2 = ax.twinx()
                ax2.plot(rx, ry, ":", color="tab:orange", lw=1.4,
                         alpha=0.9, zorder=2, label=twin_label)
                if m == "js":
                    ax2.set_yscale("log")
                    ax2.set_ylim(1e-10, 2)
                else:
                    ax2.set_ylim(0, 1.05)
                ax2.set_ylabel(twin_label, color="tab:orange", fontsize=9)
                ax2.tick_params(axis="y", labelcolor="tab:orange",
                                labelsize=8)
                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
            else:
                ax.plot(rx, ry, ":", color="tab:blue", lw=1.4, alpha=0.8,
                        zorder=2,
                        label="per-prompt detection rate")
                ax.legend(fontsize=8, loc="center right")
        flips_axis_setup(ax, xlo, xhi, rhos)
        ax.set_ylim(-0.05, 1.1)
        ax.set_ylabel("fault coverage\n(fraction of draws detected)")
        ax.set_title(f"{title}\n[{desc}]", fontsize=10)
        ax.grid(alpha=0.3)
    sub = (f"100 random probe prompts; semantic threshold = judge "
           f"P(same) floor on identical pairs ({threshold:.3f}); "
           f"JS/cos thresholds provisional (tau-95 pending)"
           if threshold is not None else
           "100 random probe prompts")
    fig.suptitle(f"Fault coverage vs. fault magnitude — Qwen3-0.6B bf16, "
                 f"uniform over 9.6 Gbit\n{sub}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    for m, title, _, _ in PANELS:
        for r in rhos:
            k, n = counts[m][r]
            if n:
                print(f"  {title:28s} rho={r:g}: {k}/{n} draws detected")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", default=os.path.join(ROOT, "results/trials.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results/coverage_vs_rho.png"))
    args = ap.parse_args()

    trials = [json.loads(line) for line in open(args.trials) if line.strip()]
    assert trials, f"no trials in {args.trials}"
    plot_coverage(trials, args.out)


if __name__ == "__main__":
    main()
