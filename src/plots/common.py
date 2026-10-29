"""Shared helpers for the plot scripts in this folder.

Imported as a sibling (`from common import ...`), which works because
script runs put their own directory first on the import path. Scripts that
also need `lib/` add src/ to the path themselves (see the shims).
"""

import os

import matplotlib
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter

# repo root: src/plots/common.py -> three levels up
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fault-space size: ~0.6B unique parameters (tied embed/lm_head counted once)
# x 16 bits per bf16 element. Converts rho -> expected flips per fault draw.
# Model-specific; if the model changes, update this constant.
N_BITS = 9.6e9


def flips_axis_setup(ax, xlo, xhi, rhos):
    """X axis: expected flipped bits per draw, ticks annotated with rho.

    One tick per rho present in the data, placed at rho*N_BITS so ticks
    align with the plotted grid points. Each tick reads '9.6\n1e-09':
    the flip count is primary, rho traceable to the grid definition.
    xlo/xhi bound the data (in flips).
    """
    ticks = [r * N_BITS for r in sorted(rhos)]
    labels = [f"{t:g}\n{t / N_BITS:.0e}" for t in ticks]
    ax.set_xscale("log")
    ax.set_xlim(xlo, xhi)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter(labels))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", labelsize=7.5)
    ax.set_xlabel("expected flipped bits per fault draw\n(lower line: fault magnitude ρ)")


# Tasks with a higher-is-better accuracy metric that can be averaged after
# normalizing to the clean baseline. wikitext (perplexity) is excluded.
ACCURACY_TASKS = ("hellaswag", "mmlu", "gsm8k", "winogrande", "arc_easy",
                  "boolq")


def pick_metric(task, metrics, preferred):
    """Return (key, value) for the task's headline metric, with a fallback
    that tolerates lm-eval version changes in metric naming."""
    if task in preferred:
        pref = preferred[task]
        for k in metrics:
            if k.startswith(pref):
                return k, metrics[k]
    if task == "wikitext":
        for k in metrics:
            if "perplexity" in k and isinstance(metrics[k], (int, float)):
                return k, metrics[k]
    # last resort: first numeric value (trials with failed tasks carry an
    # "error" string entry instead of metrics; those plot as missing)
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            return k, v
    return None, None
