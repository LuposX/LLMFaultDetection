"""Write data/fault_manifest.json: the frozen fault set every method runs on.

Deterministic fault definitions are the comparison's foundation: every
experiment (baselines, loss variants, Algorithm 1) must evaluate the same
fault population. This script samples the canonical set once — k=1 bit
flip, seeds 0..n-1, per scope — and records the full flip specification so
later runs reconstruct identical Fault objects without re-sampling. It
cross-checks the ids against results/fault_coverage.jsonl (for the seeds
that file knows), so a drift in sample_fault would fail loudly here.

--seeds extends the set additively: seeds 0..n-1 with seed0=0 means seeds
0..19 are identical to any previously frozen manifest with n=20, so cached
decodes carry over; only the new seeds are sampled fresh.

Run: python src/make_fault_manifest.py [--seeds 100]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from lib.faults import sample_faults          # noqa: E402
from lib.loading import load_model            # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"
N_FAULTS = 20
K = 1
SCOPES = ["mantissa", "exponent"]
OUT = "data/fault_manifest.json"
# Flip-multiplicity sweep: same scopes and seeds at higher flip counts, for
# the fault-multiplicity axis (coverage vs number of flips). Superset: the
# experiment config selects which k values to run (data/fault_manifest.json
# owns k=1).
K_SWEEP = [2, 4, 8, 10, 20, 50]
OUT_K_SWEEP = "data/fault_manifest_k_sweep.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=N_FAULTS,
                    help="faults per (k, scope); extending additively keeps "
                         "seeds 0..n-1 of any smaller frozen set identical")
    args = ap.parse_args()
    n_seeds = args.seeds

    model, tokenizer = load_model(MODEL, "cpu")

    faults = []
    for scope in SCOPES:
        faults.extend(f.to_dict()
                      for f in sample_faults(model, n_seeds, K, scope, 0))

    manifest = {
        "model": MODEL,
        "k": K,
        "n_per_scope": n_seeds,
        "scopes": SCOPES,
        "seed0": 0,
        "note": "frozen fault set; seeds seed0..seed0+n_per_scope-1 per "
                "scope; do not re-sample",
        "faults": faults,
    }

    # cross-check against the cached random-baseline coverage, if present
    cov_path = "results/fault_coverage.jsonl"
    if os.path.exists(cov_path):
        known = {}
        with open(cov_path) as f:
            for line in f:
                r = json.loads(line)
                known[(r["scope"], r["seed"])] = r["fault"]["id"]
        mismatch = [
            (f["scope"], f["seed"], f["id"],
             known.get((f["scope"], f["seed"])))
            for f in faults
            if (f["scope"], f["seed"]) in known
            and known[(f["scope"], f["seed"])] != f["id"]]
        if mismatch:
            raise SystemExit(
                f"fault id mismatch vs {cov_path} (sample_fault drifted?):\n"
                + "\n".join(map(str, mismatch)))
        print(f"ids match {cov_path} for the "
              f"{len(known)} seeds that file knows")
    else:
        print(f"{cov_path} not found — skipped id cross-check")

    with open(OUT, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {OUT}: {len(faults)} faults "
          f"({len(SCOPES)} scopes x {n_seeds} seeds, k={K})")

    # --- flip-multiplicity sweep manifest --------------------------------
    sweep = []
    for k in K_SWEEP:
        for scope in SCOPES:
            sweep.extend(f.to_dict() for f in
                         sample_faults(model, n_seeds, k, scope, 0))
    sweep_manifest = dict(manifest)
    sweep_manifest["k"] = K_SWEEP
    sweep_manifest["note"] = (
        "flip-multiplicity sweep; same scopes/seeds as the k=1 set at "
        "higher flip counts; do not re-sample")
    sweep_manifest["faults"] = sweep
    with open(OUT_K_SWEEP, "w") as f:
        json.dump(sweep_manifest, f, indent=1)
    print(f"wrote {OUT_K_SWEEP}: {len(sweep)} faults "
          f"(k in {K_SWEEP} x {len(SCOPES)} scopes x {n_seeds} seeds)")


if __name__ == "__main__":
    main()
