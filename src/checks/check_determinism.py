"""Determinism check for greedy decoding: the foundation of the
deterministic-detection pipeline.

Detection by exact token matching assumes the healthy model reproduces its
baseline output bitwise on every run. If batching, reloading, or the
inject/restore cycle can flip a near-tie argmax, a "detection" could be
numerical noise rather than a fault. This script tests the four ways that
could happen; any FAIL means the exact-match pipeline is unusable as
designed and must be fixed or rethought before ATPG work starts.

Checks:
  1. same-session rerun      — identical inputs twice in one process
  2. batch-size invariance   — batch 8 vs batch 1 (padding / reduction
                               order must not reach the argmax)
  3. fresh reload            — a newly loaded model reproduces the baseline
  4. inject/restore residue  — after injecting a fault and restoring the
                               weights, outputs AND weights are bitwise back
                               to baseline (4b / 4a respectively)

Exit code 0 iff all checks pass.
"""

import argparse
import json
import os
import sys

import torch

# one level below src/: add src/ to the import path for the lib imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.generation import generate_greedy  # noqa: E402
from lib.inject import (  # noqa: E402
    inject_model, snapshot_weights, restore_weights, verify_restore)
from lib.loading import load_model  # noqa: E402

MODEL_ID = "Qwen/Qwen3-0.6B"

# Anchor data paths to the repo root (three levels up from this file),
# not the cwd: the script must behave the same from anywhere.
ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))


def pick_subset(prompt_file, per_stratum):
    """First `per_stratum` prompts of each stratum.

    The file is grouped by stratum (wiki, qa, math), so a single pass with
    per-stratum counters picks deterministic, short-ish candidates without
    loading any dataset. What we test here is reproducibility mechanics,
    not model behavior, so a handful of prompts is enough.
    """
    counts, out = {}, []
    for line in open(prompt_file):
        if not line.strip():
            continue
        p = json.loads(line)
        if counts.get(p["stratum"], 0) < per_stratum:
            counts[p["stratum"]] = counts.get(p["stratum"], 0) + 1
            out.append(p["prompt"])
    return out


def first_divergences(ref_ids, ids):
    """Per prompt: position of the first differing token, None if identical.

    A shared prefix with different lengths counts as diverging right after
    the prefix (position min length) — still a token-level mismatch.
    """
    out = []
    for a, b in zip(ref_ids, ids):
        if a == b:
            out.append(None)
            continue
        n = min(len(a), len(b))
        pos = next((j for j in range(n) if a[j] != b[j]), n)
        out.append(pos)
    return out


_results = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""),
          flush=True)
    _results.append(ok)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--per-stratum", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--inject-rho", type=float, default=1e-9,
                    help="fault strength for check 4; only needs to produce "
                         ">0 flips so the restore test is non-trivial")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    prompts = pick_subset(os.path.join(ROOT, "data/prompts.jsonl"),
                          args.per_stratum)
    print(f"{len(prompts)} prompts, max_new_tokens={args.max_new_tokens}, "
          f"device={device}", flush=True)

    model, tokenizer = load_model(args.model, device)

    print("check 1: same-session rerun", flush=True)
    texts1, ids1 = generate_greedy(model, tokenizer, prompts,
                                   args.max_new_tokens, args.batch_size, device)
    texts2, ids2 = generate_greedy(model, tokenizer, prompts,
                                   args.max_new_tokens, args.batch_size, device)
    check("1 same-session rerun", ids1 == ids2)

    print("check 2: batch-size invariance (8 vs 1)", flush=True)
    _, ids3 = generate_greedy(model, tokenizer, prompts,
                              args.max_new_tokens, 1, device)
    div = first_divergences(ids1, ids3)
    check("2 batch-size invariance", ids1 == ids3,
          "" if ids1 == ids3 else f"first-divergence positions: {div}")

    print("check 3: fresh reload", flush=True)
    del model
    torch.cuda.empty_cache()
    model, tokenizer = load_model(args.model, device)
    _, ids4 = generate_greedy(model, tokenizer, prompts,
                              args.max_new_tokens, args.batch_size, device)
    div = first_divergences(ids1, ids4)
    check("3 fresh reload", ids1 == ids4,
          "" if ids1 == ids4 else f"first-divergence positions: {div}")

    print(f"check 4: inject/restore residue (rho={args.inject_rho:g})", flush=True)
    snapshot = snapshot_weights(model)
    n_flips = inject_model(model, args.inject_rho, seed=0)
    print(f"  injected {n_flips} flips", flush=True)
    # Sanity, not a pass/fail: the fault should actually change the output,
    # otherwise check 4b would pass vacuously (restore trivially matches a
    # baseline the fault never perturbed).
    _, ids_faulty = generate_greedy(model, tokenizer, prompts,
                                    args.max_new_tokens, args.batch_size, device)
    changed = sum(1 for a, b in zip(ids1, ids_faulty) if a != b)
    print(f"  fault changed {changed}/{len(prompts)} outputs "
          f"(informational)", flush=True)
    restore_weights(model, snapshot)
    check("4a weights bitwise restored", verify_restore(model, snapshot))
    _, ids5 = generate_greedy(model, tokenizer, prompts,
                              args.max_new_tokens, args.batch_size, device)
    div = first_divergences(ids1, ids5)
    check("4b outputs identical after restore", ids1 == ids5,
          "" if ids1 == ids5 else f"first-divergence positions: {div}")

    n_fail = _results.count(False)
    print(f"\n{len(_results) - n_fail}/{len(_results)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
