"""Measure deterministic exact-match coverage by fault scope.

For each sampled single-bit fault, run the fixed probe set against the healthy
and faulty model. A fault is detected by a probe when its greedy generated
continuation differs from the healthy continuation. The output stores the
per-prompt detection vector so the plot can show coverage as the probe budget
increases.

This is the deterministic baseline only. It does not use semantic judges,
multiple stochastic trials, or optimized prompts.
"""

import argparse
import json
import os
import sys
import time

import torch

# Allow execution as: python src/run_fault_coverage.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.faults import sample_fault  # noqa: E402
from lib.generation import generate_greedy  # noqa: E402
from lib.inject import snapshot_weights, restore_weights, verify_restore  # noqa: E402
from lib.loading import load_model  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_ID = "Qwen/Qwen3-0.6B"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scopes", default="mantissa,exponent",
                    help="fault scopes to compare")
    ap.add_argument("--faults", type=int, default=20,
                    help="number of single-bit faults per scope")
    ap.add_argument("--seed", type=int, default=0,
                    help="first fault seed in each scope")
    ap.add_argument("--prompt-file",
                    default=os.path.join(ROOT, "data/prompts.jsonl"))
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=os.path.join(
        ROOT, "results/fault_coverage.jsonl"))
    args = ap.parse_args()
    args.device = args.device or ("cuda" if torch.cuda.is_available()
                                   else "cpu")
    scopes = [s.strip() for s in args.scopes.split(",") if s.strip()]

    prompts = [json.loads(line)["prompt"]
               for line in open(args.prompt_file) if line.strip()]
    model, tokenizer = load_model(args.model, args.device)
    snapshot = snapshot_weights(model)

    print(f"healthy reference: {len(prompts)} prompts", flush=True)
    _, healthy = generate_greedy(model, tokenizer, prompts,
                                 args.max_new_tokens, args.batch,
                                 args.device)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as out:
        for scope in scopes:
            for seed in range(args.seed, args.seed + args.faults):
                fault = sample_fault(model, 1, seed, scope)
                t0 = time.time()
                fault.apply(model)
                try:
                    _, faulty = generate_greedy(
                        model, tokenizer, prompts, args.max_new_tokens,
                        args.batch, args.device)
                finally:
                    fault.undo(model)
                    if not verify_restore(model, snapshot):
                        raise RuntimeError("weights did not restore cleanly")

                detected = [h != f for h, f in zip(healthy, faulty)]
                record = {
                    "model": args.model,
                    "scope": scope,
                    "seed": seed,
                    "fault": fault.to_dict(),
                    "n_prompts": len(prompts),
                    "detected": detected,
                    "n_detected": sum(detected),
                    "elapsed_s": round(time.time() - t0, 1),
                }
                out.write(json.dumps(record) + "\n")
                out.flush()
                print(f"{scope:8s} seed={seed:4d} fault={fault.id} "
                      f"detected={sum(detected)}/{len(detected)} "
                      f"({record['elapsed_s']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
