"""HotFlip standalone baseline (experiment B2).

Gradient-guided discrete search as a METHOD, not as a refinement stage:
start from probe prompts (no differentiable pre-optimization) and
iteratively apply the best gradient-ranked token substitution, verifying
candidates on the real bf16 decode. This isolates what direct discrete
search achieves before any soft-stage machinery.

Budget matching: the random baseline (B1) spends 100 probes x 2 free-runs
of up to 65 forward steps ≈ 13,000 row-equivalents per fault (--budget-rows
default). HotFlip receives the same budget: it greedily optimizes
--hf-starts prompts (evenly spaced over the probe set, deterministic),
--hf-steps steps each, stopping early when the measured row-equivalent
budget (--budget-rows) is exhausted or a start detects the fault. Compute
is counted exactly by a forward hook (src/lib/compute.py), so the
comparison against other methods is on measured, not planned, cost.

Output: one JSONL record per fault with per-start results and the compute
accounting.
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.cache import platform_tag  # noqa: E402
from lib.compute import ComputeCounter  # noqa: E402
from lib.faults import load_manifest  # noqa: E402
from lib.inject import snapshot_weights, verify_restore  # noqa: E402
from lib.loading import load_model  # noqa: E402
from optimize_prompts import FaultApplier, first_divergence, free_run  # noqa: E402
from run_pattern_comparison import first_token_margin, hotflip_search  # noqa: E402

MODEL_ID = "Qwen/Qwen3-0.6B"


def free_run_verdict(model, ids, applier, m, eos):
    """Ground-truth detection: free-run divergence under the exact decode
    path of the deployment criterion."""
    h = free_run(model, ids, m, eos)
    applier.apply(model)
    try:
        f = free_run(model, ids, m, eos)
    finally:
        applier.undo(model)
    return first_divergence(h, f)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scopes", default="mantissa,exponent")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--faults", type=int, default=20)
    ap.add_argument("--manifest", default="data/fault_manifest.json",
                    help="fault manifest; the k-sweep manifest plus --ks "
                         "selects a different k stratum")
    ap.add_argument("--ks", default=None,
                    help="comma-separated subset of manifest k values")
    ap.add_argument("--n-probes", type=int, default=100)
    ap.add_argument("--prompt-file", default="data/prompts.jsonl")
    ap.add_argument("--hf-starts", type=int, default=8)
    ap.add_argument("--hf-steps", type=int, default=150,
                    help="max HotFlip steps per start prompt")
    ap.add_argument("--hf-width", type=int, default=4)
    ap.add_argument("--budget-rows", type=int, default=13000,
                    help="row-equivalent compute budget per fault, matched "
                         "to the random baseline's 100-probe free-run cost")
    ap.add_argument("--m", type=int, default=32,
                    help="max new tokens for verification free-runs")
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/exp_hotflip.jsonl")
    args = ap.parse_args()
    args.device = args.device or ("cuda" if torch.cuda.is_available()
                                  else "cpu")
    scopes = [s.strip() for s in args.scopes.split(",") if s.strip()]

    prompts = [json.loads(line)["prompt"]
               for line in open(args.prompt_file) if line.strip()]
    k_vals = ([int(k) for k in args.ks.split(",")]
              if args.ks else None)
    faults = [f for f in load_manifest(args.manifest, model_name=args.model)
              if f.scope in scopes
              and (k_vals is None or f.k in k_vals)
              and args.seed0 <= f.seed < args.seed0 + args.faults]
    assert faults, "no manifest faults match --scopes/--seed0/--faults/--ks"

    model, tokenizer = load_model(args.model, args.device)
    snapshot = snapshot_weights(model)
    counter = ComputeCounter()
    counter.attach(model)
    eos = tokenizer.eos_token_id

    start_idx = [i * len(prompts) // args.hf_starts
                 for i in range(args.hf_starts)]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as out:
        for fault in faults:
            t0 = time.time()
            c0 = counter.as_dict()   # per-fault compute delta baseline
            applier = FaultApplier(fault, "bf16")
            starts = []
            detected_any = None
            stop_reason = "starts-exhausted"
            for si, pi in enumerate(start_idx):
                rows_before = counter.row_equiv()
                start_ids = tokenizer(prompts[pi], return_tensors="pt") \
                    .input_ids[0].to(args.device)
                best_ids, n_eval, n_flips = hotflip_search(
                    model, start_ids, applier, args, tokenizer,
                    counter=counter)
                d = free_run_verdict(model, best_ids, applier,
                                     args.m, eos)
                margin = first_token_margin(model, best_ids, applier)[0]
                rec_start = {
                    "probe_idx": pi, "probe_prompt": prompts[pi],
                    "evals": n_eval, "flips": n_flips,
                    "detected": d is not None, "first_divergence": d,
                    "final_margin": margin,
                    "rows": counter.row_equiv() - rows_before,
                    "prompt_text": tokenizer.decode(
                        best_ids, skip_special_tokens=True),
                }
                starts.append(rec_start)
                if d is not None:
                    detected_any = d
                    stop_reason = f"detected-at-start-{si}"
                    break
                if counter.row_equiv() - c0["row_equiv"] >= args.budget_rows:
                    stop_reason = "budget-exhausted"
                    break

            c1 = counter.as_dict()
            compute = {k: c1[k] - c0[k]
                       for k in ("rows", "backwards", "row_equiv")}
            compute["elapsed_s"] = round(time.time() - t0, 1)

            record = {
                "model": args.model, "scope": fault.scope,
                "seed": fault.seed, "fault": fault.to_dict(),
                "method": "hotflip",
                "hf_starts": args.hf_starts, "hf_steps": args.hf_steps,
                "hf_width": args.hf_width, "budget_rows": args.budget_rows,
                "m": args.m, "margin": args.margin,
                "starts": starts, "detected": detected_any is not None,
                "first_divergence": detected_any,
                "stop_reason": stop_reason,
                "platform": platform_tag(),
                "compute": compute,
            }
            out.write(json.dumps(record) + "\n")
            out.flush()
            print(f"{fault.scope} seed={fault.seed:3d} fault={fault.id} "
                  f"hotflip={'DET' if detected_any is not None else 'no'}"
                  f" d={detected_any} starts={len(starts)} "
                  f"rows={record['compute']['row_equiv']} "
                  f"({record['compute']['elapsed_s']}s)", flush=True)

    rows_total = counter.as_dict()["row_equiv"]
    print(f"total: {rows_total} row-equivalents, "
          f"platform {platform_tag()['device']}", flush=True)


if __name__ == "__main__":
    main()
