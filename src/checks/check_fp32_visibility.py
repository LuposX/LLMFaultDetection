"""fp32 recheck of the invisible faults found by the depth trace.

The depth-trace pilot classified 206 faults invisible to the 100-probe
set; 12 leave the bf16 logits bit-identical. This check reruns the same
bf16 fault under fp32 arithmetic on CPU: weights are loaded in bf16, the
fault is applied, and both healthy and faulty snapshots are cast to fp32
(bf16 -> fp32 is lossless, so the fault is the identical bit flip). If
the fp32 logits differ, the fault's effect is mathematically present but
below the bf16 resolution of the logits; if they remain identical, the
effect is exactly zero (masked), and the fault is invisible at any
precision.

The fault sits in bf16 weight bits, so flipping bits after the fp32 cast
would change the fault; the fault is applied in bf16 and then cast.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.inject import snapshot_weights, restore_weights  # noqa: E402
from lib.faults import load_manifest  # noqa: E402
from optimize_prompts import FaultApplier  # noqa: E402
from lib.loading import load_model  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
MODEL_ID = "Qwen/Qwen3-0.6B"


def fp32_compare(model, ids, applier):
    """Forward healthy and faulty, both in fp32; compare logits.

    applier must be FaultApplier(fault, "fp32") so the flip mirrors the
    bf16 corruption at value level on the fp32 weights.
    """
    with torch.no_grad():
        h = model(input_ids=ids.unsqueeze(0)).logits[0]
        applier.apply(model)
        try:
            f = model(input_ids=ids.unsqueeze(0)).logits[0]
        finally:
            applier.undo(model)
    max_diff = float((h - f).abs().max().item())
    argmax_eq = bool(torch.equal(h.argmax(-1), f.argmax(-1)))
    return {"max_logit_diff_fp32": max_diff,
            "argmax_equal_fp32": argmax_eq}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace-file", default="results/depth_trace.jsonl",
                    help="depth-trace records; when the canonical run has "
                         "not finished yet the pilot is used")
    ap.add_argument("--pilot", default="results/depth_trace_pilot_1probe_m8.jsonl")
    ap.add_argument("--manifest", default="data/fault_manifest.json")
    ap.add_argument("--prompt-file",
                    default=os.path.join(ROOT, "data/prompts.jsonl"))
    ap.add_argument("--probe-idx", type=int, default=0)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--out", default="results/depth_trace_fp32.jsonl")
    args = ap.parse_args()

    src = args.trace_file
    if not os.path.exists(src):
        src = args.pilot
    recs = [json.loads(l) for l in open(src)]
    print(f"rechecking {len(recs)} faults from {src}")

    manifest = {f.id: f for f in
                load_manifest(args.manifest, model_name=args.model)}
    prompts = [json.loads(l)["prompt"] for l in open(args.prompt_file)]

    model, tokenizer = load_model(args.model, "cpu")
    model.float()                      # fault is mirrored at value level
    snapshot = snapshot_weights(model)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as out:
        for rec in recs:
            fault = manifest[rec["fault_id"]]
            applier = FaultApplier(fault, "fp32")
            ids = tokenizer(prompts[args.probe_idx],
                            return_tensors="pt").input_ids[0]
            res = fp32_compare(model, ids, applier)
            out.write(json.dumps({
                "scope": rec["scope"], "seed": rec["seed"],
                "fault_id": rec["fault_id"],
                "bf16_logits_bitwise_equal":
                    rec["probes"][0]["logits_bitwise_equal"],
                **res}) + "\n")
            out.flush()
    print("done")

    restore_weights(model, snapshot)


if __name__ == "__main__":
    main()
