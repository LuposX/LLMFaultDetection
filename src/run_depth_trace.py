"""Depth trace for faults invisible to the full probe set.

Some single-bit faults produce bit-identical responses on all 100 probes
(206 of 770 in the B1 run). This script traces where the fault's effect
dies: for each probe, healthy and faulty models run the same forward with
output_hidden_states, and every transformer layer's output (residual
stream) is compared elementwise in bf16. Recorded per probe:

  * first_layer_diff — first hidden state (embeddings = 0, layer output t
    = 1..28) containing at least one bf16 value that differs from healthy;
    null if the residual stream is bit-identical through the whole stack
  * max_diff_by_layer — max |healthy - faulty| per hidden state
  * n_diff_by_layer — number of differing bf16 values per hidden state
  * logits_bitwise_equal, max_logit_diff, argmax_equal — whether the flip
    reaches the final logits and the greedy choice at any prompt position
  * first_divergence — first differing token over the m-token greedy
    decode (consistency check; expected null for invisible faults)

A fault whose first_layer_diff stays null has an effect fully absorbed by
bf16 rounding below representable resolution; a fault with early-layer
diffs that vanish before the last layer is masked downstream. Both
outcomes decide whether such faults are detectable in principle.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.inject import (snapshot_weights, restore_weights,
                        verify_restore)  # noqa: E402
from lib.faults import load_manifest  # noqa: E402
from optimize_prompts import FaultApplier  # noqa: E402
from lib.loading import load_model  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_ID = "Qwen/Qwen3-0.6B"


def hidden_scan(model, ids, applier):
    """One forward healthy and faulty; compare per-layer hidden states."""
    with torch.no_grad():
        out_h = model(input_ids=ids.unsqueeze(0),
                      output_hidden_states=True)
        applier.apply(model)
        try:
            out_f = model(input_ids=ids.unsqueeze(0),
                          output_hidden_states=True)
        finally:
            applier.undo(model)

    layers = out_h.hidden_states          # tuple: emb, layer1..N
    first_diff = None
    max_diff, n_diff = [], []
    for t, (h, f) in enumerate(zip(layers, out_f.hidden_states)):
        d = (h.float() - f.float()).abs()
        nz = int((h != f).sum().item())
        max_diff.append(float(d.max().item()))
        n_diff.append(nz)
        if nz and first_diff is None:
            first_diff = t

    logits_h, logits_f = out_h.logits[0], out_f.logits[0]
    logits_eq = bool(torch.equal(logits_h, logits_f))
    max_logit = float((logits_h.float() - logits_f.float()).abs().max().item())
    argmax_eq = bool(torch.equal(logits_h.argmax(-1), logits_f.argmax(-1)))
    return {
        "first_layer_diff": first_diff,
        "max_diff_by_layer": max_diff,
        "n_diff_by_layer": n_diff,
        "logits_bitwise_equal": logits_eq,
        "max_logit_diff": max_logit,
        "argmax_equal": argmax_eq,
    }


def decode_check(model, ids, applier, tokenizer, m):
    """Greedy decode healthy vs faulty; first differing token or null."""
    with torch.no_grad():
        h = model.generate(ids.unsqueeze(0), max_new_tokens=m,
                           do_sample=False,
                           pad_token_id=tokenizer.eos_token_id)[0]
        applier.apply(model)
        try:
            f = model.generate(ids.unsqueeze(0), max_new_tokens=m,
                               do_sample=False,
                               pad_token_id=tokenizer.eos_token_id)[0]
        finally:
            applier.undo(model)
    L = ids.numel()
    for t in range(L, min(h.numel(), f.numel())):
        if h[t] != f[t]:
            return t - L
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scopes", default="mantissa,exponent")
    ap.add_argument("--manifest", default="data/fault_manifest.json")
    ap.add_argument("--invisible-file", default="results/exp_random.jsonl",
                    help="B1 run used to select the faults invisible to "
                         "the full 100-probe set; pass an empty string "
                         "to trace every k=1 manifest fault instead")
    ap.add_argument("--probes", type=int, default=5,
                    help="probes per fault (spread over the prompt file)")
    ap.add_argument("--m", type=int, default=64,
                    help="decode length for the divergence check")
    ap.add_argument("--prompt-file",
                    default=os.path.join(ROOT, "data/prompts.jsonl"))
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/depth_trace.jsonl")
    args = ap.parse_args()
    args.device = args.device or ("cuda" if torch.cuda.is_available()
                                  else "cpu")

    scopes = [s.strip() for s in args.scopes.split(",")]
    all_faults = load_manifest(args.manifest, model_name=args.model)
    faults = [f for f in all_faults if f.k == 1 and f.scope in scopes]
    if args.invisible_file:
        invisible = set()
        for l in open(args.invisible_file):
            r = json.loads(l)
            if not any(r["detected"]):
                invisible.add((r["scope"], r["seed"]))
        faults = [f for f in faults if (f.scope, f.seed) in invisible]
    print(f"faults to trace: {len(faults)} (k=1, scopes {scopes})")

    prompts = [json.loads(l)["prompt"] for l in open(args.prompt_file)]
    n_probes = min(args.probes, len(prompts))
    probe_idx = [round(i * (len(prompts) - 1) / (n_probes - 1))
                 for i in range(n_probes)] if n_probes > 1 else [0]

    model, tokenizer = load_model(args.model, args.device)
    snapshot = snapshot_weights(model)
    eos = tokenizer.eos_token_id

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    new_file = not os.path.exists(args.out)
    with open(args.out, "a" if not new_file else "w") as out:
        for fault in faults:
            applier = FaultApplier(fault, "bf16")
            probes = []
            for pi in probe_idx:
                ids = tokenizer(prompts[pi], return_tensors="pt") \
                    .input_ids[0].to(args.device)
                rec = hidden_scan(model, ids, applier)
                rec["probe_idx"] = pi
                rec["first_divergence"] = decode_check(
                    model, ids, applier, tokenizer, args.m)
                probes.append(rec)
            fd = [p["first_layer_diff"] for p in probes]
            ever = [p["first_layer_diff"] is not None for p in probes]
            logits_eq = all(p["logits_bitwise_equal"] for p in probes)
            rec = {
                "scope": fault.scope, "seed": fault.seed,
                "fault_id": fault.id,
                "flips": [[n, int(e), int(v)] for n, e, v in fault.flips],
                "n_probes": len(probes), "probes": probes,
                "first_layer_min": min((t for t in fd if t is not None),
                                       default=None),
                "first_layer_max": max((t for t in fd if t is not None),
                                       default=None),
                "residual_ever_differs": any(ever),
                "logits_bitwise_equal_all": logits_eq,
                "platform": {"device": str(args.device),
                             "torch": torch.__version__},
            }
            out.write(json.dumps(rec) + "\n")
            out.flush()
            tag = (f"layer {rec['first_layer_min']}-{rec['first_layer_max']}"
                   if rec["residual_ever_differs"] else "no effect")
            print(f"{fault.scope} seed={fault.seed} {tag} "
                  f"logits_equal={logits_eq}", flush=True)

    restore_weights(model, snapshot)
    print(f"done; weights restored: {verify_restore(model, snapshot)}")


if __name__ == "__main__":
    main()
