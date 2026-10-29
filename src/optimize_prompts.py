"""First ATPG optimization for the deterministic detection pipeline.

One known fault (lib/faults), one learnable soft prompt (lib/sdot): the
prompt is optimized so that the fault's next-token predictions deviate
maximally from the healthy model's along a teacher-forced continuation.
Then the prompt is discretized (argmax) and detection is validated in
free-running greedy decoding — the exact-match criterion of the
deterministic pipeline.

Steps:
  0. fault selection: --fault-seed if given, else scan seeds and take the
     first fault the 100 random probe prompts do NOT detect — so a
     success cannot be an accident. The scan doubles as the random-prompt
     baseline (detection count per seed).
  1. teacher prefix: the healthy model's greedy continuation of the
     initial prompt (M tokens), fixed for the whole run. Both passes are
     compared at the M positions that predict these tokens.
  2. optimization: Adam on Z, loss = -JS(healthy, faulty) averaged over
     the M positions; fault applied/undone around the faulty pass (one
     model, no copy). tau anneals linearly --tau-start -> --tau-end.
  3. discretization: argmax over Z -> real token ids.
  4. validation: free-running greedy on healthy and faulty model;
     detection = outputs differ. Mid-run checks at --check-every.

The record (curve, fault, discretized prompt, verdict) goes to
results/optimization_<fault id>.json.
"""

import argparse
import json
import os
import time

import torch

# one level below src/: add src/ to the import path for the lib imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.faults import sample_fault  # noqa: E402
from lib.generation import generate_greedy  # noqa: E402
from lib.loading import load_model  # noqa: E402
from lib.sdot import (  # noqa: E402
    soft_forward_prefix, one_hot_logits, js_positions)

MODEL_ID = "Qwen/Qwen3-0.6B"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# same encoding amplitude as the equivalence check: faithful tau band
# measured for this value
AMPLITUDE = 30.0


def free_run(model, ids, max_new_tokens, eos_id):
    """Greedy continuation of a token-id prompt (no padding, no trimming:
    teacher forcing and raw comparison want the full tail)."""
    out = model.generate(ids.unsqueeze(0), max_new_tokens=max_new_tokens,
                         do_sample=False, pad_token_id=eos_id)
    return out[0, ids.numel():].tolist()


def first_divergence(a, b):
    """Position of the first differing token between two id lists
    (None if identical; a shared prefix of different lengths diverges
    right after the prefix)."""
    if a == b:
        return None
    n = min(len(a), len(b))
    return next((j for j in range(n) if a[j] != b[j]), n)


def select_fault(model, args, tokenizer, probes, healthy_ids):
    """Return (fault, n_random_detections). If no seed was given, scan
    seeds for a fault the probe prompts do not detect."""
    if args.fault_seed is not None:
        fault = sample_fault(model, args.k, args.fault_seed, args.scope)
        fault.apply(model)
        _, faulty = generate_greedy(model, tokenizer, probes,
                                    args.probe_tokens, args.batch, args.device)
        fault.undo(model)
        n_det = sum(h != f for h, f in zip(healthy_ids, faulty))
        print(f"fault {fault.id}: {n_det}/{len(probes)} random probes detect it",
              flush=True)
        return fault, n_det

    best = None
    for seed in range(args.scan_seeds):
        fault = sample_fault(model, args.k, seed, args.scope)
        fault.apply(model)
        _, faulty = generate_greedy(model, tokenizer, probes,
                                    args.probe_tokens, args.batch, args.device)
        fault.undo(model)
        n_det = sum(h != f for h, f in zip(healthy_ids, faulty))
        print(f"scan seed {seed}: fault {fault.id} — {n_det}/{len(probes)} "
              f"random probes detect it", flush=True)
        if best is None or n_det < best[1]:
            best = (fault, n_det)
        if n_det == 0:
            return fault, 0
    print(f"no undetected fault within {args.scan_seeds} seeds — using the "
          f"least-detected one (still a valid target, but the baseline is "
          f"not zero)", flush=True)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--scope", default="mantissa",
                    choices=["all", "exponent", "mantissa", "sign"])
    ap.add_argument("--k", type=int, default=1,
                    help="flips per fault (1 = single-bit upset)")
    ap.add_argument("--fault-seed", type=int, default=None,
                    help="fixed fault seed; default: scan for a fault the "
                         "random probes do not detect")
    ap.add_argument("--scan-seeds", type=int, default=8)
    ap.add_argument("--init", type=int, default=0,
                    help="index of the probe prompt to initialize Z from")
    ap.add_argument("--n", type=int, default=16,
                    help="learnable prompt length (soft positions)")
    ap.add_argument("--m", type=int, default=32,
                    help="teacher-forced continuation positions")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--tau-start", type=float, default=2.0)
    ap.add_argument("--tau-end", type=float, default=1.0,
                    help="must stay below the faithful boundary (~1.18 at "
                         "amplitude 30, see checks/check_sdot.py)")
    ap.add_argument("--probe-tokens", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--check-every", type=int, default=50)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model(args.model, args.device)
    # gradients only ever flow into Z; skipping weight-grad computation
    # saves ~1.2 GB of .grad allocations per backward pass
    for p in model.parameters():
        p.requires_grad_(False)
    vocab = model.get_input_embeddings().num_embeddings
    eos = tokenizer.eos_token_id

    probes = [json.loads(line)["prompt"]
              for line in open(os.path.join(ROOT, "data/prompts.jsonl"))
              if line.strip()]

    # --- step 0: fault + random-prompt baseline -------------------------
    print("healthy reference on probe prompts ...", flush=True)
    _, healthy_probe_ids = generate_greedy(model, tokenizer, probes,
                                           args.probe_tokens, args.batch,
                                           args.device)
    t0 = time.time()
    fault, n_random_det = select_fault(model, args, tokenizer, probes,
                                       healthy_probe_ids)
    print(f"target fault: {fault.id} ({fault.scope}, k={fault.n_flips}) — "
          f"random-prompt baseline: {n_random_det}/{len(probes)} "
          f"({time.time() - t0:.0f}s)", flush=True)

    # --- step 1: teacher prefix -----------------------------------------
    init_ids = tokenizer(probes[args.init], return_tensors="pt")\
        .input_ids[0].to(args.device)
    prefix = free_run(model, init_ids, args.m, eos)
    print(f"teacher prefix: {args.m} tokens (healthy greedy continuation "
          f"of probe {args.init})", flush=True)

    # --- step 2: optimization -------------------------------------------
    Z = one_hot_logits(init_ids, vocab, amplitude=AMPLITUDE).requires_grad_(True)
    opt = torch.optim.Adam([Z], lr=args.lr)
    curve = []
    print(f"\noptimizing: {args.steps} steps, tau {args.tau_start} -> "
          f"{args.tau_end}, lr {args.lr}", flush=True)
    for step in range(args.steps + 1):
        tau = args.tau_start + (args.tau_end - args.tau_start) * step / args.steps
        fault.apply(model)
        faulty_logits = soft_forward_prefix(model, Z, tau, prefix)
        fault.undo(model)
        healthy_logits = soft_forward_prefix(model, Z, tau, prefix)
        n = Z.shape[0]
        js = js_positions(healthy_logits[n - 1:n - 1 + len(prefix)],
                          faulty_logits[n - 1:n - 1 + len(prefix)])
        loss = -js.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % args.log_every == 0 or step == args.steps:
            js_val = js.mean().item()
            curve.append({"step": step, "tau": tau, "js": js_val})
            msg = f"step {step:4d}  tau={tau:.3f}  JS={js_val:.4f}"
            if step % args.check_every == 0 or step == args.steps:
                # discretize + free-running spot check
                with torch.no_grad():
                    disc = Z.argmax(-1)
                h = free_run(model, disc, args.m, eos)
                fault.apply(model)
                f = free_run(model, disc, args.m, eos)
                fault.undo(model)
                d = first_divergence(h, f)
                msg += f"  free-run: {'DETECTS (div @ ' + str(d) + ')' \
                    if d is not None else 'no divergence'}"
            print(msg, flush=True)

    # --- steps 3+4: discretize + validate --------------------------------
    with torch.no_grad():
        disc = Z.argmax(-1)
    h = free_run(model, disc, args.m, eos)
    fault.apply(model)
    f = free_run(model, disc, args.m, eos)
    fault.undo(model)
    d = first_divergence(h, f)
    detected = d is not None
    prompt_text = tokenizer.decode(disc, skip_special_tokens=True)

    print(f"\nfinal: discretized prompt = {disc.tolist()}", flush=True)
    print(f"  text: {prompt_text!r}", flush=True)
    print(f"  free-running detection: {'YES, first divergence at position '
          + str(d) if detected else 'NO — outputs identical'}", flush=True)

    out = os.path.join(ROOT, f"results/optimization_{fault.id}.json")
    with open(out, "w") as fh:
        json.dump({"model": args.model, "args": vars(args),
                   "fault": fault.to_dict(),
                   "random_probe_detections": n_random_det,
                   "init_prompt": probes[args.init],
                   "prompt_ids": disc.tolist(),
                   "prompt_text": prompt_text,
                   "curve": curve,
                   "detected": detected,
                   "first_divergence": d}, fh, indent=2)
    print(f"record written to {out}", flush=True)
    sys.exit(0 if detected else 1)


if __name__ == "__main__":
    main()
