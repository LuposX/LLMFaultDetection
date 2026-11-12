"""First ATPG optimization for the deterministic detection pipeline.

One known fault (lib/faults), one learnable soft prompt (lib/sdot): the
prompt is optimized so that the fault's next-token predictions deviate
maximally from the healthy model's along a teacher-forced continuation.
Then the prompt is discretized (argmax) and detection is validated in
free-running greedy decoding — the exact-match criterion of the
deterministic pipeline.

Selectable objective (--objective):
  js     : maximize mean Jensen-Shannon divergence of the next-token
           distributions (dense baseline; can saturate on tail mass and
           never flip the argmax — see results/optimization_*.json).
  margin : hinge on the logit MARGIN t*-vs-top between healthy and faulty
           (margin_loss). Targets the discrete argmax-flip criterion
           directly; loss is 0 exactly when the faulty argmax beats the
           healthy pick by --margin. --margin sets the hinge bias.

--dtype bf16 | fp32:
  fp32 promotes the model to fp32 before fault sampling (so the fault is
  defined in fp32 representation), optimizes with un-quantized gradient
  signal, then also reports the bf16 deployment verdict by rounding the
  faulted fp32 model back down and free-running it. bf16 (default) runs
  everything at the deployment precision.

Steps:
  0. fault selection: --fault-seed if given, else scan seeds and take the
     first fault the 100 random probe prompts do NOT detect — so a
     success cannot be an accident. The scan doubles as the random-prompt
     baseline (detection count per seed).
  1. teacher prefix: the healthy model's greedy continuation of the
     initial prompt (M tokens), fixed for the whole run. Both passes are
     compared at the M positions that predict these tokens.
  2. optimization: Adam on Z, loss = objective(healthy, faulty) averaged
     over the M positions; fault applied/undone around the faulty pass
     (one model, no copy). tau anneals --tau-start -> --tau-end.
  3. discretization: argmax over Z -> real token ids.
  4. validation: free-running greedy on healthy and faulty model;
     detection = outputs differ. Mid-run checks at --check-every.

For fp32 runs the top-level 'detected' verdict is on the optimization
model's own dtype; an fp32 run additionally records a 'bf16' verdict
from the faulted model rounded down to deployment precision. The record
(curve, fault, discretized prompt, verdict) goes to
results/optimization_<fault id>_<objective>_<dtype>.json.
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
from lib.inject import unique_parameters  # noqa: E402
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


def margin_loss(logits_h, logits_f, delta):
    """Per-position argmax-flip hinge loss between healthy (H) and faulty
    (F) logit vectors. Returns a scalar to MINIMIZE.

    Detection under token match is simply argmax(H) != argmax(F). The
    discrete non-differentiable condition is replaced with a margin:
    give the faulted model credit for how strongly its OWN top token beats
    the healthy model's prediction.

        t*   = argmax(H)                    (the healthy prediction)
        score_h = F[t*]                      (what F gives the healthy pick)
        best_other = max_{t != t*} F[t]      (F's best NON-healthy pick)
        M      = score_h - best_other        (margin: >0 means F still
                                              prefers the healthy pick)
        loss   = max(0, M + delta)

    A positive margin M (> 0) means F still predicts the healthy token
    (argmax did not flip) -> loss > 0, gradient pushes F[t*] down and the
    competing non-healthy top up until the flip happens. The loss is 0
    once the faulty argmax is a non-healthy token beating t* by `delta`.

    best_other must EXCLUDE t*: max over all F would otherwise equal
    F[t*] whenever t* is still the faulty argmax, making M identically 0
    -> the two gradient terms would cancel exactly (a zero-gradient
    saddle that pins loss at `delta` forever). Masking t* restores the
    gradient that separates the healthy pick from the faulty runner-up.
    Unlike JS this targets the discrete criterion directly and does not
    saturate on tail mass.
    """
    hf = logits_h.float()
    ff = logits_f.float()
    t_star = hf.argmax(-1)                       # healthy prediction per pos
    score_h = ff.gather(-1, t_star.unsqueeze(-1)).squeeze(-1)
    # faulty's best score over tokens EXCLUDING the healthy prediction
    other = ff.masked_fill(
        torch.nn.functional.one_hot(t_star, ff.shape[-1]).bool(),
        -float("inf"))
    best_other, _ = other.max(-1)
    m = score_h - best_other                      # +: F still predicts t*
    return torch.relu(m + delta).mean()


def model_to_fp32(model):
    """In-place conversion of all float parameters to fp32 (optimization
    pass). Uses unique_parameters so aliased/tied storage (embedding and
    lm_head) is converted once. Returns the model. Buffers are left
    alone; the fault injector only touches parameters.
    """
    for name, p in unique_parameters(model):
        if p.dtype.is_floating_point and p.dtype != torch.float32:
            p.data = p.data.float()
    return model


def model_to_bf16(model):
    """Cast a (fp32 optimization) model back to bf16 for deployment-style
    free-run validation."""
    for name, p in unique_parameters(model):
        if p.dtype.is_floating_point and p.dtype != torch.bfloat16:
            p.data = p.data.bfloat16()
    return model


class FaultApplier:
    """dtype-aware fault application. bf16 uses Fault's byte-XOR directly
    (the deployment bit layout). fp32 mirrors the same bf16 corruption as a
    value: because the fp32 model is converted from bf16, each fp32 element
    equals float(bf16_value), so casting it back to bf16, flipping `bit`, and
    casting to fp32 reproduces exactly the weight a bf16 deployment would
    run after the fault — while the optimizer sees un-quantized gradient
    signal.
    """
    def __init__(self, fault, dtype):
        self.fault = fault
        self.dtype = dtype

    def _names(self, model):
        return {name: p for name, p in unique_parameters(model)}

    def apply(self, model):
        if self.dtype == "bf16":
            self.fault.apply(model)
            return
        by_name = self._names(model)
        for name, elem, bit in self.fault.flips:
            p = by_name[name]
            flat = p.data.view(-1)          # elem is a linear buffer index
            cur = flat[elem]
            bf16 = cur.to(torch.bfloat16)
            bf16 = bf16.view(torch.int16) ^ (1 << bit)
            flat[elem] = bf16.view(torch.bfloat16).to(torch.float32)

    def undo(self, model):
        if self.dtype == "bf16":
            self.fault.undo(model)
            return
        # same value-level flip twice returns the original value
        self.apply(model)


class FP32FaultApplier(FaultApplier):
    pass  # keeps the class name discoverable (== FaultApplier behaviour)


def first_divergence(a, b):
    """Position of the first differing token between two id lists
    (None if identical; a shared prefix of different lengths diverges
    right after the prefix)."""
    if a == b:
        return None
    n = min(len(a), len(b))
    return next((j for j in range(n) if a[j] != b[j]), n)


def probe_activation(model, tokenizer, probes, applier, batch, device):
    """Max |dlogit| over the probe prompts, bf16 outputs compared in fp32.

    The fault's raw visibility: how much it moves the logits at all. A
    fault with activation exactly 0 is masked by the deployment
    precision itself (its effect is below bf16 rounding everywhere) —
    NO prompt can detect it, so it is excluded from ATPG targets.
    """
    worst = 0.0
    order = sorted(range(len(probes)), key=lambda i: len(probes[i]),
                   reverse=True)
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        enc = tokenizer([probes[i] for i in idx], return_tensors="pt",
                        padding=True).to(device)
        with torch.no_grad():
            h = model(**enc).logits.float()
            applier.apply(model)
            f = model(**enc).logits.float()
            applier.undo(model)
        worst = max(worst, (h - f).abs().max().item())
    return worst


def select_fault(model, tokenizer, probes, healthy_ids, args):
    """Return (fault, n_random_detections, activation).

    Two-phase scan: (1) cheap activation prefilter per seed — faults
    whose logits never move are masked at the model precision and skipped;
    (2) among live faults, generate with the random probes and take the
    first seed with zero detections (else the least-detected live fault).
    """
    # fault selection always runs on the bf16 deployment model (conversion
    # to fp32, when requested, happens only for the optimization pass), so
    # the scans use the bf16 byte-level applier, not args.dtype.
    applier = FaultApplier
    mkl = lambda f: applier(f, "bf16")  # noqa
    live = []
    seeds = ([args.fault_seed] if args.fault_seed is not None
             else range(args.scan_seeds))
    for seed in seeds:
        fault = sample_fault(model, args.k, seed, args.scope)
        act = probe_activation(model, tokenizer, probes, mkl(fault),
                               args.batch, args.device)
        print(f"scan seed {seed}: fault {fault.id} — activation "
              f"{act:.3e}", flush=True)
        if act > 0:
            live.append((fault, act))
        elif args.fault_seed == seed:
            print("  requested fault is masked at model precision — its "
                  "effect is below the precision floor everywhere",
                  flush=True)
    if not live:
        raise RuntimeError("all scanned faults are masked — no "
                           "ATPG-detectable fault found; increase k or "
                           "scan more seeds")

    # detection counts for all live faults, then pick: among the ones the
    # random probes miss, the loudest activation gives the optimizer the
    # strongest gradient signal (a one-ulp fault is technically live but
    # its objective surface is too flat to climb — measured, see
    # optimization_69d36416964e.json)
    live_det = []
    for fault, act in live:
        fn = mkl(fault)
        fn.apply(model)
        _, faulty = generate_greedy(model, tokenizer, probes,
                                    args.probe_tokens, args.batch, args.device)
        fn.undo(model)
        n_det = sum(h != f for h, f in zip(healthy_ids, faulty))
        print(f"live fault {fault.id} (activation {act:.3e}): "
              f"{n_det}/{len(probes)} random probes detect it", flush=True)
        live_det.append((fault, act, n_det))

    if args.fault_seed is not None:
        # return the SPECIFIC requested seed's fault (live_det is built in
        # scan order, so [0] would silently pick an arbitrary earlier seed).
        for f, act, n_det in live_det:
            if f.seed == args.fault_seed:
                return f, n_det, act
        raise RuntimeError(f"requested fault seed {args.fault_seed} is "
                           f"masked at model precision; skip it or "
                           f"increase k / scan more seeds")

    undetected = [(act, f, n_det) for f, act, n_det in live_det if n_det == 0]
    if undetected:
        act, fault, n_det = max(undetected, key=lambda x: x[0])
        return fault, n_det, act
    print("no live fault is missed by the random probes — using the "
          "least-detected one (baseline not zero)", flush=True)
    fault, act, n_det = min(live_det, key=lambda x: x[2])
    return fault, n_det, act


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--scope", default="mantissa",
                    choices=["all", "exponent", "mantissa", "sign"])
    ap.add_argument("--k", type=int, default=1,
                    help="flips per fault (1 = single-bit upset)")
    ap.add_argument("--objective", default="js",
                    choices=["js", "margin"],
                    help="optimization objective (default js, the "
                         "dense-divergence baseline; margin targets the "
                         "argmax-flip detection criterion directly)")
    ap.add_argument("--margin", type=float, default=1.0,
                    help="safety margin for the margin objective (delta in "
                         "logit units; the flip must clear the healthy pick "
                         "by delta for loss == 0)")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"],
                    help="precision of the optimization pass. bf16 matches "
                         "the deployment (margin gradients quantized to "
                         "2^-8); fp32 converts the model to fp32, samples "
                         "the fault in fp32 representation, optimizes at "
                         "fp32, then casts back to bf16 for free-run "
                         "validation (recovers gradient signal lost to the "
                         "bf16 mantissa floor)")
    ap.add_argument("--fault-seed", type=int, default=None,
                    help="fixed fault seed; default: scan for a fault the "
                         "random probes do not detect")
    ap.add_argument("--scan-seeds", type=int, default=8)
    ap.add_argument("--scan-only", action="store_true",
                    help="print activation + random-probe detections for "
                         "the scanned faults, then exit (no optimization)")
    ap.add_argument("--init", type=int, default=0,
                    help="index of the probe prompt to initialize Z from; "
                         "-1 = random token init (--init-seed)")
    ap.add_argument("--init-seed", type=int, default=0)
    ap.add_argument("--positions", default="first",
                    choices=["first", "prefix"],
                    help="which positions the loss compares: 'first' = only "
                         "the first generated token (the exact free-run "
                         "detection point, no teacher prefix); 'prefix' = "
                         "the legacy teacher-forced prefix (averaged over M "
                         "positions, washes the fault out with depth)")
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
    fault, n_random_det, activation = select_fault(model, tokenizer, probes,
                                                   healthy_probe_ids, args)
    if args.scan_only:
        return
    # fp32 mode: fault selection + baseline ran in bf16 (the deployment
    # truth); promote the model to fp32 now so the OPTIMIZATION pass gets
    # un-quantized gradient signal. The FaultApplier mirrors each bf16
    # fault as a value in the fp32 weights, so the optimized prompt is
    # built against the same corruption the bf16 deployment runs.
    if args.dtype == "fp32":
        model = model_to_fp32(model)
    applier = FaultApplier(fault, args.dtype)
    print(f"target fault: {fault.id} ({fault.scope}, k={fault.n_flips}) — "
          f"activation {activation:.3e}, random-prompt baseline: "
          f"{n_random_det}/{len(probes)} ({time.time() - t0:.0f}s)", flush=True)

    # --- step 1: init Z + (prefix mode) teacher prefix -------------------
    if args.init >= 0:
        init_ids = tokenizer(probes[args.init], return_tensors="pt")\
            .input_ids[0].to(args.device)
    else:
        g = torch.Generator().manual_seed(args.init_seed)
        init_ids = torch.randint(0, vocab, (args.n,), generator=g)\
            .to(args.device)
        print(f"random init: {init_ids.tolist()}", flush=True)
    if args.positions == "prefix":
        prefix_ids = torch.tensor(free_run(model, init_ids, args.m, eos),
                                  device=args.device)
        print(f"teacher prefix: {args.m} tokens (healthy greedy continuation "
              f"of probe {args.init})", flush=True)
    else:
        prefix_ids = None

    # --- step 2: optimization -------------------------------------------
    Z = one_hot_logits(init_ids, vocab, amplitude=AMPLITUDE).requires_grad_(True)
    opt = torch.optim.Adam([Z], lr=args.lr)
    curve = []
    print(f"\noptimizing: {args.steps} steps, tau {args.tau_start} -> "
          f"{args.tau_end}, lr {args.lr}, objective {args.objective}, "
          f"dtype {args.dtype}, margin {args.margin}", flush=True)

    def free_run_check(disc, ctx):
        """Discretize Z, free-run healthy vs faulted, return divergence.
        The model is currently clean (fault undone). ctx is a label for
        logging (e.g. 'fp32' / 'bf16')."""
        h = free_run(model, disc, args.m, eos)
        applier.apply(model)
        f = free_run(model, disc, args.m, eos)
        applier.undo(model)
        return first_divergence(h, f)

    if args.positions == "prefix":
        pos = slice(Z.shape[0] - 1, Z.shape[0] - 1 + len(prefix_ids))
    else:
        # first generated token: the last soft prompt position predicts it
        pos = slice(Z.shape[0] - 1, Z.shape[0])
    gnorms = []
    for step in range(args.steps + 1):
        tau = args.tau_start + (args.tau_end - args.tau_start) * step / args.steps
        if args.positions == "prefix":
            applier.apply(model)
            faulty_logits = soft_forward_prefix(model, Z, tau, prefix_ids)
            applier.undo(model)
            healthy_logits = soft_forward_prefix(model, Z, tau, prefix_ids)
        else:
            applier.apply(model)
            faulty_logits = soft_forward(model, Z, tau)
            applier.undo(model)
            healthy_logits = soft_forward(model, Z, tau)
        hlf = healthy_logits[pos]
        flf = faulty_logits[pos]
        if args.objective == "margin":
            loss = margin_loss(hlf, flf, args.margin)
            js = js_positions(hlf, flf)   # logged as a side metric only
        else:
            js = js_positions(hlf, flf)
            loss = -js.mean()
        opt.zero_grad()
        loss.backward()
        gnorm = Z.grad.norm().item()
        gnorms.append(gnorm)
        opt.step()

        if step % args.log_every == 0 or step == args.steps:
            js_val = js.mean().item()
            entry = {"step": step, "tau": tau, "js": js_val}
            entry["gnorm"] = gnorm
            msg = f"step {step:4d}  tau={tau:.3f}  JS={js_val:.4f}" \
                f"  |gZ|={gnorm:.2e}"
            if args.objective == "margin":
                msg += f"  loss={loss.item():.4f}"
                entry["loss"] = round(loss.item(), 6)
            if step % args.check_every == 0 or step == args.steps:
                with torch.no_grad():
                    disc = Z.argmax(-1)
                d = free_run_check(disc, args.dtype)
                msg += f"  free-run: {'DETECTS (div @ ' + str(d) + ')' \
                    if d is not None else 'no divergence'}"
            curve.append(entry)
            print(msg, flush=True)

    # --- steps 3+4: discretize + validate --------------------------------
    with torch.no_grad():
        disc = Z.argmax(-1)
    prompt_text = tokenizer.decode(disc, skip_special_tokens=True)

    # primary verdict: on the optimization model's own dtype
    d = free_run_check(disc, args.dtype)
    detected = d is not None
    print(f"\nfinal: discretized prompt = {disc.tolist()}", flush=True)
    print(f"  text: {prompt_text!r}", flush=True)
    print(f"  free-running detection ({args.dtype}): "
          + (f"YES, first divergence at position {d}" if detected
             else "NO — outputs identical"), flush=True)

    # fp32 optimization also reports the bf16 deployment verdict: how the
    # faulted fp32 model behaves once rounded down to the deployment dtype.
    bf16 = {}
    if args.dtype == "fp32":
        applier.apply(model)    # fault the fp32 weights
        model_to_bf16(model)    # round the whole model down to bf16
        hb = free_run(model, disc, args.m, eos)
        fb = free_run(model, disc, args.m, eos)
        db = first_divergence(hb, fb)
        bf16["detected"] = db is not None
        bf16["first_divergence"] = db
        model_to_fp32(model)    # round back to the fp32 optimization state
        applier.undo(model)     # un-fault (value-level, fp32-safe)
        print(f"  free-running detection (bf16 deployment): "
              + (f"YES, first divergence at position {db}" if db is not None
                 else "NO — outputs identical"), flush=True)

    n_changed = int((disc.cpu() != init_ids.cpu()).sum().item())
    print(f"  prompt tokens changed from init: {n_changed}/{len(disc)}")

    out = os.path.join(ROOT, f"results/optimization_{fault.id}"
                               f"_{args.objective}_{args.dtype}.json")
    with open(out, "w") as fh:
        json.dump({"model": args.model, "args": vars(args),
                   "fault": fault.to_dict(),
                   "activation": activation,
                   "random_probe_detections": n_random_det,
                   "init_prompt": probes[args.init],
                   "prompt_ids": disc.tolist(),
                   "prompt_text": prompt_text,
                   "curve": curve,
                   "detected": detected,
                   "first_divergence": d,
                   "bf16": bf16,
                   "n_tokens_changed": n_changed,
                   "gnorms": gnorms}, fh, indent=2)
    print(f"record written to {out}", flush=True)
    sys.exit(0 if detected else 1)


if __name__ == "__main__":
    main()
