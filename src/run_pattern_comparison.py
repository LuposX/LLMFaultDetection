"""Optimized vs. random test patterns under the deterministic criterion.

Experiment: for a set of sampled single-bit faults, compare the detection
rate of

  (a) ONE RANDOM PROMPT from the fixed probe set — measured as the
      per-fault fraction of the 100 probe prompts that detect the fault
      (a uniformly drawn single prompt detects with that probability), and
  (b) ONE OPTIMIZED PROMPT — SDoT soft prompt optimized against the fault
      (margin hinge on the first generated token, the exact free-run
      detection point), discretized and validated free-running.

Detection criterion everywhere: the faulty model's greedy continuation
differs from the healthy one (exact token match). No semantic judges.

Per fault one JSONL line is written immediately (flushed), so partial
runs are usable: fault record, activation, random detection vector over
the 100 probes, optimization curve summary, optimized prompt, verdict.

The optimization loop is deliberately the same construction as
optimize_prompts.py (margin objective, tau annealing, one-hot init,
free-run validation) but inlined as a function so the experiment script
is self-contained on the cluster.
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.faults import sample_fault  # noqa: E402
from lib.generation import generate_greedy  # noqa: E402
from lib.inject import (snapshot_weights, restore_weights,  # noqa: E402
                        verify_restore, unique_parameters)
from lib.loading import load_model  # noqa: E402
from lib.sdot import (soft_forward, one_hot_logits,  # noqa: E402
                      soft_embeddings)
from optimize_prompts import (  # noqa: E402
    margin_loss, FaultApplier, free_run, first_divergence, AMPLITUDE)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_ID = "Qwen/Qwen3-0.6B"


def first_token_dlogit_scan(model, tokenizer, prompts, applier, batch):
    """Per-prompt max |dlogit| at the LAST PROMPT position (the position
    that predicts the first generated token) between healthy and faulty.
    A probe with 0 here has a bitwise-identical first-token prediction
    under the fault — the position-0 margin objective has exactly zero
    gradient when optimizing from it, so such prompts are useless inits.
    """
    device = next(model.parameters()).device
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]),
                   reverse=True)
    out = [0.0] * len(prompts)
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        enc = tokenizer([prompts[i] for i in idx], return_tensors="pt",
                        padding=True).to(device)
        with torch.no_grad():
            h = model(**enc).logits[:, -1, :].float()
            applier.apply(model)
            f = model(**enc).logits[:, -1, :].float()
            applier.undo(model)
        for row, i in enumerate(idx):
            out[i] = (h[row] - f[row]).abs().max().item()
    return out


def optimize_prompt(model, init_ids, applier, args):
    """Soft-prompt optimization against one fault. Returns
    (Z argmax ids, curve, final gnorm)."""
    vocab = model.get_input_embeddings().num_embeddings
    Z = one_hot_logits(init_ids, vocab, amplitude=AMPLITUDE).requires_grad_(True)
    opt = torch.optim.Adam([Z], lr=args.lr)
    curve = []
    gnorm = None
    # loss position: the last soft prompt position predicts the FIRST
    # generated token — the same position the free-run criterion checks
    pos = slice(Z.shape[0] - 1, Z.shape[0])
    for step in range(args.steps + 1):
        tau = args.tau_start + (args.tau_end - args.tau_start) * step / args.steps
        applier.apply(model)
        faulty_logits = soft_forward(model, Z, tau)
        applier.undo(model)
        healthy_logits = soft_forward(model, Z, tau)
        loss = margin_loss(healthy_logits[pos], faulty_logits[pos], args.margin)
        opt.zero_grad()
        loss.backward()
        gnorm = Z.grad.norm().item()
        opt.step()
        if step % args.log_every == 0 or step == args.steps:
            curve.append({"step": step, "tau": round(tau, 4),
                          "loss": round(loss.item(), 6),
                          "gnorm": gnorm})
    with torch.no_grad():
        return Z.argmax(-1), curve, gnorm


def _free_run_score(model, ids, applier, m, eos):
    """Ranking tuple for a candidate prompt under the REAL bf16 decode:
    (0, first_divergence) if detected, else (1, 0). Lower is better."""
    h = free_run(model, ids, m, eos)
    applier.apply(model)
    try:
        f = free_run(model, ids, m, eos)
    finally:
        applier.undo(model)
    d = first_divergence(h, f)
    return (0 if d is not None else 1, d if d is not None else 0)


def first_token_margin(model, ids, applier):
    """Real-bf16 margin at the first generated token: F[t*] - max_{t!=t*} F[t],
    where t* is the healthy argmax. Negative margin = the faulty model's
    argmax flipped = first-token detection. This is the exact continuous
    quantity the deployment criterion flips on, and costs two short
    forward passes (no generation).
    Returns (margin, t_star).
    """
    device = next(model.parameters()).device
    emb = model.get_input_embeddings()(ids.unsqueeze(0).to(device))
    with torch.no_grad():
        h = model(inputs_embeds=emb).logits[0, -1]
    t_star = int(h.argmax())
    applier.apply(model)
    try:
        with torch.no_grad():
            f = model(inputs_embeds=emb).logits[0, -1]
    finally:
        applier.undo(model)
    f_rest = f.clone()
    f_rest[t_star] = float("-inf")
    return (f[t_star] - f_rest.max()).item(), t_star


def hotflip_search(model, disc, applier, args, tokenizer, max_evals=None,
                   counter=None):
    """HotFlip-style greedy search over token substitutions.

    One backward pass ranks every (position, token) substitution by the
    first-order directional derivative of the margin loss w.r.t. the
    one-hot input representation: the estimated loss change of replacing
    token a at position j with token b is g[j][b] - g[j][a], with
    g = dJ/d(one-hot input). The top positions by estimated improvement
    are then verified with the REAL bf16 free-run (the deployment
    criterion) and the best improving substitution is accepted; repeat.

    The ranking backward runs on the RELAXED fp32 path (bf16 activation
    rounding would zero the gradient exactly in the bands this stage
    exists for); the fault is mirrored through FaultApplier's fp32 path
    and the model is downcast afterwards (exact, must pass bitwise
    restore). The accept step runs bf16, so accepted flips transfer.

    Returns (best_ids, n_real_evaluations, n_flips_accepted).
    """
    vocab = model.get_input_embeddings().num_embeddings
    eos = tokenizer.eos_token_id
    L = disc.numel()
    device = disc.device

    def score(ids):
        margin, t_star = first_token_margin(model, ids, applier)
        # detected candidates (margin < 0) additionally rank by earliest
        # free-run divergence; not-detected candidates rank by margin only
        if margin < 0:
            h = free_run(model, ids, args.m, eos)
            applier.apply(model)
            try:
                f = free_run(model, ids, args.m, eos)
            finally:
                applier.undo(model)
            d = first_divergence(h, f)
            return (margin, d if d is not None else args.m)
        return (margin, 0)

    best_ids = disc.clone()
    best_rank = score(best_ids)
    if best_rank[0] < 0:
        return best_ids, 1, 0
    n_eval = 1
    n_flips = 0

    for step in range(args.hf_steps):
        # --- ranking pass (fp32, one backward) -------------------------
        pre_fp32 = {n: p.detach().clone()
                    for n, p in unique_parameters(model)}
        model.to(torch.float32)
        rank_applier = FaultApplier(applier.fault, "fp32")
        try:
            dists = torch.zeros(L, vocab, device=device, dtype=torch.float32)
            dists[torch.arange(L, device=device), best_ids] = 1.0
            dists.requires_grad_(True)
            emb = soft_embeddings(dists, model)
            healthy = model(inputs_embeds=emb.unsqueeze(0)).logits[0]
            rank_applier.apply(model)
            try:
                faulty = model(inputs_embeds=emb.unsqueeze(0)).logits[0]
            finally:
                rank_applier.undo(model)
            loss = margin_loss(healthy[-1], faulty[-1], args.margin)
            g = torch.autograd.grad(loss, dists)[0]          # [L, V]
            if counter is not None:
                counter.add_backward()
        finally:
            model.to(torch.bfloat16)
            for n, p in unique_parameters(model):
                if not torch.equal(p.detach(), pre_fp32[n]):
                    raise RuntimeError("restore failed after hotflip ranking")

        # estimated loss decrease for the best flip at each position
        a_tok = best_ids.long()
        g_a = g[torch.arange(L, device=device), a_tok]      # [L]
        g_min_b, b_tok = g.min(dim=-1)                      # [L], [L]
        improve = g_a - g_min_b                              # [L]
        for j in range(L):                                   # exclude identity
            if b_tok[j] == a_tok[j]:
                improve[j] = float("-inf")
        if not torch.isfinite(improve).any() or improve.max() <= 0:
            break   # first-order estimate says no flip helps: local optimum

        # --- verify the top-W candidates with the real bf16 decode ------
        order = improve.argsort(descending=True)
        accepted = False
        for j in order[:args.hf_width].tolist():
            if max_evals is not None and n_eval >= max_evals:
                return best_ids, n_eval, n_flips
            cand = best_ids.clone()
            cand[j] = b_tok[j]
            r = score(cand)
            n_eval += 1
            if r < best_rank:
                best_ids, best_rank = cand, r
                n_flips += 1
                accepted = True
                break   # greedy: apply the best accepted flip, re-rank
        if best_rank[0] < 0:
            break                   # detected: done
        if not accepted:
            break                   # no verified flip improved: local optimum
    return best_ids, n_eval, n_flips


def discrete_search(model, disc, applier, args, tokenizer, max_evals=None):
    """Forward-only local search on the discretized prompt.

    The soft optimum is limited by bf16 rounding the gradient cannot see;
    here candidate prompts are evaluated with the REAL bf16 free-run.
    Candidates: replace each position with the top-k tokens of that
    position's softmax row (from the soft logits, recomputed once).
    Returns (best_ids, n_candidates_evaluated) — best by "detects, then
    earliest divergence". max_evals bounds the total number of candidate
    evaluations (the search is O(positions x topk) free-runs otherwise).
    """
    vocab = model.get_input_embeddings().num_embeddings
    eos = tokenizer.eos_token_id

    def score(ids):
        margin, _ = first_token_margin(model, ids, applier)
        # detected candidates (margin < 0) additionally rank by earliest
        # free-run divergence; not-detected candidates rank by margin only
        if margin < 0:
            h = free_run(model, ids, args.m, eos)
            applier.apply(model)
            try:
                f = free_run(model, ids, args.m, eos)
            finally:
                applier.undo(model)
            d = first_divergence(h, f)
            return (margin, d if d is not None else args.m)
        return (margin, 0)

    best_ids = disc.clone()
    best_rank = score(disc)
    if best_rank[0] < 0:
        return best_ids, 1
    n_eval = 1

    # top-k alternates per position, from a soft forward at tau=1 seeded by
    # the current best prompt
    with torch.no_grad():
        logits = soft_forward(model, one_hot_logits(disc, vocab).float(),
                              1.0)
    topk = logits.topk(args.topk, dim=-1).indices   # [N, topk]

    for p in range(disc.numel()):
        cands = []
        for t in topk[p].tolist():
            if t != disc[p].item():
                c = disc.clone()
                c[p] = t
                cands.append(c)
        for s in range(0, len(cands), args.batch):
            batch = cands[s:s + args.batch]
            # greedy free-run, one prompt at a time (batch-1 decode keeps
            # the exact decode path of the deployment criterion)
            for c in batch:
                if max_evals is not None and n_eval >= max_evals:
                    return best_ids, n_eval
                r = score(c)
                n_eval += 1
                if r < best_rank:
                    best_ids, best_rank = c, r
                    if best_rank[0] < 0:
                        return best_ids, n_eval
    return best_ids, n_eval


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scopes", default="mantissa,exponent")
    ap.add_argument("--faults", type=int, default=20,
                    help="faults per scope, seeds seed0 .. seed0+faults-1")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--scope-k", type=int, default=1,
                    help="flips per fault")
    ap.add_argument("--prompt-file",
                    default=os.path.join(ROOT, "data/prompts.jsonl"))
    ap.add_argument("--max-new-tokens", type=int, default=64,
                    help="probe generations (random baseline)")
    ap.add_argument("--m", type=int, default=32,
                    help="free-run tokens for detection validation")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--tau-start", type=float, default=2.0)
    ap.add_argument("--tau-end", type=float, default=1.0)
    ap.add_argument("--init", type=int, default=0,
                    help="probe index the soft prompt is initialized from "
                         "(fallback for --init-select live)")
    ap.add_argument("--init-select", default="live",
                    choices=["fixed", "live"],
                    help="live (default): initialize from the probe with "
                         "the largest first-token dlogit under the fault "
                         "(zero-gradient inits otherwise pin the loss); "
                         "fixed: always use --init")
    ap.add_argument("--discrete-search", action="store_true",
                    help="after the soft stage, refine the discretized "
                         "prompt with a forward-only local search "
                         "(top-k token substitutions)")
    ap.add_argument("--relax-rounding", action="store_true",
                    help="soft optimization with fp32 activations (model "
                         "upcast; fault stays an exact bf16 bit flip; "
                         "validation and discrete search always run the "
                         "real bf16 decode)")
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--search-max-evals", type=int, default=None,
                    help="optional cap on candidate evaluations per "
                         "discrete search (each eval is 2 bf16 free-runs)")
    ap.add_argument("--hotflip", action="store_true",
                    help="gradient-guided discrete refinement (HotFlip): one "
                         "fp32 backward pass ranks all token substitutions, "
                         "top candidates verified on the real bf16 decode; "
                         "runs before discrete search, which remains the "
                         "fallback")
    ap.add_argument("--hf-steps", type=int, default=8,
                    help="max accepted flips per hotflip search")
    ap.add_argument("--hf-width", type=int, default=4,
                    help="top-W gradient-ranked candidates verified per step")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=os.path.join(
        ROOT, "results/pattern_comparison.jsonl"))
    args = ap.parse_args()
    args.device = args.device or ("cuda" if torch.cuda.is_available()
                                  else "cpu")
    scopes = [s.strip() for s in args.scopes.split(",") if s.strip()]

    prompts = [json.loads(line)["prompt"]
               for line in open(args.prompt_file) if line.strip()]
    model, tokenizer = load_model(args.model, args.device)
    snapshot = snapshot_weights(model)
    eos = tokenizer.eos_token_id

    print(f"healthy reference: {len(prompts)} probes", flush=True)
    _, healthy = generate_greedy(model, tokenizer, prompts,
                                 args.max_new_tokens, args.batch,
                                 args.device)
    init_ids = tokenizer(prompts[args.init], return_tensors="pt")\
        .input_ids[0].to(args.device)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as out:
        for scope in scopes:
            for seed in range(args.seed0, args.seed0 + args.faults):
                fault = sample_fault(model, args.scope_k, seed, scope)
                t0 = time.time()
                applier = FaultApplier(fault, "bf16")

                # --- random baseline: per-prompt detection vector ------
                applier.apply(model)
                try:
                    _, faulty_probe = generate_greedy(
                        model, tokenizer, prompts, args.max_new_tokens,
                        args.batch, args.device)
                finally:
                    applier.undo(model)
                    if not verify_restore(model, snapshot):
                        raise RuntimeError("restore failed (baseline)")
                det_vec = [h != f for h, f in zip(healthy, faulty_probe)]
                n_random = sum(det_vec)

                # --- init selection: pick a probe where the fault is live
                # at the first-token prediction (else the margin gradient
                # is exactly zero and the prompt never moves)
                dlogit = first_token_dlogit_scan(model, tokenizer, prompts,
                                                 applier, args.batch)
                init_idx = args.init
                if args.init_select == "live" and max(dlogit) > 0:
                    init_idx = max(range(len(prompts)), key=lambda i: dlogit[i])
                init_ids = tokenizer(prompts[init_idx], return_tensors="pt")\
                    .input_ids[0].to(args.device)

                # --- optimization ---------------------------------------
                # --relax-rounding: upcast the model to fp32 for the soft
                # stage so gradients survive in the bands where bf16
                # activation rounding kills them. The fault stays the
                # exact bf16 bit flip: FaultApplier's fp32 path mirrors
                # the bf16 corruption value-for-value, and every fp32
                # weight is an exact bf16 value, so the downcast is exact
                # and verify_restore must pass bitwise.
                if args.relax_rounding:
                    model.to(torch.float32)
                    opt_applier = FaultApplier(fault, "fp32")
                    soft_ids, curve, gnorm = optimize_prompt(
                        model, init_ids, opt_applier, args)
                    model.to(torch.bfloat16)
                    if not verify_restore(model, snapshot):
                        raise RuntimeError("restore failed (relaxed opt)")
                else:
                    soft_ids, curve, gnorm = optimize_prompt(
                        model, init_ids, applier, args)
                h = free_run(model, soft_ids, args.m, eos)
                applier.apply(model)
                try:
                    f = free_run(model, soft_ids, args.m, eos)
                finally:
                    applier.undo(model)
                    if not verify_restore(model, snapshot):
                        raise RuntimeError("restore failed (validation)")
                d = first_divergence(h, f)
                soft_detected = d is not None
                soft_changed = int((soft_ids.cpu() != init_ids.cpu()).sum())
                # a tie (margin == 0) detects only by argmax tie-breaking —
                # knife-edge. Treat margin >= 0 as needing refinement even
                # when the free-run verdict says detected.
                soft_margin = first_token_margin(model, soft_ids, applier)[0]
                needs_search = (not soft_detected) or soft_margin >= 0

                # --- optional discrete refinement -----------------------
                # Staged: gradient-guided HotFlip first (cheap, one backward
                # per step ranks all substitutions), blind discrete search
                # as fallback if HotFlip fails and --discrete-search is on.
                search = {}
                final_ids = soft_ids

                def _verify(ids):
                    h = free_run(model, ids, args.m, eos)
                    applier.apply(model)
                    try:
                        f = free_run(model, ids, args.m, eos)
                    finally:
                        applier.undo(model)
                        if not verify_restore(model, snapshot):
                            raise RuntimeError("restore failed (search)")
                    return first_divergence(h, f)

                if needs_search and args.hotflip:
                    t1 = time.time()
                    final_ids, n_eval, n_flips = hotflip_search(
                        model, soft_ids, applier, args, tokenizer,
                        max_evals=args.search_max_evals)
                    d2 = _verify(final_ids)
                    search = {"method": "hotflip",
                              "evaluated": n_eval, "flips": n_flips,
                              "detected": d2 is not None,
                              "first_divergence": d2,
                              "seconds": round(time.time() - t1, 1)}
                if needs_search \
                        and not search.get("detected") \
                        and args.discrete_search:
                    t1 = time.time()
                    final_ids, n_eval = discrete_search(
                        model, soft_ids, applier, args, tokenizer,
                        max_evals=args.search_max_evals)
                    d2 = _verify(final_ids)
                    search = {"method": "discrete",
                              "evaluated": n_eval,
                              "detected": d2 is not None,
                              "first_divergence": d2,
                              "seconds": round(time.time() - t1, 1)}

                record = {
                    "model": args.model, "scope": scope, "seed": seed,
                    "fault": fault.to_dict(),
                    "n_probes": len(prompts),
                    "n_random_detected": n_random,
                    "random_single_rate": n_random / len(prompts),
                    "random_detection_vector": det_vec,
                    "max_first_token_dlogit": max(dlogit),
                    "init_probe": init_idx,
                    "init_probe_dlogit": dlogit[init_idx],
                    "steps": args.steps, "objective": "margin",
                    "relax_rounding": args.relax_rounding,
                    "gnorm_final": gnorm,
                    "loss_final": curve[-1]["loss"] if curve else None,
                    "soft_detected": soft_detected,
                    "soft_first_divergence": d,
                    "soft_n_tokens_changed": soft_changed,
                    "soft_prompt_text": tokenizer.decode(
                        soft_ids, skip_special_tokens=True),
                    "soft_first_token_margin": soft_margin,
                    "final_first_token_margin": first_token_margin(
                        model, final_ids, applier)[0],
                    "curve": curve,
                    "discrete_search": search or None,
                    "final_detected": (search.get("detected")
                                       if search else soft_detected),
                    "elapsed_s": round(time.time() - t0, 1),
                }
                out.write(json.dumps(record) + "\n")
                out.flush()
                det = record["final_detected"]
                print(f"{scope:8s} seed={seed:4d} fault={fault.id} "
                      f"random={n_random}/{len(prompts)} "
                      f"soft={'DET' if soft_detected else 'no'} "
                      f"final={'DET' if det else 'no'} "
                      f"({record['elapsed_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
