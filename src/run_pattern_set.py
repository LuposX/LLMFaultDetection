"""Joint K-pattern optimization with a train/test fault split.

The proposal's joint pattern generation (Algorithm 1): K soft patterns are
optimized simultaneously against a TRAINING set of sampled faults; each
fault is assigned to its best-matching pattern via the inner max. The
discretized pattern set is then validated free-running on held-out TEST
faults — coverage = fraction of test faults detected by at least one
pattern (deterministic exact-match criterion).

Objective per step: for each training fault f,
    loss_f = min_p margin_loss(H_p, F_p,f)
where H_p are the healthy first-token logits under pattern p and F_p,f
the faulted ones. min over patterns = the inner max over divergence (a
fault is covered once ONE pattern flips its argmax; loss_f = 0 then).
Minimizing sum_f loss_f pushes the SET to cover every training fault.
Gradients flow only into the best pattern per fault (hard min), which is
the same subgradient the proposal's max-assignment induces.

Faults are sampled with sample_fault (k flips, --scope), seeds
--seed0 .. --seed0+n-1; the first --n-train seeds are the training set,
the rest held out. If results/fault_coverage.jsonl contains the same
(scope, seed) faults, its per-prompt random detection vectors are
attached to the record so the plot can draw the random-K baseline from
the same faults.

Output: one JSON record per scope with patterns, fault split, detection
matrix [faults x patterns], coverage curves, and the optimization curve.
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
from lib.inject import snapshot_weights, restore_weights, verify_restore  # noqa: E402
from lib.loading import load_model  # noqa: E402
from lib.sdot import soft_forward, one_hot_logits  # noqa: E402
from optimize_prompts import (  # noqa: E402
    margin_loss, FaultApplier, free_run, first_divergence, AMPLITUDE)
from run_pattern_comparison import (  # noqa: E402
    first_token_dlogit_scan, optimize_prompt, discrete_search,
    hotflip_search)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_ID = "Qwen/Qwen3-0.6B"


def optimize_pattern_set(model, inits, faults, appliers, args):
    """Joint optimization of K soft patterns over the training faults.

    Returns (list of Z argmax id tensors, curve, final per-fault losses).
    """
    vocab = model.get_input_embeddings().num_embeddings
    Zs = [one_hot_logits(ids, vocab, amplitude=AMPLITUDE).requires_grad_(True)
          for ids in inits]
    opt = torch.optim.Adam(Zs, lr=args.lr)
    curve = []
    for step in range(args.steps + 1):
        tau = args.tau_start + (args.tau_end - args.tau_start) * step / args.steps
        opt.zero_grad()
        # healthy first-token logits per pattern (graph-attached)
        healthy = [soft_forward(model, Z, tau)[-1] for Z in Zs]
        total = 0.0
        fault_losses = []
        assignment = []
        for applier in appliers:
            losses = []
            applier.apply(model)
            try:
                for p in range(len(Zs)):
                    f_logits = soft_forward(model, Zs[p], tau)[-1]
                    losses.append(margin_loss(healthy[p], f_logits,
                                              args.margin))
            finally:
                applier.undo(model)
            stacked = torch.stack(losses)
            fault_losses.append(stacked.min().item())
            assignment.append(int(stacked.argmin()))
            total = total + stacked.min()   # inner max over patterns = min loss
        total.backward()
        opt.step()
        if step % args.log_every == 0 or step == args.steps:
            curve.append({"step": step, "tau": round(tau, 4),
                          "loss": round(total.item(), 6),
                          "mean_fault_loss": round(sum(fault_losses) /
                                                   max(1, len(fault_losses)), 6)})
            print(f"  step {step:4d} tau={tau:.3f} loss={total.item():.4f} "
                  f"mean/fault={sum(fault_losses) / max(1, len(fault_losses)):.4f}",
                  flush=True)
    with torch.no_grad():
        return ([Z.argmax(-1) for Z in Zs], curve, assignment, fault_losses)


def optimize_pattern_set_independent(model, prompts, train, args, tokenizer,
                                     applier_dtype="bf16"):
    """Baseline: one pattern per training fault, optimized independently
    with the single-fault optimizer (live init selection, as in the
    pattern comparison). No fault sharing, no inner max.

    Returns (list of Z argmax id tensors, curves, identity assignment).
    """
    soft_ids, curves = [], []
    for i, fault in enumerate(train):
        applier = FaultApplier(fault, applier_dtype)
        dlogit = first_token_dlogit_scan(model, tokenizer, prompts,
                                         applier, 8)
        init_idx = max(range(len(prompts)), key=lambda j: dlogit[j]) \
            if max(dlogit) > 0 else (i * 13) % len(prompts)
        init_ids = tokenizer(prompts[init_idx], return_tensors="pt") \
            .input_ids[0].to(args.device)
        print(f"  pattern {i + 1}/{len(train)}: fault seed {fault.seed} "
              f"init probe {init_idx} dlogit {max(dlogit):.3f}", flush=True)
        ids, curve, _ = optimize_prompt(model, init_ids, applier, args)
        soft_ids.append(ids)
        curves.append({"fault_seed": fault.seed, "init_probe": init_idx,
                       "curve": curve})
    return soft_ids, curves, list(range(len(train)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scopes", default="mantissa,exponent")
    ap.add_argument("--faults", type=int, default=20,
                    help="faults per scope (train + test), seeds seed0..")
    ap.add_argument("--n-train", type=int, default=8,
                    help="first n seeds are the training set; rest held out")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--k", type=int, default=1, help="flips per fault")
    ap.add_argument("--patterns", type=int, default=5,
                    help="number of jointly optimized patterns (K)"
                         " (joint mode only)")
    ap.add_argument("--mode", default="joint", choices=["joint", "independent"],
                    help="joint: Algorithm 1, K patterns optimized together, "
                         "each fault assigned to its best pattern; "
                         "independent: baseline, one pattern per training "
                         "fault optimized separately (uses --n-train as K)")
    ap.add_argument("--prompt-file",
                    default=os.path.join(ROOT, "data/prompts.jsonl"))
    ap.add_argument("--coverage-file", default=os.path.join(
        ROOT, "results/fault_coverage.jsonl"),
        help="random-baseline detection vectors (optional; attached when "
             "the (scope, seed) faults match)")
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--tau-start", type=float, default=2.0)
    ap.add_argument("--tau-end", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--relax-rounding", action="store_true",
                    help="soft optimization with fp32 activations (model "
                         "upcast; fault stays an exact bf16 bit flip; "
                         "validation and discrete search always run the "
                         "real bf16 decode)")
    ap.add_argument("--discrete-search", action="store_true",
                    help="after the soft stage, refine each pattern with "
                         "the forward-only token-substitution search on the "
                         "real bf16 decode (target: hardest assigned "
                         "training fault; set validated as a whole)")
    ap.add_argument("--topk", type=int, default=8,
                    help="candidate tokens per position in discrete search")
    ap.add_argument("--search-max-evals", type=int, default=200,
                    help="hard cap on candidate evaluations per pattern "
                         "(each eval is 2 bf16 free-runs)")
    ap.add_argument("--batch", type=int, default=8,
                    help="candidate chunking for discrete search (decode "
                         "is always batch-1)")
    ap.add_argument("--hotflip", action="store_true",
                    help="refine patterns with gradient-guided HotFlip search "
                         "instead of the blind substitution search")
    ap.add_argument("--hf-steps", type=int, default=8)
    ap.add_argument("--hf-width", type=int, default=4)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=os.path.join(
        ROOT, "results/pattern_set.json"))
    args = ap.parse_args()
    args.device = args.device or ("cuda" if torch.cuda.is_available()
                                  else "cpu")
    scopes = [s.strip() for s in args.scopes.split(",") if s.strip()]

    prompts = [json.loads(line)["prompt"]
               for line in open(args.prompt_file) if line.strip()]
    model, tokenizer = load_model(args.model, args.device)
    snapshot = snapshot_weights(model)

    # healthy reference on the probe set: each pattern is initialized from
    # a DIFFERENT probe (spread across the set, live-selection per pattern
    # comes once single-pattern behavior is characterized)
    _, healthy = generate_greedy(model, tokenizer, prompts, args.m,
                                 8, args.device)
    init_ids_all = [tokenizer(prompts[(j * 13) % len(prompts)],
                              return_tensors="pt").input_ids[0].to(args.device)
                    for j in range(args.patterns)]

    # optional random baselines from the coverage run (same faults)
    random_vec = {}
    if os.path.exists(args.coverage_file):
        for line in open(args.coverage_file):
            if not line.strip():
                continue
            r = json.loads(line)
            random_vec[(r["scope"], r["seed"])] = r["detected"]

    def det_matrix(ids_list, fs):
        """Detection matrix [faults x patterns]: free-running greedy decode,
        real bf16 model, exact token match against the healthy reference."""
        mat = []
        for f in fs:
            row = []
            applier = FaultApplier(f, "bf16")
            for ids in ids_list:
                h = free_run(model, ids, args.m, tokenizer.eos_token_id)
                applier.apply(model)
                try:
                    o = free_run(model, ids, args.m,
                                 tokenizer.eos_token_id)
                finally:
                    applier.undo(model)
                    if not verify_restore(model, snapshot):
                        raise RuntimeError("restore failed")
                row.append(first_divergence(h, o) is not None)
            mat.append(row)
        return mat

    def refine(ids_list, train, assignment, fault_losses):
        """Discrete-search refinement: per pattern, run the forward-only
        token-substitution search (real bf16 decode) against a target
        training fault — joint: the hardest fault assigned to the pattern;
        independent: the pattern's own fault. Returns (refined_ids, info).
        The refined SET is validated as a whole afterwards, so a
        substitution that helps its target but hurts the set shows up
        honestly in the refined coverage numbers.
        """
        refined, info = [], []
        for j, ids in enumerate(ids_list):
            if args.mode == "independent":
                target = j
            else:
                assigned = [i for i, a in enumerate(assignment) if a == j]
                if not assigned:
                    refined.append(ids)
                    info.append({"target": None, "evaluated": 0,
                                 "changed": 0})
                    continue
                target = max(assigned, key=lambda i: fault_losses[i])
            applier = FaultApplier(train[target], "bf16")
            if args.hotflip:
                new_ids, n_eval, n_flips = hotflip_search(
                    model, ids, applier, args, tokenizer,
                    max_evals=args.search_max_evals)
            else:
                new_ids, n_eval = discrete_search(
                    model, ids, applier, args, tokenizer,
                    max_evals=args.search_max_evals)
                n_flips = int((new_ids.cpu() != ids.cpu()).sum())
            changed = int((new_ids.cpu() != ids.cpu()).sum())
            print(f"  refine pattern {j + 1}/{len(ids_list)}: target fault "
                  f"seed {train[target].seed}, {n_eval} evals, "
                  f"{changed} tokens changed", flush=True)
            refined.append(new_ids)
            info.append({"target": target, "target_seed": train[target].seed,
                         "evaluated": n_eval, "changed": changed,
                         "method": "hotflip" if args.hotflip
                         else "substitution"})
        return refined, info

    records = []
    with open(args.out, "w") as out:
        for scope in scopes:
            print(f"=== scope {scope}", flush=True)
            faults = [sample_fault(model, args.k, seed, scope)
                      for seed in range(args.seed0,
                                        args.seed0 + args.faults)]
            train, test = faults[:args.n_train], faults[args.n_train:]
            applier_dtype = "fp32" if args.relax_rounding else "bf16"
            appliers = [FaultApplier(f, applier_dtype) for f in train]

            # --relax-rounding: upcast to fp32 for the soft stage so
            # gradients survive where bf16 activation rounding kills them.
            # The fault stays the exact bf16 bit flip (FaultApplier's fp32
            # path mirrors the corruption value-for-value), and every fp32
            # weight is an exact bf16 value, so the downcast is exact and
            # verify_restore must pass bitwise.
            if args.relax_rounding:
                model.to(torch.float32)

            t0 = time.time()
            if args.mode == "joint":
                (soft_ids, opt_curves, assignment,
                 fault_losses) = optimize_pattern_set(
                    model, init_ids_all, train, appliers, args)
            else:
                soft_ids, opt_curves, assignment = \
                    optimize_pattern_set_independent(
                        model, prompts, train, args, tokenizer,
                        applier_dtype=applier_dtype)
                fault_losses = None

            if args.relax_rounding:
                model.to(torch.bfloat16)
                if not verify_restore(model, snapshot):
                    raise RuntimeError("restore failed (relaxed opt)")

            # --- validation: detection matrix [faults x patterns] --------
            mat_train = det_matrix(soft_ids, train)
            mat_test = det_matrix(soft_ids, test)
            cov_train = sum(any(r) for r in mat_train) / len(mat_train)
            cov_test = sum(any(r) for r in mat_test) / len(mat_test)
            # full random detection vectors for the same faults, when the
            # coverage run covered them (drives the random-K baseline)
            rand_vecs = {str(f.seed): random_vec.get((scope, f.seed))
                         for f in faults}
            rand_vecs = {k: v for k, v in rand_vecs.items() if v is not None}
            rand_test = [sum(random_vec[(scope, f.seed)]) /
                         len(random_vec[(scope, f.seed)])
                         for f in test if (scope, f.seed) in random_vec] or None
            print(f"  coverage: train {cov_train:.2f} test {cov_test:.2f}"
                  + (f" (random K=1 mean on test: "
                     f"{sum(rand_test) / len(rand_test):.2f})"
                     if rand_test else ""), flush=True)

            # --- optional discrete-search refinement ---------------------
            search_rec = None
            if args.discrete_search:
                if args.mode == "independent":
                    ref_ids, ref_info = refine(soft_ids, train,
                                               assignment, None)
                else:
                    ref_ids, ref_info = refine(soft_ids, train,
                                               assignment, fault_losses)
                mat_train_r = det_matrix(ref_ids, train)
                mat_test_r = det_matrix(ref_ids, test)
                cov_train_r = sum(any(r) for r in mat_train_r) / len(mat_train_r)
                cov_test_r = sum(any(r) for r in mat_test_r) / len(mat_test_r)
                search_rec = {
                    "per_pattern": ref_info,
                    "det_matrix_train_refined": mat_train_r,
                    "det_matrix_test_refined": mat_test_r,
                    "coverage_train_refined": cov_train_r,
                    "coverage_test_refined": cov_test_r,
                    "refined_prompts": [tokenizer.decode(
                        ids, skip_special_tokens=True) for ids in ref_ids],
                }
                print(f"  refined coverage: train {cov_train_r:.2f} "
                      f"test {cov_test_r:.2f}", flush=True)

            record = {
                "model": args.model, "scope": scope,
                "args": {k: v for k, v in vars(args).items()
                         if k != "device"},
                "faults_train": [f.to_dict() for f in train],
                "faults_test": [f.to_dict() for f in test],
                "patterns": [{"init_probe": ((j * 13) % len(prompts)
                                              if args.mode == "joint" else
                                              opt_curves[j]["init_probe"]),
                              "prompt_ids": ids.tolist(),
                              "prompt_text": tokenizer.decode(
                                  ids, skip_special_tokens=True)}
                             for j, ids in enumerate(soft_ids)],
                "mode": args.mode,
                "train_assignment": assignment,
                "det_matrix_train": mat_train,
                "det_matrix_test": mat_test,
                "coverage_train": cov_train,
                "coverage_test": cov_test,
                "random_test_single_rate": rand_test,
                "random_vectors": rand_vecs,
                "opt_curves": opt_curves,
                "discrete_search": search_rec,
                "elapsed_s": round(time.time() - t0, 1),
            }
            out.write(json.dumps(record) + "\n")
            out.flush()
            records.append(record)
    print(f"records written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
