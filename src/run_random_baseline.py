"""Random-pattern baseline on the frozen fault set (experiment B1).

For each manifest fault, evaluate the probe prompts by real greedy
decoding: a prompt detects the fault when the faulty continuation differs
from the healthy one. This is the coverage reference every optimization
method is compared against, so it shares their infrastructure exactly:
faults come from data/fault_manifest.json (same ids as every other
method), and decodes are cached platform-tagged (src/lib/cache.py), so
healthy decodes are computed once per (model, prompt file) and faulty
decodes once per (fault, prompt) across all future re-runs.

Caveat: bf16 decodes are bitwise-reproducible only for a fixed batch
composition. The batch here is the probe subset (default 100 prompts in
--batch chunks); changing --n-probes or --batch can flip individual
probes whose margin sits at a bf16 tie. Cross-method comparisons are
unaffected because every method shares this decode path.

Pattern-budget mode (--n-patterns N --repeats R): instead of decoding all
100 probes, R independent random N-subsets of the probe pool are drawn
(shared across all faults, so the healthy reference is decoded once per
subset) and every fault is decoded on each subset. Each subset gets its
own cache keys — the batch composition of a genuine N-pattern decode
differs from the 100-probe run, and bf16 ties may resolve differently,
so cached 100-probe decodes must not be reused. One JSONL record is
written per (fault, repeat) with the subset, its draw seed, and the
per-pattern detection vector.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.cache import Cache, platform_tag  # noqa: E402
from lib.compute import ComputeCounter  # noqa: E402
from lib.faults import load_manifest  # noqa: E402
from lib.generation import generate_greedy  # noqa: E402
from lib.inject import snapshot_weights, verify_restore  # noqa: E402
from lib.loading import load_model  # noqa: E402

MODEL_ID = "Qwen/Qwen3-0.6B"


def load_resume(out_path, method, check):
    """Cells already recorded in out_path, keyed (scope, seed, repeat).

    Tolerates a truncated final line (interrupted write). Aborts when an
    existing record disagrees with the current invocation, so a rerun can
    never silently mix configurations in one file."""
    done = set()
    if not os.path.exists(out_path):
        return done
    for line in open(out_path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue            # truncated by an interrupted write
        if r.get("method") != method:
            continue
        err = check(r)
        if err:
            raise SystemExit(
                f"{out_path}: existing record disagrees with this "
                f"invocation ({err}). Archive the file to "
                f"results/pattern_budget/partial/ and start fresh.")
        done.add((r["scope"], r["seed"], r["repeat"]))
    return done


def run_budget_mode(args, prompts, pf_hash, faults, scopes):
    """Pattern-budget mode: R random n-subsets, all faults on each subset."""
    model, tokenizer = load_model(args.model, args.device)
    snapshot = snapshot_weights(model)
    cache = Cache(args.cache)
    counter = ComputeCounter()
    counter.attach(model)
    base_key = (args.model, args.max_new_tokens, args.batch, pf_hash)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    def check(r):
        if r.get("n_patterns") != args.n_patterns:
            return f"n_patterns {r.get('n_patterns')} != {args.n_patterns}"
        if r.get("subset_seed") != args.subset_seed + r.get("repeat"):
            return f"subset_seed {r.get('subset_seed')} does not match "
        if r.get("max_new_tokens") != args.max_new_tokens:
            return "max_new_tokens mismatch"
        return None

    done = load_resume(args.out, "random-budget", check)
    if done:
        print(f"resuming: {len(done)} fault-repeats already recorded, "
              f"skipping them", flush=True)
    with open(args.out, "a") as out:
        for r in range(args.repeats):
            rng = random.Random(args.subset_seed + r)
            subset = sorted(rng.sample(range(len(prompts)), args.n_patterns))
            sub_hash = hashlib.sha1(
                json.dumps(subset).encode()).hexdigest()[:8]
            sub_prompts = [prompts[i] for i in subset]
            print(f"== repeat {r + 1}/{args.repeats}: subset {sub_hash} "
                  f"{subset} — healthy reference", flush=True)
            healthy, h_new = decode_with_cache(
                model, tokenizer, sub_prompts,
                [(*base_key, "healthy", f"n{args.n_patterns}", f"r{r}",
                  sub_hash, i) for i in range(len(sub_prompts))],
                cache, args.max_new_tokens, args.batch, args.device)
            print(f"   healthy: {h_new} fresh, "
                  f"{len(sub_prompts) - h_new} cached", flush=True)
            bar = tqdm(faults, desc=f"faults r{r}", unit="fault",
                       dynamic_ncols=True)
            for fault in bar:
                if (fault.scope, fault.seed, r) in done:
                    continue
                t0 = time.time()
                c0 = counter.as_dict()
                faulty, n_new = decode_with_cache(
                    model, tokenizer, sub_prompts,
                    [(*base_key, "faulty", fault.id, f"n{args.n_patterns}",
                      f"r{r}", sub_hash, i) for i in range(len(sub_prompts))],
                    cache, args.max_new_tokens, args.batch, args.device,
                    corrupt=fault, snapshot=snapshot)
                detected = [h != f for h, f in zip(healthy, faulty)]
                record = {
                    "model": args.model, "scope": fault.scope,
                    "seed": fault.seed, "fault": fault.to_dict(),
                    "method": "random-budget",
                    "n_patterns": args.n_patterns, "repeat": r,
                    "subset": subset, "subset_seed": args.subset_seed + r,
                    "subset_hash": sub_hash,
                    "n_probes": len(sub_prompts), "max_new_tokens":
                        args.max_new_tokens, "batch": args.batch,
                    "manifest": args.manifest,
                    "detected": detected, "n_detected": sum(detected),
                    "n_new_decodes": n_new,
                    "platform": platform_tag(),
                    "compute": {k: counter.as_dict()[k] - c0[k]
                                for k in ("forward_calls", "rows",
                                          "context_tokens", "backwards",
                                          "row_equiv", "flops")},
                    "elapsed_s": round(time.time() - t0, 1),
                }
                out.write(json.dumps(record) + "\n")
                out.flush()
                bar.set_postfix_str(
                    f"{fault.scope} {fault.seed:3d} "
                    f"{record['n_detected']}/{len(sub_prompts)} "
                    f"[{'all cached' if n_new == 0 else f'{n_new} new'}]")

    # summary: coverage per repeat and pooled over repeats, per scope
    recs = [json.loads(l) for l in open(args.out)]
    recs = [x for x in recs if x.get("method") == "random-budget"]
    groups = defaultdict(list)
    for x in recs:
        groups[(x["n_patterns"], x["repeat"], x["scope"])].append(x)
    print("== summary (coverage per repeat):", flush=True)
    for n_pat in sorted({g[0] for g in groups}):
        for scope in scopes:
            reps = [groups[(n_pat, r, scope)] for r in range(args.repeats)]
            covs = [sum(any(x["detected"]) for x in rep) / len(rep)
                    for rep in reps]
            pooled = [x for rep in reps for x in rep]
            pcov = sum(any(x["detected"]) for x in pooled) / len(pooled)
            print(f"  n={n_pat:3d} {scope:8s}: "
                  f"{' | '.join(f'{c:.1%}' for c in covs)} "
                  f"-> pooled {pcov:.1%} ({len(pooled)} fault-repeats)",
                  flush=True)
    tot = counter.as_dict()
    print(f"total: {tot['row_equiv']} row-equivalents, "
          f"{tot['flops'] / 1e12:.1f} TFLOP (estimated), "
          f"platform {platform_tag()['device']}", flush=True)

def decode_with_cache(model, tokenizer, prompts, keys, cache, max_new_tokens,
                      batch, device, corrupt=None, snapshot=None):
    """Greedy decodes for prompts[i] stored under keys[i]. Returns
    (token-id lists, n_new) where n_new is how many prompts were actually
    decoded this call (0 = all cache hits — relevant when resuming an
    interrupted run). corrupt: optional Fault applied around the
    generation of missing entries (and undone after, with a restore
    check)."""
    missing = [i for i, k in enumerate(keys) if cache.get(*k) is None]
    if missing:
        if corrupt is not None:
            corrupt.apply(model)
        try:
            texts, ids = generate_greedy(
                model, tokenizer, [prompts[i] for i in missing],
                max_new_tokens, batch, device)
        finally:
            if corrupt is not None:
                corrupt.undo(model)
                if not verify_restore(model, snapshot):
                    raise RuntimeError("weights did not restore cleanly")
        for i, idl in zip(missing, ids):
            cache.put(idl, *keys[i])
    return [cache.get(*k) for k in keys], len(missing)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scopes", default="mantissa,exponent")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--faults", type=int, default=20,
                    help="number of faults per scope (seeds seed0..)")
    ap.add_argument("--n-probes", type=int, default=100)
    ap.add_argument("--n-patterns", type=int, default=None,
                    help="pattern budget n: decode only a random n-subset "
                         "of the probe pool (pattern-budget mode)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="pattern-budget mode: number of independently "
                         "drawn n-subsets evaluated per fault")
    ap.add_argument("--subset-seed", type=int, default=0,
                    help="pattern-budget mode: base seed for the subset "
                         "draws (subset r uses seed + r)")
    ap.add_argument("--manifest", default="data/fault_manifest.json",
                    help="frozen fault set (use fault_manifest_k_sweep.json "
                         "for the flip-multiplicity sweep)")
    ap.add_argument("--ks", default=None,
                    help="comma-separated subset of the manifest's k values "
                         "to run (k-sweep manifests only)")
    ap.add_argument("--prompt-file", default="data/prompts.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--device", default=None)
    ap.add_argument("--cache", default="results/cache/random_baseline.jsonl")
    ap.add_argument("--out", default="results/exp_random.jsonl")
    args = ap.parse_args()
    args.device = args.device or ("cuda" if torch.cuda.is_available()
                                  else "cpu")
    scopes = [s.strip() for s in args.scopes.split(",") if s.strip()]

    prompt_lines = [json.loads(line)
                    for line in open(args.prompt_file) if line.strip()]
    prompts = [p["prompt"] for p in prompt_lines][:args.n_probes]
    # probe identity in the cache keys: index + prompt-file content, so a
    # silently edited prompt file can't poison cached decodes
    pf_hash = hashlib.sha1(
        json.dumps([p["prompt"] for p in prompt_lines]).encode()
    ).hexdigest()[:8]

    budget_mode = args.n_patterns is not None
    if budget_mode:
        assert 1 <= args.n_patterns <= len(prompts), "--n-patterns out of range"
        assert args.repeats >= 1, "--repeats must be >= 1"

    faults = [f for f in load_manifest(path=args.manifest,
                                       model_name=args.model)
              if f.scope in scopes
              and args.seed0 <= f.seed < args.seed0 + args.faults]
    if args.ks is not None:
        k_sub = {int(x) for x in args.ks.split(",") if x.strip()}
        faults = [f for f in faults if f.k in k_sub]
    assert faults, "no manifest faults match --scopes/--seed0/--faults/--rhos"
    k_vals = sorted({f.k for f in faults})

    if budget_mode:
        run_budget_mode(args, prompts, pf_hash, faults, scopes)
        return

    print(f"== stage 1/2: healthy reference ({len(prompts)} prompts, "
          f"batch {args.batch})", flush=True)

    model, tokenizer = load_model(args.model, args.device)
    snapshot = snapshot_weights(model)
    cache = Cache(args.cache)
    counter = ComputeCounter()
    counter.attach(model)

    # batch is part of the key: bf16 decode is bitwise-reproducible only
    # for a fixed batch composition, so cached decodes from a different
    # batch size must not be reused
    base_key = (args.model, args.max_new_tokens, args.batch, pf_hash)
    print(f"healthy reference: {len(prompts)} prompts", flush=True)
    healthy, healthy_new = decode_with_cache(
        model, tokenizer, prompts,
        [(*base_key, "healthy", i) for i in range(len(prompts))],
        cache, args.max_new_tokens, args.batch, args.device)
    print(f"healthy reference: decoded {healthy_new} prompts fresh, "
          f"{len(prompts) - healthy_new} from cache", flush=True)
    print(f"== stage 2/2: {len(faults)} faults (k={k_vals}) — "
          f"{'no-cache' if not os.path.exists(args.cache) else 'cache: ' + args.cache}",
          flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as out:
        bar = tqdm(faults, desc="faults", unit="fault", dynamic_ncols=True)
        for fault in bar:
            t0 = time.time()
            c0 = counter.as_dict()   # per-fault compute delta baseline
            faulty, n_new = decode_with_cache(
                model, tokenizer, prompts,
                [(*base_key, "faulty", fault.id, i)
                 for i in range(len(prompts))],
                cache, args.max_new_tokens, args.batch, args.device,
                corrupt=fault, snapshot=snapshot)
            detected = [h != f for h, f in zip(healthy, faulty)]
            record = {
                "model": args.model, "scope": fault.scope,
                "seed": fault.seed, "fault": fault.to_dict(),
                "n_probes": len(prompts), "max_new_tokens":
                    args.max_new_tokens, "batch": args.batch,
                "manifest": args.manifest,
                "detected": detected, "n_detected": sum(detected),
                "n_new_decodes": n_new,
                "compute": {k: counter.as_dict()[k] - c0[k]
                            for k in ("forward_calls", "rows",
                                      "context_tokens", "backwards",
                                      "row_equiv", "flops")},
                "elapsed_s": round(time.time() - t0, 1),
            }
            out.write(json.dumps(record) + "\n")
            out.flush()
            strat = ("rho", f"{fault.rho:g}") if fault.rho is not None \
                else ("k", str(fault.k))
            cache_note = "all cached" if n_new == 0 else f"{n_new} decoded"
            bar.set_postfix_str(
                f"{fault.scope} {strat[0]}={strat[1]} "
                f"{record['n_detected']}/{len(prompts)} "
                f"[{cache_note}] {record['elapsed_s']}s")
            tqdm.write(
                f"{fault.scope} {strat[0]}={strat[1]} seed={fault.seed:3d} "
                f"fault={fault.id} random={record['n_detected']}/"
                f"{len(prompts)} ({cache_note}, {record['elapsed_s']}s)")

    # summary: per (stratum, scope) single-prompt rate and >=1-of-N coverage;
    # stratum = rho for distribution-drawn faults, else the fixed k
    recs = [json.loads(l) for l in open(args.out)]
    groups = defaultdict(list)
    for r in recs:
        f = r["fault"]
        strat = ("rho", f"{f['rho']:g}") if f.get("rho") is not None \
            else ("k", str(f["k"]))
        groups[(strat, r["scope"])].append(r)
    print("== summary (single-prompt rate | coverage by >=1 of "
          f"{len(prompts)} probes):", flush=True)
    for ((sname, sval), scope), vec in sorted(groups.items()):
        rate = sum(r["n_detected"] for r in vec) / (len(vec) * len(prompts))
        cov = sum(any(r["detected"]) for r in vec) / len(vec)
        zf = sum(not r["fault"]["flips"] for r in vec)
        zf_note = f" [{zf} zero-flip]" if zf else ""
        print(f"  {sname}={sval:>6s} {scope:8s}: {rate:6.1%} | {cov:6.1%} "
              f"({len(vec)} faults){zf_note}", flush=True)

    rows_total = counter.as_dict()["row_equiv"]
    print(f"total: {rows_total} row-equivalents, "
          f"platform {platform_tag()['device']}", flush=True)


if __name__ == "__main__":
    main()
