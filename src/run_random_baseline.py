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

Output: one JSONL record per fault with the per-prompt detection vector,
same shape as results/fault_coverage.jsonl.
"""

import argparse
import hashlib
import json
import os
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

    faults = [f for f in load_manifest(path=args.manifest,
                                       model_name=args.model)
              if f.scope in scopes
              and args.seed0 <= f.seed < args.seed0 + args.faults]
    if args.ks is not None:
        k_sub = {int(x) for x in args.ks.split(",") if x.strip()}
        faults = [f for f in faults if f.k in k_sub]
    assert faults, "no manifest faults match --scopes/--seed0/--faults/--rhos"
    k_vals = sorted({f.k for f in faults})
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
                "compute": counter.as_dict(),
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
