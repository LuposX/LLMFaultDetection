"""Orchestrate the rho-sweep experiment: inject, benchmark, restore, record.

For each flip probability rho and seed, f_bitflip(rho) is injected into the
model, the proposal's benchmarks are run, and the weights are restored and
verified. One JSON line per trial is appended to --out.
"""

import argparse
import json
import os
import time

# Reduce allocator fragmentation (eager attention has spiky allocation
# peaks); must be set before torch initializes CUDA. New-style var name:
# the old PYTORCH_CUDA_ALLOC_CONF is deprecated and did not take effect.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch

from lib.inject import inject_model, snapshot_weights, restore_weights, verify_restore
from lib.benchmarks import evaluate, PREFERRED_METRIC
from lib.probes import ProbeSet
from lib.loading import load_model

MODEL_ID = "Qwen/Qwen3-0.6B"

# Anchor data paths to the repo root (parent of src/), not the cwd, so the
# script behaves the same from anywhere.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL_TASKS = ["hellaswag", "wikitext", "mmlu", "gsm8k",
             "winogrande", "arc_easy", "boolq"]


def fmt_seconds(s):
    s = int(round(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {sec:02d}s"
    return f"{sec}s"


def metric_summary(task, metrics):
    """Compact one-line metric string for the progress output."""
    if not metrics:
        return "no numeric metrics"
    if "error" in metrics:
        return f"ERROR ({metrics['error']})"
    numeric = {k: v for k, v in metrics.items()
               if isinstance(v, (int, float))}
    pairs = []
    key = PREFERRED_METRIC.get(task)
    if key in numeric:
        pairs.append((key, numeric[key]))
    for k, v in numeric.items():
        if len(pairs) >= 2:
            break
        if "stderr" in k or "alias" in k or k in dict(pairs):
            continue
        pairs.append((k, v))
    return ", ".join(f"{k}={v:.4f}" for k, v in pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rhos", default="0,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5",
                        help="comma-separated flip probabilities")
    parser.add_argument("--seeds", type=int, default=3,
                        help="independent fault draws per rho (rho=0 runs once)")
    parser.add_argument("--min-seed", type=int, default=0,
                        help="first seed to run (resume: seeds 0..min-1 "
                             "are assumed already recorded)")
    parser.add_argument("--tasks", default=",".join(ALL_TASKS),
                        help="subset of: " + ",".join(ALL_TASKS))
    parser.add_argument("--gsm8k-limit", type=int, default=200)
    parser.add_argument("--mmlu-limit", type=int, default=None,
                        help="max examples per mmlu SUBTASK (57 subtasks); "
                             "default = --limit. --limit 500 still evaluates "
                             "~14k mmlu docs (~11 min); pass ~100 for fast trials")
    parser.add_argument("--limit", type=int, default=None,
                        help="max examples per task (smoke tests); "
                             "gsm8k-limit applies to gsm8k regardless")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="base batch size; wikitext always runs at 1 "
                             "(eager attention: batch 16 OOMs at seq~2k)")
    parser.add_argument("--sem-prompts", default=os.path.join(ROOT, "data/prompts.jsonl"),
                        help="prompt file for the fault-effect metrics "
                             "(probes); see src/make_prompts.py")
    parser.add_argument("--sem-max-new-tokens", type=int, default=96)
    parser.add_argument("--sem-batch", type=int, default=8)
    parser.add_argument("--no-sem", action="store_true",
                        help="skip the probe fault-effect metrics")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--out", default=os.path.join(ROOT, "results/trials.jsonl"))
    parser.add_argument("--device", default=None,
                        help="cuda (default) or cpu")
    args = parser.parse_args()

    rhos = [float(x) for x in args.rhos.split(",")]
    tasks = [t for t in args.tasks.split(",") if t]
    for t in tasks:
        assert t in ALL_TASKS, f"unknown task {t}"

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args.model, device)
    snapshot = snapshot_weights(model)

    # Healthy reference for the fault-effect metrics. Must run on the
    # untouched model; caches generated outputs and next-token distributions
    # in CPU RAM so each trial only pays one generation + forward pass.
    sem = None
    if not args.no_sem:
        sem = ProbeSet(model, tokenizer, prompt_file=args.sem_prompts,
                       max_new_tokens=args.sem_max_new_tokens,
                       gen_batch=args.sem_batch)
        sem.compute_baseline()
        print(f"probes: {sem.n_prompts} prompts, healthy reference cached",
              flush=True)

    meta = {
        "model": args.model,
        "dtype": "bfloat16",
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "torch": torch.__version__,
        "tasks": tasks,
        "limit": args.limit,
        "mmlu_limit": args.mmlu_limit,
        "gsm8k_limit": args.gsm8k_limit,
        "sem_prompts": None if sem is None else sem.n_prompts,
        "sem_max_new_tokens": args.sem_max_new_tokens,
        "laya_identical": None if sem is None else sem.laya_identical,
    }

    trials = [(rho, seed)
              for rho in rhos
              for seed in ([0] if rho == 0
                           else range(args.min_seed, args.seeds))]
    assert trials, "nothing to run: --min-seed >= --seeds"
    n_trials = len(trials)
    task_last_s = {}   # task -> duration of its previous run, for per-task ETA
    trial_durs = []    # completed trial wall times, for the run ETA

    def on_task(event, i, n, task, info):
        if event == "start":
            detail = []
            if info["fewshot"]:
                detail.append(f"{info['fewshot']}-shot")
            if info["limit"] is not None:
                detail.append(f"limit={info['limit']}")
            detail.append(f"batch {info['batch']}")
            known = task_last_s.get(task)
            eta = (f"last run {fmt_seconds(known)}" if known
                   else "first run (may download datasets)")
            print(f"   [{i}/{n}] {task:<9s} ({', '.join(detail)}) — {eta}",
                  flush=True)
        elif event == "error":
            print(f"   [{i}/{n}] {task:<9s} FAILED after "
                  f"{fmt_seconds(info['seconds'])}: {info['error']}", flush=True)
        else:
            dt = info["seconds"]
            task_last_s[task] = dt
            print(f"   [{i}/{n}] {task:<9s} done in {fmt_seconds(dt)} — "
                  f"{metric_summary(task, info['metrics'])}", flush=True)

    with open(args.out, "a") as f:
        for idx, (rho, seed) in enumerate(trials, 1):
            remaining = n_trials - idx + 1
            if trial_durs:
                avg = sum(trial_durs) / len(trial_durs)
                eta = f"~{fmt_seconds(avg * remaining)} left"
            else:
                eta = "unknown until first trial finishes"
            print(f"=== trial {idx}/{n_trials}: rho={rho:g} seed={seed} "
                  f"(run ETA {eta})", flush=True)

            n_flips = 0
            if rho > 0:
                t0 = time.time()
                n_flips = inject_model(model, rho, seed)
                inject_s = time.time() - t0
            else:
                inject_s = 0.0

            if rho > 0 and n_flips == 0:
                # the draw left the model bitwise unchanged: evaluating it
                # would re-measure the clean baseline. Record and skip.
                print("    zero flips — model unchanged, skipping eval",
                      flush=True)
                record = {**meta, "rho": rho, "seed": seed, "n_flips": 0,
                          "inject_s": round(inject_s, 1), "eval_s": 0.0,
                          "sem_s": 0.0,
                          "results": {"skipped": "zero flips"}, "sem": None}
                f.write(json.dumps(record) + "\n")
                f.flush()
                trial_durs.append(inject_s)
                continue

            try:
                t0 = time.time()
                metrics = evaluate(model, tasks,
                                   gsm8k_limit=args.gsm8k_limit,
                                   batch_size=args.batch_size,
                                   limit=args.limit,
                                   on_task=on_task,
                                   mmlu_limit=args.mmlu_limit)
                eval_s = time.time() - t0

                sem_s = 0.0
                sem_metrics = None
                if sem is not None and rho > 0:
                    t0 = time.time()
                    sem_metrics = sem.evaluate(model)
                    sem_s = time.time() - t0
                    o = sem_metrics["overall"]
                    print(f"    sem: match={o['token_match']:.3f} "
                          f"first_div={o['first_div_mean']} "
                          f"js={o['js']:.3g} laya={o['laya_same']:.3f} "
                          f"cos={o['embed_cos']:.4f}", flush=True)
            finally:
                # never leave corrupted weights in the model, even if the
                # trial itself raised: later trials would inherit them
                restore_weights(model, snapshot)
                if not verify_restore(model, snapshot):
                    raise RuntimeError(
                        "weights did not restore cleanly after trial")

            record = {**meta, "rho": rho, "seed": seed, "n_flips": n_flips,
                      "inject_s": round(inject_s, 1),
                      "eval_s": round(eval_s, 1), "sem_s": round(sem_s, 1),
                      "results": metrics, "sem": sem_metrics}
            f.write(json.dumps(record) + "\n")
            f.flush()
            trial_durs.append(inject_s + eval_s + sem_s)
            print(f"    trial took {fmt_seconds(inject_s + eval_s + sem_s)} — "
                  f"metrics: {json.dumps(metrics)}", flush=True)


if __name__ == "__main__":
    main()
