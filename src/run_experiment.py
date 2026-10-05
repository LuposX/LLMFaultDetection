"""Orchestrate the rho-sweep experiment: inject, benchmark, restore, record.

For each flip probability rho and seed, f_bitflip(rho) is injected into the
model, the proposal's benchmarks are run, and the weights are restored and
verified. One JSON line per trial is appended to --out.
"""

import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inject import inject_model, snapshot_weights, restore_weights, verify_restore
from evaluate import evaluate

MODEL_ID = "Qwen/Qwen3-0.6B"
ALL_TASKS = ["hellaswag", "wikitext", "mmlu", "gsm8k"]


def load_model(model_id: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)  # kept for future prompt work
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rhos", default="0,1e-9,1e-8,1e-7,1e-6,1e-5",
                        help="comma-separated flip probabilities")
    parser.add_argument("--seeds", type=int, default=3,
                        help="independent fault draws per rho (rho=0 runs once)")
    parser.add_argument("--tasks", default=",".join(ALL_TASKS),
                        help="subset of: " + ",".join(ALL_TASKS))
    parser.add_argument("--gsm8k-limit", type=int, default=200)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--out", default="results/trials.jsonl")
    parser.add_argument("--device", default=None,
                        help="cuda (default) or cpu")
    args = parser.parse_args()

    rhos = [float(x) for x in args.rhos.split(",")]
    tasks = [t for t in args.tasks.split(",") if t]
    for t in tasks:
        assert t in ALL_TASKS, f"unknown task {t}"

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(args.model, device)
    snapshot = snapshot_weights(model)

    meta = {
        "model": args.model,
        "dtype": "bfloat16",
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "torch": torch.__version__,
        "tasks": tasks,
        "gsm8k_limit": args.gsm8k_limit,
    }

    with open(args.out, "a") as f:
        for rho in rhos:
            trial_seeds = [0] if rho == 0 else list(range(args.seeds))
            for seed in trial_seeds:
                if rho > 0:
                    t0 = time.time()
                    inject_model(model, rho, seed)
                    inject_s = time.time() - t0
                else:
                    inject_s = 0.0

                t0 = time.time()
                metrics = evaluate(model, tasks,
                                   gsm8k_limit=args.gsm8k_limit,
                                   batch_size=args.batch_size)
                eval_s = time.time() - t0

                record = {**meta, "rho": rho, "seed": seed,
                          "inject_s": round(inject_s, 1),
                          "eval_s": round(eval_s, 1),
                          "results": metrics}
                f.write(json.dumps(record) + "\n")
                f.flush()
                print(f"[rho={rho:g} seed={seed}] eval {eval_s:.0f}s, "
                      f"metrics: {json.dumps(metrics)}")

                restore_weights(model, snapshot)
                if not verify_restore(model, snapshot):
                    raise RuntimeError("weights did not restore cleanly after trial")


if __name__ == "__main__":
    main()
