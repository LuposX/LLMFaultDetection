"""Run a queue of experiments from one YAML configuration file.

Each experiment entry names a method (mapped to an entry-point script in
METHODS), a fault selection, and an output file; everything else is passed
through as CLI flags to that script. Experiments run sequentially in file
order, each with its own log, and a failed experiment aborts the queue so
finished outputs stay intact for inspection.

Usage:
  python src/run_experiments.py --config experiments.yaml
  python src/run_experiments.py --config experiments.yaml --dry-run
  python src/run_experiments.py --config experiments.yaml --only L1,hotflip
  python src/run_experiments.py --config c.yaml --shard 0/4   # CPU cluster
  python src/run_experiments.py --config c.yaml --merge       # after shards

Config schema:
  common:                # defaults merged into every experiment
    model: Qwen/Qwen3-0.6B
  experiments:
    - name: L1_baseline
      method: random     # key into METHODS
      scopes: [mantissa, exponent]
      seeds: all         # all | [start, end] | count (contiguous from 0)
      out: results/exp_random.jsonl
      args:              # extra CLI flags for the method's script
        n-probes: 100
"""

import argparse
import glob
import json
import os
import shlex
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from lib.faults import manifest_meta  # noqa: E402

# method name -> (script path relative to repo root, fixed CLI args)
METHODS = {
    "random": ("src/run_random_baseline.py", []),
    "hotflip": ("src/run_hotflip_baseline.py", []),
    "naive": ("src/run_hotflip_baseline.py", ["--search", "discrete"]),
    "pattern_comparison": ("src/run_pattern_comparison.py", []),
    "pattern_set": ("src/run_pattern_set.py", []),
}


def resolve_seeds(spec, manifest):
    """seeds field -> (seed0, n_faults). 'all' defers to the manifest."""
    if spec == "all":
        return manifest["seed0"], manifest["n_per_scope"]
    if isinstance(spec, int):
        return 0, spec
    if isinstance(spec, list) and len(spec) == 2:
        return spec[0], spec[1] - spec[0] + 1
    raise ValueError(f"seeds must be 'all', [start, end], or a count: {spec}")


def build_cmd(exp, common, manifest, out_path, shard=None):
    merged = dict(common)
    merged.update(exp)
    script, fixed = METHODS[merged["method"]]
    flags = list(fixed)
    flags += ["--model", merged.get("model", "Qwen/Qwen3-0.6B")]
    flags += ["--scopes", ",".join(merged["scopes"])]
    seed0, n = resolve_seeds(merged.get("seeds", "all"), manifest)
    if shard is not None:
        k, n_shards = shard
        block = n // n_shards
        seed0, n = seed0 + k * block, block
        out_path = f"{out_path}.shard{k}"
    flags += ["--seed0", str(seed0), "--faults", str(n), "--out", out_path]
    for key, val in (merged.get("args") or {}).items():
        flag = "--" + key.replace("_", "-")
        if val is True:
            flags.append(flag)
        elif val is False or val is None:
            continue
        elif isinstance(val, list):
            flags += [flag, ",".join(str(v) for v in val)]
        else:
            flags += [flag, str(val)]
    return [sys.executable, script] + flags, out_path


def merge_shards(exp, common, manifest):
    merged = dict(common)
    merged.update(exp)
    out = merged["out"]
    # numeric suffixes only: out.shard0.log must not match
    cands = [s for s in glob.glob(out + ".shard*")
             if s.rsplit(".shard", 1)[1].isdigit()]
    shards = sorted(cands, key=lambda p: int(p.rsplit(".shard", 1)[1]))
    if not shards:
        print(f"[merge] {exp['name']}: no shard files found, skipped")
        return
    with open(out, "w") as dst:
        for s in shards:
            with open(s) as src:
                dst.write(src.read())
    print(f"[merge] {exp['name']}: {len(shards)} shards -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--only", default=None,
                    help="comma-separated experiment names to run")
    ap.add_argument("--force", action="store_true",
                    help="re-run even if the output file exists")
    ap.add_argument("--keep-going", action="store_true",
                    help="continue the queue after a failed experiment")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resolved commands and exit")
    ap.add_argument("--shard", default=None, metavar="K/N",
                    help="run only fault block K of N (contiguous split)")
    ap.add_argument("--merge", action="store_true",
                    help="concatenate *.shard* outputs and exit")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    common = cfg.get("common") or {}
    experiments = cfg["experiments"]
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        experiments = [e for e in experiments if e["name"] in wanted]
        missing = wanted - {e["name"] for e in experiments}
        if missing:
            raise SystemExit(f"no such experiment(s): {', '.join(missing)}")

    manifest = manifest_meta(model_name=common.get("model"))
    shard = tuple(int(x) for x in args.shard.split("/")) if args.shard else None

    if args.merge:
        for exp in experiments:
            merge_shards(exp, common, manifest)
        return

    failed = []
    for i, exp in enumerate(experiments):
        cmd, out_path = build_cmd(exp, common, manifest, exp["out"], shard)
        exists = os.path.exists(out_path)
        status = "SKIP (exists)" if exists and not args.force else (
            "DRY-RUN" if args.dry_run else "RUN")
        print(f"[{i + 1}/{len(experiments)}] {exp['name']}: {status}")
        print("  " + shlex.join(cmd), flush=True)
        if (exists and not args.force) or args.dry_run:
            continue
        log_path = out_path + ".log"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(log_path, "w") as log:
            # tee: child output goes to the log AND the terminal, so a long
            # run is visibly alive instead of silent for hours
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, cwd=ROOT,
                                    text=True, bufsize=1)
            for line in proc.stdout:
                log.write(line)
                sys.stdout.write(line)
                sys.stdout.flush()
            proc.wait()
        if proc.returncode == 0:
            lines = open(log_path).read().strip().splitlines()
            tail = lines[-1] if lines else "(no output)"
            print(f"  done -> {out_path}\n  ... {tail}", flush=True)
        else:
            failed.append(exp["name"])
            print(f"  FAILED (rc={proc.returncode}), log: {log_path}",
                  flush=True)
            if not args.keep_going:
                print("queue aborted; use --keep-going to continue past "
                      "failures", flush=True)
                break
    if failed:
        raise SystemExit(f"failed experiments: {', '.join(failed)}")


if __name__ == "__main__":
    main()
