# Running the comparison on bwUniCluster 3.0

The nix flake is not usable there; the environment comes from
`requirements.txt` in a plain venv instead.

## Files

- `requirements.txt` — pinned versions matching the dev environment
  (torch 2.9.1 stack). Linux PyPI wheels are CUDA builds, which is what
  the H100/A100 partitions need.
- `run_pattern_comparison.sh` — sbatch script: loads the Python module,
  creates/reuses a venv in `$HOME/atpg-venv`, pip-installs once, caches
  HuggingFace weights in `$HOME/hf_cache`, then runs
  `src/run_pattern_comparison.py` and the plot.

## Steps

```bash
# 1. upload the repo (rsync/git) to the cluster, e.g. to $HOME/llm-fault-injection
# 2. check the python module name once on the login node:
module avail devel/python   # adjust the `module load` line in the script if needed
# 3. submit:
sbatch hpc/run_pattern_comparison.sh
# 4. watch:
squeue -u $USER; tail -f slurm-<jobid>.out
```

Model weights (~1.4 GB, Qwen/Qwen3-0.6B) download automatically into
`$HF_HOME` on first run.

## What the job computes

For each scope (mantissa, exponent) and 20 single-bit faults (seeds 0–19,
matching `results/fault_coverage.jsonl`):

- random baseline: per-prompt detection vector over the 100 fixed probes →
  `random_single_rate` = P(one uniformly drawn probe detects the fault)
- optimized pattern: SDoT soft prompt, margin objective on the first
  generated token (the exact detection point), tau-annealed, discretized,
  validated free-running → `soft_detected`
- optional `--discrete-search`: forward-only local search (top-k token
  substitutions per position) that evaluates candidates under the real
  bf16 decode — add the flag to the python line in the sbatch script.

Results append to `results/pattern_comparison.jsonl` (one flushed line per
fault, partial results usable). The figure
`results/pattern_comparison.png` compares, per scope, the mean random
single-prompt detection rate against the fraction of faults the optimized
pattern detects.

Runtime estimate: ~1.5–2.5 min per fault on an H100 → ~1.5–2 h for the
default 40 faults, well within the 4 h wall time. The cluster restarts
decisions differ from the ROCm dev box; determinism of the detection
verdict is per-machine (same hardware reproduces bitwise).
