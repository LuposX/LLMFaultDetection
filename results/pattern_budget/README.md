# Pattern-budget experiments

Real runs for the results table of `reports/experiment-report.typ`
(Section 2.6): every (method, pattern budget n) combination is its own
experiment — no derived or truncated values. Each fault is evaluated
R = 5 times with independently drawn pattern budgets; the draws are
shared across faults within a repeat (so the healthy reference is
decoded once per draw).

* 385 faults per class (mantissa, exponent), k = 1 manifest
  (`data/fault_manifest.json`), seeds 0 to 384.
* n = 1, 2, 4, 8 (n = 32 planned; n = 100 = the completed B1 full run,
  `results/exp_random.jsonl`).
* Fault model: bf16 weight-bit flips, mantissa and exponent scope.
* Decode: greedy, 64 new tokens, batch 8; pattern-budget runs use their
  own cache keys (batch composition differs from the 100-probe run).
* Compute: row-equivalents (1 row = one sequence-forward, backward = 2
  forwards) and estimated FLOPs (analytical, 2 P per token forward,
  P = 596,049,920 non-embedding; backward = 2x forward of the traversed
  graph; attention and embedding lookups neglected).

## Files

* `random_n{1,2,4,8}.jsonl` — random baseline at budget n; one record
  per (fault, repeat) with `subset` (pattern indices), `subset_seed`,
  and the per-pattern detection vector `detected`.
* `hotflip_n{1,2,4,8}.jsonl` — HotFlip baseline at budget n (random
  start sampling, `--start-sampling random`); one record per
  (fault, repeat) with `start_indices`, `start_seed`, per-start results
  `starts`, and `detected`.
* `summary.json` — aggregate written by
  `python3 src/aggregate_pattern_budget.py --out results/pattern_budget/summary.json`.
* `partial/` — fragments from interrupted runs, kept for provenance but
  excluded from aggregation (they would duplicate or rag against the
  complete files above).

## Status (2025-11-14)

* Complete: `random_n1`, `random_n2`, `hotflip_n1`, `hotflip_n2`
  (desktop GPU, RX 9070; 3,850 records each, validated gap-free).
* Pending: `random_n4`, `hotflip_n4`, `random_n8`, `hotflip_n8`
  (desktop queue).
* Naive (optimized) rows n = 1, 2, 4, 8 run on the cluster GPU partition
  (H100 PCIe) as a sharded array job
  (`CONFIG=experiments_cluster_naive_gpu.yaml SHARDS=96 sbatch ...`);
  shard files `naive_n*.jsonl.shardK` merge via
  `python3 src/run_experiments.py --config experiments_cluster_naive_gpu.yaml --merge`.
  The CPU partition was ruled out by calibration (0.47 rows/s bf16
  single-thread; see `slurmlogs/archive/` on the cluster for both
  calibration logs).
* Records carry a `platform` tag (device, torch, transformers); desktop
  rows and cluster rows must pass the cross-platform check before they
  share a table.

## Reproduce

Desktop GPU (RX 9070), from the repo root:

```sh
for n in 1 2 4 8
    python3 src/run_random_baseline.py --faults 385 \
        --n-patterns $n --repeats 5 \
        --out results/pattern_budget/random_n$n.jsonl
end
for n in 1 2 4 8
    python3 src/run_hotflip_baseline.py --faults 385 \
        --hf-starts $n --start-sampling random --repeats 5 \
        --out results/pattern_budget/hotflip_n$n.jsonl
end
python3 src/aggregate_pattern_budget.py \
    --out results/pattern_budget/summary.json
```

(The fish loops above are illustrative; see `hpc/` for Slurm versions.)
