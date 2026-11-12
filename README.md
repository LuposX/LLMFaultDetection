# llm-fault-injection

Bit-flip fault injection on Qwen3-0.6B (bf16) and benchmark degradation as a
function of the flip probability rho. First experimental step of the research
proposal "Detecting Hardware Faults in AI Accelerators for Large Language
Models". Design decisions that shape the results: see DESIGN.md.

## Setup

```
direnv allow          # or: nix develop
```

The flake provides the full Python environment from nixpkgs stable
(PyTorch with ROCm for the RX 9070 XT, transformers, lm-eval, matplotlib).
First entry downloads ~5 GiB from the binary cache.

## Code map

Everything lives in `src/`; scripts are run as `python src/<path>.py`
from the repo root. Folders only where several files belong together:
`lib/` (libraries, imported never run) and `checks/` (setup/determinism
verification scripts). Data/results paths are anchored to the repo root,
so scripts work from any cwd.

`src/lib/` — libraries:

| file | purpose |
|---|---|
| `loading.py` | load model + tokenizer (bf16, eager attention — SDPA segfaults on this ROCm/RDNA4 build) |
| `generation.py` | the single greedy-generation path shared by everything |
| `inject.py` | fault mechanics: flip random bits in bf16 weights, snapshot/restore/verify weights |
| `faults.py` | named fault configurations: exact bit positions as replayable, serializable objects (scopes: all/exponent/mantissa/sign) |
| `benchmarks.py` | lm-eval harness wrapper: hellaswag, wikitext ppl, mmlu, gsm8k, ... |
| `probes.py` | fault-effect metrics on the fixed probe prompts (token match, JS divergence, laya judge, embedding cosine) |
| `sdot.py` | Soft Distribution over Tokens: differentiable soft-embedding forward pass for prompt optimization |

`src/` — runnable scripts:

| file | purpose |
|---|---|
| `checks/check_env.py` | verify GPU + download the model |
| `checks/check_determinism.py` | verify greedy decoding is bitwise reproducible (rerun, batching, reload, inject/restore) |
| `checks/check_sdot.py` | verify the soft (SDoT) and hard (token id) paths agree as τ → 0, and that the soft path is differentiable; records the sweep to `results/sdot_tau_sweep.json` |
| `make_prompts.py` | build `data/prompts.jsonl` (100 probe prompts: wiki/qa/math) |
| `run_degradation.py` | the rho sweep: inject → benchmark → probe → restore → log one JSON line per trial |
| `optimize_prompts.py` | first ATPG loop: optimize one soft prompt against one known fault (teacher-forced), discretize, validate free-running detection. `--objective js` (dense divergence baseline) or `margin` (hinge on the logit margin that drives the argmax-flip detection criterion); `--dtype bf16` (deployment precision) or `fp32` (un-quantized gradient signal, reports a separate bf16 deployment verdict) |

`src/plots/` — figure scripts (one per figure, shared axis/metric helpers in `common.py`):

| file | output |
|---|---|
| `plot_degradation.py` | `results/degradation_vs_rho.png` (benchmark scores vs fault magnitude) |
| `plot_probes.py` | `results/probes_vs_rho.png` (probe metrics vs fault magnitude) |
| `plot_coverage.py` | `results/coverage_vs_rho.png` (fault coverage: fraction of draws detected by the random probe set, one panel per probe metric — token match, JS, embedding cosine, laya — with detection rules per metric) |
| `plot_sdot_sweep.py` | `results/sdot_tau_sweep.png` (soft-vs-hard logit difference vs temperature, from the check's JSON record) |

Library self-tests run through the package mechanism, e.g.
`cd src && python -m lib.faults`.

Data flow:

```
make_prompts.py → data/prompts.jsonl
run_degradation.py ── uses ── lib/{inject, benchmarks, probes}.py
                                        ↓
                          results/trials.jsonl
checks/check_sdot.py → results/sdot_tau_sweep.json
                                        ↓
                          plots/*.py → PNGs
```

Planned next (ATPG detection pipeline, deterministic variant): the
objective/optimization refinements observed to be needed — margin-based
objective (target argmax flips directly, not JS) and/or fp32 optimization
passes — then `checks/check_detection.py` (multi-pattern validation),
coverage vs random baselines, held-out faults.

## Check the environment

```
python src/check_env.py --gpu-test --download
```

Downloads the model (~1.4 GB) into the HuggingFace cache and runs a bf16
matmul on the GPU.

## Run

```
# full grid: rho in {0, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5}, 3 draws per rho
python src/run_degradation.py

# smoke test (minutes instead of hours)
python src/run_degradation.py --tasks hellaswag,wikitext --seeds 1

# probes only (~3.5 min/trial, no benchmarks): enough for the coverage curves
python src/run_degradation.py --skip-eval

python src/plots/plot_degradation.py   # results/degradation_vs_rho.png
python src/plots/plot_probes.py        # results/probes_vs_rho.png
python src/plots/plot_coverage.py      # results/coverage_vs_rho.png
python src/plots/plot_sdot_sweep.py    # results/sdot_tau_sweep.png
```

Trials are appended to `results/trials.jsonl`; reruns skip nothing, so delete
the file before a fresh grid.
