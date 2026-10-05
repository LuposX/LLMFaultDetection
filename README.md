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

## Check the environment

```
python src/check_env.py --gpu-test --download
```

Downloads the model (~1.4 GB) into the HuggingFace cache and runs a bf16
matmul on the GPU.

## Run

```
# full grid: rho in {0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5}, 3 draws per rho
python src/run_experiment.py

# smoke test (minutes instead of hours)
python src/run_experiment.py --tasks hellaswag,wikitext --seeds 1

python src/plot_results.py    # results/benchmark_vs_rho.png
```

Trials are appended to `results/trials.jsonl`; reruns skip nothing, so delete
the file before a fresh grid.
