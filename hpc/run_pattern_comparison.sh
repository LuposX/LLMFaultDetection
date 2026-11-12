#!/bin/bash
#SBATCH --job-name=atpg-cmp
#SBATCH --partition=gpu_h100          # or gpu_a100_il / gpu_a100_short / gpu_h100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Optimized-vs-random test pattern comparison (deterministic exact-match
# criterion). bwUniCluster 3.0 does not support the repo's nix flake, so
# this script builds a plain Python venv and pip-installs requirements.txt.
#
# First-time setup is automatic but slow (~2-3 GB of pip downloads); once
# $VENV_DIR and the HF cache exist, subsequent jobs start in seconds.
#
# Adjust these two paths if needed:
VENV_DIR="$HOME/atpg-venv"                 # venv lives in $HOME (persists)
export HF_HOME="$HOME/hf_cache"            # model cache (persists)

set -euo pipefail

# Locate the repo root. sbatch runs a COPY of this script from its spool
# directory, so script-relative paths are unreliable; search the submit dir
# and script-relative candidates for a directory containing both markers.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" || SCRIPT_DIR=""
REPO_ROOT=""
for cand in "${SLURM_SUBMIT_DIR:-}" "$SCRIPT_DIR" "$SCRIPT_DIR/.." \
            "$SCRIPT_DIR/../.."; do
    [ -n "$cand" ] || continue
    if [ -f "$cand/requirements.txt" ] && \
       [ -f "$cand/src/run_pattern_comparison.py" ]; then
        REPO_ROOT="$(cd "$cand" && pwd)"
        break
    fi
done
if [ -z "$REPO_ROOT" ]; then
    echo "ERROR: could not locate the repo (need requirements.txt and " \
         "src/run_pattern_comparison.py together)"
    echo "  searched: submit dir=${SLURM_SUBMIT_DIR:-unset} script dir=$SCRIPT_DIR"
    echo "  submit dir contents:"
    ls -la "${SLURM_SUBMIT_DIR:-.}" || true
    exit 1
fi
cd "$REPO_ROOT"
echo "== repo root: $REPO_ROOT"

# Fail early with a clear message if the upload is incomplete
for f in requirements.txt src/run_pattern_comparison.py \
        src/plots/plot_pattern_comparison.py data/prompts.jsonl; do
    [ -f "$f" ] || { echo "MISSING: $f — upload the full repo to the cluster"; exit 1; }
done

echo "== job $SLURM_JOB_ID on $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi || true

# --- python -----------------------------------------------------------------
# bwUniCluster3.0 provides Python as an environment module; load it (check
# `module avail` on the login node for the exact name, e.g. devel/python).
module load devel/python 2>/dev/null || module load python 2>/dev/null || true
python3 --version

# --- venv + deps (created once, reused afterwards) ---------------------------
if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "== creating venv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
if ! python -c "import torch, transformers" 2>/dev/null; then
    echo "== installing requirements.txt"
    pip install --upgrade pip
    pip install -r requirements.txt
fi
python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "no CUDA GPU visible to the job"
EOF

# --- run ---------------------------------------------------------------------
# Fault seeds 0..19 per scope match results/fault_coverage.jsonl, so the
# random baselines in both files describe the same faults.
# Add --discrete-search to add the forward-only refinement arm.
python src/run_pattern_comparison.py \
    --scopes mantissa,exponent \
    --faults 20 --seed0 0 \
    --steps 100 \
    --out results/pattern_comparison.jsonl

# --- figure -------------------------------------------------------------------
python src/plots/plot_pattern_comparison.py

echo "== done"
