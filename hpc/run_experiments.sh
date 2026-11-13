#!/bin/bash
#SBATCH --job-name=atpg-exp
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8            # experiments run sequentially per task
#SBATCH --mem=24G                    # bf16 weights 1.2G, fp32 relax ~2.5G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Generic experiment runner for bwUniCluster: executes the YAML experiment
# queue (src/run_experiments.py) on the frozen manifest fault set.
#
# Submit as an ARRAY JOB to split the fault set into contiguous blocks, one
# block per task (each task runs every experiment on its own fault block):
#
#   sbatch --array=0-3 hpc/run_experiments.sh          # 4 shards
#   SHARDS=8 sbatch --array=0-7 hpc/run_experiments.sh # 8 shards
#
# Or as a single job to run everything sequentially in one allocation:
#
#   sbatch hpc/run_experiments.sh
#
# After all array tasks finish, merge the shard outputs (fast, login node
# or tiny job is fine):
#
#   python src/run_experiments.py --config experiments.yaml --merge
#
# Configuration: experiments.yaml (override with CONFIG=<path>). Per-task
# caches live under results/cache/ and are platform-tagged, so shard tasks
# share decodes; first-run healthy decodes are recomputed per task (cheap).
#
# For a GPU partition instead of CPU: change --partition below (e.g.
# gpu_h100) and drop the OMP thread pinning — device selection is runtime.

VENV_DIR="${ATPG_VENV:-$HOME/atpg-venv}"
export HF_HOME="${ATPG_HF_CACHE:-$HOME/hf_cache}"
CONFIG="${CONFIG:-experiments.yaml}"
SHARDS="${SHARDS:-4}"

set -euo pipefail

# --- locate the repo root (sbatch runs a script copy from its spool dir) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" || SCRIPT_DIR=""
REPO_ROOT=""
for cand in "${SLURM_SUBMIT_DIR:-}" "$SCRIPT_DIR" "$SCRIPT_DIR/.." \
            "$SCRIPT_DIR/../.."; do
    [ -n "$cand" ] || continue
    if [ -f "$cand/requirements.txt" ] && \
       [ -f "$cand/src/run_experiments.py" ]; then
        REPO_ROOT="$(cd "$cand" && pwd)"
        break
    fi
done
if [ -z "$REPO_ROOT" ]; then
    echo "ERROR: could not locate the repo (need requirements.txt and " \
         "src/run_experiments.py together)"
    echo "  searched: submit dir=${SLURM_SUBMIT_DIR:-unset} script dir=$SCRIPT_DIR"
    exit 1
fi
cd "$REPO_ROOT"
echo "== repo root: $REPO_ROOT"

# --- fail early if the upload is incomplete ---------------------------------
for f in requirements.txt src/run_experiments.py "$CONFIG" \
        data/fault_manifest.json data/prompts.jsonl; do
    [ -f "$f" ] || { echo "MISSING: $f — upload the full repo to the cluster"; exit 1; }
done

NPROC="$(nproc 2>/dev/null || echo 1)"
echo "== job ${SLURM_JOB_ID:-local} on $(hostname), $NPROC logical cores"

# --- python + venv (created once, reused afterwards) ------------------------
module load devel/python 2>/dev/null || module load python 2>/dev/null || true
python3 --version
if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "== creating venv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
if ! python -c "import torch, transformers, yaml" 2>/dev/null; then
    echo "== installing requirements.txt"
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# --- run ---------------------------------------------------------------------
# Sequential experiments per task; thread pinning keeps CPU decodes bitwise
# reproducible (thread count is part of the cache platform tag).
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "== shard $SLURM_ARRAY_TASK_ID/$SHARDS of config $CONFIG"
    OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$NPROC}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$NPROC}" \
    python src/run_experiments.py --config "$CONFIG" \
        --shard "$SLURM_ARRAY_TASK_ID/$SHARDS" --keep-going
    echo "== shard done; merge after all tasks with:"
    echo "    python src/run_experiments.py --config $CONFIG --merge"
else
    echo "== single-task run of config $CONFIG"
    OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$NPROC}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$NPROC}" \
    python src/run_experiments.py --config "$CONFIG" --keep-going
fi

echo "== done"
