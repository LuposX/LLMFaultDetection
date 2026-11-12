#!/bin/bash
#SBATCH --job-name=atpg-cmp-cpu
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64          # full Ice Lake node is 2x32 cores
#SBATCH --mem=120G                  # model + weights snapshot + torch overhead
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# CPU variant of the optimized-vs-random test pattern comparison.
# Same experiment as run_pattern_comparison.sh, but on cpu_il instead of a
# GPU partition: ~4-8 min per fault instead of ~2, hence the longer wall
# time and the parallel-across-faults strategy below.
#
# Parallelism: one process per seed block, each with disjoint faults and
# its own output file; the JSONL records are independent and the plot
# script concatenates them. NPROC_BLOCKS * BLOCK_FAULTS must equal the
# total fault count per scope.
#
# requirements.txt is shared with the GPU script; it installs CUDA torch
# wheels which also run on CPU (device selection is runtime).

VENV_DIR="$HOME/atpg-venv"
export HF_HOME="$HOME/hf_cache"

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

echo "== job $SLURM_JOB_ID on $(hostname) ($(nproc) logical cores)"

# --- python -----------------------------------------------------------------
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
assert not torch.cuda.is_available() or True  # CPU run; cuda unused
EOF

# --- parallel fault blocks ----------------------------------------------------
SCOPES="mantissa,exponent"
FAULTS_PER_SCOPE=20          # total faults per scope (must match seed blocks)
BLOCKS=4                     # parallel processes; each gets FAULTS_PER_SCOPE/BLOCKS
THREADS=$(( (SLURM_CPUS_PER_TASK + BLOCKS - 1) / BLOCKS ))
BLOCK_FAULTS=$(( FAULTS_PER_SCOPE / BLOCKS ))
SEED0=0

echo "== $BLOCKS parallel blocks x $BLOCK_FAULTS faults x $THREADS threads"

pids=()
for b in $(seq 0 $((BLOCKS - 1))); do
    s0=$(( SEED0 + b * BLOCK_FAULTS ))
    OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
    python src/run_pattern_comparison.py \
        --scopes "$SCOPES" \
        --faults "$BLOCK_FAULTS" --seed0 "$s0" \
        --steps 100 \
        --out "results/pattern_comparison_block$b.jsonl" \
        > "results/pattern_comparison_block$b.log" 2>&1 &
    pids+=($!)
    echo "== block $b: seeds $s0..$((s0 + BLOCK_FAULTS - 1)), pid ${pids[-1]}"
done

fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || { echo "== block with pid $pid FAILED (see block logs)"; fail=1; }
done
[ "$fail" -eq 0 ] || exit 1

# --- concatenate + figure ------------------------------------------------------
cat results/pattern_comparison_block*.jsonl > results/pattern_comparison.jsonl
python src/plots/plot_pattern_comparison.py

echo "== done"
