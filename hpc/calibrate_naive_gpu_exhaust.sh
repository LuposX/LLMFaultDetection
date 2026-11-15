#!/bin/bash
#SBATCH --job-name=atpg-calib2
#SBATCH --partition=gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j_%a.out
#SBATCH --error=slurm-%j_%a.err

# Second GPU calibration: measures the BUDGET-EXHAUSTED path of the naive
# search (margin-eval dominated), which the first calibration (early
# detection, free-run dominated) does not exercise. The probe fault
# (mantissa seed 1) was never detected by any of the 500 random-pattern
# draws, so the discrete search most likely exhausts its 200-eval sweep
# without detecting. Rows are unbounded here (--budget-rows 200000) so the
# full sweep completes and the rate is clean.
#
#   sbatch hpc/calibrate_naive_gpu_exhaust.sh
#
# Readout: rows, elapsed, rows/s at the bottom. This rate sizes the array.

VENV_DIR="${ATPG_VENV:-$HOME/atpg-venv}"
export HF_HOME="${ATPG_HF_CACHE:-$HOME/hf_cache}"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" || SCRIPT_DIR=""
REPO_ROOT=""
for cand in "${SLURM_SUBMIT_DIR:-}" "$SCRIPT_DIR" "$SCRIPT_DIR/.."; do
    [ -n "$cand" ] || continue
    if [ -f "$cand/requirements.txt" ] && [ -f "$cand/src/run_experiments.py" ]; then
        REPO_ROOT="$(cd "$cand" && pwd)"
        break
    fi
done
[ -n "$REPO_ROOT" ] || { echo "ERROR: repo root not found"; exit 1; }
cd "$REPO_ROOT"

module load devel/python 2>/dev/null || module load python 2>/dev/null || true
if [ ! -x "$VENV_DIR/bin/python" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -c "import yaml" 2>/dev/null || pip install -q pyyaml
python -c "import torch, transformers" 2>/dev/null || pip install -q -r requirements.txt

python src/run_hotflip_baseline.py \
    --search discrete \
    --model Qwen/Qwen3-0.6B \
    --scopes mantissa \
    --seed0 1 \
    --faults 1 \
    --out results/pattern_budget/calib_naive_gpu_exhaust.jsonl \
    --hf-starts 1 \
    --start-sampling random \
    --repeats 1 \
    --start-seed 900 \
    --m 64 \
    --budget-rows 200000 \
    --search-max-evals 200 \
    --batch 8

echo
echo "== calibration readout (exhaustion path) =="
python - <<'EOF'
import json
recs = [json.loads(l) for l in open("results/pattern_budget/calib_naive_gpu_exhaust.jsonl") if l.strip()]
for r in recs:
    c = r.get("compute", {})
    rows, s = c.get("row_equiv", 0), c.get("elapsed_s", 0)
    rate = rows / s if s else 0
    ev = sum(st.get("evals", 0) for st in r.get("starts", []))
    print(f"fault seed {r.get('seed')}: {rows} row-equiv in {s:.1f}s "
          f"-> {rate:.1f} rows/s; stop={r.get('stop_reason')}; evals={ev}")
EOF
