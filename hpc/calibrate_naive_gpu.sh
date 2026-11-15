#!/bin/bash
#SBATCH --job-name=atpg-calib
#SBATCH --partition=gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j_%a.out
#SBATCH --error=slurm-%j_%a.err

# GPU throughput calibration for the naive (discrete_search) campaign.
#
# Runs ONE fault, ONE repeat, mantissa scope only, with the exact production
# settings (bf16 decode, eager attention, m=64, budget-rows 13000,
# search-max-evals 200) on the cluster GPU. Device is selected at runtime
# (cuda when available), so this measures the H100 decode rate that sizes
# the array job:
#
#   sbatch hpc/calibrate_naive_gpu.sh
#   (override the partition with: sbatch --partition=<gpu partition> ...)
#
# Readout: the log's calibration block prints row-equivalents, elapsed
# time and rows/s; send the numbers back before submitting the array.

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

echo "== device check =="
python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
EOF

python src/run_hotflip_baseline.py \
    --search discrete \
    --model Qwen/Qwen3-0.6B \
    --scopes mantissa \
    --seed0 372 \
    --faults 1 \
    --out results/pattern_budget/calib_naive_gpu.jsonl \
    --hf-starts 1 \
    --start-sampling random \
    --repeats 1 \
    --start-seed 900 \
    --m 64 \
    --budget-rows 13000 \
    --search-max-evals 200 \
    --batch 8

echo
echo "== calibration readout =="
python - <<'EOF'
import json
recs = [json.loads(l) for l in open("results/pattern_budget/calib_naive_gpu.jsonl") if l.strip()]
for r in recs:
    c = r.get("compute", {})
    rows, s = c.get("row_equiv", 0), c.get("elapsed_s", 0)
    rate = rows / s if s else 0
    print(f"fault {r.get('fault_seed')}: {rows} row-equiv in {s:.1f}s "
          f"-> {rate:.1f} rows/s; stop={r.get('stop_reason')}")
    print("  NOTE: this fault is detected at the first candidate, so the rate "
          "reflects detection-speed paths (margin evals dominate). For "
          "budget-exhausted sizing, the whole-fault wall clock at this rate "
          "is roughly rows/13000 scaled by 5 repeats.")
EOF
