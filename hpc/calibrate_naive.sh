#!/bin/bash
#SBATCH --job-name=atpg-calib
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-%j_%a.out
#SBATCH --error=slurm-%j_%a.err

# Throughput calibration for the naive (discrete_search) campaign.
#
# Runs ONE fault, ONE repeat, mantissa scope only, with the exact production
# settings (bf16 decode, m=64, budget-rows 13000, search-max-evals 200) and
# prints the per-fault line with row-equivalents and elapsed time. From
# elapsed_s / row_equiv the shard count and walltime for the full array can
# be set from measured numbers instead of the stale B2 anchors.
#
#   sbatch hpc/calibrate_naive.sh
#
# Readout: the log ends with a line like
#   fault ... detected=... row-equiv=NNNN (X.X TFLOP est.) in NNs
# and the summary at the bottom of the script computes rows/s for you.

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

echo "== threads: OMP=$OMP_NUM_THREADS torch.get_num_threads will be set by the runner"
python src/run_hotflip_baseline.py \
    --search discrete \
    --model Qwen/Qwen3-0.6B \
    --scopes mantissa \
    --seed0 372 \
    --faults 1 \
    --out results/pattern_budget/calib_naive.jsonl \
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
recs = [json.loads(l) for l in open("results/pattern_budget/calib_naive.jsonl") if l.strip()]
for r in recs:
    c = r.get("compute", {})
    rows, s = c.get("row_equiv", 0), c.get("elapsed_s", 0)
    rate = rows / s if s else 0
    print(f"fault {r.get('fault_seed')}: {rows} row-equiv in {s:.0f}s "
          f"-> {rate:.1f} rows/s; stop={r.get('stop_reason')}")
    print(f"  worst-case shard projection at this rate: "
          f"{1.9e6 / rate / 3600:.1f} h for 240 fault-repeat units "
          f"(budget-exhausted mean 8000 rows assumption)")
    print(f"  if typical detection halves that, scale accordingly")
EOF
