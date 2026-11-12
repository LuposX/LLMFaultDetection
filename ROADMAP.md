# Project Roadmap: Next Steps for ATPG Fault Detection

**Current Status:** Foundation complete (random baseline established, ATPG optimization pipeline working but with low success rate — 8.3%)

**Bottleneck Identified:** Activation threshold prevents detection of mantissa faults. Optimization works mechanically (loss decreases) but doesn't translate to detectable output divergence for subtle faults.

---

## Phase 1: Understand the Ceiling (Short-term, 2-4 weeks)

### 1.1 Characterize the Activation Threshold

**Goal:** Empirically map the activation threshold below which faults become undetectable.

**Implementation:**
```python
# src/checks/check_activation_threshold.py (NEW)
- Load all 12 optimization_*.json files
- Extract activation values and success/failure labels
- Plot activation histogram with success/failure color coding
- Fit a logistic curve: P(detection) = 1 / (1 + exp(-k * (act - act_threshold)))
- Report: 
  • Threshold value (where P(detection) = 50%)
  • Slope (how sharp is the transition?)
  • Confidence intervals
```

**Why:** Understanding if the threshold is hard (binary) or soft (gradual) determines if we can solve this with better optimization or if it's a fundamental problem.

**Expected output:** 
- `results/activation_threshold_fit.json` with curve parameters
- `results/activation_threshold.png` showing the sigmoid fit

---

### 1.2 Sensitivity Analysis: Which Layers Matter?

**Goal:** Identify if fault location (layer depth, MLP vs attention) affects detectability.

**Implementation:**
```python
# src/checks/check_fault_sensitivity.py (NEW)
- Group the 6 faults by location:
  • Embedding layer (early, high fan-out)
  • Attention layers (low depth)
  • MLP layers (varied depth)
  • Output projection (late, narrow)
- For each group, compute:
  • Mean activation
  • Std dev of activation
  • Success rate
- Visualize: activation vs layer_depth scatter plot
```

**Why:** Easy faults might cluster in specific layers. Focus optimization efforts there first.

**Expected output:**
- `results/sensitivity_by_layer.json`
- `results/sensitivity_by_layer.png`

---

### 1.3 Compare Objectives Head-to-Head

**Goal:** Isolate whether margin or JS is fundamentally better, or if the difference is noise.

**Implementation:**
```python
# New experimental runs with tighter controls:
# Run 2-3 more optimization attempts per fault, comparing only objective
python src/optimize_prompts.py --fault-seed X --objective js --dtype bf16 --steps 250
python src/optimize_prompts.py --fault-seed X --objective margin --dtype bf16 --steps 250

# Aggregate in new analysis:
# src/plots/plot_objective_comparison.py (NEW)
- Only look at matched (fault_id, dtype, steps) pairs
- Compare success rate: margin vs js
- Perform chi-squared test for statistical significance
```

**Why:** The current sample is small (3 JS, 8 margin). Need to reduce confounders.

**Expected output:**
- `results/objective_comparison.json` with significance test
- `results/objective_comparison.png`

---

## Phase 2: Break the Floor (Medium-term, 4-8 weeks)

### 2.1 Extend Soft Prompt Length

**Goal:** Give the optimizer more capacity by increasing the learnable prompt from n=16 to n=32 or n=64.

**Rationale:** 
- Current prompt is only 16 soft tokens + 32 teacher tokens (48 total influence)
- Larger prompts have more degrees of freedom for the optimizer
- Risk: Longer prompts may reduce generalizability

**Implementation:**
```python
# Modify src/optimize_prompts.py:
# Add --n argument (default 16, try 32, 64)
# Re-run against the 6 hardest mantissa faults

python src/optimize_prompts.py --fault-seed 1 --n 16 --objective margin
python src/optimize_prompts.py --fault-seed 1 --n 32 --objective margin
python src/optimize_prompts.py --fault-seed 1 --n 64 --objective margin

# Analyze: Does success rate increase with n?
# Expected: n=64 might show 1-2 more successes
```

**Expected output:**
- `results/optimization_*_n32_*.json` and `optimization_*_n64_*.json` files
- `results/prompt_length_analysis.png` showing success vs n

---

### 2.2 Multi-Prompt Optimization

**Goal:** Learn multiple complementary prompts jointly instead of one per fault.

**Rationale:**
- Single prompt must work for one specific fault
- Multiple prompts can specialize: one per fault scope or per layer group
- Joint optimization can learn synergistic prompts

**Implementation:**
```python
# src/optimize_prompts_multi.py (NEW)
- Modify sdot.py to handle multiple soft prompts
- Define 2-3 target faults per run
- Optimize Z1, Z2, Z3 jointly against all targets
- Detection rule: any prompt detects = success
- Report per-prompt and aggregate detection rates

python src/optimize_prompts_multi.py \
  --fault-seeds 1,7,11 \
  --n-prompts 3 \
  --objective margin \
  --steps 300
```

**Expected output:**
- `results/optimization_multi_*.json` files
- Analysis showing if ensemble detection > any single prompt

---

### 2.3 Task-Specific Prompts

**Goal:** Move beyond generic text continuation. Use arithmetic, factual, or reasoning tasks.

**Rationale:**
- Random probes are generic (wikitext/SciQ/GSM8K continuations)
- Task-specific prompts might amplify fault effects for that task
- Different tasks may expose different fault interactions

**Implementation:**
```python
# src/make_task_prompts.py (NEW)
- Create task-specific prompt sets:
  • Arithmetic: "Q: 17 + 43 = A:"
  • Factual: "Q: What is the capital of France? A:"
  • Reasoning: "Q: If all cats are animals, and Fluffy is a cat, ..."
- Each task: ~20-30 examples from GSM8K, natural questions, etc.

# src/optimize_prompts.py enhancement:
# Add --task-prompts arithmetic | factual | reasoning
python src/optimize_prompts.py \
  --fault-seed 1 \
  --task-prompts arithmetic \
  --objective margin
```

**Expected output:**
- `data/prompts_arithmetic.jsonl`, `prompts_factual.jsonl`, etc.
- `results/optimization_task_*.json` files
- `results/task_effectiveness.png`

---

## Phase 3: Scale and Validate (Medium-long term, 8-16 weeks)

### 3.1 Build Optimized Fault Coverage Curve

**Goal:** Aggregate optimized-prompt results into a ρ-sweep coverage curve (like `coverage_vs_rho.png` but with learned prompts).

**Current state:**
- `coverage_vs_rho.png`: Random prompts across ρ grid (0, 1e-11, ..., 1e-5)
- Each ρ has ~25 random fault draws from f_bitflip(ρ)
- Need: Same ρ grid, but with 2-3 optimized prompts per fault

**Implementation:**
```python
# src/run_optimized_coverage.py (NEW)
- For each ρ in grid:
  - For each random draw seed:
    - Sample fault with f_bitflip(ρ)
    - Optimize prompt against that fault (if activation > threshold)
    - Record detection result
- Output: trials_optimized.jsonl (analogous to trials.jsonl)

# src/plots/plot_coverage_comparison.py (NEW)
- Load both trials.jsonl (random) and trials_optimized.jsonl
- Plot overlaid coverage curves: random vs optimized
- Show improvement (or lack thereof) as function of ρ
```

**Expected output:**
- `results/trials_optimized.jsonl`
- `results/coverage_comparison.png` (random vs optimized)

---

### 3.2 Held-Out Fault Validation

**Goal:** Ensure optimized prompts generalize to unseen faults, not just the ones they were trained on.

**Implementation:**
```python
# Split all sampled faults into train/test:
# - Train: seeds 0-10 (11 per scope)
# - Test: seeds 11-19 (9 per scope)

# Phase 1 (done): Optimize on train set (6 faults sampled)
# Phase 2 (new): Test generalization

# src/checks/check_generalization.py (NEW)
- For each optimized prompt trained on seed X:
  - Apply to faults with different seeds (Y != X)
  - Measure detection rate
  - Report: Does prompt generalize?
  
# Expected: Poor generalization (prompt overfits to one fault)
# But: Informs whether multi-prompt helps
```

**Expected output:**
- `results/generalization_report.json`
- `results/generalization.png` (heatmap: prompt × test_fault)

---

### 3.3 Multi-Bit Fault Investigation

**Goal:** Check if multiple bit flips are easier to detect than single bits.

**Rationale:**
- Single bit flips = subtle activation
- Multiple bits in same region = compound effect, possibly easier
- May require different optimization

**Implementation:**
```python
# src/lib/faults.py enhancement:
# Modify sample_fault to support k > 1 (already there, just expand usage)

# src/run_multi_bit_faults.py (NEW)
python src/run_multi_bit_faults.py \
  --scopes mantissa,exponent \
  --k 2,3,4,5 \
  --faults 5 \
  --seed 0

# Measure: Does random detection improve with k?
# Then optimize: Does ATPG help more for k > 1?
```

**Expected output:**
- `results/multi_bit_coverage.jsonl`
- `results/multi_bit_coverage.png` (coverage vs k)

---

## Phase 4: Production Readiness (Long-term, 16-24 weeks)

### 4.1 Scale to Larger Models

**Goal:** Validate that findings transfer to 1B, 7B, 70B models.

**Why:** 0.6B is a toy model. Hardware faults in production (H100s, TPUs) matter for large models.

**Implementation:**
```bash
# Test on progressively larger models:
# - Qwen3-1B
# - Llama-2 7B
# - Llama-3 70B (if resources permit)

python src/run_degradation.py --model Qwen/Qwen3-1B --rhos 0,1e-9,1e-8,1e-7 --seeds 1
python src/optimize_prompts.py --model Qwen/Qwen3-1B --fault-seed 0

# Compare activation thresholds, success rates across model sizes
```

**Expected output:**
- `results/scaling_report.md` (how does ρ-degradation scale?)
- `results/atpg_scaling.png` (success rate vs model size)

---

### 4.2 Hardware Fault Injection (Real vs Simulated)

**Goal:** Validate that simulated (software) bit flips match hardware faults.

**Implementation:**
- Partner with hardware team (if available) to inject real bit flips
- Compare:
  - Output distributions (healthy vs simulated vs real fault)
  - Detection rates (ATPG prompts on real faults)
  - Loss landscape (are local minima the same?)

**Expected output:**
- Validation report comparing sim vs real

---

### 4.3 Production Deployment

**Goal:** Package the optimized test patterns as deployable detection software.

**Implementation:**
```python
# src/deploy/prompt_library.py (NEW)
- Store all optimized prompts
- Fast lookup: (model_id, fault_scope, fault_seed) → prompt
- Batch detection API: given model + prompts, detect faults

# Create: /deploy/atpg_detector.py
# Usage:
#   from atpg_detector import ATpgDetector
#   detector = ATpgDetector("Qwen3-0.6B")
#   is_faulty = detector.check(model, tokenizer, prompts=["..."])
```

---

## Priority-Ranked Next Steps (Immediate Action Items)

### Week 1-2: Understanding (Low risk, high information)
1. ✓ **Activate threshold characterization** (`check_activation_threshold.py`)
   - Low effort, high insight
   - Tells us if problem is solvable
   - Time: 2-3 days

2. ✓ **Sensitivity analysis** (`check_fault_sensitivity.py`)
   - Identify high-value fault locations
   - Time: 2-3 days

### Week 3-4: Quick Wins (Medium effort, medium reward)
3. **Extend prompt length** (modify `optimize_prompts.py`, run n=32, n=64)
   - Could yield +1-2 successes
   - Time: 1-2 days implementation, 5-7 days compute

4. **Re-run a few hard faults with better hyperparameters**
   - Longer training (steps 250 → 500)
   - Lower learning rate (0.1 → 0.05 for mantissa)
   - Different τ schedule
   - Time: 5-7 days compute

### Week 5-8: Structural Improvements (High effort, uncertain reward)
5. **Multi-prompt optimization** (`optimize_prompts_multi.py`)
   - Requires modifying sdot.py
   - Could break existing code (high risk)
   - Time: 1 week implementation, 2 weeks testing

6. **Task-specific prompts** (new prompt sets + optimization)
   - High payoff if arithmetic/reasoning exposes faults better
   - Time: 3-5 days implementation, 2-3 weeks compute

---

## Decision Tree: What to Do Based on Phase 1 Findings

```
After characterizing activation threshold, decide:

IF threshold exists (binary boundary at ~5-10):
  → Problem is FUNDAMENTAL, not algorithmic
  → Skip Phase 2 (won't help much)
  → Move to Phase 4 (validate on larger models, investigate why)

IF threshold is soft (gradual sigmoid, 50%-90% over activation 0-20):
  → Problem is SOLVABLE with right optimization
  → Proceed with Phase 2: longer prompts, multi-prompt, task-specific
  → Prioritize: longer prompts first (fastest ROI)

IF no clear pattern (activation uncorrelated with success):
  → Other factors dominate (layer depth, fault position, objective fit)
  → Proceed with sensitivity analysis + multi-prompt approach
  → Investigate: Does ensemble approach help?
```

---

## Success Criteria for Each Phase

| Phase | Success Looks Like | Metric |
|-------|-------------------|--------|
| Phase 1 | Threshold characterized, sensitivity map created | Activation threshold ± 0.5, R² > 0.7 on threshold fit |
| Phase 2 | At least 1 more mantissa fault detected | Success rate: 2-3/6 (33%+) for mantissa |
| Phase 3 | Optimized coverage > random by 10+ percentage points | coverage_comparison.png shows +10% gap |
| Phase 4 | Generalization on larger models, real faults match sim | R² > 0.8 on sim vs real fault comparison |

---

## Resource Requirements

| Task | GPU Time | CPU Time | Storage |
|------|----------|----------|---------|
| Phase 1 analysis | 0 | 2 hours | 100 MB |
| Extend prompt length (n=32, 64) | 40 GPU hours | 1 hour | 500 MB |
| Multi-prompt (3 prompts × 6 faults) | 60 GPU hours | 2 hours | 1 GB |
| Task-specific (arithmetic + reasoning) | 80 GPU hours | 3 hours | 1.5 GB |
| **Phase 2 Total** | **~180 GPU hours** | **~6 hours** | **~3 GB** |

---

## Recommended Timeline

```
Week 1-2:    Phase 1 (threshold + sensitivity)
Week 3:      Presentation of Phase 1 findings + decision point
Week 4-8:    Phase 2 (prompt length + multi-prompt + tasks)
Week 9-12:   Phase 3 (coverage curve + generalization)
Week 13+:    Phase 4 (scaling, deployment)
```

---

## Open Questions to Resolve

1. **Is activation threshold hard or soft?**
   - Answer: Phase 1.1
   
2. **Can longer prompts overcome the threshold?**
   - Answer: Phase 2.1
   
3. **Does multi-prompt ensemble help?**
   - Answer: Phase 2.2
   
4. **Are task-specific prompts more effective?**
   - Answer: Phase 2.3
   
5. **How do findings scale to larger models?**
   - Answer: Phase 4.1
   
6. **What's the deployment strategy?**
   - Answer: Phase 4.3 (after Phases 1-3 reduce uncertainty)

---

## Appendix: Code Templates

### Template: New Check Script (src/checks/)
```python
"""Check description."""

import argparse
import json
import os

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results/check_*.json"))
    args = ap.parse_args()
    
    # Load data
    # Analyze
    # Save results
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
```

### Template: New Plot Script (src/plots/)
```python
"""Plot description."""

import matplotlib.pyplot as plt

def plot(data, out):
    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot logic
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")

if __name__ == "__main__":
    # Load, plot, save
    pass
```

---

## Conclusion

The project is at a critical juncture. The foundation (random baseline, ATPG framework) is solid. The ATPG pipeline works but fails on hard faults. **Phase 1 will determine if this is a solvable problem or a fundamental limitation.** Proceed with threshold characterization immediately.
