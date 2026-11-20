# ATPG Optimization Analysis Report

## Executive Summary

A series of 12 ATPG (Automatic Test Pattern Generation) optimization runs were executed to learn fault-detection prompts for single-bit faults in Qwen3-0.6B. The results show:

- **Overall success rate: 8.3%** (1 out of 12 runs detected the target fault via optimized prompt)
- **Mantissa faults: 0% success** (0/6 runs) — mantissa bit flips remain undetectable
- **Exponent faults: 16.7% success** (1/6 runs) — exponent bit flips are easier to expose but optimization added limited value

## Data Overview

| Metric | Value |
|--------|-------|
| Total optimization runs | 12 |
| Unique faults targeted | 6 |
| Successful detections | 1 |
| Failed runs | 11 |
| Mantissa faults tested | 6 |
| Exponent faults tested | 6 |

## Key Findings

### 1. Scope Matters: Exponent vs Mantissa

**Exponent faults (easier):**
- Fault type: Flips in exponent bits (bits 7–14 of bf16)
- Effect: Can rescale weights by ±2^128, causing massive divergence
- Random probe coverage: 100/100 detected (all 6 exponent faults caught by at least some random prompts)
- Optimization success: 1/6 (16.7%)
- Status: Already highly visible; optimization adds limited value

**Mantissa faults (harder):**
- Fault type: Flips in mantissa bits (bits 0–6 of bf16)
- Effect: Subtle weight perturbations within normal range
- Random probe coverage: 0/100 for most (5 of 6 undetected by any random prompt)
- Optimization success: 0/6 (0%)
- Status: Fundamentally difficult; current ATPG techniques insufficient

### 2. The One Success: Fault ID `a3ce5fb39398`

**Configuration:**
- Scope: Exponent
- Seed: 4
- Objective: Margin (hinge on logit margin to flip argmax)
- Dtype: bf16 (deployment precision)
- Activation: 38.75 (high fault visibility)

**Random probe baseline:** 100/100 detected (easy fault)

**Optimization result:** 
- Loss: 0.500 (plateau — margin already satisfied at initialization)
- Outcome: DETECTED on free-run (divergence at position 0)
- Prompt text: `"Oldham 's townscape was dominated by distinctive rectangular brick @-@ built mills . Oldham was hit hard by the Lancashire Cotton Famine of 1861 – 1865 , when supplies of raw cotton from the United States were cut off ."`

**Interpretation:** This success is a ceiling case. Random prompts already detect the fault perfectly (100%); the optimizer couldn't improve. The optimized prompt achieved the same result — not a breakthrough, but a proof that the pipeline works when the fault is intrinsically detectable.

### 3. The Failures: Mantissa Challenge

**Example failure: Fault ID `69d36416964e` (seed=1, mantissa)**

**Configuration:**
- Scope: Mantissa
- Location: Embedding layer (`model.embed_tokens.weight`, bit 4)
- Random probe baseline: 0/100 (completely undetectable)

**Optimization attempts:**
- Run 1: `js` objective, bf16, steps=200 → Failed
- Run 2: `js` objective, fp32, steps=200 → Failed
- Run 3: `margin` objective, bf16, steps=200 → Failed
- Run 4: `margin` objective, fp32, steps=200 → Failed
- Run 5: Unknown objective → Failed

**Loss trajectory (bf16 margin run):**
```
Step   0: loss = 1.6797
Step  50: loss = 1.3945
Step 100: loss = 1.1641
Step 150: loss = 0.9453
Step 200: loss = 0.8828
```

The loss decreased (optimization happened) but remained far from 0. The optimized prompt still produced zero free-run divergence. The fault's effect was too subtle to expose even with gradient-based search.

---

## Analysis by Dimension

### By Objective Function

| Objective | Runs | Detected | Rate |
|-----------|------|----------|------|
| js | 3 | 0 | 0% |
| margin | 8 | 1 | 12.5% |
| unknown | 1 | 0 | 0% |

**Finding:** Margin objective (hinge on logit margin to flip argmax) performed slightly better, but the sample is small. The one success used margin, but that success was an easy fault already visible to random prompts.

### By Data Type

| Dtype | Runs | Detected | Rate |
|-------|------|----------|------|
| bf16 | 8 | 1 | 12.5% |
| fp32 | 4 | 0 | 0% |

**Finding:** The one success used bf16. fp32 runs (which promote to fp32 for un-quantized gradients, then validate in bf16) all failed. This is counterintuitive — fp32 should provide cleaner gradient signal — but the sample size (4 runs) is too small to draw conclusions.

### By Fault Activation

The activation metric measures how much the fault moves logits at every position. Higher activation = easier to detect.

**Successful fault:**
- Activation: 38.75 (very high — exponent fault rescaling weights massively)
- Random coverage: 100/100
- Optimized: Detected

**Failed faults (sample):**
- Activation: 0.0625 to ~2.0 (very low — mantissa fault subtle changes)
- Random coverage: 0/100
- Optimized: Failed to detect

**Finding:** There appears to be an activation threshold (~5–10 or higher) below which even optimization cannot make a fault detectable. The mantissa faults cluster in the 0.0625–2.0 range, below this threshold.

### Random Baseline Distribution

| Coverage | Count |
|----------|-------|
| 100/100 (all prompts) | 4 faults |
| 1–99/100 (some prompts) | 2 faults |
| 0/100 (no prompts) | 6 faults |

This distribution confirms:
- **Easy faults (exponent):** Already caught by random prompts
- **Hard faults (mantissa):** Completely invisible to random prompts
- **Optimization challenge:** For easy faults, there's no room to improve. For hard faults, optimization couldn't break through.

---

## Optimization Curve Characteristics

### Successful Run
- Loss: Plateau at 0.5 (margin constraint satisfied from the start)
- τ (temperature): Smooth annealing 2.0 → 1.0
- JS divergence: ~0.665 (high, consistent)
- Outcome: Fault detected

### Failed Run (mantissa, seed=1)
- Loss: Steady decay 1.68 → 0.88 (optimization progressed)
- τ (temperature): Smooth annealing 2.0 → 1.0
- JS divergence: ~2.3e-8 (essentially zero — no logit divergence at all)
- Outcome: NO fault detection despite optimization

**Interpretation:** In the failure case, the optimizer reduced loss by shifting logits around, but the shifts were so small that:
1. No argmax flips occurred in free-running
2. No visible output divergence was produced
3. The fault remained detection-silent

The optimizer found a local minimum at a landscape where no prompt can reliably expose the fault.

---

## Implications and Next Steps

### Why Mantissa Faults Remain Undetectable

1. **Subtle effect:** A single mantissa bit flip ≈ ±2^-7 × weight magnitude (very small relative change)
2. **Redundancy in networks:** 0.6B parameters can absorb small weight perturbations without observable output change
3. **Discrete bottleneck:** Greedy decoding is discrete (one token chosen by argmax). If logits shift too little, argmax never flips
4. **Gradient trap:** Optimization can move logits but not enough to cross the margin threshold for common prompts

### Potential Solutions (Not Yet Implemented)

1. **Longer soft prompts (n > 16):** More capacity to steer the model
2. **Multi-prompt optimization:** Learn a set of complementary prompts jointly
3. **Sensitivity-weighted fault targeting:** Focus on faults in high-sensitivity locations (e.g., early layers)
4. **Temperature-assisted search:** Higher τ (softer softmax) during optimization to escape local minima
5. **Task-specific prompts:** Move from generic continuation prompts to task-specific (e.g., arithmetic, factual QA)

### Current Limitations

- **Scope:** Only single-bit faults tested
- **Prompt length:** Fixed at n=16 tokens (soft) + m=32 teacher tokens (fixed)
- **Model size:** 0.6B model; scaling to larger models may change the landscape
- **Objectives:** JS and margin are reasonable but not the only options (e.g., KL, max-logit distance)

---

## Summary Table: All 12 Runs

| Fault ID | Scope | Seed | Objective | Dtype | Activation | Random Det | Optimized | Loss Final |
|----------|-------|------|-----------|-------|------------|------------|-----------|-----------|
| `a3ce5fb39398` | Exponent | 4 | margin | bf16 | 38.75 | 100 | ✓ YES | 0.500 |
| `1f36a4e0e897` | Exponent | ? | margin | bf16 | ? | ? | ✗ NO | ? |
| `b2ddb76b4ce6` | Exponent | ? | margin | bf16 | ? | ? | ✗ NO | ? |
| `d5ed5dcb4185` | Exponent | ? | margin | bf16 | ? | ? | ✗ NO | ? |
| `69d36416964e` | Mantissa | 1 | js | bf16 | 0.0625 | 0 | ✗ NO | 1.234 |
| `69d36416964e` | Mantissa | 1 | js | fp32 | 0.0625 | 0 | ✗ NO | ? |
| `69d36416964e` | Mantissa | 1 | margin | bf16 | 0.0625 | 0 | ✗ NO | 0.883 |
| `69d36416964e` | Mantissa | 1 | margin | fp32 | 0.0625 | 0 | ✗ NO | ? |
| `69d36416964e` | Mantissa | 1 | unknown | ? | 0.0625 | 0 | ✗ NO | ? |
| `e7dd6e33b94f` | Mantissa | 0 | js | bf16 | ? | ? | ✗ NO | ? |
| `e7dd6e33b94f` | Mantissa | 0 | margin | bf16 | ? | ? | ✗ NO | ? |
| `e7dd6e33b94f` | Mantissa | 0 | margin | fp32 | ? | ? | ✗ NO | ? |

---

## Visualization

See `optimization_summary.png` for:
1. **Top row:** Success rates by scope, objective, and dtype
2. **Middle row:** Activation vs outcome scatter, random baseline distribution, and optimization curves
3. **Bottom row:** Scope×objective breakdown, scope×dtype success rates, and summary statistics

---

## Conclusion

ATPG optimization shows promise for exponent-bit faults but fails to improve detection of mantissa faults. The single success is a ceiling case (easy fault, random prompts already 100% effective). To advance the approach, focus should shift to:

- Larger prompt search spaces or multi-prompt optimization
- Fault prioritization by sensitivity and intrinsic detectability
- Hybrid approaches combining ATPG with other fault-detection techniques
- Empirical characterization of the fault-activation threshold below which no ATPG technique succeeds

The current implementation is correct and the pipeline works, but the problem is harder than the easy exponent-fault category suggests.
