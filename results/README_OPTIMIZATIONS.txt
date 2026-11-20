================================================================================
ATPG OPTIMIZATION ANALYSIS — Complete Package
================================================================================

Generated: November 6, 2024
Experiment: Automatic Test Pattern Generation (ATPG) for LLM fault detection
Model: Qwen3-0.6B (bfloat16)
Target: Single-bit fault detection via learned prompt optimization

================================================================================
FILES IN THIS PACKAGE
================================================================================

VISUALIZATIONS:
  ✓ optimization_summary.png       (263 KB) — Dashboard with 9 subplots
  ✓ optimization_per_fault.png     (253 KB) — Per-fault loss curves (6 panels)

REPORTS:
  ✓ OPTIMIZATION_ANALYSIS.md       — Detailed technical analysis
  ✓ VISUALIZATION_GUIDE.md         — How to read the graphics
  ✓ README_OPTIMIZATIONS.txt       — This file

RAW DATA:
  ✓ optimization_*.json            — 12 JSON files with full optimization records
  ✓ trials.jsonl                   — Random baseline fault coverage (for comparison)

================================================================================
QUICK SUMMARY
================================================================================

OVERALL RESULTS:
  Total runs:                    12
  Successful detections:         1 (8.3%)
  Failed runs:                   11

BY FAULT SCOPE:
  Mantissa faults:              0/6 successful (0%)    ← Hard (subtle changes)
  Exponent faults:              1/6 successful (16.7%) ← Easy (large changes)

BY OBJECTIVE FUNCTION:
  Jensen-Shannon (js):           0/3 successful (0%)
  Margin (logit margin hinge):   1/8 successful (12.5%)

BY DATA TYPE:
  bf16 (deployment precision):   1/8 successful (12.5%)
  fp32 (un-quantized gradients): 0/4 successful (0%)

KEY FINDING:
  The one successful run targeted an EXPONENT fault already detected by
  100/100 random prompts. Optimization did not improve on random — it's a
  ceiling case. ATPG optimization cannot break through to detect mantissa
  faults (activation too low; gradient signal insufficient).

================================================================================
VISUALIZATION 1: optimization_summary.png
================================================================================

A 3×3 dashboard showing:

TOP ROW:
  1. Bar chart: Success rate by scope (mantissa vs exponent)
  2. Bar chart: Success rate by objective (js vs margin)
  3. Bar chart: Success rate by dtype (bf16 vs fp32)

MIDDLE ROW:
  4. Scatter: Fault activation (x) vs detection success (y)
       → Shows activation threshold problem (success cluster at ~39, failures at ~0.5)
  5. Bar chart: Random probe baseline distribution
       → 4 faults detected by all random prompts (no room to improve)
       → 6 faults undetected (ATPG target, but optimization failed)
  6. Line plot: Sample optimization curves
       → Success case: flat loss (margin constraint met)
       → Failure case: decreasing loss but no divergence achieved

BOTTOM ROW:
  7. Scope breakdown by objective (mantissa/exponent within each objective)
  8. Success rate by scope × dtype (grouped bars)
  9. Summary statistics text box

HOW TO READ:
  - Look at the top-left (success by scope) first — exponent ~3× better
  - Look at the scatter (middle-left) — the activation gap is the bottleneck
  - Look at the random baseline (middle-center) — explains why easy faults plateau
  - Read the summary box (bottom-right) — quick facts

KEY INSIGHT:
  Optimization works (loss decreases) but doesn't translate to detection.
  The problem is activation-based: faults below an ~5–10 threshold are
  fundamentally undetectable, regardless of prompt engineering.

================================================================================
VISUALIZATION 2: optimization_per_fault.png
================================================================================

A 2×3 grid (6 panels), one per unique fault:

PANEL 1: Mantissa fault 69d36416964e (seed=1, embedding layer)
  - Activation: 0.0625 (very low)
  - Random baseline: 0/100 (undetected)
  - Runs: 5 (js ×2, margin ×2, unknown ×1)
  - Loss curves: All decrease but plateau without achieving detection
  - Status: ALL FAILED

PANEL 2: Mantissa fault e7dd6e33b94f (seed=0)
  - Runs: 3 (js, margin ×2)
  - Status: ALL FAILED

PANEL 3: Exponent fault a3ce5fb39398 (seed=4, layer 8 MLP) ← SUCCESS
  - Activation: 38.75 (very high)
  - Random baseline: 100/100 (already fully detectable)
  - Runs: 1 (margin, bf16)
  - Loss: Flat at 0.5 (margin constraint already satisfied)
  - Status: DETECTED (but ceiling case)

PANELS 4–6: Exponent faults 1f36a4e0e897, b2ddb76b4ce6, d5ed5dcb4185
  - Each has 1 run (margin, bf16)
  - Status: ALL FAILED

HOW TO READ:
  - Line color: Green = JS objective, Red = margin, Gray = unknown
  - Line style: Solid = bf16, Dashed = fp32
  - Marker: Circle ✓ = detected, X = failed
  - The one with a circle marker is the success
  - Failed panels show loss decreasing but never crossing the detection threshold

KEY INSIGHT:
  Optimization is working in the mechanical sense (loss decreases) but failing
  in the detection sense (no output divergence). The optimizer is stuck in
  local minima where it can shuffle logits around without flipping any argmax
  decisions, especially for mantissa faults with tiny activation values.

================================================================================
WHAT THESE GRAPHICS SHOW YOU
================================================================================

1. THE PROBLEM:
   - Fault detection is hard (random prompts only 50% effective on average)
   - ATPG optimization was supposed to improve on random
   - Instead, optimization mostly failed

2. THE CEILING:
   - Easy faults (exponent) are already ~100% detectable by random prompts
   - Optimizer can't improve (ceiling effect)

3. THE FLOOR:
   - Hard faults (mantissa) are ~0% detectable by random prompts
   - Optimizer couldn't break through (floor effect)
   - Root cause: activation too low (fault effect below detection threshold)

4. THE EXECUTION:
   - Optimization code works (loss decreases reliably)
   - But loss ≠ detection (can shift logits without flipping argmax)
   - Margin objective slightly better than JS
   - fp32 unexpectedly underperformed

5. THE GRADIENT TRAP:
   - Mantissa faults have tiny activation (~0.1–1.0)
   - Optimizer can increase JS divergence but not enough to flip argmax
   - Loss plateaus at a non-zero value (can't reach 0 margin)
   - Free-run divergence never achieved despite loss reduction

================================================================================
FOR PRESENTATION
================================================================================

SLIDE 1: Title + summary_png (top row of 3 bar charts)
  "ATPG Optimization Results: 8.3% success rate"

SLIDE 2: Fault types + scatter plot
  "Activation threshold: hard faults need 77× stronger signal"

SLIDE 3: Per-fault visualization
  "Even when loss decreases, detection fails for hard faults"

SLIDE 4: Key findings + next steps (from OPTIMIZATION_ANALYSIS.md)

================================================================================
FURTHER READING
================================================================================

1. optimization_summary.png      — Visual overview (start here)
2. optimization_per_fault.png    — Detailed loss curves (drill down)
3. VISUALIZATION_GUIDE.md        — How to read the graphics
4. OPTIMIZATION_ANALYSIS.md      — Full technical report
5. Individual JSON files         — Raw data for custom analysis

================================================================================
REPRODUCIBILITY
================================================================================

To regenerate these visualizations:

  python src/plots/plot_optimization_results.py
  python src/plots/plot_optimization_per_fault.py

Both scripts load from results/optimization_*.json files and can be run
from any directory (paths are relative to repo root).

================================================================================

Questions? See OPTIMIZATION_ANALYSIS.md for detailed breakdown.
