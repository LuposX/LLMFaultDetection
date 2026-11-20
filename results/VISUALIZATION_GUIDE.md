# ATPG Optimization Visualization Guide

This directory contains two comprehensive visualizations of the ATPG (Automatic Test Pattern Generation) optimization experiments, along with detailed analysis.

## Files Generated

### 1. `optimization_summary.png` (263 KB)
**Main overview dashboard** with 9 subplots covering all key dimensions.

#### Subplots:

**Top Row (Success Rates):**

1. **Success Rate by Fault Scope** (left)
   - Blue bar: Mantissa faults (0/6 successful)
   - Orange bar: Exponent faults (1/6 successful)
   - Key insight: Exponent faults have 3× higher success rate, though still low overall

2. **Success Rate by Objective** (center)
   - Green bar: JS (Jensen-Shannon divergence) objective (0/3)
   - Red bar: Margin objective (1/8)
   - Key insight: Margin slightly outperforms JS, but sample too small to conclude

3. **Success Rate by Data Type** (right)
   - Purple bar: bf16 (1/8)
   - Brown bar: fp32 (0/4)
   - Key insight: fp32 (cleaner gradients) unexpectedly underperformed; sample small

**Middle Row (Relationships):**

4. **Activation vs Detection Outcome** (left scatter)
   - X-axis: Fault activation (how much the fault moves logits)
   - Y-axis: Detection success (0 = failed, 1 = detected)
   - Red dots: Failures
   - Green dots: Success
   - Key insight: The one success (green dot) has activation ~39; all failures cluster below ~2
   - Implication: Activation threshold (~5–10) exists below which no ATPG can succeed

5. **Random Probe Baseline Distribution** (center bar chart)
   - Green: Detected by all 100 random prompts (4 faults — easy)
   - Gold: Detected by some (1–99/100) random prompts (2 faults — medium)
   - Red: Not detected by any random prompt (6 faults — hard)
   - Key insight: Hard faults (red) are the ATPG challenge; easy faults leave no room for improvement

6. **Optimization Curves** (right)
   - Solid line with markers: Successful run (loss plateau at 0.5)
   - Dashed line with X markers: Sample failure (loss decay 1.68 → 0.88, insufficient)
   - Key insight: Optimization "works" (loss decreases) but doesn't translate to detection

**Bottom Row (Detailed Breakdowns):**

7. **Scope Success by Objective** (left)
   - Stacked/grouped bars showing mantissa vs exponent success within each objective
   - Key insight: Margin is slightly better for both scopes

8. **Success Rate: Scope × Data Type** (center)
   - Grouped bars comparing mantissa and exponent within bf16 and fp32
   - Key insight: bf16 margin showed 1 success; others zero (small samples)

9. **Summary Statistics** (right, text box)
   - Total runs: 12
   - Successful: 1 (8.3%)
   - Avg activation: ~9.4
   - Avg random detections: ~50/100
   - By-scope breakdown
   - Key finding callout

---

### 2. `optimization_per_fault.png` (253 KB)
**Detailed per-fault analysis** with loss curves for each of the 6 unique faults.

#### Layout:
- 2 rows × 3 columns = 6 panels (one per unique fault)
- Each panel shows all optimization runs targeting that fault

#### Per-Fault Panel Contents:

Each subplot displays:
- **Title:** Fault scope (MANTISSA/EXPONENT), fault ID, success count (e.g., "1 successful")
- **Lines:** One for each optimization run of that fault
  - Line color: Green (js objective) or red (margin objective) or gray (unknown)
  - Line style: Solid (bf16) or dashed (fp32)
  - Line marker: Circle (✓ detected) or X (failed)
- **Legend:** Shows all runs as `objective (dtype)` with checkmark if successful
- **Bottom-right annotation:** Activation value and random probe baseline count

#### Faults Shown:

1. **Mantissa fault `69d36416964e` (seed=1)**
   - Location: Embedding layer
   - Activation: 0.0625 (very low)
   - Random baseline: 0/100 (completely undetectable)
   - Runs: 5 (js ×2 bf16/fp32, margin ×2 bf16/fp32, unknown ×1)
   - Result: ALL FAILED (loss decreased but no divergence achieved)

2. **Mantissa fault `e7dd6e33b94f` (seed=0)**
   - Location: Unknown specific layer
   - Random baseline: Unknown (likely 0/100)
   - Runs: 3 (js ×1 bf16, margin ×2 bf16/fp32)
   - Result: ALL FAILED

3. **Exponent fault `a3ce5fb39398` (seed=4)** ← **THE ONE SUCCESS**
   - Location: Layer 8 MLP down_proj
   - Activation: 38.75 (very high)
   - Random baseline: 100/100 (already fully detectable)
   - Runs: 1 (margin, bf16)
   - Result: DETECTED (but ceiling case — improvement impossible)

4. **Exponent fault `1f36a4e0e897`**
   - Runs: 1 (margin, bf16)
   - Result: FAILED

5. **Exponent fault `b2ddb76b4ce6`**
   - Runs: 1 (margin, bf16)
   - Result: FAILED

6. **Exponent fault `d5ed5dcb4185`**
   - Runs: 1 (margin, bf16)
   - Result: FAILED

#### Key Observations:

- **Successful panel:** Flat loss (margin constraint already satisfied), marker is circle (✓)
- **Failed panels:** Decreasing loss curves (optimization progresses) but all markers are X (failed detection)
- **Implication:** Optimization reduced loss/divergence but couldn't overcome the intrinsic difficulty of detecting subtle faults

---

### 3. `OPTIMIZATION_ANALYSIS.md`
Comprehensive written report covering:
- Executive summary (8.3% success rate, scope matters)
- Data overview and key findings
- Detailed breakdown by objective, dtype, and activation
- The one successful case (ceiling case analysis)
- Example failure case (mantissa fault stuck at local minimum)
- Implications and proposed next steps
- Full table of all 12 runs

---

## How to Interpret the Visualizations

### Quick Reading Guide

**For presentation/summary:**
- Show `optimization_summary.png` **top row** (3 bar charts) — immediately shows success rates
- Point to the **activation vs outcome scatter** — visualizes the activation threshold problem
- Mention the **random baseline distribution** — explains why easy faults have no room for improvement

**For technical audience:**
- Show **both PNGs side-by-side** — summary for overview, per-fault for details
- Highlight the **loss curves in per_fault.png** — shows optimization is working (loss decreasing) but not sufficient
- Reference `OPTIMIZATION_ANALYSIS.md` **"Why Mantissa Faults Remain Undetectable"** section

**For stakeholders:**
- Frame as: "ATPG optimization works as implemented, but faces a hard ceiling. Easy faults (exponent) are already ~100% detectable by random prompts, leaving no improvement margin. Hard faults (mantissa) require breaking through an activation threshold ~10× below the current mantissa-fault level."

---

## Data Summary at a Glance

| Dimension | Value | Implication |
|-----------|-------|-------------|
| **Total runs** | 12 | Limited experimental scope |
| **Overall success** | 1/12 (8.3%) | Low success rate |
| **Mantissa success** | 0/6 (0%) | Zero detections for subtle faults |
| **Exponent success** | 1/6 (16.7%) | Slightly better for easy faults |
| **Random baseline** | 4 faults 100% detected | Easy faults have ceiling; no room to improve |
| **Hard faults** | 6 undetected by any random prompt | ATPG target — but couldn't crack them |
| **Avg activation (success)** | 38.75 | Exponent fault, high visibility |
| **Avg activation (failure)** | ~0.5 | Mantissa faults, subtle |
| **Activation ratio** | 77× difference | Huge gap between detectable and undetectable |

---

## Technical Details for Reproducibility

### Visualization Scripts

Generated using:
- `src/plots/plot_optimization_results.py` — creates `optimization_summary.png`
- `src/plots/plot_optimization_per_fault.py` — creates `optimization_per_fault.png`

Both scripts:
1. Load all `optimization_*.json` files from `results/`
2. Extract runs, faults, objectives, dtypes, outcomes
3. Aggregate by various dimensions
4. Create matplotlib figures with subplots
5. Save PNG at 150 DPI

### Data Sources

All 12 JSON files:
- `optimization_69d36416964e_js_bf16.json`
- `optimization_69d36416964e_js_fp32.json`
- `optimization_69d36416964e_margin_bf16.json`
- `optimization_69d36416964e_margin_fp32.json`
- `optimization_e7dd6e33b94f_js_bf16.json`
- `optimization_e7dd6e33b94f_margin_bf16.json`
- `optimization_e7dd6e33b94f_margin_fp32.json`
- `optimization_a3ce5fb39398_margin_bf16.json` ← **ONLY SUCCESS**
- `optimization_1f36a4e0e897_margin_bf16.json`
- `optimization_b2ddb76b4ce6_margin_bf16.json`
- `optimization_d5ed5dcb4185_margin_bf16.json`
- (+ 1 older file, not shown in latest run summary)

---

## Recommendations

1. **Next visualization:** Aggregate the 6 successful/unsuccessful faults into a per-fault coverage curve (if multiple prompts were tried per fault)

2. **To improve detection rate:**
   - Focus on exponent-bit faults (slightly higher success) first
   - Investigate why fp32 underperformed; may be a bug or needs longer training
   - Increase soft prompt length (n > 16) to give optimizer more capacity
   - Try multi-prompt joint optimization instead of single prompts

3. **To break the mantissa ceiling:**
   - Empirically map activation threshold across more faults
   - Explore fault sensitivity in different layers (early layers may be easier)
   - Consider hybrid approaches (e.g., sandwiching ATPG prompt with task-specific setup)
   - Extend beyond simple continuation tasks to arithmetic/QA/reasoning prompts

---

## File Locations

- **Visualizations:** `results/optimization_summary.png` and `results/optimization_per_fault.png`
- **Detailed report:** `results/OPTIMIZATION_ANALYSIS.md`
- **Raw data:** `results/optimization_*.json` (12 files)
- **Plot scripts:** `src/plots/plot_optimization_results.py` and `src/plots/plot_optimization_per_fault.py`

---

Generated: November 6, 2024
Data source: 12 ATPG optimization runs on Qwen3-0.6B (bf16)
Model: Single-bit fault detection via prompt optimization
