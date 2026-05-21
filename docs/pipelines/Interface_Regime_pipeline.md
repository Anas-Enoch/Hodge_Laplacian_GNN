# Interface Regime Pipeline

**Location:** `scripts_gse278936/`  
**Purpose:** Two-regime operator testing and therapeutic hypothesis generation for tumor–immune interface analysis

---

## ⚠ Safety Statement

> These analyses generate **formal operator-derived hypotheses, not clinical recommendations**.  
> The physics analogies (Maxwell–Boltzmann, Euler–Bernoulli) are **interpretive operator analogies** — they do not claim that tumor tissue literally follows these equations.  
> All therapeutic principles require orthogonal experimental validation before any clinical inference can be drawn.

---

## Purpose

This pipeline tests the two-regime hypothesis:

> Tumor–immune interfaces represent a distinct non-equilibrium operator regime with elevated non-gradient interaction, instability-like activity, and a shifted constraint configuration relative to the near-equilibrium gradient-compatible tumor bulk. Therapeutic hypotheses may be generated around interface-specific operator regimes; if validated experimentally, they would suggest targeting the interface not by amplifying gradients, but by disrupting or exploiting its non-integrable interaction structure. **These are formal hypotheses, not clinical recommendations.**

---

## The Two-Regime Model

### Bulk regime — near-equilibrium, gradient-compatible

The exact (gradient) component dominates; coexact energy is low; KS-like instability is low. The interaction field is approximately integrable under a gradient-only model.

| Property | Value |
|---|---|
| Analogy | Near-equilibrium statistical dynamics |
| Operator signature | Low coexact energy, low graph curvature, low CDIS |
| Note | When `flux_exact` is unavailable, `coexact_exact_ratio` is not used as a primary quantitative claim; CDIS uses the reduced formula |

### Interface regime — non-equilibrium, constraint-dominated

The coexact component is not reducible to a gradient — enrichment observed in 40/40 TNBC sections and the majority of GSE278936 sections (verify exact count against `cohort_summary.csv`). Graph-Laplacian curvature is higher; KS-like instability is elevated (median fold ≈ 8.96 over tumor core); KTS analysis shows a strong exhaustion-directed transition bias.

| Property | Value |
|---|---|
| Analogy | Constraint-driven boundary dynamics where gradients fail |
| Operator signature | High coexact energy (or coexact/exact ratio when available), high graph curvature, high CDIS, KTS exhaustion bias |

### On the analogies

The Maxwell–Boltzmann reference characterizes the bulk as a near-equilibrium baseline. Euler–Bernoulli is used only as an analogy for constraint-dominated boundary behavior — it is not a mechanistic claim. **The claim is that the two regimes have formally different operator signatures, not that tumor tissue literally obeys these equations.**

---

## Pipeline Structure

```
scripts_gse278936/
├── step15_regime_assignment.py              Classify nodes into operator regimes
├── step16_operator_regime_metrics.py        Compute per-regime operator summaries
├── step17_two_regime_test.py                Permutation test: interface vs bulk
├── step18_bulk_equilibrium_null.py          Bulk-matched null: near-equilibrium baseline
├── step19_constraint_regime_score.py        CDIS composite constraint-dominated score
├── step20_interface_targeting_principles.py Therapeutic hypothesis table
├── step21_pre_post_therapy_template.py      Pre/post treatment response [PROSPECTIVE ONLY]
├── step22_regime_summary_table.py           Manuscript-ready two-regime model table
├── step23a_power_spectrum_test.py           Power spectrum shape [EXPERIMENTAL]
├── step23b_local_vs_global_prediction.py    Local vs global predictability [EXPERIMENTAL]
├── step23c_spatial_autocorrelation.py       Spatial ACF shape [EXPERIMENTAL]
└── README_Interface_Regime_Pipeline.md      This file
```

**Prerequisites:** Steps 01–14 of the main pipeline must be completed first. Required inputs: `*_spots_coexact_energy.csv`, `*_edges_hodge.csv`, and (for KTS analyses) `kts_transition_bias` outputs.

---

## Step-by-Step Reference

### Step 15 — Regime Assignment

Classifies each spot into one of five operator regimes.

| Regime | Classification logic |
|---|---|
| `bulk_like` | tumor_core region AND coexact_energy < section Q50 |
| `interface_adjacent` | tumor_core region AND coexact_energy ≥ section Q50 |
| `interface_like` | interface/interface_like annotated region |
| `stromal_like` | stroma annotated region |
| `immune_like` | immune annotated region |

`bulk_like` nodes serve as the near-equilibrium reference. `interface_adjacent` nodes are in the tumor core but carry elevated coexact energy.

```bash
python scripts_gse278936/step15_regime_assignment.py \
  --statsdir results_gse278936 \
  --outdir   results_interface_regime \
  --coexact-bulk-q 0.50
```

**Output:** `{sid}_regime_assignment.csv`

---

### ⚠ Pre-Step 16 — Copy required files

Step 16 requires both the regime assignment files (from Step 15) **and** the original spots/edges files in the same directory. Copy them before running:

```bash
mkdir -p results_interface_regime

cp Results_TNBC_rebuild_gse278936/*_spots_coexact_energy.csv results_interface_regime/
cp Results_TNBC_rebuild_gse278936/*_edges_hodge.csv          results_interface_regime/
# Step 15 already wrote regime_assignment files to results_interface_regime/
```

> This was the source of the "missing inputs" error when running Step 16 immediately after Step 15 without the original spots/edges files present.

---

### Step 16 — Operator Regime Metrics

Computes per-regime medians of six operator metrics for each section.

| Metric | Description |
|---|---|
| `coexact_energy` | Non-gradient interaction intensity |
| `exact_energy_node` | Gradient component (baseline) |
| `coexact_exact_ratio` | Non-gradient dominance ratio |
| `graph_curvature` | \|Lu\| — Laplacian applied to coexact field |
| `bilaplacian_mag` | \|L²u\| — higher-order stabilizing structure |
| `nonlin_grad_energy` | Squared edge differences per node |

```bash
python scripts_gse278936/step16_operator_regime_metrics.py \
  --statsdir results_interface_regime \
  --outdir   results_interface_regime
```

**Outputs:** `{sid}_operator_regime_metrics.csv`, `cohort_operator_regime_summary.csv`

---

### Step 17 — Two-Regime Permutation Test

Tests whether interface-like nodes differ from bulk-like nodes using a node-count-matched random subsample from bulk-like nodes as null. One-sided p-value: fraction of null medians ≥ observed interface median.

**Primary metrics** (always tested):

1. `coexact_energy` — non-gradient interaction intensity
2. `graph_curvature` — \|Lu\|
3. `ks_like` — \|−Lu − L²u − \|∇u\|²\|

**Conditional metric** (only when `flux_exact` is available):

4. `coexact_exact_ratio`

The output CSV includes `primary_metric` (boolean) and `exact_available` columns for transparency.

```bash
python scripts_gse278936/step17_two_regime_test.py \
  --statsdir results_interface_regime \
  --outdir   results_interface_regime \
  --n-perm 300 --seed 123
```

**Output:** `cohort_two_regime_test.csv`

**GSE278936 result (n = 23 sections):**

| Metric | Fold over bulk | Sections significant |
|---|---|---|
| Coexact energy | 6.38× | 23/23 |
| Graph curvature | 9.68× | 23/23 |
| KS-like magnitude | 17.84× | 23/23 |
| Coexact/exact ratio (conditional) | — | 22/23 |

---

### Step 18 — Bulk Equilibrium Null

Bootstraps from bulk-like nodes to construct a reference distribution for "near-equilibrium" behavior. Interface nodes are tested against this distribution to confirm they constitute a structurally distinct regime — not merely a high-energy random subset.

```bash
python scripts_gse278936/step18_bulk_equilibrium_null.py \
  --statsdir results_interface_regime \
  --outdir   results_interface_regime \
  --n-perm 300 --seed 123
```

**Output:** `cohort_bulk_vs_interface_null.csv`

**GSE278936 result (n = 23 sections, all significant):**

| Metric | Median interface/bulk fold | Sign test p |
|---|---|---|
| Coexact energy | 5.48× | 1.19 × 10⁻⁷ |
| KS-like instability | 19.54× | 1.19 × 10⁻⁷ |

---

### Step 19 — Constraint-Dominated Interface Score (CDIS)

Combines three operator metrics into a composite score using **robust z-scores** (median-centered, IQR-scaled) to prevent outlier inflation from extreme right-skewed distributions.

**Formula (conditional):**

| Condition | Formula |
|---|---|
| `flux_exact` available | `CDIS = rz(coexact/exact) + rz(\|L²u\|) + rz(nonlin_grad)` |
| `flux_exact` unavailable | `CDIS = rz(coexact_energy) + rz(\|L²u\|) + rz(nonlin_grad)` |

The formula used is written to the `cdis_formula` column for auditability.

CDIS is a **within-section configuration score**, not an absolute biological magnitude. A positive interface–bulk gap means interface nodes consistently occupy a more constraint-dominated region of operator space than bulk nodes.

```bash
python scripts_gse278936/step19_constraint_regime_score.py \
  --statsdir results_interface_regime \
  --outdir   results_interface_regime \
  --n-perm 300 --seed 123
```

**Outputs:** `{sid}_constraint_score.csv`, `cohort_constraint_score_summary.csv`

**GSE278936 result (22/23 sections significant, sign test p = 2.86 × 10⁻⁶):**

| Statistic | Value |
|---|---|
| Median interface CDIS | +6.09 |
| Median bulk CDIS | −0.17 |
| Median interface–bulk gap | +6.07 (IQR 5.14–8.17) |

> **Note:** GSM8558019 was the only non-significant section and had only 6 bulk-like nodes — the bulk estimate is statistically unreliable at this sample size, not a biological reversal.

---

### Step 20 — Therapeutic Hypothesis Table

Outputs four formal therapeutic principles as a structured CSV. Each entry includes evidence tier, supporting metric, falsifiable operator prediction, and required validation experiment.

> **This script generates formal targeting hypotheses, not clinical recommendations. It was not used as therapeutic evidence.**

```bash
python scripts_gse278936/step20_interface_targeting_principles.py \
  --outdir results_interface_regime
```

**Output:** `interface_targeting_principles.csv`

---

### Step 21 — Pre/Post Therapy Response Template

Computes signed Delta metrics (post − pre) at the interface region for paired treatment samples.

| Metric | Therapeutic prediction |
|---|---|
| Δcoexact_exact_ratio | Should decrease (Principle 3) |
| Δks_like | Instability reduction |
| ΔCDIS | Constraint regime exit |
| ΔKTS exhaustion bias | Transition dynamics change |

```bash
python scripts_gse278936/step21_pre_post_therapy_template.py \
  --pre-id  GSM_pre_treatment_ID \
  --post-id GSM_post_treatment_ID \
  --statsdir results_paired \
  --outdir   results_interface_regime \
  --label    "Patient_01_anti_VEGF"
```

**Output:** `paired_response_metrics.csv`

> **Step 21 is a prospective paired-sample template only.** It should not be cited as evidence unless true paired pre/post-treatment samples are provided. No paired treatment data were used in the current study.

---

### Step 22 — Regime Summary Table

Generates the manuscript-ready two-regime model table: all layers, metrics, TNBC and GSE278936 results, operator interpretations, analogy notes, and global safety statement.

```bash
python scripts_gse278936/step22_regime_summary_table.py \
  --outdir results_interface_regime
```

**Output:** `table_two_regime_model.csv`

---

## Expected Outputs

| File | Source |
|---|---|
| `{sid}_regime_assignment.csv` | Step 15 |
| `{sid}_operator_regime_metrics.csv` | Step 16 |
| `cohort_operator_regime_summary.csv` | Step 16 |
| `cohort_two_regime_test.csv` | Step 17 |
| `cohort_bulk_vs_interface_null.csv` | Step 18 |
| `{sid}_constraint_score.csv` | Step 19 |
| `cohort_constraint_score_summary.csv` | Step 19 (includes `cdis_formula` column) |
| `interface_targeting_principles.csv` | Step 20 |
| `paired_response_metrics.csv` | Step 21 |
| `table_two_regime_model.csv` | Step 22 |

---

## Batch Execution

```bash
# Step 1: Assign regimes
python scripts_gse278936/step15_regime_assignment.py \
  --statsdir results_gse278936 \
  --outdir   results_interface_regime \
  --coexact-bulk-q 0.50

# Step 2: Copy spots/edges files into the working directory
mkdir -p results_interface_regime
cp Results_TNBC_rebuild_gse278936/*_spots_coexact_energy.csv results_interface_regime/
cp Results_TNBC_rebuild_gse278936/*_edges_hodge.csv          results_interface_regime/

# Steps 3–8: Core two-regime analyses
STATSDIR=results_interface_regime

python scripts_gse278936/step16_operator_regime_metrics.py \
  --statsdir $STATSDIR --outdir $STATSDIR

python scripts_gse278936/step17_two_regime_test.py \
  --statsdir $STATSDIR --outdir $STATSDIR --n-perm 300 --seed 123

python scripts_gse278936/step18_bulk_equilibrium_null.py \
  --statsdir $STATSDIR --outdir $STATSDIR --n-perm 300 --seed 123

python scripts_gse278936/step19_constraint_regime_score.py \
  --statsdir $STATSDIR --outdir $STATSDIR --n-perm 300 --seed 123

python scripts_gse278936/step20_interface_targeting_principles.py \
  --outdir $STATSDIR

python scripts_gse278936/step22_regime_summary_table.py \
  --outdir $STATSDIR

# Optional: paired treatment samples only
python scripts_gse278936/step21_pre_post_therapy_template.py \
  --pre-id  PRE_SAMPLE_ID \
  --post-id POST_SAMPLE_ID \
  --statsdir results_paired \
  --label    "Patient_01_Treatment"
```

---

## Reproducibility

All stochastic steps use `--seed 123`. Results are fully deterministic given a fixed seed.

**Class-imbalance caveat:** Some sections have small bulk-like node sets (n < 10). Permutation tests preserve group sizes, so inference remains valid, but sections with `n_bulk < 10` should be interpreted with caution. Step 19 flags these in the `status` column. GSM8558019 (n_bulk = 6) is the canonical example in the GSE278936 cohort.

**Dependency chain:**

```
Step 15 → Step 16 → Step 17
Step 15 → Step 18
Step 15 + Step 16 → Step 19
Step 19 + optional KTS outputs + paired sample IDs → Step 21
Steps 17–19 → Step 22
```

---

## Step 23 Extensions (Experimental)

> Step 23 results are **experimental** and **not part of the core two-regime evidence chain** (Steps 17–19). They should be reported separately and labeled as exploratory until fully validated.

| Script | What it tests | Status |
|---|---|---|
| `step23a_power_spectrum_test.py` | Spectral shape: periodic vs random vs aperiodic candidate | Experimental |
| `step23b_local_vs_global_prediction.py` | Local neighborhood vs global spectral predictability | Experimental |
| `step23c_spatial_autocorrelation.py` | ACF shape: monotone decay vs oscillation | Experimental |

Run after Steps 15–19. Use the same `--statsdir results_interface_regime`.

### Step 23a — Power Spectrum Shape

Tests whether the coexact field's graph power spectrum has dominant global frequencies (periodic) or is continuous without peaks (aperiodic candidate). Most sections showed strong low-frequency spectral concentration; however, the coexact field was not consistently less peaked than the exact field (9/23 sections), so spectral peak analysis alone is insufficient to characterize the interface as globally aperiodic.

### Step 23b — Local vs Global Predictability

Compares local neighborhood prediction (k-hop means) against global spectral reconstruction (top-k eigenmodes). Across all 23/23 sections, local reconstruction substantially outperformed global spectral reconstruction:

| Metric | Value |
|---|---|
| Median best local R² | 0.637 |
| Median global spectral R² | 0.077 |
| Median local–global gap | +0.561 |
| Sign test p | 1.19 × 10⁻⁷ |

This demonstrates that coexact interface organization is **highly locally predictable but poorly globally compressible** — inconsistent with a globally periodic lattice, and inconsistent with random noise.

### Step 23c — Spatial Autocorrelation

Spatial autocorrelation functions (ACF) characterize the form of spatial organization. Most sections (17/23) displayed positive short-range autocorrelation with monotone decay and no oscillatory recurrence — finite-range constrained organization without long-range periodicity.

| Statistic | Value |
|---|---|
| Median lag-1 ACF | 0.482 |
| Median decay length τ | 1.47 graph hops |
| Sign test p | 0.017 |

### Step 23 Interpretation

The three analyses together support a **locally constrained, globally weakly coherent** interface regime. The coexact interaction structure is strongly determined by local neighborhood geometry and propagates over finite spatial ranges, but is not reducible to globally periodic spectral organization.

> **Important caveat:** Step 23a preliminary output may classify some sections as PERIODIC. Step 23 results should not be used to support an aperiodic-organization claim unless Step 23b–c results and final summaries jointly justify it.

---

## Relationship to Main Manuscript

The two-regime framework and associated therapeutic principles are discussed in the manuscript Discussion section ("Two dynamical regimes and therapeutic implications"). This pipeline is the computational substrate for those claims.

The analyses support a two-regime tissue organization:

1. **Bulk-like regime** — lower coexact activity, greater equilibrium-like behavior
2. **Interface regime** — elevated coexact interaction, high KS-like instability, enhanced operator constraint dominance (CDIS), locally structured but globally weakly coherent organization

Step 23 further refines this interpretation by demonstrating that interface organization is strongly locally predictable yet poorly captured by global spectral modes — consistent with a constrained nonequilibrium interaction regime rather than a passive mixing boundary.

All therapeutic principles are mechanistic hypotheses derived from operator organization and regime structure. They are not clinical recommendations. The pipeline enforces this distinction through evidence-tier classification in Step 20.

---

## Author

**Anas Enoch, MD**  
Mohammed VI University of Health Sciences (UM6SS), Casablanca  
anas_nour@um5.ac.ma
