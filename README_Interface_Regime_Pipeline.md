# Interface Regime Pipeline
## scripts_gse278936/ — Two-Regime Operator Testing and Therapeutic Hypothesis Generation

---

## SAFETY STATEMENT (Read First)

> These analyses generate formal operator-derived hypotheses, not clinical
> recommendations. The physics analogies (Maxwell–Boltzmann and Euler–Bernoulli)
> are **interpretive operator analogies** and do not claim that tumor tissue
> literally follows Maxwell–Boltzmann or Euler–Bernoulli dynamics. The pipeline
> tests gradient-compatible bulk versus non-integrable constraint-dominated
> interface using coexact enrichment, graph curvature, KS-like instability,
> and KTS exhaustion bias. All therapeutic principles require orthogonal
> experimental validation before any clinical inference can be drawn.

---

## Purpose

This pipeline tests the two-regime hypothesis derived from the main manuscript:

> Tumor–immune interfaces represent a distinct non-equilibrium operator regime
> with elevated non-gradient interaction, instability-like activity, and a shifted
> constraint configuration relative to the near-equilibrium gradient-compatible
> tumor bulk. Therapeutic hypotheses may be generated around interface-specific operator regimes;
> if validated experimentally, they would suggest targeting the interface not by amplifying
> gradients, but by disrupting or exploiting its non-integrable interaction structure.
> These are formal hypotheses, not clinical recommendations.

This is an operator-level claim. The pipeline tests whether the claim is
supported by quantitative operator metrics across spatial transcriptomics cohorts.

---

## The Two-Regime Model (Operator Formulation)

### Bulk regime (near-equilibrium, gradient-compatible)

In the tumor bulk, the interaction field is approximately integrable under a
gradient-only model. The exact (gradient) component dominates; coexact energy
is low; the KS-like instability proxy is low. In operator terms, the bulk
resembles a near-equilibrium statistical field where program interactions are
diffuse and directionally unstructured.

**Analogy:** Near-equilibrium statistical dynamics where local gradients dominate.
**Operator signature:** Low coexact energy, low graph curvature, low CDIS. Note: when flux_exact is unavailable, coexact_exact_ratio is not used as a primary quantitative claim; CDIS is computed using the reduced formula (coexact energy, bilaplacian magnitude, nonlinear gradient energy).

### Interface regime (non-equilibrium, constraint-dominated)

At the tumor–immune boundary, the interaction field is not reducible to a
gradient — coexact enrichment was observed in 40/40 TNBC sections and in the majority of GSE278936 sections; verify exact count against cohort_summary.csv. The graph-Laplacian curvature of the coexact field is higher;
the KS-like instability proxy is elevated (median fold ≈ 8.96 over tumor core);
and the KTS analysis shows a strong exhaustion-directed transition bias.

**Analogy:** Constrained, curvature-driven boundary dynamics where gradients fail
and operator-level constraints govern the field.
**Operator signature:** High coexact energy or coexact/exact ratio (when exact energy is available), high graph curvature, high CDIS,
KTS exhaustion bias.

### The analogies are operator-level, not mechanistic

The Maxwell–Boltzmann reference characterizes the bulk as a near-equilibrium
baseline against which the interface deviates. Euler–Bernoulli is used only as an analogy for constraint-dominated boundary behavior and is not a mechanistic claim; it characterizes the interface as a regime where boundary conditions and local program opposition govern dynamics in operator terms. **These are operator analogies.**
The claim is that the two regimes have formally different operator signatures —
not that tumor tissue literally obeys these physical equations.

---

## Pipeline Structure

```
scripts_gse278936/
├── step15_regime_assignment.py          Classify nodes into operator regimes
├── step16_operator_regime_metrics.py    Compute per-regime operator summaries
├── step17_two_regime_test.py            Permutation test: interface vs bulk
├── step18_bulk_equilibrium_null.py      Bulk-matched null: near-equilibrium baseline
├── step19_constraint_regime_score.py    CDIS: composite constraint-dominated score
├── step20_interface_targeting_principles.py  Therapeutic hypothesis table
├── step21_pre_post_therapy_template.py  Pre/post treatment response template [PROSPECTIVE TEMPLATE — not used as evidence unless paired treatment data are provided]
├── step22_regime_summary_table.py       Manuscript-ready two-regime model table
└── README_Interface_Regime_Pipeline.md  This file
```

**Prerequisites:** Steps 01–14 of the main pipeline must be completed first.
Specifically: spots_coexact_energy.csv, edges_hodge.csv, and (for KTS analyses)
the kts_transition_bias outputs must exist.

---

## Step-by-Step Reference

### Step 15 — Regime Assignment
Classifies each spot into: `bulk_like`, `interface_like`, `interface_adjacent`,
`stromal_like`, `immune_like`, `other`.

Classification logic:
- `bulk_like`: tumor_core region AND coexact_energy < section Q50
- `interface_adjacent`: tumor_core region AND coexact_energy ≥ section Q50
- `interface_like`: interface/interface_like annotated region
- `stromal_like`, `immune_like`: stroma/immune annotated regions

**Key distinction:** `bulk_like` nodes are the near-equilibrium reference.
`interface_adjacent` nodes are in the tumor core but carry elevated coexact energy.

```bash
python scripts_gse278936/step15_regime_assignment.py \
  --statsdir results_gse278936 \
  --outdir results_interface_regime \
  --coexact-bulk-q 0.50
```
Output: `{sid}_regime_assignment.csv`

---

### Pre-Step 16 — Copy required files into working directory

Step 16 requires both the regime assignment files (from Step 15) and the
original spots/edges files in the same `--statsdir`. Copy them before running:

```bash
mkdir -p results_interface_regime

# Copy from main GSE278936 pipeline outputs
cp results_gse278936/*_spots_coexact_energy.csv results_interface_regime/
cp results_gse278936/*_edges_hodge.csv results_interface_regime/
# Step 15 writes regime_assignment files to results_interface_regime/ directly
```

This was the source of the "missing inputs" error when running Step 16
immediately after Step 15 without the original spots/edges files present.

---

### Step 16 — Operator Regime Metrics
For each section, computes per-regime medians of:
- `coexact_energy` — non-gradient interaction intensity
- `exact_energy_node` — gradient component (baseline)
- `coexact_exact_ratio` — non-gradient dominance ratio
- `graph_curvature` — |Lu| (Laplacian applied to coexact field)
- `bilaplacian_mag` — |L²u| (higher-order stabilizing structure)
- `nonlin_grad_energy` — squared edge differences per node

```bash
python scripts_gse278936/step16_operator_regime_metrics.py 
--statsdir results_interface_regime\
--outdir results_interface_regime
```
Output: `{sid}_operator_regime_metrics.csv`, `cohort_operator_regime_summary.csv`

---

### Step 17 — Two-Regime Permutation Test
Tests whether interface-like nodes differ from bulk-like nodes in three
primary metrics (always tested) and one conditional metric:

**Primary metrics (always tested):**
1. `coexact_energy` — non-gradient interaction intensity
2. `graph_curvature` — |Lu| (Laplacian applied to coexact field)
3. `ks_like` — |−Lu − L²u − |∇u|²| (KS-like instability proxy)

**Conditional metric (tested only when flux_exact is available):**
4. `coexact_exact_ratio` — non-gradient dominance ratio

The output CSV includes a `primary_metric` boolean column and an `exact_available`
column for transparency.

Null model: node-count-matched random subsample from bulk-like nodes.
One-sided p-value: fraction of null medians ≥ observed interface median.

```bash
python scripts_gse278936/step17_two_regime_test.py \
  --statsdir results_interface_regime \
  --outdir results_interface_regime \
  --n-perm 300 --seed 123
```
Output: `cohort_two_regime_test.csv`

**GSE278936 result (n = 23 sections):**
Interface-like nodes significantly exceeded bulk-like nodes on all three primary metrics
in 23/23 sections (p < 0.05 each):
- Coexact energy fold ≈ 6.38×
- Graph curvature fold ≈ 9.68×
- KS-like magnitude fold ≈ 17.84×

The conditional coexact/exact ratio was significant in 22/23 sections.

---

### Step 18 — Bulk Equilibrium Null Model
Treats bulk-like nodes as the near-equilibrium reference distribution.
Bootstrapped comparison of interface vs bulk in coexact_energy and ks_like.

Tests whether interface nodes significantly exceed what a bulk-matched
random sample would produce — confirming that the interface constitutes a
distinct operator regime, not a continuous gradient above the bulk.

```bash
python scripts_gse278936/step18_bulk_equilibrium_null.py \
  --statsdir results_interface_regime \
  --outdir results_interface_regime \
  --n-perm 300 --seed 123
```
Output: `cohort_bulk_vs_interface_null.csv`

**GSE278936 result (n = 23 sections):**
The bulk equilibrium null was rejected in 23/23 sections for both metrics:
- Coexact energy: median interface/bulk fold ≈ 5.48×
- KS-like instability: median interface/bulk fold ≈ 19.54×
- Sign test p = 1.19 × 10⁻⁷

This rules out the explanation that interface nodes are simply a high-energy
random subset of the tissue; they constitute a structurally distinct regime.

---

### Step 19 — Constraint-Dominated Interface Score (CDIS)
CDIS uses a **conditional formula** with **robust z-scores** (median-centered, IQR-scaled):

- When exact energy is available: `CDIS = rz(coexact/exact) + rz(|L²u|) + rz(nonlin_grad)`
- When exact energy is unavailable: `CDIS = rz(coexact_energy) + rz(|L²u|) + rz(nonlin_grad)`

Robust z-scores are used to prevent outlier inflation from the extreme right-skewed ratio distributions. The formula used is written to the `cdis_formula` column for auditability.

Combines three orthogonal operator metrics into a single composite score.
CDIS is a **within-section configuration score**, not an absolute biological magnitude.
It tests whether interface nodes occupy an operator-space regime distinct from bulk-like nodes.
A positive interface–bulk gap means interface nodes consistently exceed the bulk on the
combined constraint-dominated operator signature.

Tests CDIS enrichment at interface vs bulk using permutation null.
Reports cohort-level sign test.

```bash
python scripts_gse278936/step19_constraint_regime_score.py \
  --statsdir results_interface_regime \
   --outdir results_interface_regime \
  --n-perm 300 --seed 123
```
Outputs: `{sid}_constraint_score.csv`, cohort_constraint_score_summary.csv | Step 19 — includes cdis_formula for CDIS auditability

**GSE278936 result (n = 23 sections):**
CDIS identified a distinct constraint-dominated interface regime in 22/23 sections
(sign test p = 2.86 × 10⁻⁶):
- Median interface CDIS = +6.09
- Median bulk CDIS = −0.17
- Median interface–bulk gap = +6.07 (IQR 5.14–8.17)

GSM8558019 was the only non-significant section and had only 6 bulk-like nodes, limiting the stability of the bulk estimate. It is therefore treated as a low-power section rather than evidence against the cohort-level effect.

---

### Step 20 — Therapeutic Hypothesis Table
Outputs four formal therapeutic principles as a structured CSV with:
- Evidence tier (Tier 2 = operator inference; Tier 3 = speculative)
- Supporting metric
- Operator prediction (falsifiable)
- Required validation experiment

**This script generates formal targeting hypotheses, not clinical recommendations. It was not used as therapeutic evidence.**

```bash
python scripts_gse278936/step20_interface_targeting_principles.py --outdir results_interface_regime
```
Output: `interface_targeting_principles.csv`

---

### Step 21 — Pre/Post Therapy Response Template
Computes Delta metrics between paired pre/post-treatment sections:
- Δcoexact_exact_ratio (Principle 3 prediction: should decrease)
- Δks_like (instability reduction)
- ΔCDIS (constraint regime exit)
- ΔKTS exhaustion bias (transition dynamics change)

```bash
python scripts_gse278936/step21_pre_post_therapy_template.py \
  --pre-id  GSM_pre_treatment_ID \
  --post-id GSM_post_treatment_ID \
  --statsdir results_paired \
  --outdir results_interface_regime \
  --label   "Patient_01_anti_VEGF"
```
Output: `paired_response_metrics.csv`

> **Step 21 is a prospective paired-sample template only.** It should not be
> cited as evidence unless true paired pre/post-treatment samples are provided.
> No paired treatment data were used in the current study.

---

### Step 22 — Regime Summary Table
Generates the manuscript-ready two-regime model table with all layers,
metrics, TNBC and GSE278936 results, operator interpretations, and
analogy notes. Includes the global safety statement.

```bash
python scripts_gse278936/step22_regime_summary_table.py --outdir results_interface_regime
```
Output: `table_two_regime_model.csv`

---

## Expected Outputs

| File | Source step |
|---|---|
| `{sid}_regime_assignment.csv` | Step 15 |
| `{sid}_operator_regime_metrics.csv` | Step 16 |
| `cohort_operator_regime_summary.csv` | Step 16 |
| `cohort_two_regime_test.csv` | Step 17 |
| `cohort_bulk_vs_interface_null.csv` | Step 18 |
| `{sid}_constraint_score.csv` | Step 19 |
| `cohort_constraint_score_summary.csv` | Step 19 |
| `interface_targeting_principles.csv` | Step 20 |
| `paired_response_metrics.csv` | Step 21 |
| `cohort_constraint_score_summary.csv` (with `cdis_formula` column) | Step 19 — for auditability of CDIS formula used per cohort |
| `table_two_regime_model.csv` | Step 22 |

---

## Batch Execution

mkdir -p results_interface_regime

python scripts_gse278936/step15_regime_assignment.py \
  --statsdir results_gse278936 \
  --outdir results_interface_regime \
  --coexact-bulk-q 0.50

cp results_gse278936/*_spots_coexact_energy.csv results_interface_regime/
cp results_gse278936/*_edges_hodge.csv results_interface_regime/

STATSDIR=results_interface_regime

python scripts_gse278936/step16_operator_regime_metrics.py  --statsdir $STATSDIR --outdir $STATSDIR
python scripts_gse278936/step17_two_regime_test.py          --statsdir $STATSDIR --outdir $STATSDIR --n-perm 300 --seed 123
python scripts_gse278936/step18_bulk_equilibrium_null.py    --statsdir $STATSDIR --outdir $STATSDIR --n-perm 300 --seed 123
python scripts_gse278936/step19_constraint_regime_score.py  --statsdir $STATSDIR --outdir $STATSDIR --n-perm 300 --seed 123
python scripts_gse278936/step20_interface_targeting_principles.py --outdir $STATSDIR
python scripts_gse278936/step22_regime_summary_table.py     --outdir $STATSDIR


# For paired samples (requires post-treatment data):
python scripts_gse278936/step21_pre_post_therapy_template.py \
  --pre-id  PRE_SAMPLE_ID \
  --post-id POST_SAMPLE_ID \
  --statsdir results_paired \
  --label   "Patient_01_Treatment"
```

---

## Reproducibility

All stochastic steps use `--seed 123` as the default.

**Class-imbalance caveat:** Some sections contain small bulk-like node sets
(n < 10). Permutation tests preserve group sizes, ensuring valid inference
under class imbalance. Sections with n_bulk < 10 should be interpreted with
caution; Step 19 flags these in the `status` column. GSM8558019 (n_bulk = 6)
is the canonical example in the GSE278936 cohort. Results are
deterministic given fixed seed. All steps are independent of each other
except for the dependency chain:

```
Step 15 → Step 16 → Step 17
Step 15 → Step 18
Step 15 + Step 16 → Step 19
Step 19 + optional KTS outputs + paired pre/post sample IDs → Step 21
Steps 17–19 → Step 22
```

---

## Step 23 Extensions (Experimental)

Step23a preliminary output may classify sections as PERIODIC; therefore Step23 results should not be used to support an aperiodic-organization claim unless Steps23b–c and final summaries justify it. Current Step23 analyses remain exploratory and excluded from the core manuscript claim.

| Script | Tests | Status |
|---|---|---|
| `step23a_power_spectrum_test.py` | Spectral shape (periodic vs random vs aperiodic) | Experimental |
| `step23b_local_vs_global_prediction.py` | Local rule vs global template prediction | Experimental |
| `step23c_spatial_autocorrelation.py` | ACF shape (monotone decay vs oscillation) | Experimental |

Run after Steps 15–19. Results should be reported separately from the
core two-regime evidence (Steps 17–19) and labeled as exploratory.

---

## Relationship to Main Manuscript 

The two-regime model and therapeutic principles are discussed in the Discussion
section of the manuscript ("Two dynamical regimes and therapeutic implications").
This pipeline provides the computational substrate for those claims. The
Discussion section specifies that the four therapeutic principles are formal
hypotheses (not clinical recommendations), and this pipeline enforces that
distinction through evidence tier classification in Step 20.

Tumor–immune interfaces represent a distinct non-equilibrium operator regime with elevated non-gradient interaction intensity, high KS-like instability, and a CDIS-defined constraint configuration distinct from bulk-like tissue. These results support a two-regime operator model and generate formal, experimentally testable therapeutic hypotheses; they do not constitute clinical recommendations.

---

## Author

Anas Enoch, MD
Mohammed VI University of Health Sciences (UM6SS), Casablanca
anas_nour@um5.ac.ma
