# CosMx External Validation

**Cross-technology and Cross-resolution Replication (Single-cell Spatial Transcriptomics)**

This document describes the full reproducible pipeline used to validate the Hodge–Laplacian transport framework on CosMx single-cell spatial transcriptomics data, as reported in Section 16.13 of the manuscript.

**Scientific goal:** Test whether interface-localised coexact (rotational) transport enrichment — established in the GSE210616 Visium TNBC discovery cohort at spot-level resolution — replicates at single-cell resolution in an independent cohort across a different measurement technology.

---

## What is included in this repository

The following processed summary files are **already included** and allow verification of all reported results without re-running the full pipeline or downloading raw data:

| File | Contents |
|------|----------|
| `results_cosmx/cosmx_breast_hodge_summary.csv` | Per-FOV enrichment ratios, permutation p-values, region sizes |
| `results_cosmx/cosmx_control_wedges_cohort_summary.csv` | Control operator cohort summary |
| `results_cosmx/cosmx_density_control_cohort_summary.csv` | Density control cohort summary |
| `results_cosmx/cosmx_operator_robustness_cohort_summary.csv` | Operator robustness cohort summary |
| `results_cosmx/cosmx_protein_anchoring_cohort_summary.csv` | Protein anchoring cohort summary |
| `results_cosmx/cosmx_remeshing_cohort_summary.csv` | Remeshing stability cohort summary |

**Raw CosMx expression data are not redistributed** in this repository due to data usage and licensing constraints. See [Dataset](#dataset) for download instructions.

---

## Dataset

You must download the raw data manually from the official NanoString source:

| Field | Value |
|-------|-------|
| Platform | NanoString CosMx Spatial Molecular Imager |
| Dataset | Breast Cancer Multiomic (RNA + protein) |
| URL | https://nanostring.com/products/cosmx-spatial-molecular-imager/ffpe-dataset/ |

After download, place the files in the expected location:

```
data/Breast_Multiomic/Flatfiles_RNA/flatFiles/BreastCancer/
├── BreastCancer_exprMat_file.csv.gz
└── BreastCancer_metadata_file.csv.gz
```

All outputs are written to `results_cosmx/`. All scripts read and write **relative to the repository root**.

---

## Repository Layout

```
Hodge_Laplacian_GNN/
├── scripts_cosmx/
│   ├── step01_cosmx_build_canonical_cells.py
│   ├── step02_cosmx_define_regions.py
│   ├── step03_cosmx_graph_wedge.py
│   ├── step04_cosmx_control_wedges.py
│   ├── step05_cosmx_density_control.py
│   ├── step06_cosmx_operator_robustness.py
│   ├── step07_cosmx_protein_anchoring.py
│   └── step08_cosmx_remeshing_sensitivity.py
├── data/
│   └── Breast_Multiomic/                    ← download here
│       └── Flatfiles_RNA/flatFiles/BreastCancer/
└── results_cosmx/                           ← all outputs written here
    ├── cosmx_breast_hodge_summary.csv       ← already included
    ├── cosmx_control_wedges_cohort_summary.csv
    ├── cosmx_density_control_cohort_summary.csv
    ├── cosmx_operator_robustness_cohort_summary.csv
    ├── cosmx_protein_anchoring_cohort_summary.csv
    └── cosmx_remeshing_cohort_summary.csv
```

---

## Environment Setup

```bash
source .venv/bin/activate
```

The CosMx pipeline shares the same environment as the main TNBC pipeline. Reproduce the environment from the repository's pinned environment files:

```bash
pip install -r requirements.txt   # or: pip install -e .
```

Python version and all dependency versions are pinned in the repository. No additional packages beyond those required by the main pipeline are needed.

---

## Circularity Boundary (Hard Constraint)

The following gene sets define the wedge operator (Steps 01–03) and are **forbidden from any downstream validation step** (Steps 04–08):

| Program | Genes |
|---------|-------|
| Tumor | EPCAM, KRT8, KRT18, KRT19, ERBB2, MUC1, TACSTD2 |
| Immune | PTPRC, CD3D, CD3E, NKG7, CD68, C1QA, CXCL9, CXCL10 |
| Stroma | COL1A1, COL1A2, DCN, LUM, TAGLN, ACTA2, VIM |

Any validation marker or covariate must be outside these three gene sets.

---

## Pipeline Overview

Eight sequential scripts. Each script is self-contained and reads the output of the previous step.

```
Step 01 → canonical cell table
Step 02 → region assignment
Step 03 → graph + wedge + Hodge decomposition + enrichment  ← primary result
Step 04 → control wedge analysis                            ← confound exclusion
Step 05 → density control                                   ← confound exclusion
Step 06 → operator robustness                               ← construction robustness
Step 07 → protein anchoring                                 ← cross-modality validation
Step 08 → remeshing sensitivity                             ← graph-construction invariance
```

---

## One-Command Run Order

Copy-paste sequence to run the full pipeline from scratch:

```bash
source .venv/bin/activate

python scripts_cosmx/step01_cosmx_build_canonical_cells.py \
  --expr data/Breast_Multiomic/Flatfiles_RNA/flatFiles/BreastCancer/BreastCancer_exprMat_file.csv.gz \
  --meta data/Breast_Multiomic/Flatfiles_RNA/flatFiles/BreastCancer/BreastCancer_metadata_file.csv.gz \
  --out  results_cosmx/cosmx_breast_canonical_cells.csv.gz

python scripts_cosmx/step02_cosmx_define_regions.py \
  --cells               results_cosmx/cosmx_breast_canonical_cells.csv.gz \
  --out                 results_cosmx/cosmx_breast_cells_with_regions.csv.gz \
  --tumor-quantile       0.75 \
  --immune-quantile      0.75 \
  --radius-multiplier    2.0 \
  --core-radius-multiplier 3.0

python scripts_cosmx/step03_cosmx_graph_wedge.py \
  --cells      results_cosmx/cosmx_breast_cells_with_regions.csv.gz \
  --out-cells  results_cosmx/cosmx_breast_cells_hodge.csv.gz \
  --out-summary results_cosmx/cosmx_breast_hodge_summary.csv \
  --n-perm     1000 \
  --hodge-type upper

python scripts_cosmx/step04_cosmx_control_wedges.py \
  --cells  results_cosmx/cosmx_breast_cells_hodge.csv.gz \
  --out    results_cosmx/ \
  --n-perm 500

python scripts_cosmx/step05_cosmx_density_control.py \
  --cells   results_cosmx/cosmx_breast_cells_hodge.csv.gz \
  --summary results_cosmx/cosmx_breast_hodge_summary.csv \
  --out     results_cosmx/

python scripts_cosmx/step06_cosmx_operator_robustness.py \
  --cells  results_cosmx/cosmx_breast_cells_hodge.csv.gz \
  --out    results_cosmx/ \
  --n-perm 500

python scripts_cosmx/step07_cosmx_protein_anchoring.py \
  --cells   results_cosmx/cosmx_breast_cells_hodge.csv.gz \
  --summary results_cosmx/cosmx_breast_hodge_summary.csv \
  --out     results_cosmx/

# Run on a representative subset first (recommended); omit --fovs for full cohort
python scripts_cosmx/step08_cosmx_remeshing_sensitivity.py \
  --cells results_cosmx/cosmx_breast_cells_hodge.csv.gz \
  --fovs  37,41,46,58,76,87,94,104,120,132 \
  --out   results_cosmx/
```

---

## Step-by-Step Reference

### Step 01 — Canonical Cell Table

Reads the CosMx expression matrix and metadata, aligns them, computes per-FOV z-scored program scores, and writes one row per cell.

**Key design decisions:**
- Z-scoring is **per-FOV** (not global), consistent with the per-FOV quantile thresholds in Step 02.
- Column detection uses header-only scan (`nrows=0`) to prevent gene dropout from NaN-valued early rows.
- Deduplication asserts uniqueness on the composite `(fov, cell_id_numeric)` key.

**Output:** `results_cosmx/cosmx_breast_canonical_cells.csv.gz`
Columns: `cell`, `cell_id_numeric`, `fov`, `x`, `y`, `cell_area_px`, `total_counts`, `protein_tumor_proxy`, `protein_immune_proxy`, `protein_myeloid_proxy`, `tumor_score`, `immune_score`, `stroma_score`, `log_total_counts`.

---

### Step 02 — Region Assignment

Classifies each cell as `tumor`, `immune`, `interface`, `tumor_core`, or `other` per FOV.

**Key design decisions:**
- Absolute score floor (default 0.0 = per-FOV mean) prevents background-noise cells in single-compartment FOVs from being promoted to the minority class.
- Tumor-core minimum distance = `3 × base_spacing` vs interface radius = `2 × base_spacing`, creating a clean gap zone between interface and comparison baseline.
- FOVs where > 80% of cells are `other` are flagged as potentially miscalibrated.

**Output:** `results_cosmx/cosmx_breast_cells_with_regions.csv.gz`

---

### Step 03 — Graph, Wedge, Hodge, Enrichment

Core analysis. Builds a Delaunay complex per FOV, computes the antisymmetric tumour–immune wedge, decomposes it via full Hodge theory, and tests coexact enrichment at interface versus tumour-core cells.

**Key design decisions:**
- **Upper-Hodge decomposition** (B₁ + B₂): full simplicial complex. See [Known Caveats](#known-caveats) — this is **not directly comparable** to the lower-Hodge used in the Visium pipeline.
- **Restricted permutation**: shuffles labels only within the interface + tumour-core pool. Matches the Visium Step 7 design.
- Vectorised wedge and node energy; lsqr tolerance 1e-10.

**Outputs:**
- `results_cosmx/cosmx_breast_cells_hodge.csv.gz` — adds `node_coexact_energy`, `node_exact_energy`, `node_harm_energy`
- `results_cosmx/cosmx_breast_hodge_summary.csv` — per-FOV enrichment ratios and permutation p-values

---

### Step 04 — Control Wedge Analysis

Tests whether the canonical signal is reproduced by biologically unrelated operators.

**Note:** `immune × housekeeping` is treated as a one-sided gradient diagnostic, not a null control. The verdict rests on (A) the THK/IHK asymmetry proof and (B) per-FOV canonical residual over IHK. Terminal output clearly separates these. Any `ONE_SIDED_GRADIENT_CONTROL` label in the CSV is expected and correct.

**Outputs:** `results_cosmx/cosmx_control_wedges_per_fov.csv`, `cosmx_control_wedges_cohort_summary.csv`

---

### Step 05 — Density Control

Tests whether enrichment is explained by cell density differences.

**Note:** Cell-level partial Spearman is not used here (unlike Visium Step 25) because `node_coexact_energy` is a graph property with spatial autocorrelation at single-cell resolution. The primary test is a structural sign test on per-FOV T2 density ratios.

**Outputs:** `results_cosmx/cosmx_density_control_per_fov.csv`, `cosmx_density_control_cohort_summary.csv`

---

### Step 06 — Operator Robustness

Five operators in decreasing information order: proxy → normalised → rank → threshold → sign-only.

**Outputs:** `results_cosmx/cosmx_operator_robustness_per_fov.csv`, `cosmx_operator_robustness_cohort_summary.csv`

---

### Step 07 — Protein Anchoring

Links RNA-derived coexact energy to protein-defined tumour–immune organisation.

**Note on terminal output:** pandas `SettingWithCopyWarning` messages that may appear during execution are benign — they arise from internal DataFrame handling and do not affect any numeric outputs. All reported results are unaffected.

**Outputs:** `results_cosmx/cosmx_protein_anchoring_per_fov.csv`, `cosmx_protein_anchoring_cohort_summary.csv`

---

### Step 08 — Remeshing Sensitivity

Tests whether the coexact enrichment direction is invariant to graph construction.

**Primary metric: sign-consistency**, not R>1 rate. A FOV that is consistently non-enriched (R<1) across all graph types is sign-consistent and counts as a stability finding.

**Outputs:** `results_cosmx/cosmx_remeshing_per_fov.csv`, `cosmx_remeshing_cohort_summary.csv`

---

## Reproducibility Contract

The following four invariants hold across any correct re-run of this pipeline:

1. **Seeds are fixed.** All permutation tests use `--seed 42` by default. Results may differ at the last digit under different seeds but not at the conclusion level.

2. **Delaunay options are fixed.** All Delaunay triangulations use `qhull_options="QJ Qbb Qc Qz"` with explicit ghost-vertex filtering (`0 ≤ vertex < n_points`).

3. **Upper-Hodge is not comparable to lower-Hodge.** The `coexact_fraction_global` values in CosMx (median ≈ 0.36) are systematically higher than in Visium (≈ 0.08–0.28) because Step 03 uses the full B₂ simplicial incidence matrix. This is a methodological difference, not a biological one. Cross-technology comparison of direction (R > 1) and rank order is valid; comparison of absolute coexact fractions is not.

4. **Radius graph is excluded from primary stability verdict.** Step 08 uses kNN (k=4,6,8) and Delaunay as primary and secondary graph types. The radius graph is retained descriptively; in small-core FOVs it produces near-zero tumour-core coexact medians and numerically divergent ratios.

---

## Reproducibility Checkpoint

A correct re-run of Steps 01–03 and verification against the provided summary CSVs should recover these cohort-level numbers:

| Metric | Expected value |
|--------|----------------|
| Total FOVs processed | 108 |
| Testable FOVs (n_interface ≥ 5 and n_tumor_core ≥ 5) | 96 |
| FOVs with R > 1 | 78/96 |
| Sign test p (R > 1) | 2.2 × 10⁻¹⁰ |
| Median R (IQR) | 2.09 (1.22–3.12) |
| FOVs significant at p < 0.05 | 59/96 |
| Binomial p vs 5% type-I null | < 10⁻⁵¹ |
| Generic antisymmetry verdict | EXCLUDED |
| Cell density verdict | EXCLUDED (T2 < 1 in 77/96, p < 10⁻⁹) |
| Operator robustness (all 5 variants) | SURVIVES (all p < 10⁻⁵) |
| Protein juxtaposition (positive FOVs) | 75/108 (p < 10⁻⁵) |
| Remeshing sign-consistency | 19/19 FOVs (100%) |

These values are the checksum for a successful reproduction. If your numbers differ, check: (a) Delaunay `qhull_options` match exactly; (b) per-FOV z-scoring (not global) in Step 01; (c) restricted permutation (interface+core pool only) in Step 03; (d) `--hodge-type upper` in Step 03.

---

## Known Caveats

1. **Upper-Hodge vs lower-Hodge.** `coexact_fraction_global` in CosMx (Step 03, upper-Hodge) is not numerically comparable to Visium (lower-Hodge). The cross-technology comparison is valid for direction (R > 1) and rank, not for absolute coexact fractions.

2. **Radius graph excluded from remeshing verdict.** Three FOVs in the first test batch showed divergent ratios (R > 10⁹) from the radius graph due to near-zero tumour-core coexact energy. The radius graph is retained for descriptive context but excluded from the sign-consistency and rank-stability claims.

3. **Magnitude stability is not the primary remeshing criterion.** Absolute R values vary across graph types (expected: more edges → different coexact normalisation). The primary claim is sign-consistency (direction of R − 1), not magnitude stability.

4. **Small-core FOVs produce extreme ratio values.** FOVs with n_tumor_core ≤ 5 can show R > 100 or R > 1000 because the permutation test median at the core fluctuates near zero. These are valid results but should be interpreted in terms of direction, not magnitude. The enrichment_ratio in `cosmx_breast_hodge_summary.csv` reports raw values; `perm_p` is the inferential criterion.

---

## Contact

Refer to the main repository README and manuscript (Section 16.13) for full methodological details and statistical interpretation.
