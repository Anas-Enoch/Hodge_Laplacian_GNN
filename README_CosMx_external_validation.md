# CosMx External Validation
## Cross-technology and Cross-resolution Replication (Single-cell Spatial Transcriptomics)

This document describes the full reproducible pipeline used to validate the Hodge–Laplacian framework on CosMx single-cell spatial transcriptomics data, as reported in Section 16.13 of the manuscript.

**Scientific goal:** Test whether interface-localized coexact non-gradient interaction enrichment — established in the GSE210616 Visium TNBC discovery cohort at spot-level resolution — replicates at single-cell resolution in an independent cohort across a different measurement technology. Rotational interpretation is restricted to explicit upper-Hodge / face-level construction and should not be generalized to all coexact structure.

**Framework alignment:** In the final multilayer framework (manuscript v7), CosMx is used as **cross-technology validation of the local coexact-enrichment layer only**. It is not used for corrected Zeta energy-matched nulls, graph-KS instability analysis, or KTS transition-bias analysis. Those analyses are documented in `README_GSE278936_PIPELINE.md` and `README_TNBC.md`.

**Final CosMx interpretation:** CosMx validates that coexact non-gradient interaction enrichment at tumor–immune interfaces is not specific to Visium spot-level data and can be recovered at single-cell spatial resolution. It does not extend the final manuscript's corrected spectral, KTS, or KS claims unless those analyses are explicitly added.

---

## What is Included

The following processed summary files are **already included** and allow verification of all reported results without re-running the full pipeline:

| File | Contents |
|---|---|
| `results_cosmx/cosmx_breast_hodge_summary.csv` | Per-FOV coexact enrichment ratios, permutation p-values, region sizes |
| `results_cosmx/cosmx_control_wedges_cohort_summary.csv` | Control operator cohort summary |
| `results_cosmx/cosmx_density_control_cohort_summary.csv` | Density control cohort summary |
| `results_cosmx/cosmx_operator_robustness_cohort_summary.csv` | Operator robustness cohort summary |
| `results_cosmx/cosmx_protein_anchoring_cohort_summary.csv` | Protein anchoring cohort summary |
| `results_cosmx/cosmx_remeshing_cohort_summary.csv` | Remeshing stability cohort summary |

These files support the CosMx-specific claim that coexact energy enrichment at tumor–immune interfaces replicates at single-cell resolution. They should **not** be interpreted as evidence for interface-specific spectral organization, KTS transition bias, or KS-like instability unless those analyses are explicitly performed.

**Raw CosMx expression data are not redistributed** due to data usage and licensing constraints.

---

## Dataset

| Field | Value |
|---|---|
| Platform | NanoString CosMx Spatial Molecular Imager |
| Dataset | Breast Cancer Multiomic (RNA + protein) |
| URL | https://nanostring.com/products/cosmx-spatial-molecular-imager/ffpe-dataset/ |

Place downloaded files at:

```
data/Breast_Multiomic/Flatfiles_RNA/flatFiles/BreastCancer/
├── BreastCancer_exprMat_file.csv.gz
└── BreastCancer_metadata_file.csv.gz
```

---

## Circularity Boundary (Hard Constraint)

| Program | Genes |
|---|---|
| Tumor | EPCAM, KRT8, KRT18, KRT19, ERBB2, MUC1, TACSTD2 |
| Immune | PTPRC, CD3D, CD3E, NKG7, CD68, C1QA, CXCL9, CXCL10 |
| Stroma | COL1A1, COL1A2, DCN, LUM, TAGLN, POSTN, FAP |

ACTA2 and VIM were removed from the stromal construction panel to reduce smooth-muscle/fibroblast and immune-activation overlap; POSTN and FAP are added to align with the final manuscript marker logic.

---

## Pipeline Overview

```
Step 01 → canonical cell table
Step 02 → region assignment
Step 03 → graph + wedge + upper-Hodge decomposition + coexact enrichment  ← primary CosMx result
Step 04 → control wedge analysis                                           ← confound exclusion
Step 05 → density control                                                  ← confound exclusion
Step 06 → operator robustness                                              ← construction robustness
Step 07 → protein anchoring                                                ← cross-modality validation
Step 08 → remeshing sensitivity                                            ← graph-construction invariance
```

CosMx Step 03 uses a Delaunay face complex and upper-Hodge decomposition, whereas GSE278936 and rebuilt TNBC corrected analyses use graph-based controls. Therefore, CosMx coexact enrichment should be compared as a **direction-of-effect validation**, not as a direct numerical validation of graph-curl, Zeta, KTS, or KS outputs.

---

## One-Command Run Order

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
  --out    results_cosmx/ --n-perm 500

python scripts_cosmx/step05_cosmx_density_control.py \
  --cells   results_cosmx/cosmx_breast_cells_hodge.csv.gz \
  --summary results_cosmx/cosmx_breast_hodge_summary.csv \
  --out     results_cosmx/

python scripts_cosmx/step06_cosmx_operator_robustness.py \
  --cells  results_cosmx/cosmx_breast_cells_hodge.csv.gz \
  --out    results_cosmx/ --n-perm 500

python scripts_cosmx/step07_cosmx_protein_anchoring.py \
  --cells   results_cosmx/cosmx_breast_cells_hodge.csv.gz \
  --summary results_cosmx/cosmx_breast_hodge_summary.csv \
  --out     results_cosmx/

python scripts_cosmx/step08_cosmx_remeshing_sensitivity.py \
  --cells results_cosmx/cosmx_breast_cells_hodge.csv.gz \
  --fovs  37,41,46,58,76,87,94,104,120,132 \
  --out   results_cosmx/
```

---

## Step-by-Step Reference

### Step 01 — Canonical Cell Table
Reads CosMx expression matrix and metadata; computes per-FOV z-scored program scores.
Output: `cosmx_breast_canonical_cells.csv.gz`

### Step 02 — Region Assignment
Classifies cells as tumor, immune, interface, tumor_core, or other per FOV.
Tumor-core minimum distance = 3× base_spacing; interface radius = 2× base_spacing.
Output: `cosmx_breast_cells_with_regions.csv.gz`

### Step 03 — Graph, Wedge, Hodge, Enrichment *(primary CosMx result)*
Builds Delaunay complex per FOV; computes antisymmetric tumor–immune wedge; decomposes using the **upper-Hodge simplicial formulation**; tests whether coexact energy is enriched at interface cells relative to tumor-core cells.

The upper-Hodge CosMx construction supports circulation-like interpretation within the Delaunay face complex, but the **primary cohort-level claim remains coexact energy enrichment**. This should not be conflated with the graph-curl proxy used in GSE278936.

Upper-Hodge is **not directly comparable** to lower-Hodge used in the Visium pipeline. Cross-technology comparison is valid for direction (R > 1) and rank order; comparison of absolute coexact fractions is not.

Outputs: `cosmx_breast_cells_hodge.csv.gz`, `cosmx_breast_hodge_summary.csv`

### Step 04 — Control Wedge Analysis
Tests whether signal is reproduced by biologically unrelated operators.
Note: `immune × housekeeping` is a one-sided gradient diagnostic, not a null control.
Outputs: `cosmx_control_wedges_per_fov.csv`, `cosmx_control_wedges_cohort_summary.csv`

### Step 05 — Density Control
Tests whether enrichment is explained by cell density differences.
Primary test: structural sign test on per-FOV T2 density ratios.
Outputs: `cosmx_density_control_per_fov.csv`, `cosmx_density_control_cohort_summary.csv`

### Step 06 — Operator Robustness
Five operators in decreasing information order: proxy → normalised → rank → threshold → sign-only.
Outputs: `cosmx_operator_robustness_per_fov.csv`, `cosmx_operator_robustness_cohort_summary.csv`

### Step 07 — Protein Anchoring
Links RNA-derived coexact energy to protein-defined tumor–immune organization.
Outputs: `cosmx_protein_anchoring_per_fov.csv`, `cosmx_protein_anchoring_cohort_summary.csv`

### Step 08 — Remeshing Sensitivity
Primary metric: sign-consistency (direction of R − 1), not magnitude stability.
Outputs: `cosmx_remeshing_per_fov.csv`, `cosmx_remeshing_cohort_summary.csv`

---

## Reproducibility Checkpoint

| Metric | Expected value |
|---|---|
| Total FOVs processed | 108 |
| Testable FOVs | 96 |
| FOVs with R > 1 | 78/96 |
| Sign test p (R > 1) | 2.2 × 10⁻¹⁰ |
| Median R (IQR) | 2.09 (1.22–3.12) |
| FOVs significant at p < 0.05 | 59/96 |
| Binomial p vs 5% type-I null | < 10⁻⁵¹ |
| Generic antisymmetry verdict | EXCLUDED |
| Cell density verdict | EXCLUDED (T2 < 1 in 77/96, p < 10⁻⁹) |
| Operator robustness (all 5 variants) | SURVIVES (all p < 10⁻⁵) |
| Protein juxtaposition | 75/108 (p < 10⁻⁵) |
| Remeshing sign-consistency | 19/19 FOVs (100%) |

These checkpoint values validate the CosMx **local coexact-enrichment layer**. They do not imply that CosMx has been tested for energy-matched Zeta collapse, KTS exhaustion attractors, or graph-KS instability.

---

## Known Caveats

1. **Upper-Hodge vs lower-Hodge.** `coexact_fraction_global` in CosMx (median ≈ 0.36) is systematically higher than in Visium (≈ 0.08–0.28) because Step 03 uses the full B₂ simplicial incidence matrix. Direction (R > 1) and rank are cross-technology comparable; absolute fractions are not.

2. **Radius graph excluded from remeshing verdict.** Three FOVs showed divergent ratios from the radius graph due to near-zero tumor-core coexact energy. Retained descriptively only.

3. **Magnitude stability is not the primary remeshing criterion.** The claim is sign-consistency, not magnitude stability.

4. **Small-core FOVs.** FOVs with n_tumor_core ≤ 5 can show extreme R values. Use `perm_p` as the inferential criterion.

---

## Compatibility with Manuscript v7

This README corresponds to the CosMx cross-technology validation component of manuscript v7. The final manuscript distinguishes coexact enrichment from higher-order spatial organization: CosMx supports the enrichment layer, while corrected Zeta, graph-curl, KTS, and KS analyses are reported separately for GSE278936 and rebuilt TNBC outputs.

Refer to the main repository README, `README_GSE278936_PIPELINE.md`, `README_TNBC.md`, and manuscript v7 for the full corrected multilayer interpretation.
