### Manuscript version

The code corresponding to the manuscript

*Operator-Level Transport Phenotyping with PDE-Constrained Hodge–Laplacian Graph Neural Networks*

is archived in the release:

**v1.0-manuscript**

## Repository structure

This repository contains two analysis pipelines.

### scripts_visium/
Prototype operator diagnostics on Visium spatial transcriptomics data.

### scripts_tnbc/
Full TNBC cohort pipeline including:
- operator diagnostics
- curl null tests
- PDE-constrained GNN training
- hybrid transport decomposition

# Hodge_Laplacian_GNN

Discrete Hodge-theoretic analysis of spatial transcriptomics transport structure, with an emphasis on falsifiable diagnostics rather than unconstrained prediction. The repository combines operator construction on spatial graphs, region-aware biological annotation, scalar transport null models, non-gradient wedge fluxes, and Hodge decomposition into exact, coexact, and harmonic components.

## What this repository does

This project implements a stepwise pipeline for testing whether passive transport-style explanations are adequate for structured tumor microenvironments. The current worked example uses a public Visium human breast cancer dataset and builds toward region-level tests of rotational transport structure.

Core outputs include:
- spatial graph construction from Visium spots
- marker-based tissue compartment annotation
- proxy transport residual maps
- Hodge decomposition of edge fluxes
- non-gradient wedge fluxes between tumor, stromal, and immune programs
- region-level enrichment tests with permutation controls
- manuscript-ready figures and LaTeX tables

## Current biological use case

The present analysis focuses on a breast tumor tissue section and asks whether coexact, rotationally structured flux signatures are enriched outside tumor core, especially in stromal, invasive-margin-like, and immune-enriched regions. The working logic is:

1. define biologically plausible spatial regions independently from the transport model
2. build the spatial graph and simplicial operators
3. test passive scalar transport null models
4. escalate to non-gradient flux constructions
5. quantify exact vs coexact energy and local curl structure by region

## Repository structure

```text
Hodge_Laplacian_GNN
│
├── README.md
├── figures/
├── tables/
│
├── scripts_tnbc/
├── scripts_visium/
│
├── stats/
│   ├── CSV_GSM/
│   └── CSV_visium/
│
├── GSM_visium_figures/
│
├── data/
│   ├── TNBC_GSE210616/
│   ├── raw_Visium/
│   ├── README_data.md_GSM.md
│   └── README_data.md_visium.md
│
└── reference.bib
```

## Minimal pipeline overview
## Prototype Visium pipeline (`scripts_visium/`)

The following minimal pipeline describes the original Visium operator-diagnostic workflow implemented in `scripts_visium/`.

### Step 1 — marker scoring
`step1_visium_map.py`

Loads the Visium matrix and spatial coordinates, computes first-pass tumor, stromal, and immune scores, and overlays them on the tissue image.

### Step 2 — region definition
`step2_define_regions.py`

Builds biologically usable spatial regions from marker programs:
- tumor_core
- invasive_margin
- stroma
- immune_rich
- mixed_unassigned

### Step 3 — graph construction
`step3_build_spatial_graph.py`

Constructs the spot graph and the core operators:
- node-edge incidence matrix `B1`
- adjacency matrix
- graph Laplacian

### Step 4–5 — passive transport null models
`step4_compute_proxy_flux_residuals.py` and `step5_multi_proxy_residuals.py`

Evaluate diffusion-like proxy fluxes derived from single scalar programs such as tumor, stroma, or immune scores.

### Step 6 — Hodge decomposition of scalar proxy flux
`step6_hodge_flux_decomposition.py`

Builds triangular 2-cells, constructs `B2`, and decomposes scalar-derived fluxes into:
- exact
- coexact
- harmonic

This step is mainly a sanity check. Scalar-derived fluxes are expected to be mostly exact.

### Step 7 — non-gradient wedge fluxes
`step7_non_gradient_flux.py`

Constructs antisymmetric non-gradient fluxes, including:
- immune_tumor_wedge
- stroma_tumor_wedge
- immune_stroma_wedge

These are the first fluxes in the pipeline that can generate nontrivial coexact structure.

### Step 8 — coexact energy and curl analysis
`step8_compute_coexact_and_curl.py`

Computes:
- node absolute coexact energy
- node-mapped triangle curl magnitude
- region-level tests
- permutation controls
- combined summary tables
- manuscript-ready LaTeX exports

## Main outputs currently emphasized

Recommended manuscript-facing outputs:
- `combined_bar_energyfractions.png`
- `combined_step7_summary.csv`
- `combined_step7_summary.tex`
- `step7_maps_immune_tumor_wedge.png`
- `step7_boxplots_immune_tumor_wedge.png`
- `step8_map_node_abs_coexact_immune_tumor_wedge.png`
- `step8_map_node_mean_curl_immune_tumor_wedge.png`
- `step8_region_tests_immune_tumor_wedge.csv`

## Reproducibility

The raw Visium `.h5` expression matrix and spatial image bundle are not meant to be committed to the repository root. Keep raw data in `data/raw/` or another ignored local folder. The tracked repository should prioritize:
- code
- lightweight summary tables
- manuscript-facing figures
- explicit notes about expected input filenames

See `data/README_data.md` for expected dataset placement and naming.

## Environment

Create a virtual environment and install the dependencies listed in `requirements.txt`.

Typical local workflow:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you export LaTeX tables with pandas styling or advanced formatting, make sure `jinja2` is installed.

## Scientific framing

This repository is not built to maximize prediction error metrics in isolation. Its core design principle is falsification under mechanistic structure. Elevated residuals or enriched coexact structure are treated as scientifically meaningful failure modes when a passive transport hypothesis is inadequate.

In practical terms:
- low error with high conservation inconsistency is a mechanistic warning sign
- low exactness with enriched coexact structure may indicate interface-driven or non-passive transport organization
- region-level enrichment is more important than visually pretty maps

## Data and ethics

The current spatial transcriptomics example uses publicly available data. No identifiable human information is used. Raw datasets remain under their original terms of use.

## Citation and reuse

If you reuse the code, operators, or workflow logic from this repository, cite the associated manuscript and clearly indicate any dataset-specific modifications or alternative flux definitions.


### Extension to TNBC cohort analysis

The full manuscript pipeline extends this prototype workflow to the TNBC spatial transcriptomics cohort using the scripts in `scripts_tnbc/`.

Additional stages include:

- cohort-scale region enrichment tests
- Lie-structured null diagnostics
- PDE-constrained Hodge–Laplacian graph neural network training
- operator analysis of learned transport fields
- hybrid gradient/stream potential reconstruction

These stages correspond to:

scripts_tnbc/step9_tnbc_curl_maps.py
scripts_tnbc/step10_curl_null_test.py
scripts_tnbc/step11_lie_structured_null.py
scripts_tnbc/step12_region_hotspot_lie_test.py
scripts_tnbc/step13_tnbc_prepare_gnn_data.py
scripts_tnbc/step14_tnbc_train_pde_gnn.py
scripts_tnbc/step15_tnbc_analyze_gnn_flux.py
scripts_tnbc/step16_transport_equation_figure.py
scripts_tnbc/step17_tnbc_solve_hybrid_potentials.py

To biologically validate the operator-level transport phenotype, we applied the full pipeline to spatial transcriptomics data from the TNBC cohort **GSE210616**.

Each tissue section is processed through a structured pipeline that constructs spatial transport fields and tests their geometric structure using Hodge decomposition, null-model diagnostics, and conservation-constrained learning.

### Pipeline overview

## Key manuscript figures

The figures referenced in the manuscript are located in:

GSM_visium_figures/GSM_6433618_fig/
GSM_visium_figures/GSM_6433619_fig/

These correspond to the TNBC Visium sections analyzed in the study.
The cohort analysis is implemented in `scripts_tnbc/`.

| Step | Script | Description |
|-----|------|-------------|
| 1 | `step1_tnbc_map.py` | Load TNBC Visium data and compute marker scores |
| 2 | `step2_tnbc_regions.py` | Assign tissue regions (tumor / stroma / immune / interface-like) |
| 3 | `step3_tnbc_spatial_graph.py` | Construct spatial cell complex and incidence operators |
| 4 | `step4_tnbc_flux_proxies.py` | Build proxy and residualized flux fields |
| 5 | `step5_tnbc_flux_residuals.py` | Compute conservation-style residual summaries |
| 6 | `step6_tnbc_hodge_decomposition.py` | Perform Hodge decomposition of flux fields |
| 7 | `step7_tnbc_region_enrichment.py` | Region-level enrichment analysis of transport components |
| 8 | `run_tnbc_screening.py` | Cohort-level orchestration / screening pipeline |
| 9 | `step9_tnbc_curl_maps.py` | Compute and visualize face curl structure |
| 10 | `step10_curl_null_test.py` | Randomized null test for curl structure |
| 11 | `step11_lie_structured_null.py` | Operator-derived Lie-structured null |
| 12 | `step12_region_hotspot_lie_test.py` | Region-level hotspot enrichment vs Lie null |
| 13 | `step13_tnbc_prepare_gnn_data.py` | Prepare graph data for PDE-constrained GNN training |
| 14 | `step14_tnbc_train_pde_gnn.py` | Train PDE-constrained Hodge-Laplacian GNN |
| 15 | `step15_tnbc_analyze_gnn_flux.py` | Analyze learned GNN flux with operator diagnostics |
| 16 | `step16_transport_equation_figure.py` | Generate transport-equation summary figure |
| 17 | `step17_tnbc_solve_hybrid_potentials.py` | Solve hybrid gradient/stream potential decomposition |
---

### Key outputs

Representative outputs are written to `stats/` and `visium_figures/`.

Examples include:

**Proxy and learned flux summaries**
- `stats/GSM_6433618_step14_gnn_summary_flux_tumor_immune.csv`
- `stats/GSM_6433618_step15_gnn_operator_summary_flux_tumor_immune.csv`

**Curl and hotspot statistics**
- `stats/GSM_6433618_step15_face_curl_gnn_flux_tumor_immune.csv`
- `stats/GSM_6433618_step15_hotspot_enrichment_gnn_flux_tumor_immune.csv`

**Hybrid potential decomposition**
- `stats/GSM_6433618_step17_proxy_hybrid_flux_tumor_immune_summary.csv`
- `stats/GSM_6433618_step17_gnn_hybrid_flux_tumor_immune_summary.csv`

**Figures**
- `GSM_visium_figures/GSM_6433618_fig/GSM_6433618_step14_gnn_training_history_flux_tumor_immune.png`
- `GSM_visium_figures/GSM_6433618_fig/GSM_6433618_step16_transport_equation_figure_flux_tumor_immune.png`
- `GSM_visium_figures/GSM_6433618_fig/GSM_6433618_Hybrid_potential_decomposition_of_learned_transport_field.png`


---

### Figures

Generated figures are stored in:
GSM_visium_figures/

Important outputs include:

| Figure | Description |
|------|-------------|
| `step6_hodge_maps_flux_*` | Hodge decomposition visualization |
| `step9_curl_maps_flux_*` | Curl density maps |
| `step11_lie_hotspots_flux_*` | High-curl hotspot locations |
| `step11_lie_null_hist_*` | Lie-null calibration plots |

---

### Example result

For sample **GSM_6433618**:
Interface hotspot enrichment = 1.456
p ≈ 0.002

For sample **GSM_6433619**:
Interface hotspot enrichment = 1.053
p ≈ 0.002

These results indicate that rotational transport motifs are **significantly enriched at tumor–immune interfaces**, while global curl statistics remain consistent with the Lie null model.

This pattern supports the interpretation that transport geometry becomes structured specifically at biological boundaries.