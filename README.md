### Manuscript version

# Non-passive Transport Organization at Tumor–Immune Interfaces

This repository accompanies the manuscript:

**“Non-passive transport organization at tumor–immune interfaces revealed by operator-based analysis”**

---

## Overview

Spatial transcriptomics is widely used to study tumor microenvironments, yet most analyses implicitly assume that molecular transport follows passive, diffusion-like behavior.

This repository provides a computational framework to **test this assumption directly**.

We show that tumor–immune interfaces exhibit **structured rotational transport organization** that cannot be explained by passive, gradient-driven models. These findings suggest that spatial organization at the invasive margin is governed by **interaction-driven biological processes** rather than diffusion alone.

---

## Core Idea

The pipeline implements a **two-stage design**:

1. **Measurement**  
   Detect transport structure directly from spatial transcriptomics data using operator-based decomposition:
   - gradient-driven (exact)
   - rotational (coexact)

2. **Mechanistic testing**  
   Evaluate whether the observed structure can be reproduced under **passive transport constraints** using a PDE-constrained graph neural network.

If the model reproduces gradient structure but fails to reproduce rotational structure, this provides a **falsification signal**:
> the observed spatial organization is incompatible with passive transport assumptions.

---

## Biological Insight

Across Visium TNBC samples, we observe:

- **Minimal rotational structure** for single-program (tumor / stroma / immune) transport  
- **Strong rotational enrichment** in interaction-derived (tumor–immune) flux  
- **Localization of rotational structure at tumor–immune interfaces**

These results indicate that the invasive margin is not a passive boundary, but a **structured interaction zone** driven by competing biological programs.

---

## Minimal Pipeline Overview

### Step 1 — marker scoring  
`scripts_visium/step1_visium_map.py`

Compute tumor, stromal, and immune scores and map them spatially.

---

### Step 2 — region definition  
`scripts_visium/step2_define_regions.py`

Define biologically meaningful regions:
- tumor_core  
- invasive_margin  
- stroma  
- immune_rich  

---

### Step 3 — spatial graph construction  
`scripts_visium/step3_build_spatial_graph.py`

Construct graph representation and discrete operators:
- incidence matrices  
- graph Laplacian  

---

### Step 4–5 — passive transport proxy  
`scripts_visium/step4_compute_proxy_flux_residuals.py`  
`scripts_visium/step5_multi_proxy_residuals.py`

Compute diffusion-like fluxes from scalar biological programs.

---

### Step 6 — Hodge decomposition  
`scripts_visium/step6_hodge_flux_decomposition.py`

Decompose flux into:
- exact (gradient)  
- coexact (rotational)  
- harmonic  

---

### Step 7 — interaction-driven flux  
`scripts_visium/step7_non_gradient_flux.py`

Construct wedge-based fluxes:
- tumor–immune  
- tumor–stroma  
- immune–stroma  

These are the first signals capable of generating rotational structure.

---

### Step 8 — coexact and curl analysis  
`scripts_visium/step8_compute_coexact_and_curl.py`

Compute:
- coexact energy  
- curl magnitude  
- region-level statistics  
- permutation tests  

---

## PDE-Constrained Learning

`scripts_tnbc/step14_tnbc_train_pde_gnn.py`

A graph neural network is trained under:
- conservation constraints  
- diffusion-like dynamics  

Result:
- gradient structure is preserved  
- rotational structure collapses  

This demonstrates that the observed interface organization is **not reproducible under passive transport assumptions**.

---

## Hybrid Transport Model

`scripts_tnbc/step17_tnbc_solve_hybrid_potentials.py`

We introduce a minimal representation:

- gradient component (global structure)  
- interaction-driven component (interface dynamics)

This hybrid model captures both aspects of spatial organization.

---

## Main Outputs

Key manuscript figures are generated from:

- `visium_figures/`
- `GSM_visium_figures/`

Important outputs include:
- coexact energy maps  
- curl maps  
- region-level statistics  
- GNN training diagnostics  
- hybrid decomposition  

---

## Reproducibility

The repository includes:

- all analysis scripts  
- intermediate outputs (`stats/`)  
- figure generation pipelines  

Raw Visium data should be placed in:
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