# Non-passive Transport Organization at Tumor–Immune Interfaces

**"Non-passive transport organization at tumor–immune interfaces revealed by operator-based analysis"**  
Anas Enoch, MD · Mohammed VI University of Health Sciences (UM6SS), Casablanca  
Target journal: *Bioinformatics Advances* (Oxford) · Submission BIOINF-2026-0777

---

## Overview

Spatial transcriptomics is widely used to study tumor microenvironments, yet most analyses implicitly assume that molecular transport follows passive, diffusion-like behavior. This repository provides a computational framework to **test this assumption directly** using operator-level Hodge–Laplacian decomposition and PDE-constrained graph neural networks.

Applied to **GSE210616** — 43 Visium sections from 22 primary TNBC patients — we show that tumor–immune interfaces consistently exhibit **structured rotational transport organization** that cannot be reproduced by any gradient-only passive transport model. A subsequent stochastic Hodge framework provides Bayesian evidence quantifying both the presence and interface-localisation of this non-gradient structure.

---

## Core Scientific Claim

Across the TNBC cohort (40 valid sections for enrichment, 19 for stochastic analysis):

| Result | Value | N | Sign test *p* |
|--------|-------|---|---------------|
| Coexact enrichment > 1.0 at tumor–immune interfaces | 40/40 sections | 40 | < 10⁻¹² |
| Coexact/exact differential > 1.0 (median 2.54) | 38/40 sections | 40 | < 10⁻⁹ |
| NCG: ρ(NC, coexact) at interface = 0.832 median | 18/18 sections | 18 | < 10⁻⁵ |
| Zeta: Z(s=1) = 1.89, f_low = 0.653 | 19/19 sections | 19 | < 10⁻⁵ |
| Biomarker validation: cytotoxic/CD8 enrichment | 14–15/19 | 19 | < 0.04 |
| log B(M1a/M0): non-gradient structure present | 19/19 sections | 19 | < 10⁻⁵ |
| log B(M1b/M1a): interface-localised (median +517.6) | 19/19 sections | 19 | < 10⁻⁵ |
| GNN falsification: coexact collapses to 2.7×10⁻¹² | — | — | — |

Finding is independent of race, neoadjuvant chemotherapy, age, and RFS (all *p* > 0.18, *n* = 22 patients).

---

## Design Principle

The pipeline implements a **three-stage falsification design**:

1. **Geometric detection** — Hodge decomposition of proxy transport fields into exact (gradient), coexact (rotational), and harmonic components. Enrichment of the coexact component at annotated tumor–immune interfaces is the primary geometric signal.

2. **Operator-class falsification** — A PDE-constrained Hodge–Laplacian GNN is trained to fit the observed flux field under conservation constraints. If the model reproduces the exact component but the coexact component collapses, this is a **falsification signal**: the observed spatial organization is incompatible with the passive transport class.

3. **Probabilistic quantification** — A stochastic Hodge decomposition places Gaussian priors over the flux field and computes analytic Bayes factors comparing passive (M0), uniform active (M1a), and interface-localised active (M1b) model classes.

> The GNN is not a predictor of biology — it is the best possible passive transport explanation.  
> Collapse of the coexact component under conservation-constrained learning is falsification, not fitting failure.

---

## Repository Structure

```
Hodge_Laplacian_GNN/
│
├── README.md                         ← this file
├── README_TNBC.md                    ← TNBC-specific pipeline and results
├── reference.bib
│
├── scripts_visium/                   ← Prototype pipeline (single Visium section)
│   ├── step1_visium_map.py
│   ├── step2_define_regions.py
│   ├── step3_build_spatial_graph.py
│   ├── step4_compute_proxy_flux_residuals.py
│   ├── step5_multi_proxy_residuals.py
│   ├── step6_hodge_flux_decomposition.py
│   └── step7_non_gradient_flux.py
│
├── scripts_tnbc/                     ← Full TNBC cohort pipeline (Steps 1–24)
│   ├── step1_tnbc_map.py
│   ├── step2_tnbc_regions.py
│   ├── step3_tnbc_spatial_graph.py
│   ├── step4_tnbc_flux_proxies.py
│   ├── step5_tnbc_flux_residuals.py
│   ├── step6_tnbc_hodge_decomposition.py
│   ├── step7_tnbc_region_enrichment.py
│   ├── step9_tnbc_curl_maps.py
│   ├── step10_curl_null_test.py
│   ├── step11_lie_structured_null.py
│   ├── step12_region_hotspot_lie_test.py
│   ├── step13_tnbc_prepare_gnn_data.py
│   ├── step14_tnbc_train_pde_gnn.py
│   ├── step15_tnbc_analyze_gnn_flux.py
│   ├── step16_transport_equation_figure.py
│   ├── step17_tnbc_solve_hybrid_potentials.py
│   ├── step18_ablation_no_constraint.py
│   ├── step19_coexact_bio_correlation.py
│   ├── step19_generate_csv.py
│   ├── step20_ncg_commutator.py          ← NCG commutator grounding
│   ├── step21_zeta_spectral.py           ← Zeta spectral diagnostic
│   ├── step22_ncg_bio_validation.py      ← Independent biomarker validation
│   ├── step23_operator_robustness.py     ← Alternative antisymmetric constructions
│   └── step24_stochastic_hodge_v2.py    ← Stochastic Hodge + Bayesian model comparison
│
├── stats/
│   ├── CSV_GSM/                      ← Per-sample CSVs (all steps)
│   └── gnn_data/                     ← GNN input/output arrays
│
├── visium_figures/                   ← Spatial maps and diagnostic plots
├── GSM_visium_figures/               ← Per-GSM figure directories
│
├── data/
│   ├── TNBC_GSE210616/
│   │   └── GSM_xxxxx/               ← One directory per sample (see inputs below)
│   ├── raw_Visium/
│   └── README_data.md
│
├── figures/
├── tables/
└── requirements.txt
```

---

## Prototype Pipeline (scripts_visium/)

The `scripts_visium/` directory contains a self-contained prototype applied to a single 10x Genomics Visium breast cancer section, demonstrating the core geometric workflow before TNBC cohort extension.

| Step | Script | Description |
|------|--------|-------------|
| 1 | `step1_visium_map.py` | Marker scoring: tumor, stromal, and immune programs |
| 2 | `step2_define_regions.py` | Region annotation: tumor core / invasive margin / stroma / immune-rich |
| 3 | `step3_build_spatial_graph.py` | Spatial graph, incidence matrices, discrete Laplacians |
| 4–5 | `step4_*.py`, `step5_*.py` | Proxy flux residuals |
| 6 | `step6_hodge_flux_decomposition.py` | Hodge decomposition: exact / coexact / harmonic |
| 7 | `step7_non_gradient_flux.py` | Wedge-based interaction fluxes (tumor–immune, tumor–stroma, immune–stroma) |

The prototype confirms that wedge-derived interaction fields generate substantial coexact content (20–23% of total energy) concentrated outside the homogeneous tumor core.

---

## TNBC Cohort Pipeline (scripts_tnbc/)

Full 24-step pipeline applied to GSE210616 (22 TNBC patients, 43 Visium sections). See `README_TNBC.md` for step-by-step commands and expected outputs.

### Steps 1–7: Core geometric pipeline
Marker scoring → region annotation → spatial graph → wedge flux → Hodge decomposition → interface enrichment testing.

### Steps 9–18: Diagnostics and falsification
Curl maps, Lie-structured null, region hotspot tests, PDE-constrained GNN training, GNN flux analysis, transport equation figure, hybrid potential decomposition, ablation study.

### Steps 19–22: Validation against construction artifacts
Three independent diagnostics rule out the construction artifact interpretation:

- **Step 19** — Within-interface Spearman correlation: coexact energy vs. residualized biological programs (tumor, immune, stroma). Cohort result: tumor residual median ρ = 0.349 (17/18, *p* = 7.2×10⁻⁵).
- **Step 20** — NCG commutator grounding: per-node non-commutativity norm vs. coexact energy. Interface correlation median ρ = 0.832 (18/18); specificity Δρ = +0.198 (*p* < 10⁻⁵). Rules out *spatial* construction artifact.
- **Step 21** — Zeta spectral diagnostic: coexact energy concentrated in low-frequency eigenmodes of the node graph Laplacian. f_low = 0.653 (19/19); Z(s=1) = 1.89 (18/19). Rules out *spectral* construction artifact.
- **Step 22** — Independent biomarker validation: top-10% NC nodes enriched for cytotoxic, CD8, chemokine, antigen presentation markers (14–15/19, *p* < 0.04); hypoxia null control passed (6/19, *p* = 0.97). Rules out *biological* circularity.

### Step 23: Operator robustness
Alternative antisymmetric constructions (normalized, rank-based, thresholded wedge) confirm that coexact interface enrichment is invariant to the specific operator algebraic form.

### Step 24: Stochastic Hodge decomposition
Gaussian priors over the flux field; analytic Bayes factors comparing three model classes:

- **M0** (passive): zero cycle-space variance
- **M1a** (uniform active): 30 low-frequency cycle modes, uniform spatial distribution
- **M1b** (interface-localised): same modes, variance redistributed to interface edges (trace-normalised)

Results: log B(M1a/M0) median +45.95 (19/19); log B(M1b/M1a) median +517.6 (19/19, *p* < 10⁻⁵). Probabilistic counterpart of the Step 7 sign test.

---

## Key Manuscript-Facing Outputs

### Primary result files (stats/CSV_GSM/)

```
*_step6_nodes_hodge_flux_tumor_immune_region_interface_weighted.csv  ← node-level coexact energy + region
*_step6_edges_hodge_flux_tumor_immune_region_interface_weighted.csv  ← edge-level flux components
*_step7_region_enrichment_flux_tumor_immune_region_interface_weighted.csv  ← enrichment + permutation p
*_step19_coexact_bio_*.csv                ← biological anchoring (cohort: 18 sections)
*_step20_ncg_*.csv                        ← NCG commutator results
*_step21_zeta_*.csv                       ← Zeta spectral diagnostic
*_step22_topk_*.csv                       ← Independent biomarker validation
*_step24v2_summary_*.csv                  ← Stochastic Hodge Bayes factors (per section)
cohort_step24v2_summary_*.csv             ← Cohort-level stochastic summary
```

### Key figures (visium_figures/ and GSM_visium_figures/)

| Figure pattern | Description |
|----------------|-------------|
| `*_step6_hodge_maps_*.png` | Hodge decomposition spatial maps |
| `*_step9_curl_maps_*.png` | Curl density maps |
| `*_step11_lie_hotspots_*.png` | Lie-structured null: curl hotspot locations |
| `*_step11_lie_null_hist_*.png` | Lie null calibration histograms |
| `*_step14_gnn_training_history_*.png` | GNN conservation-constrained training |
| `*_step16_transport_equation_figure_*.png` | Transport equation summary |
| `*_step18_ablation_plot_*.png` | GNN ablation: coexact collapse |
| `*_step19_coexact_bio_plot_*.png` | Biological anchoring scatter |
| `Hybrid_potential_decomposition_*.png` | Hybrid gradient/stream decomposition |

---

## Circularity Boundary (hard constraint)

The following gene sets define the wedge flux and are **forbidden** from any downstream validation step (Steps 19–22, Step 24):

- **Tumor score:** EPCAM, KRT8, KRT18, KRT19, ERBB2, MUC1, TACSTD2
- **Immune score:** PTPRC, CD3D, CD3E, NKG7, CD68, C1QA, CXCL9, CXCL10
- **Stroma score:** COL1A1, COL1A2, DCN, LUM, POSTN, FAP, TAGLN

All validation markers in Step 22 were selected specifically to be outside these lists.

---

## Environment

```bash
python >= 3.9
pip install scanpy numpy pandas matplotlib scipy scikit-learn networkx
```

For the stochastic pipeline (Step 24), no additional dependencies beyond scipy and numpy are required.

Full installation:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Scientific Framing

This repository is not built to maximize prediction error metrics in isolation. Its core design principle is **falsification under mechanistic structure**:

- Low error with high conservation inconsistency is a mechanistic warning sign.
- Enriched coexact structure at interfaces, under three independent diagnostics and one stochastic framework, constitutes evidence of non-passive transport organization.
- The framework identifies incompatibility with the passive transport model class — it does not assert a unique biological mechanism.

> "Identifies incompatibility with passive transport, without asserting a unique underlying biological mechanism."

---

## Data and Ethics

All data used in this study are publicly available from NCBI Gene Expression Omnibus (GSE210616). No identifiable human information is accessed or stored. Raw datasets remain under their original terms of use.

---

## Reproducibility

Raw Visium data should be placed in `data/TNBC_GSE210616/GSM_xxxxx/`. See `README_TNBC.md` for the full step-by-step reproducibility commands.

Recommended files to track in version control:
```
scripts_tnbc/          ← all analysis scripts
stats/CSV_GSM/*.csv    ← lightweight per-sample summaries
visium_figures/        ← manuscript figures
reference.bib
requirements.txt
```

Avoid committing `.npy`/`.npz` intermediates unless required for end-to-end reruns.

---

## Citation

If you use this pipeline, please cite:

> Anas Enoch. *Non-passive transport organization at tumor–immune interfaces revealed by operator-based analysis.* Bioinformatics Advances, 2026. Submission BIOINF-2026-0777.

Dataset:

> Bassiouni R et al. *Spatial Transcriptomic Analysis of a Diverse Patient Cohort Reveals a Conserved Architecture in Triple-Negative Breast Cancer.* Cancer Research 83(1):34–48, 2023. GEO: GSE210616.

---

## Contact

Open a GitHub issue or contact via the repository: [github.com/Anas-Enoch/Hodge_Laplacian_GNN](https://github.com/Anas-Enoch/Hodge_Laplacian_GNN)
