# Non-passive Transport Organization at Tumor–Immune Interfaces

**"Non-passive transport organization at tumor–immune interfaces revealed by operator-based analysis"**
Anas Enoch, MD · Mohammed VI University of Health Sciences (UM6SS), Casablanca
Target journal: *Bioinformatics Advances* (Oxford) · Submission BIOINF-2026-0777

---

## Overview

Spatial transcriptomics is widely used to study tumor microenvironments, yet most analyses implicitly assume that molecular transport follows passive, diffusion-like behavior. This repository provides a computational framework to **test this assumption directly** using operator-level Hodge–Laplacian decomposition and PDE-constrained graph neural networks.

Applied to **GSE210616** and externally validated on **GSE278936**, this framework shows that tumor–immune interfaces robustly concentrate non-gradient coexact interaction intensity. Corrected analyses demonstrate that this does not imply consistent rotational organization or interface-specific spectral coherence. Instead, the strongest biological signal lies in transition-level dynamics: KTS analysis reveals exhaustion-directed bias, while exploratory graph-KS analysis supports an instability-like interface regime.

**Final interpretation:** tumor–immune interfaces are high-intensity non-gradient interaction zones. They do not show robust rotational or interface-specific spectral organization after correction, but they exhibit strong exhaustion-directed transition dynamics and instability-like operator activity.

---

## Core Scientific Claim (Corrected Multilayer Framework)

| Layer | Result | Cohort evidence | Interpretation |
|---|---|---|---|
| **Coexact enrichment** | Strong interface enrichment | TNBC 40/40; GSE 21/23 | Interface is a high non-gradient intensity zone |
| **Spectral (energy-matched null)** | Not interface-specific | TNBC 2/43; GSE 2/23; sign test p = 1.0 | Spectral effect explained by energy, not distinct geometry |
| **Graph-curl proxy** | Weak / inconsistent | GSE median fold ≈ 1.08; TNBC not reliable under rebuilt geometry | No robust rotational phenotype |
| **KTS transition bias** | Strong exhaustion-directed bias | GSE near-universal; TNBC immune-active/stroma → exhausted | Biological structure lies in transition dynamics |
| **Graph-KS proxy** | Exploratory positive | GSE interface/tumor fold ≈ 8–10 | Interface behaves as instability-like nonlinear zone |

Additional results (TNBC legacy):

| Result | Value | N | Sign test *p* |
|---|---|---|---|
| Coexact enrichment > 1.0 | 40/40 sections | 40 | < 10⁻¹² |
| Coexact/exact differential > 1.0 (median 2.54) | 38/40 | 40 | < 10⁻⁹ |
| NCG: ρ(NC, coexact) at interface | 18/18 | 18 | < 10⁻⁵ |
| GNN falsification: coexact collapses to 2.7×10⁻¹² | — | — | — |
| log B(M1b/M1a): interface-localised (median +517.6) | 19/19 | 19 | < 10⁻⁵ |

Finding is independent of race, neoadjuvant chemotherapy, age, and RFS (all *p* > 0.18, *n* = 22).

---

## Four-Layer Falsification Design

1. **Local operator layer** — Hodge decomposition identifies non-gradient coexact interaction intensity. Enrichment of the coexact component at annotated tumor–immune interfaces is the primary geometric signal. Decomposition yields exact (gradient), coexact (non-gradient), and harmonic components. *A rotational interpretation requires explicit curl or face-level construction.*

2. **Spectral control layer** — Normalized Zeta with energy-matched nulls tests whether coexact enrichment reflects independent spectral organization. Both TNBC and GSE278936 show 2/43 and 2/23 sections significant respectively (sign test p = 1.0 in both). Coexact intensity is not explained by spectral geometry after energy control.

3. **Geometric control layer** — Graph-curl proxy tests whether coexact enrichment corresponds to localized rotational structure. GSE278936 cohort-level result: median fold ≈ 1.08. No robust rotational phenotype. TNBC reconstructed geometry does not support a corrected cohort-level curl claim.

4. **Dynamical layer** — KTS transition-bias analysis tests whether spatially disorganized interactions exhibit directional biological fate bias. Result: universal exhaustion attractor in GSE278936; pathway-specific (immune-active, stromal) exhaustion bias in TNBC.

> The GNN is not a predictor of biology — it is the best possible passive transport explanation.
> Collapse of the coexact component under conservation-constrained learning is falsification, not fitting failure.

---

## Repository Structure

```
Hodge_Laplacian_GNN/
│
├── README.md                            ← this file
├── README_TNBC.md                       ← TNBC legacy and rebuilt-validation pipeline
├── README_GSE278936_PIPELINE.md         ← external validation pipeline (v7)
├── README_CosMx_external_validation.md  ← cross-technology coexact enrichment validation
├── reference.bib
│
├── scripts_visium/          Prototype pipeline (single Visium section)
├── scripts_tnbc/            Full TNBC cohort pipeline (Steps 1–24)
├── scripts_gse278936/       External validation pipeline (Steps 01–14, 27)
├── scripts_tnbc_rebuild/    Legacy TNBC-to-modern schema conversion
├── scripts_cosmx/           CosMx cross-technology validation
│
├── stats/
│   ├── CSV_GSM/             Per-sample CSVs (TNBC legacy, Steps 1–24)
│   └── gnn_data/            GNN input/output arrays
│
├── results_gse278936/       GSE278936 validation outputs
│   ├── cohort_summary.csv
│   ├── cohort_zeta_energy_matched_null.csv
│   ├── step27_graph_curl_proxy_summary.csv
│   ├── cohort_ks_operator_summary.csv
│   ├── kts_transition_bias_summary.csv
│   └── kts_transition_bias_grouped_summary.csv
│
├── results_tnbc_rebuild/    Corrected TNBC validation outputs
│   ├── cohort_zeta_energy_matched_null.csv
│   ├── kts_transition_bias_summary.csv
│   └── kts_transition_bias_grouped_summary.csv
│
├── results_cosmx/           CosMx outputs (coexact enrichment layer only)
│
├── visium_figures/
├── data/
│   ├── TNBC_GSE210616/
│   └── Breast_Multiomic/
└── requirements.txt
```

---

## External Validation Pipeline (GSE278936)

The GSE278936 prostate cancer Visium cohort provides independent validation under the corrected multilayer framework. The `scripts_gse278936/` pipeline includes marker scoring, interface assignment, Hodge decomposition, coexact enrichment, normalized Zeta analysis with energy-matched nulls, graph-curl proxy analysis, KTS transition-bias analysis, and exploratory graph-KS instability analysis.

**Key GSE278936 results:**
- Spectral energy-matched null: 2/23 significant; sign test p = 1.0
- Graph-curl proxy: weak/inconsistent interface localization; median fold ≈ 1.08
- KTS transition bias: exhaustion-directed attractor from stromal, immune-active, tumor, and mixed states; 21–23/23 significant depending on source state
- Graph-KS proxy: interface/tumor-core absolute KS magnitude fold-change median ≈ 8.96, mean ≈ 10.15

Dataset URL: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE278936

---

## Corrected TNBC Rebuild

Legacy TNBC outputs were converted into the modern validation schema to allow corrected energy-matched Zeta and KTS analysis. Region labels were mapped as:

| Legacy | Modern |
|---|---|
| `interface_like` | `interface` |
| `tumor_enriched` | `tumor_core` |
| `stroma_enriched` | `stroma` |
| `immune_enriched` | `immune` |
| `other` | `other` |

**Corrected TNBC results:**
- Energy-matched Zeta null: 2/43 significant; sign test p = 1.0
- KTS: IMMUNE_ACTIVE → IMMUNE_EXHAUSTED median bias 5.68 (16/28 significant); STROMA → IMMUNE_EXHAUSTED median bias 2.41 (17/29 significant); TUMOR → IMMUNE_EXHAUSTED not enriched
- Graph-curl proxy from rebuilt TNBC geometry is **not** reported as a cohort-level phenotype

---

## TNBC Cohort Pipeline (scripts_tnbc/)

Full 24-step pipeline applied to GSE210616. See `README_TNBC.md` for commands.

### Steps 1–7: Core geometric pipeline
Marker scoring → region annotation → spatial graph → wedge flux → Hodge decomposition → interface enrichment testing.

### Steps 9–18: Diagnostics and falsification
Curl maps, Lie-structured null, region hotspot tests, PDE-constrained GNN training, GNN flux analysis, transport equation figure, hybrid potential decomposition, ablation study.

### Steps 19–22: Legacy operator-grounding and biological anchoring

Legacy diagnostics provide operator-grounding and biological anchoring. In the final corrected framework, spectral claims are interpreted through normalized Zeta with energy-matched nulls; these show that interface coexact enrichment reflects increased interaction intensity rather than independent spectral geometry.

- **Step 19** — Within-interface Spearman: tumor residual median ρ = 0.349 (17/18)
- **Step 20** — NCG commutator grounding: ρ = 0.832 (18/18). *Retained as legacy operator-grounding diagnostic; not the primary biological interpretation layer in the final manuscript.*
- **Step 21** — Legacy Zeta. *Did not implement the final energy-matched null. Corrected TNBC Zeta: 2/43 sections significant (sign test p = 1.0). Should not be cited as evidence of interface spectral enrichment.*
- **Step 22** — Independent biomarker: cytotoxic/CD8 enrichment (14–15/19, *p* < 0.04)

### Step 23: Operator robustness
Five antisymmetric operator variants confirm enrichment is not algebraically specific.

### Step 24: Stochastic Hodge decomposition
Gaussian priors; Bayes factors comparing M0 (passive), M1a (uniform active), M1b (interface-localised). log B(M1b/M1a) median +517.6 (19/19, *p* < 10⁻⁵).

---

## Key Manuscript-Facing Outputs

### Legacy TNBC (stats/CSV_GSM/)

```
*_step6_nodes_hodge_*.csv          node-level coexact energy + region
*_step6_edges_hodge_*.csv          edge-level flux components
*_step7_region_enrichment_*.csv    enrichment + permutation p
*_step20_ncg_*.csv                 NCG commutator (legacy operator grounding)
*_step22_topk_*.csv                independent biomarker validation
*_step24v2_summary_*.csv           stochastic Hodge Bayes factors
```

### Corrected validation outputs

```
results_gse278936/cohort_summary.csv
results_gse278936/cohort_zeta_energy_matched_null.csv
results_gse278936/step27_graph_curl_proxy_summary.csv
results_gse278936/cohort_ks_operator_summary.csv
results_gse278936/kts_transition_bias_summary.csv
results_gse278936/kts_transition_bias_grouped_summary.csv
results_tnbc_rebuild/cohort_zeta_energy_matched_null.csv
results_tnbc_rebuild/kts_transition_bias_summary.csv
results_tnbc_rebuild/kts_transition_bias_grouped_summary.csv
```

---

## Circularity Boundary (Hard Constraint)

```
Tumor:  EPCAM, KRT8, KRT18, KRT19, ERBB2, MUC1, TACSTD2
Immune: PTPRC, CD3D, CD3E, NKG7, CD68, C1QA, CXCL9, CXCL10
Stroma: COL1A1, COL1A2, DCN, LUM, POSTN, FAP, TAGLN
```

These gene sets define the wedge flux and are **forbidden** from any downstream validation step. KTS state definitions and downstream transition-bias analyses must not reuse validation markers as construction features without disclosure. The final interpretation treats KTS as a transition-level biological analysis, not as a direct validation of wedge construction.

---

## Scientific Framing

This repository implements a falsification-driven multilayer framework. The central result is not that tumor–immune interfaces are organized at every level. Rather, the framework separates intensity, geometry, spectrum, and dynamics:

- Coexact energy localizes to interfaces
- Corrected spectral and curl analyses do not support higher-order spatial organization
- KTS reveals exhaustion-directed transition bias
- Graph-KS analysis suggests instability-like interface behavior

> "Identifies incompatibility with passive transport, without asserting a unique underlying biological mechanism."

---

## Data and Ethics

All datasets used are publicly available.
- TNBC spatial transcriptomics: GEO accession **GSE210616** (Bassiouni R et al., Cancer Research 83(1):34–48, 2023)
- Prostate cancer external validation: GEO accession **GSE278936** — https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE278936

No identifiable human information is accessed or stored. Raw datasets remain under their original terms of use.

---

## Reproducibility

See `README_TNBC.md` for the TNBC legacy and rebuilt-validation pipeline, and `README_GSE278936_PIPELINE.md` for the external validation pipeline. The corrected reproducibility-critical outputs include energy-matched Zeta null summaries, KTS transition-bias summaries, graph-curl proxy summaries, and graph-KS instability summaries.

Recommended files to track in version control:

```
scripts_tnbc/
scripts_gse278936/
scripts_tnbc_rebuild/
results_gse278936/*.csv
results_tnbc_rebuild/*.csv
reference.bib
requirements.txt
```

---

## Citation

> Anas Enoch. *Non-passive transport organization at tumor–immune interfaces revealed by operator-based analysis.* Bioinformatics Advances, 2026. Submission BIOINF-2026-0777.

> Bassiouni R et al. *Spatial Transcriptomic Analysis of a Diverse Patient Cohort Reveals a Conserved Architecture in Triple-Negative Breast Cancer.* Cancer Research 83(1):34–48, 2023. GEO: GSE210616.

---

## Contact

[github.com/Anas-Enoch/Hodge_Laplacian_GNN](https://github.com/Anas-Enoch/Hodge_Laplacian_GNN)
