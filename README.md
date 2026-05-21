# Hodge-Laplacian Operator Framework for Tumor–Immune Interface Analysis

**Manuscript:** *Non-passive transport organization at tumor–immune interfaces revealed by operator-based analysis*
**Author:** Anas Enoch, MD
**Affiliation:** Mohammed VI University of Health Sciences (UM6SS), Casablanca, Morocco


---

## Overview

This repository contains the reproducible code, documentation, and processed outputs for a multilayer operator-based analysis of tumor–immune interfaces in spatial transcriptomics.

The final framework separates four quantities that are often conflated:

1. **Coexact interaction intensity** — local non-gradient signal detected by Hodge decomposition.
2. **Geometric/curl organization** — whether coexact signal forms consistent rotational structure.
3. **Spectral organization** — whether interface signal has independent low-frequency structure after energy control.
4. **Transition dynamics** — whether spatially disordered interactions nevertheless show biased biological state transitions.

The central conclusion is:

> Tumor–immune interfaces are high-intensity non-gradient interaction zones. They do not show robust rotational or interface-specific spectral organization after correction, but they exhibit exhaustion-directed transition dynamics and instability-like operator activity.

---

## Main Results

| Layer | Result | Cohort evidence | Interpretation |
| --- | --- | --- | --- |
| Coexact enrichment | Strong interface enrichment | TNBC discovery and GSE278936 validation | Interface is a high non-gradient interaction zone |
| Spectral energy-matched null | Not interface-specific | TNBC: 2/43 significant; GSE278936: 2/23 significant; sign test p = 1.0 | Spectral signal is explained by energy magnitude, not independent interface geometry |
| Graph-curl proxy | Weak / inconsistent | GSE278936 median fold ≈ 1.08; TNBC rebuilt graph-curl not used as cohort phenotype | No robust rotational phenotype |
| KTS transition bias | Exhaustion-directed transition bias | GSE278936: near-universal exhaustion attractor; TNBC: immune-active/stroma → exhausted | Biological structure lies in transition dynamics |
| Graph-KS proxy | Exploratory positive | GSE278936 interface/tumor fold ≈ 8–10 | Interface behaves as an instability-like nonlinear zone |

---

## Repository Structure

```
Hodge_Laplacian_GNN/
│
├── README.md                            ← this file
├── README_TNBC.md                       ← TNBC pipeline: Steps 1–24 + corrected rebuild
├── README_GSE278936_PIPELINE.md         ← GSE278936 external validation pipeline
├── README_CosMx_external_validation.md  ← CosMx cross-technology validation
├── environment.yml + requirements.txt                       
│
├── scripts_tnbc/
│   └── Full TNBC legacy pipeline (Steps 1–24): marker scoring, Hodge decomposition,
│       GNN falsification, NCG, Zeta, biomarker validation, stochastic Hodge.
│
├── scripts_gse278936/
│   └── External validation pipeline (Steps 01–14, 27): marker scoring, interface
│       assignment, Hodge decomposition, normalized Zeta with energy-matched nulls,
│       graph-curl proxy, KTS transition-bias, graph-KS instability.
│
├── scripts_tnbc_rebuild/
│   └── Utilities converting legacy TNBC outputs to the modern validation schema
│       for corrected Zeta and KTS analysis.
│
├── scripts_cosmx/
│   └── CosMx single-cell spatial transcriptomics validation pipeline.
│
├── spatial_hallmark/ biological-validation and baseline-comparison layer.
│ 
|── results_interface_regime/ Location: scripts_gse278936/step15 to step23c
│
├── stats/
│   └── CSV_GSM/Per-sample intermediate CSVs from TNBC legacy pipeline (Steps 1–24).
│
├── results_cosmx/
│   └── CosMx enrichment outputs (already included for reproducibility).
│
└── Results_TNBC_rebuild/
└── Results_gse278936/ 
        Curated final CSV outputs for both the GSE278936 validation and the
        corrected TNBC rebuild — the primary reproducibility directory.
        Key files:
          cohort_summary.csv
          cohort_zeta_energy_matched_null.csv
          step27_graph_curl_proxy_summary.csv
          cohort_ks_operator_summary.csv
          kts_transition_bias_summary.csv
          kts_transition_bias_grouped_summary.csv
```

Raw datasets are not stored in the repository. Public accession links are provided below.

---

## Pipelines

### 1. TNBC Discovery (GSE210616)

The `scripts_tnbc/` pipeline implements the full 24-step TNBC analysis. See `README_TNBC.md` for step-by-step commands.

Highlights:
- Steps 1–7: marker scoring, region annotation, wedge flux, Hodge decomposition, enrichment testing
- Steps 9–18: curl maps, Lie null, PDE-constrained GNN, ablation, hybrid decomposition
- Steps 19–22: biological anchoring, NCG commutator, Zeta spectral diagnostic, biomarker validation
- Step 23: operator robustness (5 antisymmetric constructions)
- Step 24: stochastic Hodge / Bayesian model comparison

### 2. GSE278936 External Validation

The `scripts_gse278936/` pipeline implements the standardized external validation workflow for prostate cancer Visium data. See `README_GSE278936_PIPELINE.md` for full documentation.

It includes: marker-score construction, interface assignment, wedge flux, Hodge decomposition, coexact energy, interface enrichment, normalized Zeta with energy-matched nulls, graph-curl proxy, KTS transition-bias, exploratory graph-KS instability.

### 3. TNBC Corrected Rebuild

Legacy TNBC outputs in `stats/CSV_GSM/` are converted to the modern schema by:

```bash
python scripts_tnbc_rebuild/convert_tnbc_legacy_to_modern.py
python scripts_tnbc_rebuild/normalize_tnbc_regions.py
```

Region mapping: `interface_like → interface`, `tumor_enriched → tumor_core`,
`stroma_enriched → stroma`, `immune_enriched → immune`, `other → other`.

Corrected outputs go to `Results_TNBC_rebuild/`.

### 4. CosMx External Validation

The CosMx workflow is documented in `README_CosMx_external_validation.md`. It validates local coexact non-gradient enrichment at single-cell resolution. It is **not** used as evidence for corrected Zeta, KTS, or KS claims.

---

## Key Reproducibility Commands

### GSE278936 — energy-matched Zeta null

```bash
python scripts_gse278936/step10_zeta_energy_matched_null.py \
  --statsdir Results_gse278936 \
  --out Results_gse278936/cohort_zeta_energy_matched_null.csv \
  --n-perm 300 --k-eigs 50
# Expected: 2/23 significant; sign test p = 1.0
```

### GSE278936 — KTS transition bias

```bash
python scripts_gse278936/step13b_kts_transition_bias.py \
  --statsdir Results_gse278936 \
  --out Results_gse278936/kts_transition_bias_summary.csv \
  --n-perm 300 --seed 123
# Expected: IMMUNE_EXHAUSTED attractor in 21–23/23 sections per transition
```

### GSE278936 — graph-KS operator

```bash
python scripts_gse278936/step14_ks_operator.py \
  --statsdir Results_gse278936 \
  --outdir Results_gse278936
# Expected: median fold ≈ 8.96 (23/23 fold > 1)
```

### TNBC rebuild — corrected Zeta null

```bash
python scripts_tnbc_rebuild/step10_zeta_energy_matched_null.py \
  --statsdir Results_TNBC_rebuild \
  --out Results_TNBC_rebuild/cohort_zeta_energy_matched_null.csv \
  --n-perm 300 --k-eigs 50
# Expected: 2/43 significant; sign test p = 1.0
```

### TNBC rebuild — KTS transition bias

```bash
python scripts_tnbc_rebuild/step11_kts_state_assignment.py \
  --statsdir Results_TNBC_rebuild --outdir Results_TNBC_rebuild

python scripts_tnbc_rebuild/step12_kts_transition_matrix.py \
  --statsdir Results_TNBC_rebuild

python scripts_tnbc_rebuild/step13b_kts_transition_bias.py \
  --statsdir Results_TNBC_rebuild \
  --out Results_TNBC_rebuild/kts_transition_bias_summary.csv \
  --n-perm 300 --seed 123
# Expected: IMMUNE_ACTIVE→IE bias 5.68 (16/28); STROMA→IE bias 2.41 (17/29); TUMOR→IE not enriched
```

---

## Corrected Spectral Interpretation

Normalized Zeta with energy-matched null models replaces earlier unnormalized or size-matched approaches.

Final result:

```
TNBC:      2/43 significant; sign test p = 1.0
GSE278936: 2/23 significant; sign test p = 1.0
```

> Tumor–immune interfaces are not spectrally more organized than equally energetic regions.
> Interface coexact enrichment reflects increased interaction intensity rather than distinct spectral geometry.

---

## KTS Interpretation

KTS analysis tests whether spatially disorganized interactions still exhibit directional biological transition bias.

**GSE278936:** near-universal exhaustion-directed attractor across all compartments (21–23/23 per transition).

**TNBC rebuilt:**

```
IMMUNE_ACTIVE → IMMUNE_EXHAUSTED: median bias ratio 5.68; 16/28 significant
STROMA        → IMMUNE_EXHAUSTED: median bias ratio 2.41; 17/29 significant
TUMOR         → IMMUNE_EXHAUSTED: not enriched; median bias ratio 0.42
```

> Exhaustion is a conserved dynamical endpoint, but its transition structure differs by cohort.
> In TNBC, exhaustion is mediated through immune-active and stromal pathways, not direct tumor transitions.

---

## Graph-Curl and KS Notes

Graph-curl proxy analyses are treated as geometric controls. They do not support a robust cohort-level rotational phenotype. The robust spatial signal is coexact energy enrichment.

The graph-Kuramoto–Sivashinsky operator is exploratory: it quantifies instability-like behavior of the coexact-energy field. It is **not** a claim that the biological system solves the KS PDE.

---

## Circularity Boundary

Construction panels (forbidden from downstream validation):

- **Tumor:** EPCAM, KRT8, KRT18, KRT19, ERBB2, MUC1, TACSTD2
- **Immune:** PTPRC, CD3D, CD3E, NKG7, CD68, C1QA, CXCL9, CXCL10
- **Stroma:** COL1A1, COL1A2, DCN, LUM, POSTN, FAP, TAGLN

KTS state definitions and transition-bias analyses are treated as transition-level biological analysis, not as direct validation of wedge construction.

---

## Datasets

| Dataset | Description | Access |
|---|---|---|
| GSE210616 | TNBC discovery (22 patients, 43 Visium sections) | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE210616 |
| GSE278936 | Prostate cancer external validation (23 Visium sections) | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE278936 |
| CosMx Breast Multiomic | Cross-technology single-cell validation | https://nanostring.com/products/cosmx-spatial-molecular-imager/ffpe-dataset/ |

---

## Environment

```bash
python >= 3.9
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# or: pip install numpy pandas scipy scikit-learn matplotlib networkx tqdm scanpy
```

---

## Citation


> Anas Enoch. *Non-passive transport organization at tumor–immune interfaces revealed by operator-based analysis.
=======
> Anas Enoch. *Non-passive transport organization at tumor–immune interfaces revealed by operator-based analysis.* 

> Bassiouni R et al. *Spatial Transcriptomic Analysis of a Diverse Patient Cohort Reveals a Conserved Architecture in Triple-Negative Breast Cancer.* Cancer Research 83(1):34–48, 2023. GEO: GSE210616.
