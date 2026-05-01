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
|---|---|---|---|
| Coexact enrichment | Strong interface enrichment | TNBC discovery and GSE278936 validation | Interface is a high non-gradient interaction zone |
| Spectral energy-matched null | Not interface-specific | TNBC: 2/43 significant; GSE278936: 2/23 significant; sign test p = 1.0 | Spectral signal is explained by energy magnitude, not independent interface geometry |
| Graph-curl proxy | Weak / inconsistent | GSE278936 median fold ≈ 1.08; TNBC rebuilt graph-curl not used as cohort phenotype | No robust rotational phenotype |
| KTS transition bias | Exhaustion-directed transition bias | GSE278936: near-universal exhaustion attractor; TNBC: immune-active/stroma → exhausted | Biological structure lies in transition dynamics |
| Graph-KS proxy | Exploratory positive | GSE278936 interface/tumor fold ≈ 8–10 | Interface behaves as an instability-like nonlinear zone |

---

## Repository Structure

```text
Hodge_Laplacian_GNN/
│
├── README.md
├── README_TNBC.md
├── README_GSE278936_PIPELINE.md
├── README_CosMx_external_validation.md
├── reference.bib
│
├── scripts_gse278936/
│   ├── step01_visium_build_marker_scores.py
│   ├── step02_visium_interface_detection.py
│   ├── step03_visium_wedge_flux.py
│   ├── step04_visium_hodge_decomposition.py
│   ├── step05_visium_coexact_energy.py
│   ├── step06_visium_interface_enrichment.py
│   ├── step07_cohort_summary.py
│   ├── step08_visium_zeta_interface.py
│   ├── step09_zeta_interface_null_and_correlation.py
│   ├── step10_zeta_energy_matched_null.py
│   ├── step11_kts_state_assignment.py
│   ├── step12_kts_transition_matrix.py
│   ├── step13b_kts_transition_bias.py
│   ├── step14_ks_operator.py
│   └── step27_face_bivector_orientation.py
│
├── scripts_tnbc_rebuild/
│   ├── convert_tnbc_legacy_to_modern.py
│   └── normalize_tnbc_regions.py
│
├── scripts_cosmx/
│   └── CosMx cross-technology validation scripts
│
├── legacy_visium_pipeline/
│   └── Archived early Visium prototype scripts retained for provenance only
│
├── stats/CSV_GSM/
│   └── Legacy TNBC intermediate outputs required for TNBC rebuild reproducibility
│
├── results_gse278936/
│   └── GSE278936 processed validation outputs
│
├── results_tnbc_rebuild/
│   └── Corrected TNBC rebuilt validation outputs
│
├── results_cosmx/
│   └── CosMx processed validation outputs
│
└── Results_TNBC_rebuild_gse278936/
    └── Curated final CSV outputs for repository upload / manuscript reproducibility
```

Raw datasets are not stored as final repository dependencies. Public accession links are provided below.

---

## Pipelines

### 1. GSE278936 External Validation

The `scripts_gse278936/` pipeline implements the standardized external validation workflow for prostate cancer Visium data.

It includes:

- marker-score construction
- interface assignment
- wedge flux construction
- Hodge decomposition and coexact energy computation
- interface enrichment analysis
- normalized Zeta analysis with energy-matched nulls
- graph-curl proxy analysis
- Kripke Transition System (KTS) transition-bias analysis
- exploratory graph-Kuramoto–Sivashinsky instability proxy

Key outputs:

```text
results_gse278936/cohort_summary.csv
results_gse278936/cohort_zeta_energy_matched_null.csv
results_gse278936/step27_graph_curl_proxy_summary.csv
results_gse278936/cohort_ks_operator_summary.csv
results_gse278936/kts_transition_bias_summary.csv
results_gse278936/kts_transition_bias_grouped_summary.csv
```

### 2. TNBC Discovery and Rebuilt Validation

The original TNBC discovery analysis used legacy intermediate outputs stored in:

```text
stats/CSV_GSM/
```

These are required by:

```text
scripts_tnbc_rebuild/convert_tnbc_legacy_to_modern.py
scripts_tnbc_rebuild/normalize_tnbc_regions.py
```

to reconstruct the modern TNBC validation schema used by corrected Step10 and KTS analyses.

Legacy region labels are mapped as:

```text
interface_like  → interface
tumor_enriched  → tumor_core
stroma_enriched → stroma
immune_enriched → immune
other            → other
```

Corrected TNBC outputs are stored in:

```text
results_tnbc_rebuild/
```

Key outputs:

```text
results_tnbc_rebuild/cohort_zeta_energy_matched_null.csv
results_tnbc_rebuild/kts_transition_bias_summary.csv
results_tnbc_rebuild/kts_transition_bias_grouped_summary.csv
```

### 3. CosMx External Validation

The CosMx workflow is documented separately in:

```text
README_CosMx_external_validation.md
```

CosMx is used as cross-technology validation of local coexact non-gradient enrichment. It is not used as evidence for corrected Zeta, KTS, or KS claims unless those analyses are explicitly performed.

---

## Key Reproducibility Commands

### GSE278936 — energy-matched Zeta null

```bash
python scripts_gse278936/step10_zeta_energy_matched_null.py \
  --statsdir results_gse278936 \
  --out results_gse278936/cohort_zeta_energy_matched_null.csv \
  --n-perm 300 \
  --k-eigs 50
```

### GSE278936 — KTS transition bias

```bash
python scripts_gse278936/step13b_kts_transition_bias.py \
  --statsdir results_gse278936 \
  --out results_gse278936/kts_transition_bias_summary.csv \
  --n-perm 300 \
  --seed 123
```

### GSE278936 — exploratory graph-KS operator

```bash
python scripts_gse278936/step14_ks_operator.py \
  --statsdir results_gse278936 \
  --outdir results_gse278936
```

### TNBC rebuild

```bash
python scripts_tnbc_rebuild/convert_tnbc_legacy_to_modern.py
python scripts_tnbc_rebuild/normalize_tnbc_regions.py
```

### TNBC — corrected Zeta null

```bash
python scripts_gse278936/step10_zeta_energy_matched_null.py \
  --statsdir results_tnbc_rebuild \
  --out results_tnbc_rebuild/cohort_zeta_energy_matched_null.csv \
  --n-perm 300 \
  --k-eigs 50
```

### TNBC — KTS transition bias

```bash
python scripts_gse278936/step11_kts_state_assignment.py \
  --statsdir results_tnbc_rebuild \
  --outdir results_tnbc_rebuild

python scripts_gse278936/step12_kts_transition_matrix.py \
  --statsdir results_tnbc_rebuild

python scripts_gse278936/step13b_kts_transition_bias.py \
  --statsdir results_tnbc_rebuild \
  --out results_tnbc_rebuild/kts_transition_bias_summary.csv \
  --n-perm 300 \
  --seed 123
```

---

## Corrected Spectral Interpretation

Earlier unnormalized or size-matched Zeta diagnostics were sensitive to total coexact energy. The corrected manuscript uses normalized Zeta with energy-matched null models.

Final result:

```text
TNBC:      2/43 significant; sign test p = 1.0
GSE278936: 2/23 significant; sign test p = 1.0
```

Interpretation:

> Tumor–immune interfaces are not spectrally more organized than equally energetic regions. Interface coexact enrichment reflects increased interaction intensity rather than distinct spectral geometry.

---

## KTS Interpretation

KTS analysis tests whether spatially disorganized interactions still exhibit directional biological transition bias.

### GSE278936

The GSE278936 cohort shows a near-universal exhaustion-directed attractor, with transitions toward `IMMUNE_EXHAUSTED` enriched from stromal, immune-active, tumor, and mixed states.

### TNBC

The rebuilt TNBC cohort shows pathway-specific exhaustion bias:

```text
IMMUNE_ACTIVE → IMMUNE_EXHAUSTED: median bias ratio 5.68; 16/28 significant
STROMA        → IMMUNE_EXHAUSTED: median bias ratio 2.41; 17/29 significant
TUMOR         → IMMUNE_EXHAUSTED: not enriched; median bias ratio 0.42
```

Interpretation:

> Exhaustion is a conserved dynamical endpoint, but its transition structure differs by cohort. In TNBC, exhaustion is mediated primarily through immune-active and stromal pathways rather than direct tumor-to-exhaustion transitions.

---

## Graph-Curl and KS Notes

Graph-curl proxy analyses are treated as geometric controls. They do not support a robust cohort-level rotational phenotype. The robust spatial signal is coexact energy enrichment, not curl enrichment.

The graph-Kuramoto–Sivashinsky operator is exploratory. It is an operator analogy used to quantify instability-like behavior of the coexact-energy field, not a claim that the biological system solves the Kuramoto–Sivashinsky PDE.

---

## Circularity Boundary

The construction gene panels define tumor, immune, and stromal programs used to construct the interface flux field. Downstream validation and transition interpretation must not reuse these markers as independent biological validation without disclosure.

Construction panels:

- **Tumor score:** EPCAM, KRT8, KRT18, KRT19, ERBB2, MUC1, TACSTD2
- **Immune score:** PTPRC, CD3D, CD3E, NKG7, CD68, C1QA, CXCL9, CXCL10
- **Stroma score:** COL1A1, COL1A2, DCN, LUM, POSTN, FAP, TAGLN

KTS state definitions and transition-bias analyses are treated as transition-level biological analysis, not as direct validation of wedge construction.

---

## Environment

```bash
python >= 3.9
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy pandas scipy scikit-learn matplotlib networkx tqdm
```

If `requirements.txt` is provided:

```bash
python -m pip install -r requirements.txt
```

---

## Data Availability

All datasets used are publicly available.

- TNBC spatial transcriptomics discovery cohort: GSE210616
- Prostate cancer spatial transcriptomics external validation cohort: GSE278936

Raw datasets remain under their original terms of use. Processed reproducibility outputs are provided in the results directories listed above.

---

## Scientific Framing

This repository implements a falsification-driven multilayer framework. The central result is not that tumor–immune interfaces are organized at every level. Instead, the framework separates intensity, geometry, spectrum, and dynamics:

- coexact energy localizes to interfaces;
- corrected spectral analyses do not support interface-specific spectral organization;
- graph-curl analyses do not support a robust rotational phenotype;
- KTS reveals exhaustion-directed transition bias;
- graph-KS analysis suggests exploratory instability-like interface behavior.

Final interpretation:

> Tumor–immune interfaces are high-intensity non-gradient interaction zones without consistent rotational or interface-specific spectral organization, but with strongly biased transition dynamics toward exhaustion-like states and elevated instability-like operator activity.

---

## Citation

Anas Enoch. *Non-passive transport organization at tumor–immune interfaces revealed by operator-based analysis*. Manuscript in preparation, 2026.
