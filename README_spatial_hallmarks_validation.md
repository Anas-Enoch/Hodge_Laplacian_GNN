# Spatial Hallmarks Validation and Baseline Comparison Framework

## Overview

This repository module contains the biological-validation and baseline-comparison layer for the operator-based spatial hallmarks framework described in:

> Non-passive transport organization at tumor–immune interfaces revealed by operator-based analysis

The purpose of this module is twofold:

1. Biological grounding
2. Non-redundancy benchmarking

The framework evaluates whether operator-derived spatial hallmarks correspond to canonical tumour–immune biology and whether the coexact interface signal is reducible to standard spatial-analysis baselines.

---

# Repository structure

```text
spatial_hallmark/
├── build_biological_validation.py
├── baseline_comparison.py
├── build_spatial_hallmarks_kts_edges.py
└── results_spatial_hallmarks/
    ├── spatial_hallmarks_hodge_interface.csv
    ├── spatial_hallmarks_kts_edges.csv
    ├── tier1_module_correlation.csv
    ├── tier2_exhaustion_endpoint.csv
    ├── tier3_stromal_mediation.csv
    ├── results_baseline_comparison.csv
    └── baseline_comparison.png
```

---

# Input data

The framework expects:

```text
data/spatial_hallmarks_scored.h5ad
```

This AnnData object is generated reproducibly from the upstream preprocessing and operator-analysis pipeline.
see : https://github.com/Anas-Enoch/tumor-immune-operator-geometry/tree/main/scripts/spatial_hallmarks
The object contains:
- spatial coordinates,
- operator-derived coexact metrics,
- tumour/immune/stromal scores,
- module scores,
- section metadata,
- interface annotations.

---

# Biological validation framework

The biological-validation layer tests whether operator-derived spatial hallmarks correspond to biologically meaningful tumour–immune interaction programs.

Validation is intentionally separated from:
- wedge-flux construction,
- Hodge decomposition,
- spectral diagnostics,
- KTS edge generation,

preventing circular interpretation between operator construction and downstream biological grounding.

---

## Tier 1 — Module-score grounding

Tests whether coexact interface regions correlate with:
- cytotoxic programs,
- exhaustion programs,
- stromal activation,
- myeloid suppression,
- interferon signaling,
- hypoxia controls.

### Cytotoxic markers
CD8A, CD8B, GZMB, PRF1, GNLY, IFNG, NKG7

### Exhaustion markers
PDCD1, CTLA4, LAG3, HAVCR2, TIGIT, TOX, CXCL13

### Stromal markers
FAP, POSTN, COL1A1, COL3A1, TGFB1, CXCL12

### Myeloid markers
SPP1, C1QA, MRC1, CD163, TREM2, APOE

### Hypoxia controls
HIF1A, VEGFA, LDHA

Output:
```text
tier1_module_correlation.csv
```

---

## Tier 2 — Exhaustion endpoint validation

Tests whether KTS-defined exhaustion endpoints correspond to canonical exhaustion biology.

Markers:
- PDCD1
- CTLA4
- LAG3
- HAVCR2
- TIGIT
- TOX
- CXCL13

Output:
```text
tier2_exhaustion_endpoint.csv
```

---

## Tier 3 — Stromal mediation

Tests whether exhaustion-associated transition-bias structure preferentially occurs within reactive stromal environments.

Markers:
- TGFB1
- FAP
- CXCL12
- COL1A1
- COL3A1

Output:
```text
tier3_stromal_mediation.csv
```

---

# KTS edge generation

The biological-validation framework requires edge-resolved KTS graphs.

Generate them using:

```bash
python3 spatial_hallmark/build_spatial_hallmarks_kts_edges.py     --adata data/spatial_hallmarks_scored.h5ad     --out spatial_hallmark/results_spatial_hallmarks/spatial_hallmarks_kts_edges.csv
```

---

# Running biological validation

```bash
python3 spatial_hallmark/build_biological_validation.py     --adata   data/spatial_hallmarks_scored.h5ad     --hodge   spatial_hallmark/results_spatial_hallmarks/spatial_hallmarks_hodge_interface.csv     --kts     spatial_hallmark/results_spatial_hallmarks/spatial_hallmarks_kts_edges.csv     --out-dir spatial_hallmark/results_spatial_hallmarks/     --fig-dir spatial_hallmark/results_spatial_hallmarks/
```

---

# Statistical interpretation

Validation analyses emphasize:
- cross-section consistency,
- effect-size reproducibility,
- directional coherence,

rather than isolated nominal p-values.

Cohort-level directional consistency is evaluated using one-sided sign tests.

The framework should be interpreted as:
> biologically grounded operator validation performed on static spatial transcriptomics data.

The framework does not establish:
- causal mechanisms,
- temporal trajectories,
- experimentally validated transport dynamics.

---

# Baseline comparison framework

The baseline-comparison layer tests whether the coexact interface signal is reducible to standard spatial-analysis baselines.

This benchmark is a non-redundancy analysis, not a prediction-accuracy benchmark.

The framework therefore does not claim superiority to:
- CellChat,
- NicheNet,
- COMMOT,
- ligand–receptor inference systems.

Instead, it evaluates whether the operator captures:
> spatial interaction structure orthogonal to standard spatial statistics.

---

# Baselines tested

- Moran’s I spatial autocorrelation
- Boundary differential-expression score
- Ligand–receptor proximity proxy
- Laplacian smoothness
- Neighborhood-enrichment proxy

---

# Running baseline comparison

```bash
python3 spatial_hallmark/baseline_comparison.py     --adata   data/spatial_hallmarks_scored.h5ad     --hodge   spatial_hallmark/results_spatial_hallmarks/spatial_hallmarks_hodge_interface.csv     --kts     spatial_hallmark/results_spatial_hallmarks/spatial_hallmarks_kts_edges.csv     --out     spatial_hallmark/results_spatial_hallmarks/results_baseline_comparison.csv     --fig     spatial_hallmark/results_spatial_hallmarks/baseline_comparison.png
```

---

# Interpretation of baseline comparison

Low correlations with:
- Moran’s I,
- boundary DE,
- ligand–receptor proximity,
- Laplacian smoothness

indicate that the coexact operator captures:
> spatial interaction structure not reducible to smoothness, differential-expression, or ligand–receptor proximity statistics.

Partial overlap with neighborhood enrichment is expected because both quantities reflect local interface-associated concentration structure.

---

# Conceptual interpretation

The integrated biological-validation and baseline-comparison results support the interpretation that tumour–immune interfaces behave as:

- high-intensity,
- cytotoxic-rich,
- partially exhaustion-associated,
- stromally constrained,
- spatially disorganized confrontation zones.

The framework therefore supports:
> operator-defined confrontation organization

rather than:
- smooth gradients,
- coherent manifolds,
- purely ligand–receptor-defined structure.

---

# Important limitation

Because all analyses are derived from static spatial transcriptomics:

KTS structures should be interpreted as:
> spatial transition-bias organization

rather than:
- direct temporal trajectories,
- causal transitions,
- experimentally validated transport dynamics.

---

# Manuscript correspondence

This module supports:
- Section 2.7 — Independent biological validation of spatial hallmarks
- baseline-comparison analyses
- KTS biological grounding
- operator non-redundancy benchmarking
