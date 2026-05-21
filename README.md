# Non-Passive Transport Organization at Tumour–Immune Interfaces
### Operator-Geometric Falsification Framework · Spatial Transcriptomics

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

> **Paper:** *Non-passive transport organization at tumour–immune interfaces
> revealed by operator-based analysis.*  


---

## Scientific Summary

Standard spatial transcriptomics analyses ask how many immune cells are
near a tumour, or whether co-localisation exceeds a random baseline.
Neither question determines whether the interaction *field* at the boundary
is **geometrically organised** — whether it has non-gradient, rotationally
structured components that cannot arise from passive diffusion or scalar
potential functions.

This framework answers that harder question by decomposing the
tumour–immune wedge-flux operator into **exact (gradient)** and
**coexact (non-gradient)** components via discrete Hodge theory, then
falsifying the passive-transport null hypothesis using a
PDE-constrained graph neural network under four independent confound controls.
Across TNBC Visium (GSE210616, 43 sections), CosMx single-cell (108 FOVs),
and pan-cancer Spatial Hallmarks (26 sections), the coexact component is
systematically enriched at tumour–immune interfaces in a manner
not reducible to cell density, phenotype shuffling, generic antisymmetry,
or graph discretisation.

---

## Core Contribution

The framework introduces:

1. **Wedge-flux operator** — a bilinear, antisymmetric, edge-supported
   discrete 1-form built from spatial programme scores; the unique local
   operator antisymmetric under programme exchange.

2. **Discrete Hodge decomposition** — exact separation of the field into
   gradient (passive-compatible) and coexact (non-gradient) components
   on the tumour–immune spatial graph.

3. **PDE-constrained GNN falsification** — a physics-informed GNN trained
   under the conservation-law constraint `div(flux) = 0`; reconstruction
   failure falsifies passive diffusion.

4. **Null-model battery** — four independent confound exclusions: density,
   phenotype shuffle, generic antisymmetry, graph remeshing.

5. **Spatial hallmarks biological validation** — 4-tier exhaustion marker
   alignment across pan-cancer sections.

---

## Mathematical Framework

```
Wedge-flux operator (discrete 1-form):
  ω(u,v) = A(u)·B(v) − A(v)·B(u)

Hodge decomposition:
  ω  =  dα          +  δβ       +  γ
        (exact /         (coexact /    (harmonic)
         gradient)        non-gradient)

Falsification criterion:
  R = E_coexact(interface) / E_coexact(background)
  R > 1  →  non-passive organization
  R → 1  under all four null models

Stochastic Hodge Bayes factors:
  log B(M1a / M0)  = +45.95
  log B(M1b / M1a) = +517.6
```

**Uniqueness of the wedge:** Among all local, edge-supported, bilinear
operators antisymmetric under programme exchange, the wedge product is unique
up to scale. The coexact component is non-zero if and only if the field is
non-integrable and cannot arise from a scalar potential.

---

## Biological Interpretation

The coexact component at the tumour–immune interface represents
**rotational, non-gradient interaction structure** — the spatial fingerprint
of locally constrained, coherent interaction dynamics that:
- cannot be explained by immune cell density alone,
- cannot arise from passive diffusion of secreted molecules,
- are specifically enriched in sections with high exhaustion marker expression
  (Tier 2: all 7 exhaustion markers, ratios 3.36–4.75×, p = 1.49×10⁻⁸),
- are reproducible across Visium, CosMx, and IMC platforms.

> **KTS interpretation note:** KTS outputs are **spatial transition-bias
> structures** inferred from static spatial transcriptomics data.
> They are not direct temporal trajectories, causal state transitions,
> or experimentally proven transport dynamics.
> They represent geometric organisation consistent with directed interface
> interaction under spatial constraints.

---

## What This Framework Is Not

- **Not a CellChat / NicheNet replacement.** Ligand–receptor inference
  methods operate on molecular interaction databases and predict signalling
  events. This framework detects non-gradient operator structure and
  spatial transition-bias organisation at the boundary — a complementary
  geometric layer invisible to LR methods.

- **Not a COMMOT / spatial autocorrelation replacement.** COMMOT models
  optimal transport of ligand–receptor pairs. Moran's I / Geary's C measure
  scalar autocorrelation. Neither captures antisymmetric, non-gradient
  field geometry. This framework is structurally non-redundant (coexact
  vs. Moran's I LOO AUC: 0.929 vs. 0.817; ρ = +0.65, shared variance only).

- **Not a claim of literal molecular transport measurement.** The framework
  falsifies the passive-diffusion null for the interaction *field*; it does
  not measure individual molecule trajectories.

- **Not a causal signalling model.** No causal graph is constructed.
  The coexact component is a geometric property of the programme interaction
  field at a static spatial snapshot.

- **Not a temporal lineage model.** KTS states are spatial transition-bias
  structures inferred from spatial co-occurrence, not from time-series data.

---

## Core Claim

> The framework separates **local non-gradient interaction intensity**
> (coexact field enrichment at the boundary) from spectral organisation
> (Hodge exact component), geometric/curl organisation (KTS transition
> bias), and biological transition-bias structure (exhaustion marker
> alignment). Each layer is independently falsifiable and structurally
> distinct from classical spatial metrics.

---

## Repository Architecture

```
raw spatial transcriptomics  (Visium / CosMx / IMC)
          ↓
    marker scoring  (tumour · immune · exhaustion · myeloid · EMT)
          ↓
    spatial graph / cell complex construction  (kNN k=6 / Delaunay)
          ↓
    wedge-flux operator construction  ω(u,v) = A(u)B(v) − A(v)B(u)
          ↓
    Hodge decomposition  ω = dα + δβ + γ
          ↓
    exact / coexact separation  (gradient vs. non-gradient)
          ↓
    spectral and geometric controls  (remeshing · Bayes factors · density)
          ↓
    spatial hallmarks  (pan-cancer 26 sections)
          ↓
    biological validation  (4-tier exhaustion markers)
          ↓
    baseline comparison  (Moran's I · NE score · spectral entropy · Node2Vec)
          ↓
    manuscript figures
```

### Directory Structure

```
Hodge_Laplacian_GNN/
│
├── core/                          # Mathematical operator primitives
│   ├── operator_geometry/         # Wedge field, antisymmetry
│   ├── hodge_decomposition/       # Discrete Hodge solver
│   ├── pde_constraints/           # PDE-constrained GNN
│   ├── graph_construction/        # kNN / Delaunay builders
│   └── transport_models/          # Passive diffusion nulls
│
├── spatial_hallmark/              # Pan-cancer validation module
│   ├── build_biological_validation.py
│   ├── baseline_comparison.py
│   ├── build_spatial_hallmarks_kts_edges.py
│   └── results_spatial_hallmarks/
│       ├── spatial_hallmarks_hodge_interface.csv
│       ├── spatial_hallmarks_kts_edges.csv
│       ├── tier1_module_correlation.csv
│       ├── tier2_exhaustion_endpoint.csv
│       ├── tier3_stromal_mediation.csv
│       ├── results_baseline_comparison.csv
│       └── baseline_comparison.png
│
├── scripts/
│   ├── preprocessing/             # AnnData construction, programme scoring
│   ├── analysis/                  # Operator, Hodge, GNN, nulls
│   ├── visualization/             # Figure generation
│   └── utilities/                 # Shared helpers, repo tools
│
├── datasets/
│   ├── raw/                       # Immutable source data (Git LFS / Zenodo)
│   └── processed/                 # Scored AnnData objects
│
├── results/
│   ├── final/                     # Manuscript-level summary tables
│   └── intermediate/              # Per-sample outputs (gitignored)
│
├── paper/
│   ├── figures/                   # Final manuscript figures
│   ├── supplementary/             # Supplementary materials
│   └── manuscript/                # LaTeX source, bibliography
│
├── docs/
│   ├── reviewer_guide.md          # Figure-to-script mapping
│   ├── methodology/               # Mathematical derivations
│   ├── architecture/              # Pipeline design notes
│   └── reproducibility/           # Step-by-step guides
│   └── pipelines/
|       ├── TNBC_pipeline.md
|       ├── CosMx_pipeline.md
|       ├── GSE278936_pipeline.md
|       └── Interface_regime_pipeline.md
└── archive/
    ├── legacy_outputs/            # Pre-refactor outputs
    ├── generated_outputs/         # Bulky reproducible artifacts
    ├── exploratory/               # Development-phase scripts
    └── deprecated/                # Superseded approaches
```

---

## Environment Setup

### Option A — Conda (recommended)

```bash
git clone https://github.com/Anas-Enoch/Hodge_Laplacian_GNN.git
cd Hodge_Laplacian_GNN
conda env create -f environment.yml
conda activate hodge-operator
```

### Option B — pip

```bash
git clone https://github.com/Anas-Enoch/Hodge_Laplacian_GNN.git
cd Hodge_Laplacian_GNN
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Data Requirements

| Dataset | Platform | Sections | Access |
|---|---|---|---|
| GSE210616 | Visium CytAssist | 43 (22 patients) | GEO public |
| CosMx Breast Multiomic | CosMx SMI | 108 FOVs | Nanostring AtoMx |
| Spatial Hallmarks | Visium | 26 pan-cancer | Zenodo 14044964 |
| MERFISH cortex | MERFISH | 3 | Allen Brain Atlas |

> **Large-file policy:** Raw `.h5ad` files and dense edge tables are **not
> committed to git**. They must be regenerated from pipeline scripts or
> downloaded as GitHub Release / Zenodo assets.
> GitHub-tracked files are limited to: scripts, documentation, summary CSVs,
> and manuscript-critical figure outputs.

```bash
# Download Spatial Hallmarks (Zenodo)
wget https://zenodo.org/record/14044964/files/spatial_hallmarks_scored.h5ad \
     -O datasets/processed/spatial_hallmarks_scored.h5ad
```

---

## Reproduction Workflow

### 1. Programme Scoring

```bash
python scripts/preprocessing/score_programmes.py \
    --input  datasets/raw/GSE210616/ \
    --output datasets/processed/tnbc_scored.h5ad
# Runtime: ~8 min/section · Output: tnbc_scored.h5ad
```

### 2. Hodge Decomposition

```bash
python scripts/analysis/hodge_decomposition.py \
    --adata  datasets/processed/tnbc_scored.h5ad \
    --k      6 \
    --output results/final/hodge_summary.csv
# Runtime: ~3 min/section · Output: hodge_summary.csv
```

### 3. PDE-Constrained GNN Falsification

```bash
python scripts/analysis/pde_gnn_falsification.py \
    --adata      datasets/processed/tnbc_scored.h5ad \
    --hodge      results/final/hodge_summary.csv \
    --null-tests density shuffle antisymmetry remeshing \
    --output     results/final/gnn_falsification.csv
# Runtime: ~45 min · log B M1a/M0 = +45.95; log B M1b/M1a = +517.6
```

### 4. Spatial Hallmarks Biological Validation

```bash
python3 spatial_hallmark/build_biological_validation.py \
  --adata   datasets/processed/spatial_hallmarks_scored.h5ad \
  --hodge   spatial_hallmark/results_spatial_hallmarks/spatial_hallmarks_hodge_interface.csv \
  --kts     spatial_hallmark/results_spatial_hallmarks/spatial_hallmarks_kts_edges.csv \
  --out-dir spatial_hallmark/results_spatial_hallmarks/ \
  --fig-dir spatial_hallmark/results_spatial_hallmarks/
# Runtime: ~15 min · Tier 2: all 7 exhaustion markers p = 1.49×10⁻⁸
```

### 5. Baseline Comparison

```bash
python3 spatial_hallmark/baseline_comparison.py \
  --adata   datasets/processed/spatial_hallmarks_scored.h5ad \
  --hodge   spatial_hallmark/results_spatial_hallmarks/spatial_hallmarks_hodge_interface.csv \
  --kts     spatial_hallmark/results_spatial_hallmarks/spatial_hallmarks_kts_edges.csv \
  --out     spatial_hallmark/results_spatial_hallmarks/results_baseline_comparison.csv \
  --fig     spatial_hallmark/results_spatial_hallmarks/baseline_comparison.png
# Coexact AUC 0.929 · NE 0.517 · spectral entropy 0.400 · Node2Vec 0.375
```

### 6. Figure Generation

```bash
python scripts/visualization/generate_manuscript_figures.py \
    --results results/final/ \
    --output  paper/figures/
```

---

## Spatial Hallmark Validation Module

The `spatial_hallmark/` module provides standalone pan-cancer biological
validation and baseline comparison, independent of the full TNBC pipeline.

```
spatial_hallmark/
├── build_biological_validation.py    # 4-tier exhaustion marker alignment
├── baseline_comparison.py            # Moran's I · NE · entropy · Node2Vec
├── build_spatial_hallmarks_kts_edges.py  # KTS spatial transition-bias edges
└── results_spatial_hallmarks/
    ├── spatial_hallmarks_hodge_interface.csv
    ├── spatial_hallmarks_kts_edges.csv
    ├── tier1_module_correlation.csv
    ├── tier2_exhaustion_endpoint.csv
    ├── tier3_stromal_mediation.csv
    ├── results_baseline_comparison.csv
    └── baseline_comparison.png
```

**Validation results:**

| Tier | Markers | Sections | Result |
|---|---|---|---|
| T1 | Cytotoxic module | 25/26 | ρ = 0.240 |
| T2 | 7 exhaustion markers | 26/26 | Ratio 3.36–4.75×, p = 1.49×10⁻⁸ |
| T3 | TGFB1 / FAP / CXCL12 | Significant | Stromal mediation |
| T4 | Extended panel | Pending | Not yet validated |

---

## Baseline Comparison

| Metric | LOO AUC | Spearman ρ vs. coexact | Interpretation |
|---|---|---|---|
| Interface coexact energy | **0.929** | — | Reference |
| Moran's I | 0.817 | +0.65 | Shared spatial clustering; 0.11 AUC gap |
| NE score (Giotto) | 0.517 | — | Near chance |
| Graph spectral entropy | 0.400 | — | Below chance |
| Node2Vec embedding | 0.375 | — | Below chance |

---

## Figures

See [`paper/figures/FIGURE_MANIFEST.md`](paper/figures/FIGURE_MANIFEST.md)
for complete figure → script → input → output mapping.

---

## Reviewer Guide

See [`docs/reviewer_guide.md`](docs/reviewer_guide.md) for:
- figure reproduction workflow,
- key statistical result checklist,
- legacy filename mapping,
- hardware and runtime requirements.

---

## Citation

```bibtex
@article{enoch2026nonpassive,
  author  = {Enoch, Anas},
  title   = {Non-passive transport organization at tumour--immune interfaces
             revealed by operator-based analysis},
  journal = {Bioinformatics Advances},
  year    = {2026},
  note    = {Manuscript ID: BIOINF-2026-0777}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Contact

**Anas Enoch** · MD, Mohammed VI University of Health Sciences (UM6SS), Casablanca  
GitHub: [@Anas-Enoch](https://github.com/Anas-Enoch/Hodge_Laplacian_GNN)
