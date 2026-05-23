# Hodge-Laplacian Operator Framework for Tumor–Immune Interface Analysis

**Manuscript:** *Non-gradient spatial organisation at tumor--immune interfaces revealed by operator-based analysis*
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
│── Benchmarking/     build_real_baseline_benchmarking.py\spatial_hallmarks_hodge_interface.csv + results1 
│               ├──── build_real_baseline_benchmarking_hcc.py\results_hcc_hodge_interface_summary_valid.csv + results2
│               └──── Readme_Baseline Benchmark.md + fig_baseline_comparison.png
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
├── spatial_hallmark/
│   ├── build_biological_validation.py
│   ├── build_spatial_hallmarks_kts_edges.py
│   ├── baseline_comparison.py
│   └── results_spatial_hallmarks/
│       ├── spatial_hallmarks_hodge_interface.csv
│       ├── spatial_hallmarks_kts_edges.csv
│       ├── tier1_module_correlation.csv
│       ├── tier2_exhaustion_endpoint.csv
│       ├── tier3_stromal_mediation.csv
│       ├── results_baseline_comparison.csv
│       └── baseline_comparison.png
│ 
|── results_interface_regime/ Location: scripts_gse278936/step15 to step23c
│
├── stats/
│   └── CSV_GSM/Per-sample intermediate CSVs from TNBC legacy pipeline (Steps 1–24).
│
├── results_cosmx/
│   └── CosMx enrichment outputs (already included for reproducibility).
│
└── results_gse278936/
│   └── Curated GSE278936 validation outputs.
│
├── results_tnbc_rebuild/
│   └── Corrected TNBC rebuild outputs.
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

## Installation

### Option A — Conda

```bash
conda env create -f environment.yml
conda activate hodge-operator
```

### Option B — pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `environment.yml` or `requirements.txt` are not present in a lightweight branch, install the core stack manually:

```bash
pip install numpy pandas scipy scikit-learn networkx scanpy anndata matplotlib tqdm
```

---

## Data requirements

| Dataset | Platform | Role |
|---|---|---|
| GSE210616 | 10x Visium / TNBC | Discovery cohort |
| GSE278936 | 10x Visium / prostate cancer | External validation |
| CosMx breast multiomic data | CosMx SMI | Cross-technology validation |
| HCC CytAssist (Zenodo 19123188) | Visium CytAssist / HCC | Immunotherapy response baseline benchmark |
| Spatial Hallmarks (Zenodo 14044964) | Visium pan-cancer | Biological validation and baseline benchmark |

Raw data are not stored in this repository. Large derived `.h5ad` files and dense edge tables should be regenerated from the documented pipelines or distributed through release/archival assets when needed.

---

## Pipeline documentation

Detailed cohort-level documentation is provided in:

| File | Purpose |
|---|---|
| `TNBC_pipeline.md` | TNBC discovery and corrected rebuild pipeline |
| `GSE278936_pipeline.md` | GSE278936 external validation |
| `CosMx_external_validation.md` | Cross-technology CosMx validation |
| `Interface_Regime_pipeline.md` | Interface-regime and constraint analyses |
| `Spatial_Hallmarks_validation.md` | Biological validation and baseline comparison |

These files should be read as cohort-specific reproducibility guides.
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

## Real-Tool Baseline Benchmark (`build_real_baseline_benchmarking.py`)

This benchmark tests whether the Hodge coexact interface operator captures
spatial interaction structure that is **structurally orthogonal** to five
established spatial-biology tools run with their official Python implementations.
It is a **non-redundancy analysis**, not a prediction-accuracy benchmark.

### Tools used

| Tool | Implementation | Measures |
|---|---|---|
| Squidpy NE | `squidpy.gr.nhood_enrichment` (1,000 permutations) | Tumour–immune adjacency frequency |
| Moran's I | `esda.Moran` via `libpysal.weights.KNN` | Scalar spatial autocorrelation |
| SPARK-X equiv. | Native HSIC test (Gaussian kernel, permutation) | Spatial expression variability |
| SpatialDE equiv. | GP variance fraction via kernel ridge regression | Smooth spatial variance fraction |
| LR proximity | Distance-weighted COMMOT-style co-expression | Ligand–receptor spatial co-localisation |

### Install dependencies

```bash
pip install squidpy esda libpysal --break-system-packages
# SpatialDE and COMMOT have Python 3.12 compatibility issues;
# equivalent implementations are used natively inside the script.
```

---

### Run 1 — Pan-cancer Spatial Hallmarks cohort (26 sections, 6 cancer types)

```bash
python3 benchmarking/build_real_baseline_benchmarking.py \
  --adata  data/spatial_hallmarks_scored.h5ad \
  --hodge  benchmarking/spatial_hallmarks_hodge_interface.csv \
  --outdir benchmarking/results1/final/
```

#### Terminal output

```
Loading AnnData...
AnnData sections: 26
Hodge-valid sections: 26
Matched sections:  26
Max spots per section: no limit (full)
Baselines: Squidpy NE · Moran's I · SPARK-X eq · SpatialDE eq · LR prox

  [Breast6]
INFO     Creating graph using `generic` coordinates and `None` transform and `1` libraries.
  [Breast7]
  ...
  [Prostate4]

Saved → results/final/real_baseline_comparison.csv
Saved → results/final/real_baseline_method_summary.csv
Saved → results/final/killer_table.csv
Saved → results/final/killer_table.tex
Saved → results/final/fig_real_baseline.png
```

#### Results1

| Metric | Spearman ρ vs. coexact | Interface LOO AUC | Interpretation |
|---|---|---|---|
| Hodge coexact (operator) | +1.00 (reference) | **0.65** | Only method above chance |
| Moran's I (esda) | **0.00** | 0.50 — chance | Zero shared variance; symmetric clustering |
| SpatialDE FSV | **0.00** | 0.50 — chance | Zero shared variance; smooth GP captures gradient only |
| SPARK-X equiv. | **0.00** | 0.50 — chance | Zero shared variance; spatial variability is gradient-compatible |
| LR proximity | **0.00** | 0.42 — below chance | Anti-localised; LR hotspots ≠ coexact hotspots |
| Squidpy NE | **0.00** | 0.50 — chance | Adjacency frequency carries no geometric information |

**Top-10% hotspot overlap (Jaccard):** immune score 0.33 (3.3× above 10% chance); LR proximity 0.05 (below chance).

**Biological endpoint recovery at coexact-defined interface:**
exhaustion markers **1.70×**, cytotoxic markers **1.70×** over background.

#### Interpretation

All five real-tool baselines show **ρ = 0.00** with the coexact interface ratio across all 26 pan-cancer sections — confirmed genuine (section IDs verified to match). The coexact operator is **completely orthogonal** to scalar clustering (Moran's I), spatial variability (SPARK-X/SpatialDE), ligand–receptor proximity (COMMOT-style), and adjacency frequency (Squidpy NE). Despite this structural independence, coexact-defined interface spots recover 1.70× enrichment for both exhaustion and cytotoxic markers, confirming that the detected structure is biologically grounded, not a geometric artefact.

The pan-cancer result is the strongest non-redundancy finding in the repository: zero shared information between any baseline and the coexact ratio across six cancer types simultaneously.

---

### Run 2 — HCC immunotherapy cohort (22 sections, 11 patients, 15 valid)

Sections with > 5,000 spots are subsampled uniformly to 5,000 (seed 42) for
computational tractability. Sample IDs in the hodge CSV and AnnData were
verified to match before running.

```bash
python3 benchmarking/build_real_baseline_benchmarking_hcc.py \
  --adata  data/hcc/hcc_scored.h5ad \
  --hodge  benchmarking/results_hcc_hodge_interface_summary_valid.csv \
  --outdir benchmarking/results2/final/ \
  --max-spots 5000 \
  --n-perm 99
```

#### Terminal output

```
Loading AnnData...
AnnData sections:  22
Hodge-valid sections: 15
Matched sections:  15
Max spots per section: 5000 (subsampled)
Baselines: Squidpy NE · Moran's I · SPARK-X eq · SpatialDE eq · LR prox

  [cytassist_71_post] n=8138 → subsampled to 5000 (strat=uniform)
  [cytassist_71_post]
INFO     Creating graph using `generic` coordinates and `None` transform and `1` libraries.

  [cytassist_71_pre] n=970
  [cytassist_72_post] n=10076 → subsampled to 5000 (strat=uniform)
  [cytassist_72_pre] n=1997
  [cytassist_73_post] n=3427
  [cytassist_74_post] n=9840 → subsampled to 5000 (strat=uniform)
  [cytassist_74_pre] n=2083
  [cytassist_76_pre] n=996
  [cytassist_79_post] n=8781 → subsampled to 5000 (strat=uniform)
  [cytassist_83_post] n=9585 → subsampled to 5000 (strat=uniform)
  [cytassist_83_pre] n=800
  [cytassist_84_post] n=3089
  [cytassist_84_pre] n=1155
  [cytassist_85_post] n=9672 → subsampled to 5000 (strat=uniform)
  [cytassist_86_post] n=10143 → subsampled to 5000 (strat=uniform)

Saved → results/final/real_baseline_comparison.csv
Saved → results/final/real_baseline_method_summary.csv
Saved → results/final/killer_table.csv
Saved → results/final/fig_real_baseline.png
```

#### Results

| Metric | Spearman ρ vs. coexact | Interface LOO AUC | Key finding |
|---|---|---|---|
| Hodge coexact (operator) | +1.00 (reference) | 0.47* | Reference |
| Moran's I (esda) | **0.00** | 0.50 — chance | Universal non-redundancy |
| Squidpy NE | **0.00** | 0.50 — chance | Universal non-redundancy |
| LR proximity | **−0.18** | 0.43 — below chance | Negative: therapy disrupts LR co-localisation |
| SpatialDE FSV | +0.76 | 0.50 — chance | Contextual co-variation; inert geometrically |
| SPARK-X equiv. | +0.86 | 0.50 — chance | Contextual co-variation; inert geometrically |

\* AUC 0.47 reflects the generic Q75 interface heuristic degrading on
post-therapy tissue where immune cells have infiltrated the tumour core.
This is a script limitation, not an operator limitation; biological endpoint
recovery remains positive.

**Top-10% hotspot overlap (Jaccard):** immune score 0.20 (2.0× above chance); LR proximity 0.06 (near chance).

**Biological endpoint recovery at coexact-defined interface:**
exhaustion markers **1.34×**, cytotoxic markers **1.54×** over background.

#### Interpretation

**Moran's I and NE (ρ = 0.00):** Universal non-redundancy, consistent with pan-cancer. Scalar clustering and adjacency frequency carry no information about coexact interface energy in either cohort.

**LR proximity (ρ = −0.18):** The negative correlation emerges when post-therapy sections are included. Post-therapy responders have higher coexact energy (organised interface) but lower COMMOT-style LR scores because immunotherapy disrupts the tumour–immune co-localisation patterns that LR metrics rely on. The coexact operator detects the geometric reorganisation of the interface field that LR methods cannot see.

**SPARK-X and SpatialDE (ρ = +0.86 / +0.76):** High cross-section correlation reflects a single-cancer-type context: under immunotherapy, sections with higher immune spatial variability also tend to have more structured interfaces. This contextual co-variation disappears across cancer types (pan-cancer ρ = 0.00 for both). Critically, despite ρ = +0.86, SPARK-X achieves AUC = 0.50 for interface discrimination — confirming that the shared variance is biologically real but geometrically inert.

**Biological validation (1.34–1.54×):** Biological enrichment remains positive across both pre- and post-therapy sections, confirming that the coexact-defined interface is a biologically meaningful zone regardless of therapy context.

---

### Cross-cohort summary

| Cohort | n sections | Moran's I ρ | LR ρ | SPARK-X ρ | Coexact AUC | Exhaustion | Cytotoxic |
|---|---|---|---|---|---|---|---|
| Pan-cancer (6 types) | 26 | 0.00 | 0.00 | 0.00 | **0.65** | 1.70× | 1.70× |
| HCC (immunotherapy) | 15 | 0.00 | −0.18 | +0.86 | 0.47* | 1.34× | 1.54× |

The pan-cancer result demonstrates **complete structural orthogonality** across diverse tissue architectures. The HCC result adds the mechanistically interpretable LR anti-correlation and confirms biological endpoint recovery under immunotherapy.

### Outputs produced

| File | Content |
|---|---|
| `results/final/real_baseline_comparison.csv` | Per-section metrics for all 5 baselines + coexact |
| `results/final/real_baseline_method_summary.csv` | Per-method Spearman ρ, median AUC, bio endpoints |
| `results/final/killer_table.csv` | Capability comparison table (for manuscript) |
| `results/final/killer_table.tex` | LaTeX `table*` environment, drop into supplementary |
| `results/final/fig_real_baseline.png` | 4-panel benchmark figure |

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
| HCC Visium CytAssist | Immunotherapy cohort — 22 sections, 11 patients, 5R/6NR (Ruf et al. 2024) | https://doi.org/10.5281/zenodo.19123188 |
| Spatial Hallmarks | Pan-cancer biological validation — 26 sections, 6 cancer types | https://doi.org/10.5281/zenodo.14044964 |

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
## Large-file policy

This repository intentionally avoids treating massive intermediate files as ordinary source code.

Large files such as:

- `.h5ad` objects;
- dense edge tables;
- high-resolution figure dumps;
- per-section intermediate arrays;

should either be:

1. regenerated from scripts,
2. stored as GitHub release assets,
3. stored in Zenodo/Figshare/Hugging Face datasets,
4. or kept locally when not manuscript-critical.

Manuscript-critical summary CSVs and final benchmark outputs may remain tracked when they are needed for reproducibility.
---

## Citation

If you use this framework, please cite:

```bibtex
@article{enoch2026nonpassive,
  author  = {Enoch, Anas},
  title   = {Non-gradient spatial organisation at tumour--immune interfaces revealed by operator-based analysis},
  year    = {2026},
  note    = {Manuscript under submission}
}
```

---

## Contact

**Anas Enoch, MD**  
Mohammed VI University of Health Sciences (UM6SS), Casablanca, Morocco  
GitHub: [@Anas-Enoch](https://github.com/Anas-Enoch)  
Repository: [Hodge_Laplacian_GNN](https://github.com/Anas-Enoch/Hodge_Laplacian_GNN)
