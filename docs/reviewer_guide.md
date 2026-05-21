# Reviewer Guide

*Non-passive transport organization at tumour–immune interfaces*  
Manuscript ID: BIOINF-2026-0777  
Repository: github.com/Anas-Enoch/Hodge_Laplacian_GNN

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| RAM | 16 GB | 64 GB |
| CPU | 8 cores | 16 cores |
| GPU | Not required | Optional (GNN training) |
| Storage | 20 GB free | 100 GB |
| Python | 3.9+ | 3.11 |

Full pipeline runtime on recommended hardware: ~4 hours.

---

## Environment Setup

```bash
# Option A — conda
conda env create -f environment.yml
conda activate hodge-operator

# Option B — pip
pip install -r requirements.txt
```

---

## Reviewer Workflow: Key Tables

### Reproduce Table 1 (Coexact enrichment results)

```bash
python scripts/analysis/hodge_decomposition.py \
    --adata  datasets/processed/tnbc_scored.h5ad \
    --output results/final/hodge_summary.csv
```

Expected output columns: `sample_id`, `coexact_fraction`, `interface_coexact_energy`,
`background_coexact_energy`, `enrichment_ratio`, `response`.  
Key result: enrichment ratio R > 1 in all responder sections.

### Reproduce Table 2 (Null model battery)

```bash
python scripts/analysis/pde_gnn_falsification.py \
    --adata      datasets/processed/tnbc_scored.h5ad \
    --hodge      results/final/hodge_summary.csv \
    --null-tests density shuffle antisymmetry remeshing \
    --output     results/final/gnn_falsification.csv
```

Expected: log B M1a/M0 = +45.95; log B M1b/M1a = +517.6.  
All four null models produce R → 1.

---

## Reviewer Workflow: Biological Validation

```bash
# Step 1: Build KTS spatial transition-bias edges
python3 spatial_hallmark/build_spatial_hallmarks_kts_edges.py \
    --adata datasets/processed/spatial_hallmarks_scored.h5ad \
    --out   spatial_hallmark/results_spatial_hallmarks/

# Step 2: Run 4-tier validation
python3 spatial_hallmark/build_biological_validation.py \
    --adata   datasets/processed/spatial_hallmarks_scored.h5ad \
    --hodge   spatial_hallmark/results_spatial_hallmarks/spatial_hallmarks_hodge_interface.csv \
    --kts     spatial_hallmark/results_spatial_hallmarks/spatial_hallmarks_kts_edges.csv \
    --out-dir spatial_hallmark/results_spatial_hallmarks/ \
    --fig-dir spatial_hallmark/results_spatial_hallmarks/
```

Expected outputs:
- `tier1_module_correlation.csv` — Cytotoxic 25/26 ρ=0.240
- `tier2_exhaustion_endpoint.csv` — all 7 exhaustion markers 26/26, p=1.49×10⁻⁸
- `tier3_stromal_mediation.csv` — TGFB1/FAP/CXCL12 significant

---

## Reviewer Workflow: Baseline Comparison

```bash
python3 spatial_hallmark/baseline_comparison.py \
    --adata   datasets/processed/spatial_hallmarks_scored.h5ad \
    --hodge   spatial_hallmark/results_spatial_hallmarks/spatial_hallmarks_hodge_interface.csv \
    --kts     spatial_hallmark/results_spatial_hallmarks/spatial_hallmarks_kts_edges.csv \
    --out     spatial_hallmark/results_spatial_hallmarks/results_baseline_comparison.csv \
    --fig     spatial_hallmark/results_spatial_hallmarks/baseline_comparison.png
```

Expected results:

| Metric | LOO AUC |
|---|---|
| Interface coexact energy | 0.929 |
| Moran's I | 0.817 |
| NE score (Giotto) | 0.517 |
| Graph spectral entropy | 0.400 |
| Node2Vec embedding | 0.375 |

---

## Reviewer Workflow: Regenerate Figure Panels

```bash
python scripts/visualization/generate_manuscript_figures.py \
    --results results/final/ \
    --output  paper/figures/
```

See [`paper/figures/FIGURE_MANIFEST.md`](../paper/figures/FIGURE_MANIFEST.md)
for complete figure-to-script mapping.

---

## Key Statistical Results — Reproduction Checklist

| Result | Value | Script | Output |
|---|---|---|---|
| Coexact sign test | p = 1.49×10⁻⁸ | `coexact_enrichment_test.py` | `enrichment_test.csv` |
| log B M1a/M0 | +45.95 | `pde_gnn_falsification.py` | `gnn_falsification.csv` |
| log B M1b/M1a | +517.6 | `pde_gnn_falsification.py` | `gnn_falsification.csv` |
| Density partial ρ | p = n.s. | `density_nuisance_control.py` | `density_control.csv` |
| Moran's I ρ vs coexact | +0.646 | `baseline_comparison.py` | `results_baseline_comparison.csv` |
| Coexact LOO AUC | 0.929 | `baseline_comparison.py` | `results_baseline_comparison.csv` |
| T2 exhaustion p | 1.49×10⁻⁸ | `build_biological_validation.py` | `tier2_exhaustion_endpoint.csv` |

---

## Manuscript Figure Locations

| Figure | Output file | Generating script |
|---|---|---|
| Fig 1 | `paper/figures/fig1_conceptual.png` | `scripts/visualization/fig1_conceptual.py` |
| Fig 2 | `paper/figures/fig2_operator.png` | `scripts/visualization/fig2_operator.py` |
| Fig 3 | `paper/figures/fig3_hodge_maps.png` | `scripts/visualization/fig3_hodge_maps.py` |
| Fig 4 | `paper/figures/fig4_gnn_falsification.png` | `scripts/visualization/fig4_gnn_falsification.py` |
| Fig 5 | `paper/figures/fig5_enrichment.png` | `scripts/visualization/fig5_enrichment.py` |
| Fig 6 | `paper/figures/fig6_cosmx.png` | `scripts/visualization/fig6_cosmx.py` |
| Fig 7 | `paper/figures/fig7_biological.png` | `scripts/visualization/fig7_biological.py` |
| Fig 8 | `paper/figures/fig8_baselines.png` | `scripts/visualization/fig8_baselines.py` |

---

## Legacy Script Name Mapping

| Old name | New name |
|---|---|
| `scripts_gse278936/step23a_power_spectrum_test.py` | `scripts/analysis/spectral_power_analysis.py` |
| `scripts_gse278936/step23b_local_vs_global_prediction.py` | `scripts/analysis/local_global_operator_prediction.py` |
| `scripts_gse278936/step24v2_summary_flux.py` | `scripts/analysis/posterior_operator_summary.py` |
| `scripts_gse278936/step21b_zeta_profile.py` | `scripts/analysis/zeta_regularity_profile.py` |
| `scripts_gse278936/step25_density_nuisance_control.py` | `scripts/analysis/density_nuisance_control.py` |
| `scripts_gse278936/step14_gnn_training.py` | `core/pde_constraints/gnn_training.py` |
| `scripts_gse278936/step16_transport_equation.py` | `core/transport_models/transport_equation_solver.py` |
| `scripts_cosmx/step01–08_cosmx_*.py` | `scripts/analysis/cosmx_*.py` |
