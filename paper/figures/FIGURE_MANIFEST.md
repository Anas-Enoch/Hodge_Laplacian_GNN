# Figure Manifest
## Non-passive transport organization at tumour–immune interfaces

Complete provenance for every manuscript figure and supplementary table.

---

## Main Figures

### Figure 1 — Gradient vs. Non-Gradient Interaction Fields

```
Script:   scripts/visualization/fig1_conceptual.py
Inputs:   None (schematic)
Outputs:  paper/figures/fig1_conceptual.png
Runtime:  < 1 min
Claim:    Conceptual separation of gradient (passive-compatible) from coexact
          (non-gradient, non-passive) tumour–immune interaction geometries.
```

### Figure 2 — Wedge-Flux Operator Construction

```
Script:   scripts/analysis/hodge_decomposition.py
          scripts/visualization/fig2_operator.py
Inputs:   datasets/processed/tnbc_scored.h5ad
Outputs:  results/final/hodge_summary.csv
          paper/figures/fig2_operator.png
Runtime:  ~30 min
Key stat: Coexact fraction elevated at tumour–immune interfaces
```

### Figure 3 — Discrete Hodge Decomposition Maps

```
Script:   scripts/visualization/fig3_hodge_maps.py
Inputs:   results/final/hodge_summary.csv
          datasets/processed/tnbc_scored.h5ad
Outputs:  paper/figures/fig3_hodge_maps.png
Runtime:  ~10 min
Key stat: Responder sections show concentrated coexact hotspots;
          non-responder sections show diffuse maps
```

### Figure 4 — PDE-Constrained GNN Falsification

```
Script:   scripts/analysis/pde_gnn_falsification.py
          scripts/visualization/fig4_gnn_falsification.py
Inputs:   datasets/processed/tnbc_scored.h5ad
          results/final/hodge_summary.csv
Outputs:  results/final/gnn_falsification.csv
          results/final/null_model_battery.csv
          paper/figures/fig4_gnn_falsification.png
Runtime:  ~45 min
Key stat: log B M1a/M0 = +45.95; log B M1b/M1a = +517.6
Nulls:    density · phenotype shuffle · generic antisymmetry · remeshing
```

### Figure 5 — Coexact Interface Enrichment

```
Script:   scripts/analysis/coexact_enrichment_test.py
          scripts/visualization/fig5_enrichment.py
Inputs:   results/final/hodge_summary.csv
Outputs:  results/final/enrichment_test.csv
          paper/figures/fig5_enrichment.png
Runtime:  ~5 min
Key stat: Sign test p = 1.49×10⁻⁸ (26 pan-cancer sections)
```

### Figure 6 — CosMx Cross-Platform Validation

```
Script:   scripts/analysis/cosmx_validation.py
          scripts/visualization/fig6_cosmx.py
Inputs:   datasets/processed/cosmx_scored.h5ad
Outputs:  results/final/cosmx_validation.csv
          paper/figures/fig6_cosmx.png
Runtime:  ~25 min
Key stat: Coexact enrichment replicates at CosMx single-cell resolution
          (108 FOVs)
```

### Figure 7 — Biological Validation (4-Tier Exhaustion Markers)

```
Script:   spatial_hallmark/build_biological_validation.py
          scripts/visualization/fig7_biological.py
Inputs:   datasets/processed/spatial_hallmarks_scored.h5ad
          spatial_hallmark/results_spatial_hallmarks/spatial_hallmarks_hodge_interface.csv
          spatial_hallmark/results_spatial_hallmarks/spatial_hallmarks_kts_edges.csv
Outputs:  spatial_hallmark/results_spatial_hallmarks/tier1_module_correlation.csv
          spatial_hallmark/results_spatial_hallmarks/tier2_exhaustion_endpoint.csv
          spatial_hallmark/results_spatial_hallmarks/tier3_stromal_mediation.csv
          paper/figures/fig7_biological.png
Runtime:  ~15 min
Key stat: Tier 1: Cytotoxic 25/26 ρ=0.240
          Tier 2: 7 exhaustion markers 26/26, ratios 3.36–4.75×, p=1.49×10⁻⁸
          Tier 3: TGFB1/FAP/CXCL12 significant
```

### Figure 8 — Baseline Comparison

```
Script:   spatial_hallmark/baseline_comparison.py
          scripts/visualization/fig8_baselines.py
Inputs:   datasets/processed/spatial_hallmarks_scored.h5ad
          spatial_hallmark/results_spatial_hallmarks/spatial_hallmarks_hodge_interface.csv
          spatial_hallmark/results_spatial_hallmarks/spatial_hallmarks_kts_edges.csv
Outputs:  spatial_hallmark/results_spatial_hallmarks/results_baseline_comparison.csv
          spatial_hallmark/results_spatial_hallmarks/baseline_comparison.png
          paper/figures/fig8_baselines.png
Runtime:  ~20 min
Key stat: Coexact 0.929 · Moran's I 0.817 · NE 0.517 · entropy 0.400 · Node2Vec 0.375
```

---

## Supplementary Figures

| Figure | Script | Key result |
|---|---|---|
| S1 Null battery detail | `pde_gnn_falsification.py --detailed` | 4 null models, all R→1 |
| S2 Remeshing invariance | `scripts/analysis/remeshing_invariance.py` | Sign-consistency > 0.80, k ∈ {4,6,8,10,12} |
| S3 Stochastic Hodge posterior | `scripts/analysis/stochastic_hodge_bayes.py` | log B +45.95, +517.6 |
| S4 MERFISH validation | `scripts/analysis/merfish_validation.py` | Cross-resolution robustness |
| S5 Density confound | `scripts/analysis/density_nuisance_control.py` | Partial ρ density p = n.s. |

---

## Supplementary Tables

| Table | Content | Output CSV |
|---|---|---|
| S1 | Per-section Hodge summary | `results/final/hodge_summary.csv` |
| S2 | Null model battery | `results/final/null_model_battery.csv` |
| S3 | Biological validation tiers | `spatial_hallmark/results_spatial_hallmarks/tier*.csv` |
| S4 | Baseline comparison LOO AUC | `spatial_hallmark/results_spatial_hallmarks/results_baseline_comparison.csv` |
| S5 | CosMx FOV-level results | `results/final/cosmx_validation.csv` |
