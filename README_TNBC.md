# Hodge-Laplacian GNN — Operator-Based Transport Phenotyping for TNBC Spatial Transcriptomics

**Non-passive transport organization at tumor–immune interfaces revealed by operator-based analysis**  
Anas Enoch, MD · Mohammed VI University of Health Sciences (UM6SS), Casablanca  
Target journal: *Bioinformatics Advances* (Oxford) · Submission BIOINF-2026-0777

---

## Overview

This file documents the full TNBC cohort pipeline (Steps 1–24) applied to **GSE210616** (22 primary TNBC patients, 43 Visium sections). For the general framework and scientific framing, see `README.md`.

**Canonical flux tag:** `flux_tumor_immune_region_interface_weighted`  
**Region labels:** `interface_like`, `tumor_enriched`, `stroma_enriched`, `immune_enriched`, `other`

---

## Cohort-Level Results (summary)

| Diagnostic | Key metric | Sections | Sign test *p* |
|---|---|---|---|
| Step 7: coexact enrichment > 1.0 | 40/40 valid sections | 40 | < 10⁻¹² |
| Step 7: coexact/exact differential > 1.0, median 2.54 | 38/40 sections | 40 | < 10⁻⁹ |
| Step 19: tumor residual ρ = 0.349 within interface | 17/18 sections | 18 | 7.2×10⁻⁵ |
| Step 20 (NCG): ρ(NC, coexact) at interface = 0.832 | 18/18 sections | 18 | < 10⁻⁵ |
| Step 20 (NCG): Δρ = +0.198 (interface vs. other) | 18/18 sections | 18 | < 10⁻⁵ |
| Step 21 (Zeta): f_low = 0.653 | 19/19 sections | 19 | < 10⁻⁵ |
| Step 21 (Zeta): Z(s=1) = 1.89 | 18/19 sections | 19 | < 10⁻⁴ |
| Step 22: cytotoxic enrichment in top-10% NC nodes | 15/19 sections | 19 | 0.0096 |
| Step 22: hypoxia null control | 6/19 sections | 19 | 0.97 |
| Step 24: log B(M1a/M0), median +45.95 | 19/19 sections | 19 | < 10⁻⁵ |
| Step 24: log B(M1b/M1a), median +517.6 | 19/19 sections | 19 | < 10⁻⁵ |
| Step 24: posterior R under M1b, median 9.81 | 19/19 sections | 19 | < 10⁻⁵ |
| GNN falsification: coexact collapse = 2.7×10⁻¹² | — | — | — |

All cohort-level findings are independent of race, neoadjuvant chemotherapy, age, and recurrence-free survival (all *p* > 0.18, *n* = 22 patients).

---

## Required Inputs

Each sample directory must contain exactly four files:

```
data/TNBC_GSE210616/GSM_xxxxx/
├── GSMxxxxx_filtered_feature_bc_matrix.h5
├── GSMxxxxx_tissue_positions_list.csv
├── GSMxxxxx_tissue_hires_image.png
└── GSMxxxxx_scalefactors_json.json
```

Marker gene lists are built into the scripts — no external reference files required.

---

## Pipeline Architecture

| Layer | Steps | Role |
|-------|-------|------|
| **Geometric probe** | 1–7 | Construct proxy transport field; decompose into exact / coexact / harmonic; test interface enrichment |
| **Mechanistic diagnostics** | 9–12 | Curl maps, Lie-structured null, hotspot tests |
| **Operator falsification** | 13–18 | PDE-constrained GNN; ablation; hybrid potential decomposition |
| **Artifact diagnostics** | 19–23 | Biological anchoring (spatial, spectral, biological independence) |
| **Probabilistic framework** | 24 | Stochastic Hodge; Bayesian model comparison |

---

## Step-by-Step Commands

### Steps 1–7: Core geometric pipeline

```bash
source .venv/bin/activate

SAMPLE=GSM_6433619
SAMPLE_DIR=data/TNBC_GSE210616/${SAMPLE}

# Step 1 — Marker scoring
python -m scripts_tnbc.step1_tnbc_map \
  --sample_dir ${SAMPLE_DIR}

# Step 2 — Region annotation
python -m scripts_tnbc.step2_tnbc_regions \
  --sample_id ${SAMPLE} \
  --sample_dir ${SAMPLE_DIR} \
  --statsdir stats/CSV_GSM \
  --figdir visium_figures

# Step 3 — Spatial graph and incidence matrices
python -m scripts_tnbc.step3_tnbc_spatial_graph \
  --sample_id ${SAMPLE} \
  --sample_dir ${SAMPLE_DIR} \
  --statsdir stats/CSV_GSM

# Step 4 — Wedge flux construction
python -m scripts_tnbc.step4_tnbc_flux_proxies \
  --sample-id ${SAMPLE}

# Step 6 — Hodge decomposition
python -m scripts_tnbc.step6_tnbc_hodge_decomposition \
  --sample-id ${SAMPLE} \
  --flux-col flux_tumor_immune_region_interface_weighted

# Step 7 — Interface enrichment testing
python -m scripts_tnbc.step7_tnbc_region_enrichment \
  --sample-id ${SAMPLE} \
  --flux-col flux_tumor_immune_region_interface_weighted \
  --min-nodes-per-region 10
```

> **Note on Step 5:** Optional metabolic proxy step. Skip unless metabolic flux variants are needed.

### Steps 9–18: Diagnostics and falsification

```bash
# Steps 9–12: curl maps, Lie null, hotspot tests
python scripts_tnbc/step9_tnbc_curl_maps.py --sample-id ${SAMPLE}
python scripts_tnbc/step10_curl_null_test.py --sample-id ${SAMPLE}
python scripts_tnbc/step11_lie_structured_null.py --sample-id ${SAMPLE}
python scripts_tnbc/step12_region_hotspot_lie_test.py --sample-id ${SAMPLE}

# Steps 13–14: GNN data preparation and training
python scripts_tnbc/step13_tnbc_prepare_gnn_data.py --sample-id ${SAMPLE}
python scripts_tnbc/step14_tnbc_train_pde_gnn.py --sample-id ${SAMPLE}

# Steps 15–17: GNN analysis, figures, hybrid decomposition
python scripts_tnbc/step15_tnbc_analyze_gnn_flux.py --sample-id ${SAMPLE}
python scripts_tnbc/step16_transport_equation_figure.py --sample-id ${SAMPLE}
python scripts_tnbc/step17_tnbc_solve_hybrid_potentials.py --sample-id ${SAMPLE}

# Step 18: ablation (unconstrained baseline)
python scripts_tnbc/step18_ablation_no_constraint.py --sample-id ${SAMPLE}
```

### Steps 19–22: Validation against construction artifacts

```bash
# Step 19 — Biological anchoring (proxy coexact vs. residualized programs)
python scripts_tnbc/step19_generate_csv.py --sample-id ${SAMPLE}
python -m scripts_tnbc.step19_coexact_bio_correlation \
  --mode sample \
  --sample-id ${SAMPLE} \
  --flux-tag flux_tumor_immune_region_interface_weighted

# Cohort mode (after running all samples):
python -m scripts_tnbc.step19_coexact_bio_correlation \
  --mode cohort \
  --flux-tag flux_tumor_immune_region_interface_weighted

# Step 20 — NCG commutator grounding
python scripts_tnbc/step20_ncg_commutator.py \
  --mode sample --sample-id ${SAMPLE} \
  --flux-tag flux_tumor_immune_region_interface_weighted

python scripts_tnbc/step20_ncg_commutator.py \
  --mode cohort \
  --flux-tag flux_tumor_immune_region_interface_weighted

# Step 21 — Zeta spectral diagnostic
python scripts_tnbc/step21_zeta_spectral.py \
  --mode sample --sample-id ${SAMPLE} \
  --flux-tag flux_tumor_immune_region_interface_weighted

python scripts_tnbc/step21_zeta_spectral.py \
  --mode cohort \
  --flux-tag flux_tumor_immune_region_interface_weighted

# Step 22 — Independent biomarker validation
python scripts_tnbc/step22_ncg_bio_validation.py \
  --mode sample --sample-id ${SAMPLE} \
  --flux-tag flux_tumor_immune_region_interface_weighted

python scripts_tnbc/step22_ncg_bio_validation.py \
  --mode cohort \
  --flux-tag flux_tumor_immune_region_interface_weighted
```

### Step 23: Operator robustness

```bash
python scripts_tnbc/step23_operator_robustness.py \
  --mode sample --sample-id ${SAMPLE} \
  --flux-tag flux_tumor_immune_region_interface_weighted

python scripts_tnbc/step23_operator_robustness.py \
  --mode cohort \
  --flux-tag flux_tumor_immune_region_interface_weighted
```

### Step 24: Stochastic Hodge decomposition

```bash
# Per sample (auto-sigma, k_cycle=30, lower-Hodge cycle split)
python scripts_tnbc/step24_stochastic_hodge_v2.py \
  --mode sample --sample-id ${SAMPLE} \
  --flux-tag flux_tumor_immune_region_interface_weighted \
  --statsdir stats/CSV_GSM --outdir stats/CSV_GSM \
  --sigma-mode auto --sigma-fraction 0.5 \
  --k-cycle 30 --interface-weight 3.0

# Full cohort loop
while read sid; do
  python scripts_tnbc/step24_stochastic_hodge_v2.py \
    --mode sample --sample-id "$sid" \
    --flux-tag flux_tumor_immune_region_interface_weighted \
    --statsdir stats/CSV_GSM --outdir stats/CSV_GSM \
    --sigma-mode auto --sigma-fraction 0.5 \
    --k-cycle 30 --interface-weight 3.0
done < valid_sample_ids.txt

# Cohort aggregate
python scripts_tnbc/step24_stochastic_hodge_v2.py \
  --mode cohort \
  --flux-tag flux_tumor_immune_region_interface_weighted \
  --statsdir stats/CSV_GSM --outdir stats/CSV_GSM

# Optional: full Hodge via Delaunay triangulation (uses B2 from spatial coordinates)
python scripts_tnbc/step24_stochastic_hodge_v2.py \
  --mode sample --sample-id ${SAMPLE} \
  --flux-tag flux_tumor_immune_region_interface_weighted \
  --use-delaunay-faces \
  --statsdir stats/CSV_GSM --outdir stats/CSV_GSM
```

**Step 24 key parameters:**

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `--sigma-mode` | `auto` | Calibrate σ per section to cycle-component std of normalised flux |
| `--sigma-fraction` | `0.5` | σ = fraction × std(P_c @ Y_norm); 0.5 → noise = 50% of cycle signal amplitude |
| `--k-cycle` | `30` | Low-rank cycle prior: 30 lowest-eigenvalue cycle modes (avoids O(n_edges) complexity penalty) |
| `--interface-weight` | `3.0` | M1b: coexact prior 3× stronger at interface edges, 1/3× elsewhere (trace-normalised) |
| `--obs-model` | `identity` | Y = edge flux values (wedge flux is solenoidal; divergence model gives Y≈0) |
| `--use-delaunay-faces` | off | Full Hodge split via Delaunay triangulation; requires x_fullres, y_fullres in node file |

---

## Expected Outputs

### Steps 1–7

| Step | Key output files |
|------|-----------------|
| 1 | `visium_figures/GSM_xxxx_step1_marker_scores.csv`, spatial maps |
| 2 | `stats/CSV_GSM/GSM_xxxx_step2_region_assignments.csv`, region map |
| 3 | `*_step3_nodes.csv`, `*_edges.csv`, `*_faces.csv`, `*_B1.npz`, `*_B2.npz`, `*_L1_edge_hodge.npz` |
| 4 | `*_step4_edge_fluxes.csv` (all wedge variants) |
| 6 | `*_step6_edges_hodge_*.csv`, `*_step6_nodes_hodge_*.csv`, `*_step6_energy_summary_*.csv` |
| 7 | `*_step7_region_summary_*.csv`, `*_step7_region_enrichment_*.csv` |

The Step 7 enrichment CSV contains one row per metric (exact, coexact, harmonic) with columns: `observed_ratio`, `perm_p_two_sided`, `null_mean`, `null_std`, `n_focus`, `n_ref`, `note`, `harmonic_suppressed`.

### Steps 19–24

| Step | Key output files |
|------|-----------------|
| 19 | `*_step19_coexact_bio_*.csv`, cohort sign test table |
| 20 | `*_step20_ncg_*.csv`, cohort NCG summary |
| 21 | `*_step21_zeta_*.csv`, cohort Zeta summary |
| 22 | `*_step22_topk_*.csv`, hypoxia null result |
| 23 | `*_step23_robustness_*.csv` |
| 24 | `*_step24v2_summary_*.csv` (per section), `cohort_step24v2_summary_*.csv`, `*_step24v2_posterior_nodes_*.csv` |

The Step 24 per-section summary includes: `log_B_M1a_vs_M0`, `log_B_M1b_vs_M1a`, `log_B_M1b_vs_M0`, `post_R_M1b`, `post_R_M0`, `det_enrichment_ratio`, `sigma_used`, `flux_scale`, `std_Yc_norm`, `k_cycle`, `hodge_type`, `obs_model`.

---

## Step 19 — Biological Anchoring Detail

Step 19 tests whether proxy-derived coexact phenotype aligns with biological variation within interface nodes, over and above gross region composition. It uses Step 6 outputs (proxy field), not GNN outputs.

Five statistics are computed in sequence:
1. Raw Spearman — descriptive only; not reported as primary
2. Region-demeaned Spearman — primary nonparametric check; removes region-composition confound
3. OLS with region covariate — standardised β; effect size only
4. Within-interface analysis — interface-like nodes only
5. FDR correction (BH) — applied separately to global and subset analyses

Cohort result (18 valid sections): tumor residual median ρ = 0.349 (17/18, *p* = 7.2×10⁻⁵); immune residual median ρ = 0.212 (16/18, *p* = 6.6×10⁻⁴); stroma median ρ = 0.146 (14/18, *p* = 0.015).

---

## Step 20 — NCG Commutator Grounding Detail

Tests whether the coexact component is the geometric image of operator non-commutativity $[M_A, M_B]$ and whether this relationship is specifically elevated at interface nodes (rules out spatial construction artifact).

Per-node non-commutativity norm: $\mathrm{NC}_i = \frac{1}{|\mathcal{N}(i)|}\sum_{j \in \mathcal{N}(i)} f_{ij}^2$

Cohort result: interface Spearman ρ(NC, E_coexact) median = 0.832 (18/18); Δρ (interface vs. other) = +0.198 (18/18, *p* < 10⁻⁵).

---

## Step 21 — Zeta Spectral Diagnostic Detail

Tests whether coexact energy is carried by geometrically meaningful large-scale eigenmodes (low eigenvalue λ) rather than high-frequency noise — rules out spectral construction artifact.

$Z(s) = \frac{\sum_k \lambda_k^{-s} E_k}{\sum_k E_k}$ where $E_k$ = coexact energy in eigenmode $k$.

Spatial permutation null: 1000 shuffles of node labels preserving graph topology.

Cohort result: f_low = 0.653 (19/19, *p* < 10⁻⁵); Z(s=1) = 1.89 (18/19, *p* < 10⁻⁴).

---

## Step 22 — Independent Biomarker Validation Detail

**Circularity boundary.** The following genes were used to construct the wedge flux and are **strictly forbidden** from Step 22:
- Tumor: EPCAM, KRT8, KRT18, KRT19, ERBB2, MUC1, TACSTD2
- Immune: PTPRC, CD3D, CD3E, NKG7, CD68, C1QA, CXCL9, CXCL10
- Stroma: COL1A1, COL1A2, DCN, LUM, POSTN, FAP, TAGLN

Six independent marker sets used for validation:
1. Cytotoxic activity: GZMB, PRF1, GNLY, IFNG
2. CD8 T cells: CD8A, CD8B
3. Chemokine recruitment: CCL5, CXCL13, CCL4
4. Antigen presentation: HLA-DRA, HLA-DRB1, CD74
5. M2 macrophage polarity: MRC1, CD163, TGFB1
6. Hypoxia (null control): HIF1A, VEGFA, LDHA, ENO1, GLUT1

Cohort result: all five immune sets enriched in top-10% NC nodes (14–15/19, *p* < 0.04); hypoxia null control 6/19 (*p* = 0.97).

---

## Step 24 — Stochastic Hodge Decomposition Detail

**Three model classes:**

- **M0 (passive):** $C_c = 0$ — zero cycle-space variance; passive transport assumption.
- **M1a (sparse uniform active):** $C_c = V_k \mathrm{diag}((1+\lambda_k)^{-\beta_c}) V_k^\top$ restricted to $k = 30$ lowest-eigenvalue cycle modes. Low-rank prior avoids O(n_edges) complexity penalty (otherwise −185 nats, data-independent).
- **M1b (interface-localised active):** Same low-rank basis; variance redistributed to interface edges (weight 3.0 up, 1/3 down, trace-normalised so model complexity = M1a).

**Auto-sigma:** Y normalised per section to std(Y) = 1; σ = 0.5 × std(P_c @ Y_norm). This ensures σ² ∝ var(cycle component), rendering the Bayes factor scale-invariant. Without normalisation, raw flux values O(10⁻⁵) cause σ ≈ 10⁻⁵ and a data-independent penalty of −273 nats.

**Two Bayes factors:**
- log B(M1a/M0): evidence that any non-gradient (cycle-space) structure is required.
- log B(M1b/M1a): evidence that this structure is interface-localised vs. uniform — probabilistic counterpart of the Step 7 sign test.

**Cohort results (19 sections):**

| Metric | Median | N positive | Sign test *p* |
|--------|--------|------------|---------------|
| log B(M1a/M0) | +45.95 | 19/19 | < 10⁻⁵ |
| log B(M1b/M1a) | +517.6 | 19/19 | < 10⁻⁵ |
| Posterior R under M1b | 9.81 | 19/19 | < 10⁻⁵ |
| Posterior R under M0 (null) | 1.00 | 11/19 | 0.32 |

---

## Exclusion Criteria

Samples are excluded from enrichment analysis (Step 7) if either `interface_like` or `tumor_enriched` regions contain fewer than **10 nodes**. Threshold is pre-specified; excluded samples are retained for global Hodge statistics.

Step 24 requires at least 30 edges per section.

---

## Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `FileNotFoundError` | Missing `.h5`, positions, image, or JSON | Verify all four files in sample directory |
| `WARNING: B2 is None – falling back to Delaunay` | No face schema in step3_faces.csv | Normal — Delaunay fallback is correct |
| `Variable names are not unique` (scanpy) | Duplicate gene IDs | Non-fatal; `.var_names_make_unique()` in Step 1 |
| Step 7 `low_sample_size` | Interface or tumor region < 10 nodes | Excluded from enrichment; global stats valid |
| Slow permutations | Default n=1000 | Reduce to `--n-perm 100` for testing |
| Step 24 log B(M1a/M0) still negative | sigma_fraction too low | Increase `--sigma-fraction` to 1.0; check `std_Yc_norm` in CSV (should be > 0.1) |
| Step 24 `enrichment_ratio` is NaN | Missing `tumor_enriched` or `interface_like` nodes | Verify region labels: both `tumor_enriched` and `tumor_core` are accepted |
| `harmonic_suppressed = True` (Step 7) | E_harmonic/E_coexact < 10⁻⁶ | Correct — harmonic at numerical noise; p-value suppressed |
| Step 24 P_c trace ≈ 0 | Tree-structured graph (no cycles) | Use `--use-delaunay-faces` to construct full Hodge via triangulation |

---

## Statistical Notes

**Permutation null (Step 7):** Region labels shuffled (1000 permutations); enrichment ratio recomputed each time. Two-sided empirical p-value: `p = (count + 1) / (n_perm + 1)`.

**Harmonic suppression:** When E_harmonic / E_coexact < 10⁻⁶, harmonic is at numerical noise from lsqr projection. Nominal significance for `node_energy_harmonic` in that regime is suppressed.

**Aggregate fractions:** Region-level `agg_frac_*` uses sum(E_component) / sum(E_exact + E_coexact + E_harmonic) — guarantees fractions in [0, 1].

**Step 24 Bayes factor interpretation:** On finite complexes, mutual singularity of M0 and M1 (which holds in the projective-limit Abstract Wiener Space construction) is replaced by the Bayes factor. log B > 10 = strong evidence; 3–10 = moderate; 0–3 = weak. Negative = lower model favoured. The scale of log B(M1b/M1a) (+517 nats) reflects strong spatial alignment of the flux field with the interface-weighted prior; the trace normalisation ensures this is a pure spatial-pattern comparison, not a complexity advantage.

---

## Reproducibility

To reproduce the full cohort analysis:

```bash
# Run Steps 1–7 for all 43 GSM accessions in GSE210616
# Then aggregate Step 7:
python -c "
import pandas as pd, glob
dfs = [pd.read_csv(f) for f in
       glob.glob('stats/CSV_GSM/*_step7_region_enrichment_flux_tumor_immune_region_interface_weighted.csv')]
pd.concat(dfs).to_csv('cohort_enrichment_all_sections.csv', index=False)
"

# Aggregate Step 24:
python scripts_tnbc/step24_stochastic_hodge_v2.py \
  --mode cohort \
  --flux-tag flux_tumor_immune_region_interface_weighted \
  --statsdir stats/CSV_GSM --outdir stats/CSV_GSM
```

Recommended files to track in version control:
```
scripts_tnbc/step6_tnbc_hodge_decomposition.py
scripts_tnbc/step7_tnbc_region_enrichment.py
scripts_tnbc/step18_ablation_no_constraint.py
scripts_tnbc/step19_coexact_bio_correlation.py
scripts_tnbc/step20_ncg_commutator.py
scripts_tnbc/step21_zeta_spectral.py
scripts_tnbc/step22_ncg_bio_validation.py
scripts_tnbc/step23_operator_robustness.py
scripts_tnbc/step24_stochastic_hodge_v2.py
valid_sample_ids.txt
stats/CSV_GSM/cohort_step24v2_summary_flux_tumor_immune_region_interface_weighted.csv
```

---

## Environment

```bash
python >= 3.9
scanpy
numpy
pandas
matplotlib
scipy
scikit-learn
```

Install:
```bash
pip install scanpy numpy pandas matplotlib scipy scikit-learn
```

---

## Citation

> Anas Enoch. *Non-passive transport organization at tumor–immune interfaces revealed by operator-based analysis.* Bioinformatics Advances, 2026. Submission BIOINF-2026-0777.

> Bassiouni R et al. *Spatial Transcriptomic Analysis of a Diverse Patient Cohort Reveals a Conserved Architecture in Triple-Negative Breast Cancer.* Cancer Research 83(1):34–48, 2023. GEO: GSE210616.

---

## Contact

Open a GitHub issue or contact via the repository: [github.com/Anas-Enoch/Hodge_Laplacian_GNN](https://github.com/Anas-Enoch/Hodge_Laplacian_GNN)
