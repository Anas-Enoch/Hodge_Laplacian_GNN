# Hodge-Laplacian GNN — TNBC Discovery Pipeline (GSE210616)
## Non-passive transport organization at tumor–immune interfaces revealed by operator-based analysis

**Anas Enoch, MD · Mohammed VI University of Health Sciences (UM6SS), Casablanca**
**Contact:** anas_nour@um5.ac.ma

---

## Overview

This README documents the TNBC discovery and rebuilt-validation pipeline for GSE210616. The legacy TNBC pipeline provides the core Hodge/coexact enrichment results, while the rebuilt modern schema enables corrected normalized Zeta energy-matched null testing and KTS transition-bias analysis.

Applied to 43 Visium sections from 22 primary TNBC patients:
- Coexact enrichment > 1.0 at tumor–immune interfaces in **40/40 valid sections** (sign test p < 10⁻¹²)
- Enrichment is consistent across all 22 patients, independent of race, chemotherapy, and survival
- After corrected energy-matched Zeta null: **2/43 sections significant** (sign test p = 1.0) — no interface-specific spectral organization
- KTS analysis: **pathway-specific exhaustion bias** through immune-active and stromal states (not direct tumor-driven)

---

## Final TNBC Interpretation

Tumor–immune interfaces robustly concentrate **non-gradient coexact interaction intensity**, but do not exhibit interface-specific spectral organization after energy control. KTS analysis reveals that the biologically meaningful structure lies in **transition bias**: immune-active and stromal states preferentially transition toward immune-exhausted states, while direct tumor-to-exhaustion transitions are not enriched.

---

## Corrected Multi-Layer Cohort Summary

| Layer | Diagnostic | TNBC result | Interpretation |
|---|---|---|---|
| **Local** | Coexact enrichment | 40/40 valid; sign test p < 10⁻¹² | Interface concentrates non-gradient interaction intensity |
| **Spectral** | Energy-matched normalized Zeta | 2/43 significant; sign test p = 1.0 | No interface-specific spectral organization after energy control |
| **KTS dynamics** | IMMUNE_ACTIVE → IMMUNE_EXHAUSTED | Median bias 5.68; 16/28 significant | Exhaustion emerges from immune activation dynamics |
| **KTS dynamics** | STROMA → IMMUNE_EXHAUSTED | Median bias 2.41; 17/29 significant | Stromal context contributes to exhaustion bias |
| **KTS dynamics** | TUMOR → IMMUNE_EXHAUSTED | Median bias 0.42; not enriched | Exhaustion is not directly tumor-driven in TNBC |
| **Geometric curl** | Graph-curl proxy | Not used as cohort phenotype | Reconstructed TNBC geometry does not support a corrected curl cohort claim |

---

## Region-Label Mapping (Legacy to Modern Schema)

Legacy TNBC region labels are mapped to the modern validation schema as follows:

| Legacy label | Modern label |
|---|---|
| `interface_like` | `interface` |
| `tumor_enriched` | `tumor_core` |
| `stroma_enriched` | `stroma` |
| `immune_enriched` | `immune` |
| `other` | `other` |

Edge columns: `tail → i`, `head → j`, `flux_coexact → flux_coexact`
Node columns: `x_fullres → x`, `y_fullres → y`, `region_step2 → region`, `node_energy_coexact → coexact_energy`

This mapping enables corrected Step 10 Zeta and Step 13b KTS analysis using the same scripts as the GSE278936 validation pipeline.

---

## Pipeline Architecture

| Layer | Component | Role |
|---|---|---|
| 1 | Wedge flux + Hodge decomposition | Geometric probe: exact / coexact / harmonic components |
| 2 | Region enrichment, interface vs. tumor | Spatial localization: statistical validation |
| 3 | PDE-constrained Hodge–Laplacian GNN | Operator-class falsification: null model collapse |
| 3 | Lie algebra + discrete advection | Mechanistic interpretation: antisymmetric dynamics |
| 4 | Corrected Zeta energy-matched null | Spectral test: global vs. interface-specific organization |
| 5 | KTS transition-bias analysis | Dynamical test: functional transition directionality |

---

## Repository Structure

```
.
├── scripts_tnbc/           Legacy TNBC pipeline (Steps 1–23)
├── scripts_gse278936/      Modern validation pipeline (Steps 01–14, 27)
│   ├── step10_zeta_energy_matched_null.py
│   ├── step11_kts_state_assignment.py
│   ├── step12_kts_transition_matrix.py
│   ├── step13_kts_attractor_analysis.py
│   ├── step13b_kts_transition_bias.py
│   └── step14_ks_operator.py
├── results_tnbc_rebuild/   Modern-schema outputs from rebuilt TNBC analysis
│   ├── cohort_zeta_energy_matched_null.csv
│   ├── kts_transition_bias_summary.csv
│   └── kts_transition_bias_grouped_summary.csv
├── data/TNBC_GSE210616/
├── stats/CSV_GSM/          Per-sample CSVs (legacy pipeline)
└── visium_figures/
```

---

## Running the Legacy Pipeline

```bash
SAMPLE=GSM_xxxx
SAMPLE_DIR=data/TNBC_GSE210616/${SAMPLE}

python -m scripts_tnbc.step1_tnbc_map --sample_dir ${SAMPLE_DIR}
python -m scripts_tnbc.step2_tnbc_regions \
  --sample_id ${SAMPLE} --sample_dir ${SAMPLE_DIR} \
  --statsdir stats/CSV_GSM --figdir visium_figures
python -m scripts_tnbc.step3_tnbc_spatial_graph \
  --sample_id ${SAMPLE} --sample_dir ${SAMPLE_DIR} --statsdir stats/CSV_GSM
python -m scripts_tnbc.step4_tnbc_flux_proxies --sample-id ${SAMPLE}
python -m scripts_tnbc.step6_tnbc_hodge_decomposition \
  --sample-id ${SAMPLE} --flux-col flux_tumor_immune_region_interface_weighted
python -m scripts_tnbc.step7_tnbc_region_enrichment \
  --sample-id ${SAMPLE} --flux-col flux_tumor_immune_region_interface_weighted \
  --min-nodes-per-region 10
```

---

## TNBC Legacy-to-Modern Schema Conversion

Legacy TNBC outputs were converted into the modern validation schema to enable
corrected Step 10 Zeta and Step 13b KTS analysis:

```bash
python scripts_tnbc_rebuild/convert_tnbc_legacy_to_modern.py \
  --statsdir stats/CSV_GSM \
  --outdir results_tnbc_rebuild \
  --sample-ids $(cat valid_sample_ids.txt | tr '\n' ',')
```

This script maps column names and region labels as documented above.

---

## Corrected Zeta and KTS Commands (TNBC Rebuild)

```bash
# Corrected TNBC normalized Zeta energy-matched null
python scripts_gse278936/step10_zeta_energy_matched_null.py \
  --statsdir results_tnbc_rebuild \
  --out results_tnbc_rebuild/cohort_zeta_energy_matched_null.csv \
  --n-perm 300 --k-eigs 50
# Result: 2/43 significant; sign test p = 1.0

# TNBC KTS transition-bias analysis
python scripts_gse278936/step11_kts_state_assignment.py \
  --statsdir results_tnbc_rebuild --outdir results_tnbc_rebuild

python scripts_gse278936/step12_kts_transition_matrix.py \
  --statsdir results_tnbc_rebuild

python scripts_gse278936/step13b_kts_transition_bias.py \
  --statsdir results_tnbc_rebuild \
  --out results_tnbc_rebuild/kts_transition_bias_summary.csv \
  --n-perm 300 --seed 123
# Result: See TNBC KTS Results below
```

---

## TNBC KTS Results

Corrected TNBC KTS analysis revealed **pathway-specific exhaustion bias** rather than a universal attractor:

| Transition | Median bias | Significant |
|---|---|---|
| IMMUNE_ACTIVE → IMMUNE_EXHAUSTED | 5.68× | 16/28 |
| STROMA → IMMUNE_EXHAUSTED | 2.41× | 17/29 |
| TUMOR → IMMUNE_EXHAUSTED | 0.42× | not enriched |

Exhaustion emerges preferentially through immune-active and stromal pathways. Direct tumor-to-exhaustion transitions are not enriched, indicating exhaustion is not simply tumor-driven in TNBC.

---

## Step 20 — NCG Commutator Grounding (Legacy Diagnostic)

Step 20 is retained as a legacy operator-grounding diagnostic, but it is not used as the primary biological interpretation layer in the final manuscript. The current final framework prioritizes coexact enrichment, corrected Zeta nulls, and KTS transition-bias analysis. NCG results remain available for supplementary reference.

---

## Step 21 — Zeta Diagnostic (Legacy)

The legacy Step 21 normalized spectral diagnostic did not implement the final energy-matched null. Under the corrected null applied in the rebuilt schema, only 2/43 TNBC sections showed significant enrichment (sign test p = 1.0). TNBC does not support interface-specific spectral organization after energy control. Legacy Step 21 results should not be cited as evidence of interface spectral enrichment.

---

## Graph-Curl Warning

Graph-curl proxy outputs from the rebuilt TNBC schema should **not** be reported as a valid cohort-level geometric result. The TNBC rebuild was sufficient for spectral and KTS validation, but reconstructed geometry does not support a corrected graph-curl phenotype. Curl-related claims should remain restricted to representative legacy visualization, not cohort-level inference.

---

## Exclusion Criterion

Samples excluded from enrichment analysis (Step 7) when interface_like or tumor_enriched contains fewer than 10 nodes. Pre-specified prior to examining enrichment statistics. Excluded samples retained for global Hodge statistics (Step 6).

---

## Legacy / Obsolete Claims

The following are marked **legacy-only** and should not be used as primary results in the final manuscript:

- Step 20 NCG as primary biological result → now a supporting diagnostic
- Step 21 Zeta as evidence of interface spectral organization → corrected null shows 2/43
- Curl hotspot enrichment as a cohort-level phenotype → no cohort-level curl result available for TNBC
- Bivector/curl structure as confirmed TNBC cohort finding → not supported by corrected analysis

---

## Expected Outputs

| Step | Key outputs |
|---|---|
| 1 | `*_step1_marker_scores.csv`, marker maps |
| 2 | `*_step2_region_assignments.csv` |
| 3 | `*_step3_nodes.csv`, `*_edges.csv`, `*_faces.csv`, `*_B1.npz`, `*_B2.npz` |
| 4 | `*_step4_edge_fluxes.csv` |
| 6 | `*_edges_hodge_*.csv`, `*_nodes_hodge_*.csv`, `*_energy_summary_*.csv` |
| 7 | `*_step7_region_enrichment_*.csv` |
| Rebuild | `results_tnbc_rebuild/cohort_zeta_energy_matched_null.csv` |
| Rebuild | `results_tnbc_rebuild/kts_transition_bias_summary.csv` |
| Rebuild | `results_tnbc_rebuild/kts_transition_bias_grouped_summary.csv` |

---

## Reproducibility

Recommended files to track in version control:

```
scripts_tnbc/step6_tnbc_hodge_decomposition.py
scripts_tnbc/step7_tnbc_region_enrichment.py
scripts_tnbc/step18_ablation_no_constraint.py
scripts_tnbc/step19_coexact_bio_correlation.py
scripts_gse278936/step10_zeta_energy_matched_null.py
scripts_gse278936/step13b_kts_transition_bias.py
scripts_tnbc_rebuild/convert_tnbc_legacy_to_modern.py
results_tnbc_rebuild/cohort_zeta_energy_matched_null.csv
results_tnbc_rebuild/kts_transition_bias_summary.csv
results_tnbc_rebuild/kts_transition_bias_grouped_summary.csv
```

---

## Common Issues

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError` | Missing `.h5`, positions, image, or JSON | Verify all four files present |
| Step 7 `low_sample_size` | Interface or tumor region < 10 nodes | Sample excluded from enrichment; global stats valid |
| `harmonic_suppressed = True` | E_harmonic / E_coexact < 10⁻⁶ | Correct — harmonic p-value set to NaN |
| KTS `state_absent` | Transition pair not present in section | Logged, not counted toward significant |

---

## Citation

> Anas Enoch. *Non-passive transport organization at tumor–immune interfaces revealed by operator-based analysis.* Mohammed VI University of Health Sciences (UM6SS), 2026.

Dataset: Bassiouni R et al. *Spatial Transcriptomic Analysis of a Diverse Patient Cohort Reveals a Conserved Architecture in Triple-Negative Breast Cancer.* Cancer Research 83(1):34–48, 2023. GEO: GSE210616.
