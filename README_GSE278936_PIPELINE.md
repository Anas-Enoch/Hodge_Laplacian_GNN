# External Validation Cohort — Operator-Based Geometry of Tumor–Immune Interfaces
## Dataset: GSE278936 (Visium, n = 23)

**Associated manuscript:** *Non-passive transport organization at tumor–immune interfaces revealed by operator-based analysis*
**Primary cohort:** GSE210616 (TNBC, n = 22 patients, 43 Visium sections)
**This pipeline:** Independent validation on GSE278936 (prostate cancer progression
cohort; benign / treatment-naïve / neoadjuvant / CRPC)

---

## 1. Purpose and Honest Scope

This pipeline applies the operator-level Hodge decomposition framework to an
independent Visium cohort. It tests two claims established in the primary TNBC
analysis and reports a null result on a third:

| Scale | Claim | Test | Outcome |
|---|---|---|---|
| Local | Coexact energy enriched at tumor–immune interfaces | Permutation enrichment ratio R | **Confirmed: 21/23** |
| Geometric | Graph-curl proxy: coexact component interface localization | Graph-curl fold-change | **Not confirmed: median fold ≈ 1.08** |
| Spectral | Interface coexact signal has independent spectral organization | Energy-matched Zeta null | **Not confirmed: 2/23** |
| Dynamical | KTS transition bias toward IMMUNE_EXHAUSTED | Permutation null | **Confirmed: 21–23/23 per transition** |
| Instability | Graph-KS proxy interface/tumor magnitude | Fold-change | **Exploratory positive: median ≈ 8.96** |

The spectral result and its full trajectory are documented in §6. The KTS and KS results are documented in §13b and §14.

### What this pipeline does not do

- GNN falsification, NCG commutator grounding, and biomarker validation belong
  to the primary TNBC analysis (GSE210616, Steps 1–23) and are not reproduced here.
- No survival or outcome association is attempted.
- Numerical Zeta values (truncated k=50 approximation) are not directly comparable
  to the full-spectrum normalized Z(s=1) from GSE210616.

---

## 2. Mathematical Framework

### 2.1 Antisymmetric Interaction Field

For a spatial edge (u, v):

```
F_AB(u,v) = A(u)·B(v) − A(v)·B(u)
```

where A = tumor score, B = immune score (log-normalized, z-scored per section).

This is a **1-cochain** on oriented edges — an antisymmetric bilinear form with
bivector-like algebraic origin. The cochain degree is 1, not 2. The grade-2
geometric object arises only after applying B₂ᵀ to the coexact component (§2.3).

### 2.2 Hodge Decomposition

```
F = dα + δβ + γ
```

- `dα` — exact: reducible to a scalar potential φ (gradient flow)
- `δβ` — coexact: non-gradient, non-exact component. A rotational/curl interpretation requires explicit graph-curl or face-level construction; it is not implied by coexact decomposition alone
- `γ`  — harmonic: zero on simply connected planar graphs

Solved via:

```
L₀ φ = B f
f_exact   = Bᵀ φ
f_coexact = f − f_exact
```

Regularization ε = 1e-6 on L₀. Solver: `scipy.sparse.linalg.spsolve`.

### 2.3 Graph-Curl Proxy

Because the GSE278936 validation pipeline is graph-based and does not use a full face complex for cohort-level inference, rotational structure is assessed using a **graph-curl proxy** derived from signed coexact edge-flux imbalance:

```
curl_proxy_i = (1/deg_i) Σ_{j~i} flux_coexact(i,j)
```

This is **not equivalent to exact DEC face-level circulation** and must not be interpreted as confirming rotational organization at cohort scale.

**Cohort result:** Median fold interface/tumor_core ≈ 1.08; weak/inconsistent across 23 sections. No cohort-level rotational phenotype.

### 2.4 Normalized Zeta Spectral Statistic

```
Z = [Σ_k α_k λ_k⁻¹] / [Σ_k α_k]
```

where α_k = ⟨signal, φ_k⟩². Normalization by Σ_k α_k removes scale dependence
on total signal magnitude. Truncated to k=50 smallest nonzero eigenmodes.

### 2.5 Energy-Matched Null (Step 10)

The correct null for the spectral independence claim. Random subsets are drawn
from nodes whose coexact energy falls within the IQR of the actual interface nodes:

```
q_low, q_high = percentile(signal[interface], [25, 75])
candidates    = nodes where q_low ≤ signal ≤ q_high
perm_mask     = random sample of n_interface nodes from candidates
```

This isolates the question: does the interface have spectral structure *beyond*
its elevated energy? If not rejected, spectral enrichment is fully explained by
energy, not by the spatial topology of the interface.

---

## 3. Circularity Boundary

No gene from these sets may appear in any downstream validation or enrichment step:

```python
TUMOR_GENES  = ["EPCAM", "KRT8", "KRT18", "KRT19", "ERBB2", "MUC1", "TACSTD2"]
IMMUNE_GENES = ["PTPRC", "CD3D", "CD3E", "NKG7", "CD68", "C1QA", "CXCL9", "CXCL10"]
STROMA_GENES = ["COL1A1", "COL1A2", "DCN", "LUM", "POSTN", "FAP", "TAGLN"]
```

Identical to GSE210616 main manuscript panels. ACTA2 and VIM (earlier draft)
removed; POSTN and FAP added.

---

## 4. Dataset

| Field | Value |
|---|---|
| GEO Accession | GSE278936 |
| Platform | 10x Genomics Visium |
| n_samples | 23 |
| Conditions | benign, treatment-naïve, neoadjuvant, CRPC |
| Graph construction | kNN, k=6, pixel coordinates |
| Triangle mesh | Delaunay triangulation on spot centroids |

Raw data: download from GEO into `data/GSE278936/{GSM_ID}/`. Each directory
requires: `*_barcodes.tsv.gz`, `*_features.tsv.gz`, `*_matrix.mtx.gz`,
`*_tissue_positions_list.csv`.

---

## 5. Pipeline Steps

### Step 01 — Marker Scoring

**Script:** `step01_visium_build_marker_scores.py`
**Output:** `Results_TNBC_rebuild_gse278936/{sample}_spots.csv`

```python
x = log1p(expression[genes])
x = (x − mean(x)) / (std(x) + 1e-8)
score = mean(x, axis=genes)
```

```bash
python step01_visium_build_marker_scores.py \
    --sample_dir data/GSE278936/GSM_ID \
    --sample_id  GSM_ID \
    --out_dir    Results_TNBC_rebuild_gse278936
```

---

### Step 02 — Region Annotation

**Script:** `step02_visium_regions.py`
**Output:** `*_spots_regions.csv`

Neighborhood-based: a spot is `interface` if it has neighbors of both tumor and
immune type, regardless of its own score.

| Region | Criterion |
|---|---|
| `tumor_core` | score ≥ Q75, no immune neighbor |
| `immune_core` | score ≥ Q75, no tumor neighbor |
| `interface` | has both tumor and immune neighbors |
| `other` | none of the above |

Parameters: K=6, TUMOR_Q=IMMUNE_Q=0.75.

```bash
python step02_visium_regions.py \
    --input_csv  Results_TNBC_rebuild_gse278936/GSM_ID_spots.csv \
    --output_csv Results_TNBC_rebuild_gse278936/GSM_ID_spots_regions.csv
```

---

### Step 03 — Wedge Flux

**Script:** `step03_visium_wedge_flux.py`
**Output:** `*_edges_wedge.csv`

```
F_ij = tumor_i · immune_j − tumor_j · immune_i
```

```bash
python step03_visium_wedge_flux.py \
    --input_csv  Results_TNBC_rebuild_gse278936/GSM_ID_spots_regions.csv \
    --output_csv Results_TNBC_rebuild_gse278936/GSM_ID_edges_wedge.csv
```

---

### Step 04 — Hodge Decomposition

**Script:** `step04_visium_hodge_decomposition.py`
**Output:** `*_edges_hodge.csv` (columns: `i, j, flux_wedge, flux_exact, flux_coexact`)

Verify: `‖f_exact‖² + ‖f_coexact‖²` ≈ `‖f‖²`. Large discrepancy indicates
harmonic content from disconnected components.

```bash
python step04_visium_hodge_decomposition.py \
    --edges_csv  Results_TNBC_rebuild_gse278936/GSM_ID_edges_wedge.csv \
    --spots_csv  Results_TNBC_rebuild_gse278936/GSM_ID_spots_regions.csv \
    --output_csv Results_TNBC_rebuild_gse278936/GSM_ID_edges_hodge.csv
```

---

### Step 05 — Coexact Energy

**Script:** `step05_visium_coexact_energy.py`
**Output:** `*_spots_coexact_energy.csv`

```
E_i = (1/deg_i) · Σ_{j~i} f_coexact(i,j)²
```

```bash
python step05_visium_coexact_energy.py \
    --edges_csv  Results_TNBC_rebuild_gse278936/GSM_ID_edges_hodge.csv \
    --spots_csv  Results_TNBC_rebuild_gse278936/GSM_ID_spots_regions.csv \
    --output_csv Results_TNBC_rebuild_gse278936/GSM_ID_spots_coexact_energy.csv
```

---

### Step 06 — Interface Enrichment Test

**Script:** `step06_visium_interface_enrichment.py`
**Output:** `*_enrichment.csv`

```
R = mean(E_coexact | interface) / mean(E_coexact | tumor_core)
```

Null: 1000 region-label permutations. One-sided p-value.

**Cohort result (n=23):**

| Metric | Value |
|---|---|
| Median R | 4.94 |
| R > 1 | 21/23 |
| Sign test p | < 10⁻⁵ |
| Individually significant (p < 0.05) | 21/23 |

GSM8557977 (R=1.25, p=0.41) and GSM8558019 (R=1.19, p=0.47) not significant;
both retained in all cohort aggregates.

```bash
python step06_visium_interface_enrichment.py \
    --input_csv  Results_TNBC_rebuild_gse278936/GSM_ID_spots_coexact_energy.csv \
    --output_csv Results_TNBC_rebuild_gse278936/GSM_ID_enrichment.csv
```

---

### Step 07 — Cohort Enrichment Summary

**Script:** `step07_visium_cohort_summary.py`
**Output:** `cohort_summary.csv`

```bash
python step07_visium_cohort_summary.py \
    --input_dir  Results_TNBC_rebuild_gse278936 \
    --output_csv Results_TNBC_rebuild_gse278936/cohort_summary.csv
```

---

### Step 08 — Interface-Restricted Zeta (Diagnostic)

**Script:** `step08_visium_zeta_interface.py`
**Output:** `*_zeta_interface.csv`

Computes normalized Z_global and Z_interface. Output includes `normalization`
and `truncation_note` columns making the approximation explicit in the record.

**Do not report Z_density from this step as a significance claim.** The
size-corrected null (Step 09) does not control for energy. Use Step 10.

```bash
python step08_visium_zeta_interface.py \
    --spots_csv  Results_TNBC_rebuild_gse278936/GSM_ID_spots_coexact_energy.csv \
    --edges_csv  Results_TNBC_rebuild_gse278936/GSM_ID_edges_hodge.csv \
    --output_csv Results_TNBC_rebuild_gse278936/GSM_ID_zeta_interface.csv \
    --k-eigs 50
```

---

### Step 09 — Size-Corrected Null (Diagnostic Only — Confounded)

**Script:** `step09_zeta_interface_null_and_correlation.py`
**Output:** `cohort_zeta_interface_null_summary.csv`

**This test is confounded and should not be used for inference.**

Compares the interface signal against random masks of equal size drawn from all
nodes. Does not control for the energy difference between interface and
non-interface nodes. Retained for documentation and reproducibility; results
should not be reported as spectral evidence in any form.

---

### Step 10 — Energy-Matched Null (Primary Spectral Test)

**Script:** `step10_zeta_energy_matched_null.py`
**Output:** `cohort_zeta_energy_matched_null.csv`

**Cohort result (n=23):**

| Metric | Value |
|---|---|
| Interface > energy-matched null median | 2/23 |
| p_energy_matched < 0.05 | 2/23 |
| Sign test p | 1.000 |

Energy-matched random regions are more spectrally concentrated than the interface
in 21/23 sections. The spectral organization of the coexact signal is a global
property of the tissue graph — any set of high-energy nodes exhibits comparable
spectral concentration. The interface is not spectrally privileged beyond being
a region of elevated coexact energy.

```bash
python step10_zeta_energy_matched_null.py \
    --statsdir Results_TNBC_rebuild_gse278936 \
    --out      Results_TNBC_rebuild_gse278936/cohort_zeta_energy_matched_null.csv \
    --n-perm   1000 \
    --k-eigs   50 \
    --seed     123 \
    --low-q    25 \
    --high-q   75
```

---

### Step 11 — KTS State Assignment
**Script:** `step11_kts_state_assignment.py`
Assigns each spot to TUMOR, IMMUNE_ACTIVE, IMMUNE_EXHAUSTED, STROMA, or MIXED.
Output: `{sid}_kts_states.csv`

### Step 12 — KTS Transition Matrix
**Script:** `step12_kts_transition_matrix.py`
Coexact-flux-weighted transition matrix. Output: `{sid}_kts_transition_matrix.csv`

### Step 13 — KTS Entropy
**Script:** `step13_kts_attractor_analysis.py`
Output: `kts_summary.csv`
**Result: Mean entropy H = 6.75 ± 0.29 nats (6.01–7.26). System globally disordered.**

### Step 13b — KTS Transition Bias *(PRIMARY DYNAMICAL TEST)*
**Script:** `step13b_kts_transition_bias.py`
Seven pre-specified transition pairs; 300 permutations, seed 123.
Outputs: `kts_transition_bias_summary.csv`, `kts_transition_bias_grouped_summary.csv`

| Source | Target | Median bias | Significant |
|---|---|---|---|
| IMMUNE_ACTIVE | IMMUNE_EXHAUSTED | 3.63× | 22/23 |
| STROMA | IMMUNE_EXHAUSTED | 2.96× | 23/23 |
| TUMOR | IMMUNE_EXHAUSTED | 2.64× | 21/23 |
| MIXED | IMMUNE_EXHAUSTED | 1.91× | 21/23 |
| MIXED→TUMOR, TUMOR→STROMA, MIXED→STROMA | — | ≤1.10× | ≤7/23 (null) |

### Step 14 — Exploratory Graph-KS Instability
**Script:** `step14_ks_operator.py`
KS(u) = −Lu − L²u − |∇u|²; operator analogy only.
Output: `cohort_ks_operator_summary.csv`
**Result: Median fold ≈ 8.96, mean ≈ 10.15 (23/23 fold > 1). Exploratory — does not claim the system solves the KS PDE.**


### Step 27 — Face-Level Bivector Density

**Script:** `step27_face_bivector_orientation.py`
**Output:** `{sample}_step27_face_bivector_orientation.csv`,
           `step27_face_bivector_orientation_summary.csv`

**Result (4 sections tested):**

| Metric | Value |
|---|---|
| Interface abs_bivector_density fold-change | 4–9× over next-highest region |
| Signed mean at interface | ≈ 0 in all 4 sections |
| frac_positive | 0.50 ± 0.04 |
| orientation_bias | `balanced_orientation` in all 4 sections |

The interface has high absolute circulation magnitude with balanced CW/CCW
orientation — locally antisymmetric interaction structure that cancels globally.
This is the expected signature of non-integrability at multicellular Visium
resolution: the field resists reduction to a gradient but imposes no preferred
direction at the tissue scale.

**Pending:** Region-label permutation null for the full 23-section cohort required
before this enters the manuscript as a cohort-level confirmed finding.

```bash
python step27_face_bivector_orientation.py \
    --sample-ids GSM_6433618,GSM_6433619,GSM_6433601,GSM_6433624 \
    --statsdir   stats/CSV_GSM \
    --outdir     stats/CSV_GSM
```

---

## 6. The Spectral Null Result: Full Trajectory

This section documents the complete history of the spectral analysis so that
the confound cannot be reintroduced in future work.

### Version history

**v1 — unnormalized Z, size-corrected null:** 21/23 significant.
Z_old = Σ αk/λk scales with total coexact energy. The interface has 3–15×
higher energy (Step 06). The null detected this energy difference, not spectral
geometry. Result was an artifact of the unnormalized formulation.

**v2 — normalized Z, size-corrected null:** 2/23 significant.
After fixing Z = [Σ αk/λk] / [Σ αk], the signal collapsed. Normalization
removed most of the energy confound, but the null still drew masks from all
nodes, which have lower mean energy than interface nodes. Partially controlled,
not fully.

**v3 — normalized Z, energy-matched null (Step 10):** 2/23 significant.
Energy-matched random regions outperform the interface spectrally in 21/23
cases. The null is definitive: there is no residual interface-specific spectral
structure after energy is controlled.

### Mechanistic explanation

In a regular kNN graph (k=6), the spectral concentration of any node signal
is determined primarily by total energy and spatial smoothness — not by which
region the nodes occupy. High-energy nodes project similarly onto low-frequency
eigenmodes regardless of their tissue location. The interface's apparent spectral
enrichment in v1 and v2 was a consequence of its being a high-energy region,
not a consequence of its boundary topology.

This is consistent with the spectral collapse finding in the TNBC pipeline
(Step 21b): Z'(1) and Z(1) are collinear at ρ=0.983 across sections, indicating
that the kNN graph regime induces a low-dimensional spectral manifold where
higher-order Zeta statistics carry no additional stratificatory information.

### Open question for the manuscript

The GSE210616 Zeta uses a spatial permutation null (shuffles spatial arrangement,
preserves energy). Whether that result would survive an energy-matched null is
not yet tested. This should be acknowledged as a limitation: the primary cohort
spectral claim rests on the spatial permutation null, which controls spatial
structure but not the energy-spectral confound identified here. This does not
invalidate the primary result — but the comparison deserves a caveat.

---

## 7. Batch Execution

```bash
OUTDIR=Results_TNBC_rebuild_gse278936
mkdir -p $OUTDIR

for SID in $(ls data/GSE278936/); do
    python step01_visium_build_marker_scores.py \
        --sample_dir data/GSE278936/$SID --sample_id $SID --out_dir $OUTDIR

    python step02_visium_regions.py \
        --input_csv $OUTDIR/${SID}_spots.csv \
        --output_csv $OUTDIR/${SID}_spots_regions.csv

    python step03_visium_wedge_flux.py \
        --input_csv $OUTDIR/${SID}_spots_regions.csv \
        --output_csv $OUTDIR/${SID}_edges_wedge.csv

    python step04_visium_hodge_decomposition.py \
        --edges_csv $OUTDIR/${SID}_edges_wedge.csv \
        --spots_csv $OUTDIR/${SID}_spots_regions.csv \
        --output_csv $OUTDIR/${SID}_edges_hodge.csv

    python step05_visium_coexact_energy.py \
        --edges_csv $OUTDIR/${SID}_edges_hodge.csv \
        --spots_csv $OUTDIR/${SID}_spots_regions.csv \
        --output_csv $OUTDIR/${SID}_spots_coexact_energy.csv

    python step06_visium_interface_enrichment.py \
        --input_csv $OUTDIR/${SID}_spots_coexact_energy.csv \
        --output_csv $OUTDIR/${SID}_enrichment.csv

    python step08_visium_zeta_interface.py \
        --spots_csv $OUTDIR/${SID}_spots_coexact_energy.csv \
        --edges_csv $OUTDIR/${SID}_edges_hodge.csv \
        --output_csv $OUTDIR/${SID}_zeta_interface.csv --k-eigs 50
done

python step07_visium_cohort_summary.py \
    --input_dir $OUTDIR --output_csv $OUTDIR/cohort_summary.csv

python step09_zeta_interface_null_and_correlation.py \
    --dir $OUTDIR \
    --out $OUTDIR/cohort_zeta_interface_null_summary.csv \
    --n-perm 1000 --k-eigs 50 --seed 123

python step10_zeta_energy_matched_null.py \
    --statsdir $OUTDIR \
    --out $OUTDIR/cohort_zeta_energy_matched_null.csv \
    --n-perm 1000 --k-eigs 50 --seed 123 --low-q 25 --high-q 75
```

---

## 8. Expected Outputs

| File | Content | Role |
|---|---|---|
| `{sid}_spots.csv` | Program scores | Required |
| `{sid}_spots_regions.csv` | + region labels | Required |
| `{sid}_edges_wedge.csv` | Edge-level wedge flux | Required |
| `{sid}_edges_hodge.csv` | + exact/coexact | Required |
| `{sid}_spots_coexact_energy.csv` | Node coexact energy | Required |
| `{sid}_enrichment.csv` | R, p-value | Required |
| `{sid}_zeta_interface.csv` | Z_global, Z_interface, alpha_sum | Diagnostic |
| `cohort_summary.csv` | Cohort enrichment | Required |
| `cohort_zeta_interface_null_summary.csv` | Size-corrected null | Diagnostic only — confounded |
| `cohort_zeta_energy_matched_null.csv` | Energy-matched null | Required — primary spectral test |
| `{sid}_step27_face_bivector_orientation.csv` | Triangle bivector density | Required |
| `step27_face_bivector_orientation_summary.csv` | Signed orientation summary | Required |

---

## 9. Reproducibility Parameters

| Parameter | Value | Script |
|---|---|---|
| kNN k | 6 | step02, step03 |
| Region quantile | 0.75 | step02 |
| Laplacian regularization ε | 1e-6 | step04 |
| Energy normalization | degree | step05 |
| Enrichment permutations | 1000 | step06 |
| Zeta eigenmodes k | 50 | step08–10 |
| Zeta normalization | alpha_sum | step08–10 |
| Energy-matched quantile window | IQR [25, 75]; widen to [10, 90] if insufficient | step10 |
| Null permutations | 1000 | step09, step10 |
| Random seed | 123 + sum(ord(c) for c in sample_id) | step10 |
| Stroma genes | COL1A1, COL1A2, DCN, LUM, POSTN, FAP, TAGLN | step01 |

**Reproducibility assertion:**

```python
import pandas as pd

enrich   = pd.read_csv("Results_TNBC_rebuild_gse278936/cohort_summary.csv")
spectral = pd.read_csv("Results_TNBC_rebuild_gse278936/cohort_zeta_energy_matched_null.csv")
ok       = spectral[spectral["status"] == "ok"]

assert (enrich["R"] > 1).sum() >= 20,                      "Local: R > 1 in fewer than 20/23"
assert (enrich["p_value"] < 0.05).sum() >= 20,             "Local: sig in fewer than 20/23"
assert (ok["p_energy_matched"] < 0.05).sum() <= 3,         "Spectral: unexpectedly many significant"

print("Reproducibility check passed.")
print(f"  Local enrichment R > 1:         {(enrich['R'] > 1).sum()}/23")
print(f"  Spectral energy-matched p<0.05: {(ok['p_energy_matched'] < 0.05).sum()}/23")
```

---

## 10. Confirmed Results Summary

**Local (Step 06) — confirmed, 21/23 sections.**
Coexact energy is 3–15× enriched at tumor–immune interfaces (median R = 4.94,
sign test p < 10⁻⁵). The non-gradient antisymmetric interaction component
concentrates at the tumor–immune boundary across conditions.

**Geometric (Step 27) — confirmed descriptively, 4/4 sections tested.**
Face-level bivector density is 4–9× higher at interfaces than any other region.
Signed circulation is balanced (mean ≈ 0, frac_positive ≈ 0.50) — locally
rotational, globally symmetric interaction structure. Full cohort confirmation
(region-label permutation null, all 23 sections) is pending.

**Spectral (Steps 08–10) — not confirmed.**
After energy matching, interface regions show no spectral organization beyond
that expected from any set of nodes with equivalent coexact energy levels
(2/23 significant, sign test p = 1.000). Spectral organization is a global
property of the coexact field in the kNN Visium graph regime, not an
interface-specific structural feature. Reported as a null result.

---

## 11. Differences from Primary TNBC Pipeline (GSE210616)

| Aspect | GSE210616 | GSE278936 |
|---|---|---|
| Zeta | Normalized, full eigendecomposition | Normalized, k=50 truncation |
| Spectral null | Spatial permutation (energy preserved) | Energy-matched (Step 10) |
| Spectral result | 18/19 significant | 2/23 significant |
| Steps | 1–23 | 01–10, 27 |
| Stroma genes | COL1A1/2, DCN, LUM, POSTN, FAP, TAGLN | Identical |
| Output dir | stats/CSV_GSM | Results_TNBC_rebuild_gse278936 |

The GSE210616 Zeta uses a spatial permutation null, which preserves energy
but shuffles spatial arrangement. Whether the GSE210616 spectral result would
survive an energy-matched null has not been tested and should be acknowledged
as a limitation in the manuscript.

---

## 12. Author

Anas Enoch
Mohammed VI University of Health Sciences (UM6SS), Casablanca
