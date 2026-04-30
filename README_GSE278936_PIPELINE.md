# External Validation Cohort — GSE278936 (Prostate Cancer, Visium, n = 23)
## Operator-Based Multi-Scale Characterization of Tumor–Immune Interfaces

**Associated manuscript:** *Non-passive transport organization at tumor–immune interfaces revealed by operator-based analysis*
**Primary cohort:** GSE210616 (TNBC, n = 22 patients, 43 Visium sections)
**This pipeline:** Independent validation on GSE278936 (prostate cancer; benign / treatment-naïve / neoadjuvant / CRPC)

---

## Final Interpretation

Tumor–immune interfaces in GSE278936 are **high-intensity non-gradient interaction zones** without consistent rotational or interface-specific spectral organization, but with strong transition-level bias toward exhaustion-like immune states and elevated instability-like graph-KS activity. This validates the central multilayer conclusion of the primary TNBC analysis.

---

## 1. Purpose and Scope

| Layer | Diagnostic | Test | Outcome |
|---|---|---|---|
| **Local** | Coexact enrichment | Permutation enrichment ratio R | **Confirmed: 21/23** |
| **Geometric** | Graph-curl proxy | Interface/tumor fold-change | **Not confirmed: median fold ≈ 1.08** |
| **Spectral** | Energy-matched Zeta null | Z_density vs energy-matched null | **Not confirmed: 2/23** |
| **Dynamical** | KTS transition bias | Bias toward IMMUNE_EXHAUSTED | **Confirmed: 21–23/23 per transition** |
| **Instability** | Graph-KS proxy | Interface/tumor KS-magnitude fold | **Exploratory positive: median ≈ 8.96** |

GNN falsification, NCG commutator grounding, and biomarker validation belong to the primary TNBC analysis and are not reproduced here. Graph-curl uses a signed flux-imbalance proxy — not true DEC face-level circulation. Zeta values (k=50 truncation) are not numerically comparable to full-spectrum Z(s=1) from GSE210616.

---

## 2. Mathematical Framework

### 2.1 Antisymmetric Interaction Field

```
F_AB(u,v) = A(u)·B(v) − A(v)·B(u)
```

A 1-cochain on oriented edges. Antisymmetric bilinear form with algebraic bivector-like origin. Cochain degree is 1, not 2.

### 2.2 Hodge Decomposition

```
F = dα + δβ + γ
```

- `dα` — exact: gradient component, reducible to a scalar potential
- `δβ` — coexact: non-gradient, non-exact component. **A rotational/curl interpretation requires explicit graph-curl or face-level construction; it is not implied by coexact decomposition alone.**
- `γ` — harmonic: zero on simply connected graphs

Solved via L₀φ = Bf (regularization ε = 1e-6); f_exact = Bᵀφ; f_coexact = f − f_exact.

### 2.3 Graph-Curl Proxy

Because the GSE278936 validation pipeline is graph-based and does not use a full face complex for cohort-level inference, rotational structure is assessed using a **graph-curl proxy** derived from signed coexact edge-flux imbalance:

```
curl_proxy_i = (1/deg_i) Σ_{j~i} flux_coexact(i,j)
```

This is **not equivalent to exact DEC face-level circulation** and must not be interpreted as confirming rotational organization at cohort scale.

**Cohort result:** Median fold interface/tumor_core ≈ 1.08; weak/inconsistent across 23 sections. No cohort-level rotational phenotype.

### 2.4 Normalized Zeta Spectral Statistic

```
Z(s) = [Σ_k α_k λ_k^{-s}] / [Σ_k α_k],   α_k = ⟨E_coexact, φ_k⟩²
```

Normalized form removes dependence on total coexact energy, testing spectral geometry independently of signal magnitude. Only 2/23 sections exceeded the energy-matched null (sign test p = 1.0). k=50 truncation applies; hypothesis tests are valid (same truncation for observed and permuted).

### 2.5 Interface-Normalized Spectral Density

```
Z_density = Z_ratio / frac_interface
```

Z_density > 1 after energy-matched normalization indicates interface-specific spectral geometry beyond energy effects. **This condition was not met at cohort level (2/23).**

### 2.6 Kripke Transition System (KTS)

Spot-level biological states: TUMOR, IMMUNE_ACTIVE, IMMUNE_EXHAUSTED, STROMA, MIXED. Transitions along spatial edges weighted by |flux_coexact|. State-label permutation null preserves graph topology, edge weights, and global state frequencies. Tests whether spatially disorganized coexact interactions exhibit directional biological transition bias.

**Cohort result:** IMMUNE_EXHAUSTED is a near-universal attractor (bias 1.91–3.63×, 21–23/23 sections per transition).

### 2.7 Exploratory Graph-Kuramoto–Sivashinsky Proxy

```
KS(u) = −Lu − L²u − |∇u|²_graph
```

where L = weighted graph Laplacian (edge weights = |flux_coexact|), L²u captures higher-order stabilizing/damping structure, |∇u|²_graph = squared edge differences accumulated per node. **Operator analogy only — no claim that the biological system solves the KS PDE.**

**Cohort result:** Median fold interface/tumor_core ≈ 8.96 (mean 10.15, range 1.75–28.40); all 23 sections fold > 1. Stroma comparisons unavailable (stroma annotations absent in GSE278936).

---

## 3. Circularity Boundary

```python
TUMOR_GENES  = ["EPCAM","KRT8","KRT18","KRT19","ERBB2","MUC1","TACSTD2"]
IMMUNE_GENES = ["PTPRC","CD3D","CD3E","NKG7","CD68","C1QA","CXCL9","CXCL10"]
STROMA_GENES = ["COL1A1","COL1A2","DCN","LUM","POSTN","FAP","TAGLN"]
```

Identical to primary TNBC manuscript (GSE210616). No validation marker may appear in any of these sets.

---

## 4. Dataset

| Field | Value |
|---|---|
| GEO Accession | GSE278936 |
| URL | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE278936 |
| Platform | 10x Genomics Visium |
| n_samples | 23 |
| Conditions | benign, treatment-naïve, neoadjuvant, CRPC |
| Graph construction | kNN k=6 on pixel coordinates |

---

## 5. Pipeline Steps

### Step 01 — Marker Scoring
**Script:** `step01_visium_build_marker_scores.py`
Output: `{sid}_spots.csv`

### Step 02 — Region Annotation
**Script:** `step02_visium_regions.py`
kNN neighborhood-based (K=6, quantile=0.75). Regions: tumor_core, immune_core, interface, other.
Output: `{sid}_spots_regions.csv`

### Step 03 — Wedge Flux
**Script:** `step03_visium_wedge_flux.py`
Output: `{sid}_edges_wedge.csv`

### Step 04 — Hodge Decomposition
**Script:** `step04_visium_hodge_decomposition.py`
Output: `{sid}_edges_hodge.csv`

### Step 05 — Coexact Energy
**Script:** `step05_visium_coexact_energy.py`
Degree-normalized: E_i = (1/deg_i)·Σ f_coexact(i,j)²
Output: `{sid}_spots_coexact_energy.csv`

### Step 06 — Interface Enrichment Test
**Script:** `step06_visium_interface_enrichment.py`
R = mean(E_coexact|interface) / mean(E_coexact|tumor_core); 300-permutation null.
Output: `{sid}_enrichment.csv`

### Step 07 — Cohort Enrichment Summary
**Script:** `step07_visium_cohort_summary.py`
Output: `cohort_summary.csv`
**Result: 21/23 enriched; sign test p < 10⁻⁵**

### Step 08 — Zeta Interface (Historical)
**Script:** `step08_visium_zeta_interface.py`
Retained for comparison only. Not the primary spectral result.

### Step 09 — Null Model and Dissociation Classification
**Script:** `step09_zeta_interface_null_and_correlation.py`
Label-permutation null; dissociation_type column in output.

### Step 10 — Energy-Matched Normalized Zeta *(PRIMARY SPECTRAL TEST)*
**Script:** `step10_zeta_energy_matched_null.py`
Normalized Z(s); energy-matched permutation null.
Output: `cohort_zeta_energy_matched_null.csv`
**Result: 2/23 significant; sign test p = 1.0 — interface not spectrally special after energy control.**

### Step 11 — KTS State Assignment
**Script:** `step11_kts_state_assignment.py`
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
Output: `cohort_ks_operator_summary.csv`
**Result: Median fold ≈ 8.96, mean ≈ 10.15 (23/23 fold > 1). Exploratory — operator analogy only.**

### Step 27 — Graph-Curl Proxy
**Script:** `step27_face_bivector_orientation.py`
Signed coexact edge-flux imbalance summarized by region.
Output: `step27_graph_curl_proxy_summary.csv`
**Result: Median fold ≈ 1.08. Weak/inconsistent; no cohort-level rotational phenotype.**

---

## 6. Expected Outputs

| File | Content |
|---|---|
| `cohort_summary.csv` | Per-sample R and p-value |
| `cohort_zeta_energy_matched_null.csv` | Z_density, p_null, dissociation_type |
| `step27_graph_curl_proxy_summary.csv` | Per-sample curl proxy fold-change |
| `cohort_ks_operator_summary.csv` | Per-sample KS magnitude fold-change |
| `kts_transition_bias_summary.csv` | Per-sample × per-transition bias and p-value |
| `kts_transition_bias_grouped_summary.csv` | Cohort-level grouped transition summary |

---

## 7. Reproducibility Parameters

| Parameter | Value |
|---|---|
| kNN k | 6 |
| Region quantile threshold | 0.75 |
| Regularization ε | 1e-6 |
| Zeta eigenmodes k | 50 |
| Enrichment permutations | 300 |
| KTS permutations | 300 |
| Random seed | 123 |

Verification:

```python
import pandas as pd
df = pd.read_csv("results_gse278936/cohort_zeta_energy_matched_null.csv")
assert (df["Z_null_p"] < 0.05).sum() == 2
df2 = pd.read_csv("results_gse278936/cohort_summary.csv")
assert (df2["p_enrichment"] < 0.05).sum() >= 20
df3 = pd.read_csv("results_gse278936/kts_transition_bias_grouped_summary.csv")
assert df3[df3["target"]=="IMMUNE_EXHAUSTED"]["n_p_lt_005"].min() >= 21
print("Reproducibility checks passed.")
```

---

## 8. Differences from Primary TNBC Pipeline

Both TNBC (GSE210616) and GSE278936 were evaluated under the corrected normalized Zeta energy-matched null. Both cohorts show no interface-specific spectral enrichment after energy control:
- **TNBC:** 2/43 significant; sign test p = 1.0
- **GSE278936:** 2/23 significant; sign test p = 1.0

KTS differs by cohort:
- **GSE278936:** near-universal exhaustion attractor (21–23/23 per transition)
- **TNBC:** pathway-specific exhaustion bias through immune-active and stromal states; tumor-to-exhaustion transitions not enriched

| Aspect | GSE210616 (TNBC) | GSE278936 (Prostate) |
|---|---|---|
| Full eigendecomposition | Yes | No (k=50 truncation) |
| Steps | 1–23 (full pipeline) | 01–14, 27 |
| Graph-curl | Not used as cohort claim | Proxy; weak/inconsistent |
| GNN, NCG, biomarkers | Yes | No |

---

## 9. Author

Anas Enoch, MD
Mohammed VI University of Health Sciences (UM6SS), Casablanca
anas_nour@um5.ac.ma
