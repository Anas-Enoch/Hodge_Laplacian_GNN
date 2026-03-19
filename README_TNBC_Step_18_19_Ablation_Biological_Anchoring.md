##Ablation Study: Effect of Conservation Constraint

A critical component of this framework is the conservation-constrained learning formulation, which enforces gradient-driven (passive) transport dynamics. In the main manuscript, we observe a near-complete collapse of the coexact (rotational) component under this constraint.

A natural concern is whether this collapse arises from:
	•	architectural smoothing effects of the graph neural network, or
	•	the imposed conservation constraint itself.

# Diagnostic experiment

To resolve this ambiguity, we performed an ablation experiment in which the exact same architecture and training procedure were used, but the conservation constraint was removed.

This produces two directly comparable models:
	•	Constrained model — enforces passive (gradient-driven) transport
	•	Unconstrained model — no conservation constraint

# Key observation

The unconstrained model retains a substantial coexact component, whereas the constrained model exhibits a near-total collapse of coexact energy.

This demonstrates that:

The suppression of rotational structure is not an intrinsic property of the neural architecture or optimization procedure, but is specifically induced by the conservation constraint.

# Interpretation

The conservation-constrained model effectively restricts the solution space to the exact subspace of the Hodge decomposition, corresponding to gradient-driven transport fields.

In contrast, removing the constraint restores the model’s ability to represent coexact (rotational) components, confirming that the collapse observed under constraint is structural and constraint-driven, rather than a generic smoothing artifact.

# Important clarification

The unconstrained model does not enforce conservation laws and therefore does not represent a physically valid transport model. Its role in this pipeline is purely diagnostic:

it isolates the effect of the conservation constraint on the learned field.

# Implication for the framework

This ablation provides direct causal evidence supporting the central falsification logic of the framework:
	•	Rotational structure is present in the data (via wedge flux construction)
	•	Conservation-constrained learning suppresses this structure
	•	Removing the constraint restores it

Therefore:

Passive, gradient-driven transport models are structurally incapable of representing the observed rotational organization at tumor–immune interfaces.




A key question in this framework is whether the observed collapse of the coexact (rotational) component under conservation-constrained learning is:
	•	an artifact of neural network smoothing, or
	•	a direct consequence of the imposed physical constraint.

This step provides an explicit diagnostic experiment to resolve this ambiguity.

⸻

## Step 18 — Constraint ablation

### Experimental design

We train the same graph neural network architecture under two conditions:

1. **Constrained model**
   - enforces the conservation-constrained, gradient-dominant transport formulation used in the manuscript
   - serves as the passive-model test case in this implementation

2. **Unconstrained model**
   - removes the conservation constraint
   - keeps the same architecture, input data, and optimization pipeline

This ablation isolates the effect of the conservation constraint within the current model class.

### Script
`scripts_tnbc/step18_ablation_no_constraint.py`

### Required inputs

The script expects precomputed GNN data from Step 13:

```text
stats/gnn_data/
  GSM_6433618_flux_tumor_immune_edge_index.npy
  GSM_6433618_flux_tumor_immune_edge_attr.npy
  GSM_6433618_flux_tumor_immune_y_edge.npy
  GSM_6433618_flux_tumor_immune_B1.npz
  GSM_6433618_flux_tumor_immune_B2.npz

  And constrained results from Step 15:
  stats/GSM_6433618_step15_gnn_operator_summary_flux_tumor_immune.csv

 ## How to run:

python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy pandas matplotlib scipy
python scripts_tnbc/step18_ablation_no_constraint.py


## Outputs
1. Unconstrained learned flux : stats/GSM_6433618_step18_unconstrained_flux_flux_tumor_immune.csv

2. Hodge energy decomposition (unconstrained) : stats/GSM_6433618_step18_unconstrained_energy_flux_tumor_immune.csv

3. Direct comparison (constrained vs unconstrained) : stats/GSM_6433618_step18_ablation_comparison_flux_tumor_immune.csv

4. Visualization:
stats/GSM_6433618_step18_ablation_plot_flux_tumor_immune.png

## Key result

The ablation produces the following qualitative pattern:

Component      Constrained      Unconstrained

Exact           ~1.0             reduced (~0.8–0.9)

Coexact          ~0              substantial (~0.4–0.5)

Harmonic         ~0               ~0

## Interpretation
In this ablation, the unconstrained model retains a substantial coexact component, whereas the constrained model exhibits near-complete collapse of coexact energy.

This indicates that, within the current implementation, the suppression of rotational structure is driven by the conservation constraint rather than by generic architectural smoothing alone.

## Implication for the framework

This experiment strengthens the falsification logic used in the manuscript:
	1.	the proxy wedge construction produces nontrivial coexact structure
	2.	conservation-constrained learning suppresses that structure
	3.	removing the constraint restores the model’s capacity to represent it

Accordingly, the ablation supports the interpretation that the observed collapse is constraint-induced in this model class, rather than a trivial artifact of network architecture alone.




# TNBC Step 18–19 Manual: Ablation and Biological Anchoring

This section documents the additional analyses introduced after the core Hodge / PDE-constrained workflow:

- **Step 18** — ablation without conservation constraint
- **Step 19** — biological anchoring of proxy coexact energy
- **Step 19 helper** — generation of missing node-level CSV files for new samples

These steps are intended to strengthen the interpretation of the operator-derived coexact phenotype and to address two key reviewer concerns:

1. whether coexact collapse under constrained learning is merely an architectural smoothing artifact  
2. whether proxy coexact structure has biologically structured anchoring beyond gross region composition

---

---

### Step 19 — Biological anchoring
`step19_coexact_bio_correlation.py`

This script tests whether **proxy node-level coexact energy** is associated with biologically structured variation.

Purpose:
- anchor the operator phenotype to biological structure
- distinguish gross region composition effects from within-region structure
- test whether coexact energy behaves like a marker of interface heterogeneity

Important:
- this analysis does **not** establish that coexact energy is a mechanistic transport readout
- the current interpretation is more conservative:
  - **coexact energy indexes interface heterogeneity / local tumor–immune contrast**
  - it is **not** treated as an immune-specific biomarker

---

### Step 19 helper — CSV generation
`step19_generate_csv.py`

This helper script generates missing files needed to run Step 19 on a new sample, especially:

- node-level proxy Hodge summaries
- `nodes_for_gnn` biological annotation CSV

This was used for **GSM_6433619**.

---

# Files introduced in these steps

## Figures

### GSM_6433618
- `GSM_6433618_fig/GSM_6433618_step18_ablation_plot_flux_tumor_immune.png`
- `GSM_6433618_fig/GSM_6433618_step19_coexact_bio_plot_flux_tumor_immune.png`

### GSM_6433619
- `GSM_6433619_fig/GSM_6433619_step19_coexact_bio_plot_flux_tumor_immune.png`

## CSV

### Biological annotation
- `stats/gnn_data/GSM_6433619_flux_tumor_immune_nodes_for_gnn.csv`

### Proxy node-level Hodge summary
- `stats/CSV_GSM/GSM_6433619_step6_nodes_hodge_flux_tumor_immune.csv`

## Scripts
- `scripts_tnbc/step18_ablation_no_constraint.py`
- `scripts_tnbc/step19_coexact_bio_correlation.py`
- `scripts_tnbc/step19_generate_csv.py`

---

# Environment setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy pandas matplotlib scipy statsmodels



'''Step 19 helper — Generate missing files for a new sample

This helper was required for GSM_6433619.

Run: python scripts_tnbc/step19_generate_csv.py

Purpose

Generate:
	•	stats/CSV_GSM/GSM_6433619_step6_nodes_hodge_flux_tumor_immune.csv
	•	stats/gnn_data/GSM_6433619_flux_tumor_immune_nodes_for_gnn.csv

Why this exists

Some TNBC samples already had the needed files for Step 19.
Others, like GSM_6433619, required reconstruction from:
	•	edge-level Hodge decomposition outputs
	•	node-level residualized biological annotation tables


Step 19 — Run biological anchoring:
Run : python scripts_tnbc/step19_coexact_bio_correlation.py

Before running on a specific sample, set the target sample inside the script:
SAMPLE_ID = "GSM_6433618"
or
SAMPLE_ID = "GSM_6433619"

Inputs

For a sample like GSM_6433619, Step 19 expects:
	•	stats/CSV_GSM/GSM_6433619_step6_nodes_hodge_flux_tumor_immune.csv
	•	stats/gnn_data/GSM_6433619_flux_tumor_immune_nodes_for_gnn.csv

Outputs

Typical outputs:
	•	merged node-level analysis CSV
	•	region-aware stats table
	•	biological anchoring scatter plot

Examples:
	•	stats/GSM_6433618_step19_coexact_bio_stats_flux_tumor_immune.csv
	•	stats/GSM_6433619_step19_coexact_bio_stats_flux_tumor_immune.csv

and
	•	GSM_6433618_fig/GSM_6433618_step19_coexact_bio_plot_flux_tumor_immune.png
	•	GSM_6433619_fig/GSM_6433619_step19_coexact_bio_plot_flux_tumor_immune.png

⸻

Statistical logic of Step 19

Step 19 does not rely on raw global Spearman alone.

The script computes:
	1.	raw Spearman
	•	descriptive only
	2.	region-demeaned Spearman
	•	primary nonparametric robustness check
	•	addresses gross region-composition confounding
	3.	OLS with region covariate
	•	provides region-adjusted effect-size estimate
	•	used cautiously because very large n can produce tiny but highly significant coefficients
	4.	within-interface analysis
	•	interface-like nodes only
	•	tests whether coexact energy tracks biological variation inside the active boundary zone itself
	5.	FDR correction
	•	applied separately to global and subset analyses

⸻

Current biological interpretation

What the analyses support

Across GSM_6433618 and GSM_6433619, the current evidence supports:
	•	proxy coexact energy is not well interpreted as a globally immune-specific signal
	•	within interface-like nodes, coexact energy tracks tumor–immune contrast
	•	the operator-derived phenotype is best interpreted as a marker of:
	•	interface heterogeneity
	•	local tumor–microenvironment contrast

What the analyses do not support

The current results do not establish that coexact energy is:
	•	a mechanistic readout of immune transport
	•	a cytokine flow proxy
	•	a dedicated immune infiltration biomarker

That stronger claim would require:
	•	larger cohorts
	•	independent molecular or clinical annotations
	•	preferably replication beyond the current proof-of-concept TNBC sections



⸻

Reproducibility notes
	1.	step19_generate_csv.py may be needed only for samples lacking:
	•	node-level proxy Hodge CSV
	•	nodes_for_gnn annotation CSV
	2.	step19_coexact_bio_correlation.py is sample-specific through:   SAMPLE_ID = "..."

	3.	Step 19 uses the proxy-field node-level coexact energy, not the constrained GNN node file, because the biological anchoring question concerns the proxy phenotype, not the collapsed constrained output.
	4.	The Step 19 figure should be interpreted alongside the stats table, not in isolation.

Practical Git tracking recommendation

Track:
	•	the three Step 18 / Step 19 scripts
	•	the final Step 19 CSV for GSM_6433619
	•	the final figure PNGs

Avoid blindly tracking every .npy / .npz intermediate unless they are required for end-to-end reruns and are reasonably sized.

Recommended tracked additions:
	•	scripts_tnbc/step18_ablation_no_constraint.py
	•	scripts_tnbc/step19_coexact_bio_correlation.py
	•	scripts_tnbc/step19_generate_csv.py
	•	stats/gnn_data/GSM_6433619_flux_tumor_immune_nodes_for_gnn.csv
	•	stats/CSV_GSM/GSM_6433619_step6_nodes_hodge_flux_tumor_immune.csv
	•	GSM_6433618_fig/GSM_6433618_step18_ablation_plot_flux_tumor_immune.png
	•	GSM_6433618_fig/GSM_6433618_step19_coexact_bio_plot_flux_tumor_immune.png
	•	GSM_6433619_fig/GSM_6433619_step19_coexact_bio_plot_flux_tumor_immune.png

