Reproducible analysis for the manuscript:

"A PDE-Constrained Hodge–Laplacian Graph Neural Network for Falsifiable Inference in Transport-Dominated Systems"

# Data Description

This repository contains processed outputs from spatial transcriptomics analysis of TNBC tumors (GSE210616).

The analysis pipeline constructs spatial transport fields and evaluates their geometric structure using discrete Hodge operators.

---

# Directory structure
data/
TNBC_GSE210616/
GSM_6433618/
GSM_6433619/

stats/
CSV_GSM/

GSM_visium_figures/
visium_figures/
---

# Raw dataset

Spatial transcriptomics data originate from:

**GSE210616 – Spatial transcriptomics of triple negative breast cancer**

Platform: **10x Genomics Visium**

Each sample includes:
filtered_feature_bc_matrix.h5
tissue_positions_list.csv
tissue_hires_image.png
scalefactors_json.json

---

# Processed outputs

Each sample generates the following files.

### Graph construction
step3_nodes.csv
step3_edges.csv
step3_faces.csv
step3_B1.npz
step3_B2.npz
step3_L1_edge_hodge.npz

### Flux construction
step4_edge_fluxes.csv
step4_node_residualized_fields.csv

### Transport decomposition
step6_edges_hodge_flux_.csv
step6_nodes_hodge_flux_.csv
step6_faces_hodge_flux_*.csv

### Curl analysis
step9_face_curl_*.csv

### Null model
step11_lie_null_distribution_.csv
step11_lie_null_summary_.csv

### Region tests
step12_region_hotspot_lie_test_*.csv

---

# Reproducibility

The full pipeline can be executed sequentially using the scripts in
scripts_tnbc/
Example:

python -m scripts_tnbc.step11_lie_structured_null 
–sample_id GSM_6433618 
–sample_dir data/TNBC_GSE210616/GSM_6433618 
–flux_name flux_tumor_immune

---

# Key result

The analysis identifies **localized rotational transport structure** at tumor–immune interfaces, detected through the coexact component of the Hodge decomposition and validated against a spatially aware Lie-structured null model.

