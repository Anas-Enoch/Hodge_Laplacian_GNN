import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from PIL import Image

# -----------------------------
# 1) Paths
# -----------------------------
DATA_DIR = Path(".")
H5_FILE = DATA_DIR / "Visium_Human_Breast_Cancer_filtered_feature_bc_matrix.h5"
SPATIAL_DIR = DATA_DIR / "spatial"

# -----------------------------
# 2) Load expression matrix
# -----------------------------
adata = sc.read_10x_h5(H5_FILE)
adata.var_names_make_unique()

print("Loaded expression matrix:", adata.shape)

# -----------------------------
# 3) Load Visium spatial metadata
# -----------------------------
# 10x older format: no header
pos_file = SPATIAL_DIR / "tissue_positions_list.csv"
pos = pd.read_csv(
    pos_file,
    header=None,
    names=[
        "barcode",
        "in_tissue",
        "array_row",
        "array_col",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
    ],
)

# Keep only barcodes that exist in expression matrix
pos = pos[pos["barcode"].isin(adata.obs_names)].copy()
pos = pos.set_index("barcode").loc[adata.obs_names].copy()

# Add coordinates to adata.obs
adata.obs["in_tissue"] = pos["in_tissue"].astype(int).values
adata.obs["array_row"] = pos["array_row"].values
adata.obs["array_col"] = pos["array_col"].values
adata.obs["y_fullres"] = pos["pxl_row_in_fullres"].values
adata.obs["x_fullres"] = pos["pxl_col_in_fullres"].values

print("Matched spatial rows:", pos.shape[0])

# -----------------------------
# 4) Load image + scale factors
# -----------------------------
with open(SPATIAL_DIR / "scalefactors_json.json", "r") as f:
    scalefactors = json.load(f)

hires_scale = scalefactors.get("tissue_hires_scalef", 1.0)
lowres_scale = scalefactors.get("tissue_lowres_scalef", 1.0)

img = np.array(Image.open(SPATIAL_DIR / "tissue_hires_image.png"))

print("Hires image shape:", img.shape)
print("Scale factor (hires):", hires_scale)

# -----------------------------
# 5) Simple marker scoring
# -----------------------------
# Adjust later if some genes are absent. This is just the first pass.
tumor_genes = ["EPCAM", "KRT8", "KRT18", "KRT19", "ERBB2", "MUC1", "TACSTD2"]
stroma_genes = ["COL1A1", "COL1A2", "DCN", "LUM", "POSTN", "FAP", "TAGLN"]
immune_genes = ["PTPRC", "CD3D", "CD3E", "NKG7", "CD68", "C1QA", "CXCL9", "CXCL10"]

def keep_present(genes, adata):
    return [g for g in genes if g in adata.var_names]

tumor_genes_present = keep_present(tumor_genes, adata)
stroma_genes_present = keep_present(stroma_genes, adata)
immune_genes_present = keep_present(immune_genes, adata)

print("Tumor genes present:", tumor_genes_present)
print("Stroma genes present:", stroma_genes_present)
print("Immune genes present:", immune_genes_present)

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

if tumor_genes_present:
    sc.tl.score_genes(adata, gene_list=tumor_genes_present, score_name="tumor_score", use_raw=False)
else:
    adata.obs["tumor_score"] = 0.0

if stroma_genes_present:
    sc.tl.score_genes(adata, gene_list=stroma_genes_present, score_name="stroma_score", use_raw=False)
else:
    adata.obs["stroma_score"] = 0.0

if immune_genes_present:
    sc.tl.score_genes(adata, gene_list=immune_genes_present, score_name="immune_score", use_raw=False)
else:
    adata.obs["immune_score"] = 0.0

# -----------------------------
# 6) First-pass region labels
# -----------------------------
# Crude initial assignment. We refine later.
scores = adata.obs[["tumor_score", "stroma_score", "immune_score"]].copy()
max_label = scores.idxmax(axis=1)

region0 = pd.Series("unassigned", index=adata.obs_names)
region0[max_label == "tumor_score"] = "tumor_like"
region0[max_label == "stroma_score"] = "stroma_like"
region0[max_label == "immune_score"] = "immune_like"

adata.obs["region0"] = region0.values

print("\nInitial region counts:")
print(adata.obs["region0"].value_counts())

# -----------------------------
# 7) Plot tissue image + labels
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# background tissue image
axes[0, 0].imshow(img)
axes[0, 0].scatter(
    adata.obs["x_fullres"] * hires_scale,
    adata.obs["y_fullres"] * hires_scale,
    s=8,
    c=adata.obs["tumor_score"],
)
axes[0, 0].set_title("Tumor score")
axes[0, 0].invert_yaxis()
axes[0, 0].axis("off")

axes[0, 1].imshow(img)
axes[0, 1].scatter(
    adata.obs["x_fullres"] * hires_scale,
    adata.obs["y_fullres"] * hires_scale,
    s=8,
    c=adata.obs["stroma_score"],
)
axes[0, 1].set_title("Stroma score")
axes[0, 1].invert_yaxis()
axes[0, 1].axis("off")

axes[1, 0].imshow(img)
axes[1, 0].scatter(
    adata.obs["x_fullres"] * hires_scale,
    adata.obs["y_fullres"] * hires_scale,
    s=8,
    c=adata.obs["immune_score"],
)
axes[1, 0].set_title("Immune score")
axes[1, 0].invert_yaxis()
axes[1, 0].axis("off")

color_map = {
    "tumor_like": "red",
    "stroma_like": "blue",
    "immune_like": "green",
    "unassigned": "gray",
}
colors = [color_map[r] for r in adata.obs["region0"]]

axes[1, 1].imshow(img)
axes[1, 1].scatter(
    adata.obs["x_fullres"] * hires_scale,
    adata.obs["y_fullres"] * hires_scale,
    s=8,
    c=colors,
)
axes[1, 1].set_title("Initial region labels")
axes[1, 1].invert_yaxis()
axes[1, 1].axis("off")

plt.tight_layout()
plt.savefig("step1_visium_region_scores.png", dpi=200)
print("\nSaved figure: step1_visium_region_scores.png")
