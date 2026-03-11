from pathlib import Path
import json

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from PIL import Image

from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors


# ============================================================
# 1) Paths
# ============================================================
DATA_DIR = Path(".")
H5_FILE = DATA_DIR / "Visium_Human_Breast_Cancer_filtered_feature_bc_matrix.h5"
SPATIAL_DIR = DATA_DIR / "spatial"


# ============================================================
# 2) Load Visium expression matrix
# ============================================================
adata = sc.read_10x_h5(H5_FILE)
adata.var_names_make_unique()

print("Loaded expression matrix:", adata.shape)


# ============================================================
# 3) Load spatial metadata
# ============================================================
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

# Keep only barcodes in expression matrix and preserve order
pos = pos[pos["barcode"].isin(adata.obs_names)].copy()
pos = pos.set_index("barcode").loc[adata.obs_names].copy()

adata.obs["in_tissue"] = pos["in_tissue"].astype(int).values
adata.obs["array_row"] = pos["array_row"].values
adata.obs["array_col"] = pos["array_col"].values
adata.obs["y_fullres"] = pos["pxl_row_in_fullres"].values
adata.obs["x_fullres"] = pos["pxl_col_in_fullres"].values

print("Matched spatial rows:", pos.shape[0])

with open(SPATIAL_DIR / "scalefactors_json.json", "r") as f:
    scalefactors = json.load(f)

hires_scale = scalefactors.get("tissue_hires_scalef", 1.0)
img = np.array(Image.open(SPATIAL_DIR / "tissue_hires_image.png"))
print("Hires image shape:", img.shape)


# ============================================================
# 4) Normalize and compute marker scores
# ============================================================
tumor_genes = ["EPCAM", "KRT8", "KRT18", "KRT19", "ERBB2", "MUC1", "TACSTD2"]
stroma_genes = ["COL1A1", "COL1A2", "DCN", "LUM", "POSTN", "FAP", "TAGLN"]
immune_genes = ["PTPRC", "CD3D", "CD3E", "NKG7", "CD68", "C1QA", "CXCL9", "CXCL10"]

def keep_present(genes, adata_obj):
    return [g for g in genes if g in adata_obj.var_names]

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


# ============================================================
# 5) Restrict to spots in tissue
# ============================================================
in_tissue_mask = adata.obs["in_tissue"].astype(bool).values
obs = adata.obs.copy()

coords = obs.loc[in_tissue_mask, ["x_fullres", "y_fullres"]].to_numpy(dtype=float)
barcodes_in_tissue = obs.index[in_tissue_mask].to_numpy()

tumor_score = obs.loc[in_tissue_mask, "tumor_score"].to_numpy()
stroma_score = obs.loc[in_tissue_mask, "stroma_score"].to_numpy()
immune_score = obs.loc[in_tissue_mask, "immune_score"].to_numpy()

print("In-tissue spots:", coords.shape[0])


# ============================================================
# 6) Build kNN graph
# ============================================================
# k=6 is a reasonable first pass for Visium spot neighborhoods
k = 6
nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
nbrs.fit(coords)
distances, indices = nbrs.kneighbors(coords)

# remove self-neighbor (column 0)
neighbor_indices = indices[:, 1:]
neighbor_distances = distances[:, 1:]

n = coords.shape[0]
rows = np.repeat(np.arange(n), k)
cols = neighbor_indices.reshape(-1)
data = np.ones(rows.shape[0], dtype=np.float32)

A = sparse.coo_matrix((data, (rows, cols)), shape=(n, n))
A = A.maximum(A.T).tocsr()  # symmetrize

print("kNN graph built:", A.shape, "edges =", A.nnz)


# ============================================================
# 7) Threshold masks
# ============================================================
# These thresholds are deliberately simple and should be refined later.
# Tumor mask:
#   - above median tumor score
#   - tumor score exceeds stroma score
tumor_thresh = np.quantile(tumor_score, 0.50)
tumor_mask_raw = (tumor_score >= tumor_thresh) & (tumor_score > stroma_score)

# Stroma mask:
#   - above 65th percentile stroma score
#   - stroma score exceeds tumor score
stroma_thresh = np.quantile(stroma_score, 0.65)
stroma_mask = (stroma_score >= stroma_thresh) & (stroma_score > tumor_score)

# Immune mask:
#   - use upper quantile, not argmax
immune_thresh = np.quantile(immune_score, 0.90)
immune_mask = immune_score >= immune_thresh

print("Raw tumor spots:", int(tumor_mask_raw.sum()))
print("Raw stroma spots:", int(stroma_mask.sum()))
print("Immune-rich spots:", int(immune_mask.sum()))


# ============================================================
# 8) Keep largest connected tumor component
# ============================================================
tumor_subgraph = A[tumor_mask_raw][:, tumor_mask_raw]

n_comp, labels = connected_components(tumor_subgraph, directed=False)
component_sizes = np.bincount(labels)

largest_component = component_sizes.argmax()
tumor_keep_local = labels == largest_component

tumor_indices_global = np.where(tumor_mask_raw)[0]
tumor_keep_global = tumor_indices_global[tumor_keep_local]

tumor_mask = np.zeros(n, dtype=bool)
tumor_mask[tumor_keep_global] = True

print("Tumor connected components:", n_comp)
print("Largest tumor component size:", int(tumor_mask.sum()))


# ============================================================
# 9) Define invasive margin and tumor core
# ============================================================
# Margin = tumor spots adjacent to stroma or immune neighborhoods.
non_tumor_interface_mask = stroma_mask | immune_mask

margin_mask = np.zeros(n, dtype=bool)

tumor_nodes = np.where(tumor_mask)[0]
for i in tumor_nodes:
    neigh = A[i].indices
    if np.any(non_tumor_interface_mask[neigh]):
        margin_mask[i] = True

core_mask = tumor_mask & (~margin_mask)

print("Tumor core spots:", int(core_mask.sum()))
print("Invasive margin spots:", int(margin_mask.sum()))


# ============================================================
# 10) Final priority-based region assignment
# ============================================================
# Priority:
#   immune_rich > stroma > invasive_margin > tumor_core > other
full_region = pd.Series("outside_tissue", index=adata.obs_names)

region[core_mask] = "tumor_core"
region[margin_mask] = "invasive_margin"
region[stroma_mask] = "stroma"
region[immune_mask] = "immune_rich"

region_series_in_tissue = pd.Series(region, index=barcodes_in_tissue)

# Full-size series aligned to all spots
full_region = pd.Series("outside_tissue", index=adata.obs_names)
full_region.loc[barcodes_in_tissue] = region_series_in_tissue
adata.obs["region_step2"] = full_region.values

print("\nFinal region counts:")
print(adata.obs["region_step2"].value_counts())


# ============================================================
# 11) Save spot-level CSV
# ============================================================
out_df = adata.obs.copy()
out_df["barcode"] = out_df.index
out_df = out_df[
    [
        "barcode",
        "in_tissue",
        "array_row",
        "array_col",
        "x_fullres",
        "y_fullres",
        "tumor_score",
        "stroma_score",
        "immune_score",
        "region_step2",
    ]
]
out_df.to_csv("step2_region_assignments.csv", index=False)
print("\nSaved: step2_region_assignments.csv")


# ============================================================
# 12) Plot final regions on tissue image
# ============================================================
color_map = {
    "tumor_core": "#d62728",       # red
    "invasive_margin": "#ff7f0e",  # orange
    "stroma": "#1f77b4",           # blue
    "immune_rich": "#2ca02c",      # green
    "other": "#7f7f7f",            # gray
    "outside_tissue": "#cccccc",   # light gray
}

plot_df = adata.obs.copy()
plot_df["plot_x"] = plot_df["x_fullres"] * hires_scale
plot_df["plot_y"] = plot_df["y_fullres"] * hires_scale
plot_df["plot_color"] = plot_df["region_step2"].map(color_map).fillna("#000000")

fig, ax = plt.subplots(figsize=(10, 9))
ax.imshow(img)
ax.scatter(
    plot_df["plot_x"],
    plot_df["plot_y"],
    s=14,
    c=plot_df["plot_color"],
    linewidths=0,
    alpha=0.9,
)
ax.set_title("Step 2 region definitions")
ax.invert_yaxis()
ax.axis("off")
plt.tight_layout()
plt.savefig("step2_regions_map.png", dpi=220)
print("Saved: step2_regions_map.png")


# ============================================================
# 13) Plot score distributions by region
# ============================================================
region_order = ["tumor_core", "invasive_margin", "stroma", "immune_rich", "other"]
plot_scores = adata.obs[adata.obs["region_step2"].isin(region_order)].copy()

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)

for ax, score_col, title in zip(
    axes,
    ["tumor_score", "stroma_score", "immune_score"],
    ["Tumor score by region", "Stroma score by region", "Immune score by region"],
):
    data_to_plot = [
        plot_scores.loc[plot_scores["region_step2"] == reg, score_col].values
        for reg in region_order
    ]
    ax.boxplot(data_to_plot, tick_labels=region_order, showfliers=False)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig("step2_region_score_boxplots.png", dpi=220)
print("Saved: step2_region_score_boxplots.png")


# ============================================================
# 14) Print quick summary stats
# ============================================================
summary = (
    adata.obs[adata.obs["region_step2"].isin(region_order)]
    .groupby("region_step2")[["tumor_score", "stroma_score", "immune_score"]]
    .median()
)
print("\nMedian scores by region:")
print(summary)
