from pathlib import Path
import json

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from PIL import Image

from scipy import sparse
from scipy.stats import mannwhitneyu


# ============================================================
# 1) Paths
# ============================================================
DATA_DIR = Path(".")
H5_FILE = DATA_DIR / "Visium_Human_Breast_Cancer_filtered_feature_bc_matrix.h5"
SPATIAL_DIR = DATA_DIR / "spatial"

REGION_CSV = DATA_DIR / "step2_region_assignments.csv"
EDGES_CSV = DATA_DIR / "step3_edges.csv"
B1_FILE = DATA_DIR / "step3_B1_incidence.npz"


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

pos = pos[pos["barcode"].isin(adata.obs_names)].copy()
pos = pos.set_index("barcode").loc[adata.obs_names].copy()

adata.obs["in_tissue"] = pos["in_tissue"].astype(int).values
adata.obs["array_row"] = pos["array_row"].values
adata.obs["array_col"] = pos["array_col"].values
adata.obs["y_fullres"] = pos["pxl_row_in_fullres"].values
adata.obs["x_fullres"] = pos["pxl_col_in_fullres"].values

with open(SPATIAL_DIR / "scalefactors_json.json", "r") as f:
    scalefactors = json.load(f)

hires_scale = scalefactors.get("tissue_hires_scalef", 1.0)
img = np.array(Image.open(SPATIAL_DIR / "tissue_hires_image.png"))

print("Matched spatial rows:", pos.shape[0])
print("Hires image shape:", img.shape)


# ============================================================
# 4) Recompute marker scores
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
# 5) Restrict to in-tissue spots and preserve graph order
# ============================================================
obs = adata.obs.copy()
obs_in = obs.loc[obs["in_tissue"].astype(bool)].copy()

barcodes_in = obs_in.index.to_numpy()
n_nodes = len(barcodes_in)

print("In-tissue spots:", n_nodes)


# ============================================================
# 6) Load Step 2 regions and align to in-tissue spots
# ============================================================
regions_df = pd.read_csv(REGION_CSV)
regions_df = regions_df.set_index("barcode").loc[barcodes_in].copy()

obs_in["region_step2"] = regions_df["region_step2"].values

# Rename 'other' to a better manuscript-facing label
obs_in["region_step2"] = obs_in["region_step2"].replace({"other": "mixed_unassigned"})

print("\nRegion counts:")
print(obs_in["region_step2"].value_counts())


# ============================================================
# 7) Load Step 3 graph objects
# ============================================================
edges_df = pd.read_csv(EDGES_CSV)
B1 = sparse.load_npz(B1_FILE).tocsr()

print("\nLoaded graph objects:")
print("edges_df shape:", edges_df.shape)
print("B1 shape:", B1.shape)

assert B1.shape[0] == n_nodes, "B1 row count must match number of in-tissue nodes"
assert B1.shape[1] == edges_df.shape[0], "B1 column count must match number of edges"


# ============================================================
# 8) Define proxy scalar field u
# ============================================================
# First pass: tumor score as a scalar field
u = obs_in["tumor_score"].to_numpy(dtype=float)

print("\nProxy field summary (tumor_score):")
print("mean =", float(np.mean(u)))
print("std  =", float(np.std(u)))
print("min  =", float(np.min(u)))
print("max  =", float(np.max(u)))


# ============================================================
# 9) Compute edge flux f
# ============================================================
# Edge orientation comes from step3_edges.csv:
# tail -> head
#
# Diffusion-like proxy flux:
# f_e = -(u_head - u_tail) / length
#
# This is a modeling choice, not a library contract.
tail = edges_df["tail"].to_numpy(dtype=int)
head = edges_df["head"].to_numpy(dtype=int)
length = edges_df["length"].to_numpy(dtype=float)

u_tail = u[tail]
u_head = u[head]

f = -(u_head - u_tail) / np.maximum(length, 1e-8)

edges_df["u_tail"] = u_tail
edges_df["u_head"] = u_head
edges_df["proxy_flux"] = f

print("\nFlux summary:")
print("mean =", float(np.mean(f)))
print("std  =", float(np.std(f)))
print("min  =", float(np.min(f)))
print("max  =", float(np.max(f)))


# ============================================================
# 10) Compute node-wise conservation residual r = B1 f
# ============================================================
r = B1 @ f
abs_r = np.abs(r)

obs_in["proxy_residual"] = r
obs_in["abs_proxy_residual"] = abs_r

print("\nResidual summary:")
print("mean(|r|) =", float(np.mean(abs_r)))
print("median(|r|) =", float(np.median(abs_r)))
print("max(|r|) =", float(np.max(abs_r)))


# ============================================================
# 11) Save node-level residual table
# ============================================================
residual_df = obs_in.copy()
residual_df["barcode"] = residual_df.index

keep_cols = [
    "barcode",
    "array_row",
    "array_col",
    "x_fullres",
    "y_fullres",
    "tumor_score",
    "stroma_score",
    "immune_score",
    "region_step2",
    "proxy_residual",
    "abs_proxy_residual",
]
residual_df[keep_cols].to_csv("step4_proxy_residuals_by_spot.csv", index=False)
print("\nSaved: step4_proxy_residuals_by_spot.csv")


# ============================================================
# 12) Save edge-level flux table
# ============================================================
edges_df.to_csv("step4_proxy_flux_by_edge.csv", index=False)
print("Saved: step4_proxy_flux_by_edge.csv")


# ============================================================
# 13) Plot residual map on tissue
# ============================================================
plot_x = obs_in["x_fullres"].to_numpy(dtype=float) * hires_scale
plot_y = obs_in["y_fullres"].to_numpy(dtype=float) * hires_scale

fig, ax = plt.subplots(figsize=(10, 9))
ax.imshow(img)
scat = ax.scatter(
    plot_x,
    plot_y,
    s=18,
    c=abs_r,
    alpha=0.95,
)
ax.set_title("Step 4 proxy conservation residual |r|")
ax.invert_yaxis()
ax.axis("off")
plt.colorbar(scat, ax=ax, fraction=0.03, pad=0.02, label="|r|")
plt.tight_layout()
plt.savefig("step4_proxy_residual_map.png", dpi=220)
print("Saved: step4_proxy_residual_map.png")


# ============================================================
# 14) Plot residuals by region
# ============================================================
region_order = [
    "tumor_core",
    "invasive_margin",
    "stroma",
    "immune_rich",
    "mixed_unassigned",
]

plot_df = obs_in[obs_in["region_step2"].isin(region_order)].copy()

fig, ax = plt.subplots(figsize=(11, 6))
data_to_plot = [
    plot_df.loc[plot_df["region_step2"] == reg, "abs_proxy_residual"].values
    for reg in region_order
]
ax.boxplot(data_to_plot, tick_labels=region_order, showfliers=False)
ax.set_title("Proxy residual magnitude by region")
ax.set_ylabel("|r|")
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig("step4_proxy_residual_boxplots.png", dpi=220)
print("Saved: step4_proxy_residual_boxplots.png")


# ============================================================
# 15) Region summary statistics
# ============================================================
summary = (
    plot_df.groupby("region_step2")[["abs_proxy_residual", "proxy_residual"]]
    .agg(["count", "median", "mean", "std"])
)

summary.to_csv("step4_proxy_residual_region_summary.csv")
print("Saved: step4_proxy_residual_region_summary.csv")

print("\nResidual summary by region:")
print(
    plot_df.groupby("region_step2")["abs_proxy_residual"]
    .agg(["count", "median", "mean", "std"])
    .loc[region_order]
)


# ============================================================
# 16) Simple pairwise tests
# ============================================================
def compare_regions(df, region_a, region_b):
    xa = df.loc[df["region_step2"] == region_a, "abs_proxy_residual"].values
    xb = df.loc[df["region_step2"] == region_b, "abs_proxy_residual"].values

    res = mannwhitneyu(xa, xb, alternative="two-sided")
    return {
        "region_a": region_a,
        "region_b": region_b,
        "n_a": len(xa),
        "n_b": len(xb),
        "median_a": float(np.median(xa)),
        "median_b": float(np.median(xb)),
        "mean_a": float(np.mean(xa)),
        "mean_b": float(np.mean(xb)),
        "u_stat": float(res.statistic),
        "p_value": float(res.pvalue),
    }

comparisons = [
    compare_regions(plot_df, "tumor_core", "invasive_margin"),
    compare_regions(plot_df, "tumor_core", "stroma"),
    compare_regions(plot_df, "tumor_core", "immune_rich"),
]

comp_df = pd.DataFrame(comparisons)
comp_df.to_csv("step4_proxy_residual_pairwise_tests.csv", index=False)
print("\nSaved: step4_proxy_residual_pairwise_tests.csv")
print("\nPairwise tests:")
print(comp_df)


# ============================================================
# 17) Overlay regions + residual hotspots
# ============================================================
# Mark top 10% residual spots
thr90 = np.quantile(abs_r, 0.90)
obs_in["high_residual"] = abs_r >= thr90

region_color_map = {
    "tumor_core": "#d62728",
    "invasive_margin": "#ff7f0e",
    "stroma": "#1f77b4",
    "immune_rich": "#2ca02c",
    "mixed_unassigned": "#7f7f7f",
}

base_colors = obs_in["region_step2"].map(region_color_map).fillna("#000000")

fig, ax = plt.subplots(figsize=(10, 9))
ax.imshow(img)

# base regions
ax.scatter(
    plot_x,
    plot_y,
    s=14,
    c=base_colors,
    alpha=0.45,
    linewidths=0,
)

# top residual hotspots
hot = obs_in["high_residual"].to_numpy()
ax.scatter(
    plot_x[hot],
    plot_y[hot],
    s=28,
    facecolors="none",
    edgecolors="black",
    linewidths=0.6,
)

ax.set_title("High proxy residual spots over region map")
ax.invert_yaxis()
ax.axis("off")
plt.tight_layout()
plt.savefig("step4_proxy_residual_hotspots.png", dpi=220)
print("Saved: step4_proxy_residual_hotspots.png")
