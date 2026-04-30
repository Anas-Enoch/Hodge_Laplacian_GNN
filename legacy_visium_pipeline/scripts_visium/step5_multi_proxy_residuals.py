from pathlib import Path
import json

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from PIL import Image

from scipy import sparse


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
DATA_DIR = Path(".")

H5_FILE = "Visium_Human_Breast_Cancer_filtered_feature_bc_matrix.h5"
SPATIAL_DIR = "spatial"

REGION_FILE = "step2_region_assignments.csv"
EDGES_FILE = "step3_edges.csv"
B1_FILE = "step3_B1_incidence.npz"


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------
adata = sc.read_10x_h5(H5_FILE)
adata.var_names_make_unique()

print("Dataset:", adata.shape)


# ------------------------------------------------------------
# Spatial metadata
# ------------------------------------------------------------
pos = pd.read_csv(
    f"{SPATIAL_DIR}/tissue_positions_list.csv",
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

pos = pos[pos["barcode"].isin(adata.obs_names)]
pos = pos.set_index("barcode").loc[adata.obs_names]

adata.obs["in_tissue"] = pos["in_tissue"].values
adata.obs["x"] = pos["pxl_col_in_fullres"].values
adata.obs["y"] = pos["pxl_row_in_fullres"].values


# ------------------------------------------------------------
# Image
# ------------------------------------------------------------
with open(f"{SPATIAL_DIR}/scalefactors_json.json") as f:
    scalef = json.load(f)

scale = scalef["tissue_hires_scalef"]
img = np.array(Image.open(f"{SPATIAL_DIR}/tissue_hires_image.png"))


# ------------------------------------------------------------
# Marker scoring
# ------------------------------------------------------------
tumor_genes = ["EPCAM","KRT8","KRT18","KRT19","ERBB2","MUC1","TACSTD2"]
stroma_genes = ["COL1A1","COL1A2","DCN","LUM","POSTN","FAP","TAGLN"]
immune_genes = ["PTPRC","CD3D","CD3E","NKG7","CD68","C1QA","CXCL9","CXCL10"]

def present(g):
    return [x for x in g if x in adata.var_names]

tumor_genes = present(tumor_genes)
stroma_genes = present(stroma_genes)
immune_genes = present(immune_genes)

sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

sc.tl.score_genes(adata, tumor_genes, score_name="tumor_score")
sc.tl.score_genes(adata, stroma_genes, score_name="stroma_score")
sc.tl.score_genes(adata, immune_genes, score_name="immune_score")


# ------------------------------------------------------------
# Restrict to tissue
# ------------------------------------------------------------
obs = adata.obs
obs = obs[obs["in_tissue"] == 1].copy()

barcodes = obs.index.values
N = len(barcodes)

print("In tissue:", N)


# ------------------------------------------------------------
# Load region annotations
# ------------------------------------------------------------
regions = pd.read_csv(REGION_FILE)
regions = regions.set_index("barcode").loc[barcodes]

obs["region"] = regions["region_step2"].values
obs["region"] = obs["region"].replace({"other": "mixed_unassigned"})


# ------------------------------------------------------------
# Load graph
# ------------------------------------------------------------
edges = pd.read_csv(EDGES_FILE)
B1 = sparse.load_npz(B1_FILE)


# ------------------------------------------------------------
# Proxy fields
# ------------------------------------------------------------
proxy_fields = [
    "tumor_score",
    "stroma_score",
    "immune_score",
]


# ------------------------------------------------------------
# Run residuals
# ------------------------------------------------------------
for field in proxy_fields:

    print("\n======================")
    print("Proxy field:", field)

    u = obs[field].values

    tail = edges["tail"].values
    head = edges["head"].values
    length = edges["length"].values

    u_tail = u[tail]
    u_head = u[head]

    flux = -(u_head - u_tail) / np.maximum(length, 1e-8)

    residual = B1 @ flux
    abs_r = np.abs(residual)

    obs[f"residual_{field}"] = abs_r


    # --------------------------------------------
    # Region summary
    # --------------------------------------------
    summary = (
        obs.groupby("region")[f"residual_{field}"]
        .agg(["count","median","mean","std"])
        .sort_values("median")
    )

    print(summary)


    summary.to_csv(f"step5_summary_{field}.csv")


    # --------------------------------------------
    # Map
    # --------------------------------------------
    fig, ax = plt.subplots(figsize=(10,9))

    ax.imshow(img)

    x = obs["x"].values * scale
    y = obs["y"].values * scale

    sca = ax.scatter(
        x,
        y,
        c=abs_r,
        s=18,
    )

    plt.colorbar(sca, ax=ax)

    ax.set_title(f"Residual map — {field}")

    ax.invert_yaxis()
    ax.axis("off")

    plt.tight_layout()

    plt.savefig(f"step5_map_{field}.png", dpi=220)

    plt.close()


print("\nDone.")
