from pathlib import Path
import json

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from PIL import Image

from scipy import sparse
from scipy.sparse import linalg as spla


# ============================================================
# 0) User choice
# ============================================================
# Best first candidate from your step5 results:
PROXY_FIELD = "immune_score"
# Alternatives:
# PROXY_FIELD = "tumor_score"
# PROXY_FIELD = "stroma_score"

RIDGE = 1e-6  # numerical stabilization


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
# 2) Load Visium matrix
# ============================================================
adata = sc.read_10x_h5(H5_FILE)
adata.var_names_make_unique()

print("Loaded expression matrix:", adata.shape)


# ============================================================
# 3) Spatial metadata
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
# 5) Restrict to in-tissue spots and align region labels
# ============================================================
obs = adata.obs.copy()
obs_in = obs.loc[obs["in_tissue"].astype(bool)].copy()
barcodes_in = obs_in.index.to_numpy()

regions_df = pd.read_csv(REGION_CSV).set_index("barcode").loc[barcodes_in].copy()
obs_in["region_step2"] = regions_df["region_step2"].replace({"other": "mixed_unassigned"}).values

n_nodes = len(obs_in)
print("In-tissue spots:", n_nodes)
print("\nRegion counts:")
print(obs_in["region_step2"].value_counts())


# ============================================================
# 6) Load graph objects
# ============================================================
edges_df = pd.read_csv(EDGES_CSV)
B1 = sparse.load_npz(B1_FILE).tocsr()

n_edges = edges_df.shape[0]
print("\nLoaded graph objects:")
print("edges_df shape:", edges_df.shape)
print("B1 shape:", B1.shape)

assert B1.shape[0] == n_nodes, "B1 rows must match in-tissue node count"
assert B1.shape[1] == n_edges, "B1 cols must match edge count"


# ============================================================
# 7) Build edge lookup and adjacency
# ============================================================
tail = edges_df["tail"].to_numpy(dtype=int)
head = edges_df["head"].to_numpy(dtype=int)
length = edges_df["length"].to_numpy(dtype=float)

# Edge orientation convention inherited from step3:
# edge e = (i, j) with i < j, tail=i, head=j
edge_lookup = {}
neighbors = [set() for _ in range(n_nodes)]

for e_id, (i, j) in enumerate(zip(tail, head)):
    edge_lookup[(i, j)] = e_id
    neighbors[i].add(j)
    neighbors[j].add(i)

print("Built edge lookup and adjacency.")


# ============================================================
# 8) Detect triangles and build B2
# ============================================================
# Triangle orientation convention:
# for ordered triangle (i, j, k) with i < j < k,
# boundary = (j,k) - (i,k) + (i,j)
#
# Because our edge orientations are always from smaller -> larger node:
#   e_ij : i -> j
#   e_ik : i -> k
#   e_jk : j -> k
#
# So the oriented triangle column is:
#   +1 on e_ij
#   -1 on e_ik
#   +1 on e_jk
triangle_rows = []
triangle_cols = []
triangle_data = []
triangle_list = []

tri_id = 0
for i in range(n_nodes):
    nbrs_i = sorted([j for j in neighbors[i] if j > i])
    nbrs_i_set = set(nbrs_i)

    for idx_j, j in enumerate(nbrs_i):
        # intersection of neighbors(i) and neighbors(j), only k > j
        common = nbrs_i_set.intersection(neighbors[j])
        common = [k for k in common if k > j]

        for k in common:
            e_ij = edge_lookup[(i, j)]
            e_ik = edge_lookup[(i, k)]
            e_jk = edge_lookup[(j, k)]

            triangle_rows.extend([e_ij, e_ik, e_jk])
            triangle_cols.extend([tri_id, tri_id, tri_id])
            triangle_data.extend([+1.0, -1.0, +1.0])

            triangle_list.append((i, j, k))
            tri_id += 1

n_triangles = tri_id
print("Detected triangles:", n_triangles)

if n_triangles == 0:
    raise RuntimeError("No triangles detected. Cannot build B2 / coexact component meaningfully.")

B2 = sparse.coo_matrix(
    (triangle_data, (triangle_rows, triangle_cols)),
    shape=(n_edges, n_triangles),
).tocsr()

print("B2 shape:", B2.shape)


# ============================================================
# 9) Edge Hodge Laplacian
# ============================================================
L1 = (B1.T @ B1) + (B2 @ B2.T)
print("L1 shape:", L1.shape)


# ============================================================
# 10) Build proxy edge flux
# ============================================================
if PROXY_FIELD not in ["tumor_score", "stroma_score", "immune_score"]:
    raise ValueError(f"Unsupported PROXY_FIELD: {PROXY_FIELD}")

u = obs_in[PROXY_FIELD].to_numpy(dtype=float)
u_tail = u[tail]
u_head = u[head]

# Same proxy flux logic as Step 4/5
f = -(u_head - u_tail) / np.maximum(length, 1e-8)

print(f"\nProxy field: {PROXY_FIELD}")
print("Flux summary:")
print("mean =", float(np.mean(f)))
print("std  =", float(np.std(f)))
print("min  =", float(np.min(f)))
print("max  =", float(np.max(f)))


# ============================================================
# 11) Exact component: projection onto im(B1^T)
# ============================================================
# Solve alpha from:
#   (B1 B1^T + λI) alpha = B1 f
# then exact = B1^T alpha
BBt = (B1 @ B1.T).tocsr()
rhs_exact = B1 @ f
A_exact = BBt + RIDGE * sparse.eye(BBt.shape[0], format="csr")

alpha = spla.spsolve(A_exact, rhs_exact)
f_exact = B1.T @ alpha

print("\nExact component computed.")


# ============================================================
# 12) Coexact component: projection onto im(B2)
# ============================================================
# Solve beta from:
#   (B2^T B2 + λI) beta = B2^T (f - f_exact)
# then coexact = B2 beta
res_after_exact = f - f_exact

BtB = (B2.T @ B2).tocsr()
rhs_coexact = B2.T @ res_after_exact
A_coexact = BtB + RIDGE * sparse.eye(BtB.shape[0], format="csr")

beta = spla.spsolve(A_coexact, rhs_coexact)
f_coexact = B2 @ beta

print("Coexact component computed.")


# ============================================================
# 13) Harmonic remainder
# ============================================================
f_harmonic = f - f_exact - f_coexact
print("Harmonic component computed.")


# ============================================================
# 14) Diagnostics
# ============================================================
def norm_sq(x):
    return float(np.dot(x, x))

tot = norm_sq(f)
exact_e = norm_sq(f_exact)
coexact_e = norm_sq(f_coexact)
harmonic_e = norm_sq(f_harmonic)

print("\nEnergy decomposition:")
print("||f||^2           =", tot)
print("||f_exact||^2     =", exact_e)
print("||f_coexact||^2   =", coexact_e)
print("||f_harmonic||^2  =", harmonic_e)
print("fraction exact    =", exact_e / tot if tot > 0 else np.nan)
print("fraction coexact  =", coexact_e / tot if tot > 0 else np.nan)
print("fraction harmonic =", harmonic_e / tot if tot > 0 else np.nan)
print("fraction sum      =", (exact_e + coexact_e + harmonic_e) / tot if tot > 0 else np.nan)

# Divergence and curl-style diagnostics
div_total = B1 @ f
div_exact = B1 @ f_exact
div_coexact = B1 @ f_coexact
div_harm = B1 @ f_harmonic

curl_total = B2.T @ f
curl_exact = B2.T @ f_exact
curl_coexact = B2.T @ f_coexact
curl_harm = B2.T @ f_harmonic

print("\nComponent diagnostics:")
print("mean |div total|    =", float(np.mean(np.abs(div_total))))
print("mean |div exact|    =", float(np.mean(np.abs(div_exact))))
print("mean |div coexact|  =", float(np.mean(np.abs(div_coexact))))
print("mean |div harmonic| =", float(np.mean(np.abs(div_harm))))

print("mean |curl total|    =", float(np.mean(np.abs(curl_total))))
print("mean |curl exact|    =", float(np.mean(np.abs(curl_exact))))
print("mean |curl coexact|  =", float(np.mean(np.abs(curl_coexact))))
print("mean |curl harmonic| =", float(np.mean(np.abs(curl_harm))))


# ============================================================
# 15) Save edge-level decomposition
# ============================================================
edges_out = edges_df.copy()
edges_out["flux_total"] = f
edges_out["flux_exact"] = f_exact
edges_out["flux_coexact"] = f_coexact
edges_out["flux_harmonic"] = f_harmonic

edges_out["abs_total"] = np.abs(f)
edges_out["abs_exact"] = np.abs(f_exact)
edges_out["abs_coexact"] = np.abs(f_coexact)
edges_out["abs_harmonic"] = np.abs(f_harmonic)

edges_out.to_csv(f"step6_edges_hodge_{PROXY_FIELD}.csv", index=False)
print(f"\nSaved: step6_edges_hodge_{PROXY_FIELD}.csv")


# ============================================================
# 16) Map edge values to nodewise energies
# ============================================================
# node_energy(i) = mean absolute incident edge component around node i
inc_abs_total = np.zeros(n_nodes, dtype=float)
inc_abs_exact = np.zeros(n_nodes, dtype=float)
inc_abs_coexact = np.zeros(n_nodes, dtype=float)
inc_abs_harmonic = np.zeros(n_nodes, dtype=float)
inc_degree = np.zeros(n_nodes, dtype=float)

for e_id, (i, j) in enumerate(zip(tail, head)):
    a0 = abs(f[e_id])
    a1 = abs(f_exact[e_id])
    a2 = abs(f_coexact[e_id])
    a3 = abs(f_harmonic[e_id])

    inc_abs_total[i] += a0
    inc_abs_total[j] += a0

    inc_abs_exact[i] += a1
    inc_abs_exact[j] += a1

    inc_abs_coexact[i] += a2
    inc_abs_coexact[j] += a2

    inc_abs_harmonic[i] += a3
    inc_abs_harmonic[j] += a3

    inc_degree[i] += 1
    inc_degree[j] += 1

node_total = inc_abs_total / np.maximum(inc_degree, 1)
node_exact = inc_abs_exact / np.maximum(inc_degree, 1)
node_coexact = inc_abs_coexact / np.maximum(inc_degree, 1)
node_harmonic = inc_abs_harmonic / np.maximum(inc_degree, 1)

obs_in["node_energy_total"] = node_total
obs_in["node_energy_exact"] = node_exact
obs_in["node_energy_coexact"] = node_coexact
obs_in["node_energy_harmonic"] = node_harmonic

obs_in["abs_div_total"] = np.abs(div_total)
obs_in["abs_div_exact"] = np.abs(div_exact)
obs_in["abs_div_coexact"] = np.abs(div_coexact)
obs_in["abs_div_harmonic"] = np.abs(div_harm)

nodes_out = obs_in.copy()
nodes_out["barcode"] = nodes_out.index

keep_cols = [
    "barcode",
    "array_row",
    "array_col",
    "x_fullres",
    "y_fullres",
    "region_step2",
    "tumor_score",
    "stroma_score",
    "immune_score",
    "node_energy_total",
    "node_energy_exact",
    "node_energy_coexact",
    "node_energy_harmonic",
    "abs_div_total",
    "abs_div_exact",
    "abs_div_coexact",
    "abs_div_harmonic",
]
nodes_out[keep_cols].to_csv(f"step6_nodes_hodge_{PROXY_FIELD}.csv", index=False)
print(f"Saved: step6_nodes_hodge_{PROXY_FIELD}.csv")


# ============================================================
# 17) Region summaries
# ============================================================
region_order = [
    "tumor_core",
    "invasive_margin",
    "stroma",
    "immune_rich",
    "mixed_unassigned",
]

summary = (
    obs_in.groupby("region_step2")[
        ["node_energy_total", "node_energy_exact", "node_energy_coexact", "node_energy_harmonic"]
    ]
    .agg(["count", "median", "mean", "std"])
)

summary.to_csv(f"step6_region_summary_{PROXY_FIELD}.csv")
print(f"Saved: step6_region_summary_{PROXY_FIELD}.csv")

print("\nNode energy medians by region:")
print(
    obs_in.groupby("region_step2")[
        ["node_energy_total", "node_energy_exact", "node_energy_coexact", "node_energy_harmonic"]
    ]
    .median()
    .loc[region_order]
)


# ============================================================
# 18) Spatial maps
# ============================================================
plot_x = obs_in["x_fullres"].to_numpy(dtype=float) * hires_scale
plot_y = obs_in["y_fullres"].to_numpy(dtype=float) * hires_scale

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
maps = [
    ("node_energy_total", "Total edge energy"),
    ("node_energy_exact", "Exact / gradient energy"),
    ("node_energy_coexact", "Coexact / rotational energy"),
    ("node_energy_harmonic", "Harmonic energy"),
]

for ax, (col, title) in zip(axes.ravel(), maps):
    ax.imshow(img)
    scat = ax.scatter(
        plot_x,
        plot_y,
        s=18,
        c=obs_in[col].to_numpy(),
        alpha=0.95,
    )
    ax.set_title(title)
    ax.invert_yaxis()
    ax.axis("off")
    plt.colorbar(scat, ax=ax, fraction=0.03, pad=0.02)

plt.suptitle(f"Step 6 Hodge decomposition maps — {PROXY_FIELD}", y=0.98)
plt.tight_layout()
plt.savefig(f"step6_hodge_maps_{PROXY_FIELD}.png", dpi=220)
print(f"\nSaved: step6_hodge_maps_{PROXY_FIELD}.png")


# ============================================================
# 19) Region boxplots
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
box_cols = [
    ("node_energy_total", "Total edge energy"),
    ("node_energy_exact", "Exact / gradient energy"),
    ("node_energy_coexact", "Coexact / rotational energy"),
    ("node_energy_harmonic", "Harmonic energy"),
]

for ax, (col, title) in zip(axes.ravel(), box_cols):
    data_to_plot = [
        obs_in.loc[obs_in["region_step2"] == reg, col].values
        for reg in region_order
    ]
    ax.boxplot(data_to_plot, tick_labels=region_order, showfliers=False)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)

plt.suptitle(f"Step 6 region-wise Hodge energies — {PROXY_FIELD}", y=0.98)
plt.tight_layout()
plt.savefig(f"step6_hodge_boxplots_{PROXY_FIELD}.png", dpi=220)
print(f"Saved: step6_hodge_boxplots_{PROXY_FIELD}.png")


# ============================================================
# 20) Save B2 and L1
# ============================================================
sparse.save_npz(f"step6_B2_faces_{PROXY_FIELD}.npz", B2)
sparse.save_npz(f"step6_L1_edge_hodge_{PROXY_FIELD}.npz", L1)
print(f"Saved: step6_B2_faces_{PROXY_FIELD}.npz")
print(f"Saved: step6_L1_edge_hodge_{PROXY_FIELD}.npz")


# ============================================================
# 21) Triangle metadata
# ============================================================
tri_df = pd.DataFrame(triangle_list, columns=["i", "j", "k"])
tri_df["triangle_id"] = np.arange(len(triangle_list), dtype=int)
tri_df.to_csv(f"step6_triangles_{PROXY_FIELD}.csv", index=False)
print(f"Saved: step6_triangles_{PROXY_FIELD}.csv")
