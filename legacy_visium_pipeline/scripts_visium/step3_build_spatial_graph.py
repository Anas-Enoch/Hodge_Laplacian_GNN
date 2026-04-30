from pathlib import Path
import json

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from PIL import Image

from scipy import sparse
from sklearn.neighbors import NearestNeighbors


# ============================================================
# 1) Paths
# ============================================================
DATA_DIR = Path(".")
H5_FILE = DATA_DIR / "Visium_Human_Breast_Cancer_filtered_feature_bc_matrix.h5"
SPATIAL_DIR = DATA_DIR / "spatial"


# ============================================================
# 2) Load expression matrix
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

# Keep only barcodes present in adata, preserve order of adata.obs_names
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
# 4) Restrict to in-tissue spots
# ============================================================
in_tissue_mask = adata.obs["in_tissue"].astype(bool).values
obs = adata.obs.copy()

obs_in = obs.loc[in_tissue_mask].copy()
barcodes = obs_in.index.to_numpy()

# Full-resolution pixel coordinates
x = obs_in["x_fullres"].to_numpy(dtype=float)
y = obs_in["y_fullres"].to_numpy(dtype=float)

coords = np.column_stack([x, y])

n_nodes = coords.shape[0]
print("In-tissue spots:", n_nodes)


# ============================================================
# 5) Build kNN graph
# ============================================================
# For Visium, k=6 is a reasonable first pass.
k = 6

nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
nbrs.fit(coords)
distances, indices = nbrs.kneighbors(coords)

# Remove self-neighbor at column 0
neighbor_indices = indices[:, 1:]
neighbor_distances = distances[:, 1:]

print("neighbor_indices shape:", neighbor_indices.shape)
print("neighbor_distances shape:", neighbor_distances.shape)


# ============================================================
# 6) Build unique undirected edges
# ============================================================
# Important logic:
# - kNN gives directed neighbor relations
# - we convert them to unique undirected edges by storing (min(i,j), max(i,j))
# - later we impose an orientation tail->head using i<j
edge_set = set()

for i in range(n_nodes):
    for j in neighbor_indices[i]:
        a = min(i, j)
        b = max(i, j)
        if a != b:
            edge_set.add((a, b))

edge_list = sorted(edge_set)
n_edges = len(edge_list)

print("Unique undirected edges:", n_edges)


# ============================================================
# 7) Build edge dataframe with geometry
# ============================================================
edge_rows = []

for e_id, (i, j) in enumerate(edge_list):
    xi, yi = coords[i]
    xj, yj = coords[j]

    dx = xj - xi
    dy = yj - yi
    length = float(np.sqrt(dx * dx + dy * dy))

    edge_rows.append(
        {
            "edge_id": e_id,
            "tail": i,
            "head": j,
            "tail_barcode": barcodes[i],
            "head_barcode": barcodes[j],
            "x_tail": xi,
            "y_tail": yi,
            "x_head": xj,
            "y_head": yj,
            "dx": dx,
            "dy": dy,
            "length": length,
        }
    )

edges_df = pd.DataFrame(edge_rows)
print("edges_df shape:", edges_df.shape)


# ============================================================
# 8) Build node dataframe
# ============================================================
nodes_df = pd.DataFrame(
    {
        "node_id": np.arange(n_nodes, dtype=int),
        "barcode": barcodes,
        "x_fullres": x,
        "y_fullres": y,
        "array_row": obs_in["array_row"].to_numpy(),
        "array_col": obs_in["array_col"].to_numpy(),
    }
)

print("nodes_df shape:", nodes_df.shape)


# ============================================================
# 9) Build oriented node-edge incidence matrix B1
# ============================================================
# Convention:
# - each edge e = (tail=i, head=j) with i<j
# - B1[node, edge] = -1 at tail, +1 at head
#
# This is a deductible linear-algebra convention we choose for the script,
# not a Scanpy API contract.
rows = []
cols = []
data = []

for e_id, (i, j) in enumerate(edge_list):
    rows.extend([i, j])
    cols.extend([e_id, e_id])
    data.extend([-1.0, +1.0])

B1 = sparse.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_edges)).tocsr()

print("B1 shape:", B1.shape)


# ============================================================
# 10) Build simple weighted adjacency and graph Laplacian
# ============================================================
# Weight choice:
# w_ij = 1 / length
#
# This is just a first-pass geometric weighting.
adj_rows = []
adj_cols = []
adj_data = []

for _, row in edges_df.iterrows():
    i = int(row["tail"])
    j = int(row["head"])
    length = float(row["length"])

    # Avoid divide-by-zero, though it should not happen here
    w = 1.0 / max(length, 1e-8)

    adj_rows.extend([i, j])
    adj_cols.extend([j, i])
    adj_data.extend([w, w])

W = sparse.coo_matrix((adj_data, (adj_rows, adj_cols)), shape=(n_nodes, n_nodes)).tocsr()
degree = np.array(W.sum(axis=1)).ravel()
D = sparse.diags(degree)
L = D - W

print("Adjacency W shape:", W.shape)
print("Graph Laplacian L shape:", L.shape)


# ============================================================
# 11) Save CSV outputs
# ============================================================
nodes_df.to_csv("step3_nodes.csv", index=False)
edges_df.to_csv("step3_edges.csv", index=False)

print("Saved: step3_nodes.csv")
print("Saved: step3_edges.csv")


# ============================================================
# 12) Save sparse matrices
# ============================================================
# Save as .npz sparse matrices
sparse.save_npz("step3_B1_incidence.npz", B1)
sparse.save_npz("step3_W_adjacency.npz", W)
sparse.save_npz("step3_L_graph_laplacian.npz", L)

print("Saved: step3_B1_incidence.npz")
print("Saved: step3_W_adjacency.npz")
print("Saved: step3_L_graph_laplacian.npz")


# ============================================================
# 13) Save combined light metadata file
# ============================================================
np.savez(
    "step3_graph_matrices.npz",
    x=x,
    y=y,
    barcodes=barcodes,
    n_nodes=n_nodes,
    n_edges=n_edges,
    hires_scale=hires_scale,
)

print("Saved: step3_graph_matrices.npz")


# ============================================================
# 14) Plot graph over tissue image
# ============================================================
fig, ax = plt.subplots(figsize=(10, 9))
ax.imshow(img)

# Draw a subset of edges if you want faster plotting on large graphs.
# Here we draw all edges because this graph is still manageable.
for _, row in edges_df.iterrows():
    ax.plot(
        [row["x_tail"] * hires_scale, row["x_head"] * hires_scale],
        [row["y_tail"] * hires_scale, row["y_head"] * hires_scale],
        linewidth=0.25,
        alpha=0.20,
    )

ax.scatter(
    x * hires_scale,
    y * hires_scale,
    s=8,
    alpha=0.8,
)

ax.set_title("Step 3 spatial graph")
ax.invert_yaxis()
ax.axis("off")

plt.tight_layout()
plt.savefig("step3_spatial_graph.png", dpi=220)
print("Saved: step3_spatial_graph.png")


# ============================================================
# 15) Print summary diagnostics
# ============================================================
edge_lengths = edges_df["length"].to_numpy()
print("\nGraph summary")
print("-------------")
print("Nodes:", n_nodes)
print("Edges:", n_edges)
print("Mean edge length:", float(edge_lengths.mean()))
print("Median edge length:", float(np.median(edge_lengths)))
print("Min edge length:", float(edge_lengths.min()))
print("Max edge length:", float(edge_lengths.max()))

node_degrees = np.diff(W.indptr)
print("Mean node degree:", float(node_degrees.mean()))
print("Median node degree:", float(np.median(node_degrees)))
print("Min node degree:", int(node_degrees.min()))
print("Max node degree:", int(node_degrees.max()))
