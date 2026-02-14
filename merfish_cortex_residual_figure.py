# merfish_cortex_residual_figure.py
import numpy as np
import matplotlib.pyplot as plt

import scanpy as sc
import squidpy as sq
from scipy.spatial import Delaunay, ConvexHull
from scipy.sparse import coo_matrix

def get_spatial_coords(adata):
    # Common keys in spatial AnnData objects
    for k in ["spatial", "X_spatial"]:
        if k in adata.obsm:
            return np.asarray(adata.obsm[k]).astype(float)
    raise KeyError("No spatial coordinates found in adata.obsm['spatial'] or adata.obsm['X_spatial'].")

def build_delaunay_edges(xy):
    tri = Delaunay(xy)
    edges = set()
    for simplex in tri.simplices:
        for a, b in [(0,1), (1,2), (2,0)]:
            i, j = int(simplex[a]), int(simplex[b])
            if i != j:
                edges.add((min(i,j), max(i,j)))
    edges = np.array(list(edges), dtype=int)
    return edges

def compute_placeholder_flux_from_gradient(u, edges):
    """
    Placeholder oriented edge flux:
    f_ij = u_j - u_i for each undirected edge (i,j), oriented i->j.
    Replace u and/or f_ij with your model's inferred 1-form flux.
    """
    i = edges[:, 0]
    j = edges[:, 1]
    f_ij = u[j] - u[i]
    return i, j, f_ij

def divergence_from_oriented_flux(n, i, j, f_ij):
    div = np.zeros(n, dtype=float)
    div[i] += f_ij
    div[j] -= f_ij
    return div

def main():
    # 1) Load MERFISH dataset (preprocessed subset from Moffitt et al.)
    adata = sq.datasets.merfish()  # provided by Squidpy datasets API
    xy = get_spatial_coords(adata)

    # 2) Define a stable scalar potential u for the placeholder flux
    #    (Use PC1 from expression; replace with your learned latent field if available.)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=min(2000, adata.n_vars), subset=True)
    sc.pp.pca(adata, n_comps=5)
    u = adata.obsm["X_pca"][:, 0].astype(float)

    # 3) Build a reasonable mesh (Delaunay)
    edges = build_delaunay_edges(xy)

    # 4) Placeholder oriented flux + conservation residual (replace with your inferred flux)
    i, j, f_ij = compute_placeholder_flux_from_gradient(u, edges)
    r = divergence_from_oriented_flux(xy.shape[0], i, j, f_ij)
    r_abs = np.abs(r)

    # 5) Plot panels A–D
    fig = plt.figure(figsize=(12, 10))

    # (A) raw coords (color by total counts)
    ax1 = fig.add_subplot(2, 2, 1)
    counts = np.array(adata.X.sum(axis=1)).ravel()
    scA = ax1.scatter(xy[:, 0], xy[:, 1], c=counts, s=6)
    ax1.set_title("(A) MERFISH coordinates (total counts)")
    ax1.set_aspect("equal")
    ax1.invert_yaxis()
    plt.colorbar(scA, ax=ax1, fraction=0.046, pad=0.04)

    # (B) mesh overlay (thin edges)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(xy[:, 0], xy[:, 1], c="k", s=1, alpha=0.5)
    for (a, b) in edges[::4]:
        ax2.plot([xy[a, 0], xy[b, 0]], [xy[a, 1], xy[b, 1]], linewidth=0.2, alpha=0.25)
    ax2.set_title("(B) Delaunay mesh (edges)")
    ax2.set_aspect("equal")
    ax2.invert_yaxis()

    # (C) residual heatmap
    ax3 = fig.add_subplot(2, 2, 3)
    scC = ax3.scatter(xy[:, 0], xy[:, 1], c=r_abs, s=6)
    ax3.set_title("(C) Conservation residual |div flux|")
    ax3.set_aspect("equal")
    ax3.invert_yaxis()
    plt.colorbar(scC, ax=ax3, fraction=0.046, pad=0.04)

    # (D) overlay + boundary proxy (convex hull) as a structural reference (not a biological label)
    ax4 = fig.add_subplot(2, 2, 4)
    scD = ax4.scatter(xy[:, 0], xy[:, 1], c=r_abs, s=6)
    ax4.set_title("(D) Residual overlay + structural boundary proxy")
    ax4.set_aspect("equal")
    ax4.invert_yaxis()
    hull = ConvexHull(xy)
    hp = xy[hull.vertices]
    ax4.plot(np.r_[hp[:, 0], hp[0, 0]], np.r_[hp[:, 1], hp[0, 1]], linewidth=1.0)
    plt.colorbar(scD, ax=ax4, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig("merfish_real_world_conservation_residual.png", dpi=300)
    plt.close(fig)
    print("Saved: merfish_real_world_conservation_residual.png")

if __name__ == "__main__":
    main()
