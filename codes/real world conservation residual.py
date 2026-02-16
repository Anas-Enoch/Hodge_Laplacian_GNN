# real_world_residual_figure.py
import numpy as np
import matplotlib.pyplot as plt

import scanpy as sc
import squidpy as sq
from scipy.spatial import Delaunay
from scipy.sparse import coo_matrix

def build_delaunay_adjacency(xy):
    tri = Delaunay(xy)
    edges = set()
    for simplex in tri.simplices:
        for a, b in [(0,1), (1,2), (2,0)]:
            i, j = simplex[a], simplex[b]
            if i != j:
                edges.add((min(i,j), max(i,j)))
    rows, cols = zip(*list(edges))
    n = xy.shape[0]
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A = A + A.T  # undirected adjacency
    return A.tocsr(), np.array(list(edges), dtype=int)

def compute_placeholder_flux_from_gradient(X, edges):
    """
    Placeholder: define a scalar potential u_i from first PC (or total counts),
    then set flux f_ij = u_j - u_i along each edge (a gradient flow).
    Replace this with your model's learned 1-form flux.
    """
    u = X[:, 0].astype(float)  # e.g., PC1
    i = edges[:, 0]
    j = edges[:, 1]
    f_ij = u[j] - u[i]
    # oriented flux for both directions:
    return u, i, j, f_ij

def divergence_from_oriented_flux(n, i, j, f_ij):
    """
    For each undirected edge (i,j) with oriented flux f_ij (i -> j),
    divergence at node k: sum outgoing.
    """
    div = np.zeros(n, dtype=float)
    div[i] += f_ij
    div[j] -= f_ij
    return div

def main():
    # 1) Load public Visium dataset (mouse brain)
    adata = sq.datasets.visium_hne_adata()  # if this fails, switch to another squidpy dataset
    # If you prefer mouse brain specifically, use:
    # adata = sq.datasets.visium_mouse_brain()  # depending on squidpy version

    # 2) Basic preprocessing for a stable potential (PC1)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
    sc.pp.pca(adata, n_comps=5)

    xy = adata.obsm["spatial"].astype(float)
    Xpca = adata.obsm["X_pca"]

    # 3) Build mesh (Delaunay) and edges
    A, edges = build_delaunay_adjacency(xy)

    # 4) Placeholder flux and residual divergence (replace with your inferred flux)
    u, i, j, f_ij = compute_placeholder_flux_from_gradient(Xpca, edges)
    r = divergence_from_oriented_flux(xy.shape[0], i, j, f_ij)
    r_abs = np.abs(r)

    # 5) Plot figure panels A–D
    fig = plt.figure(figsize=(12, 10))

    # (A) raw coords (color by total counts)
    ax1 = fig.add_subplot(2, 2, 1)
    counts = np.array(adata.X.sum(axis=1)).ravel()
    scA = ax1.scatter(xy[:,0], xy[:,1], c=counts, s=8)
    ax1.set_title("(A) Spatial coordinates (total counts)")
    ax1.set_aspect("equal")
    ax1.invert_yaxis()
    plt.colorbar(scA, ax=ax1, fraction=0.046, pad=0.04)

    # (B) mesh overlay
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(xy[:,0], xy[:,1], c="k", s=2, alpha=0.5)
    # draw a subset of edges for clarity
    for (a,b) in edges[::3]:
        ax2.plot([xy[a,0], xy[b,0]], [xy[a,1], xy[b,1]], linewidth=0.2, alpha=0.3)
    ax2.set_title("(B) Delaunay mesh (edges)")
    ax2.set_aspect("equal")
    ax2.invert_yaxis()

    # (C) residual heatmap
    ax3 = fig.add_subplot(2, 2, 3)
    scC = ax3.scatter(xy[:,0], xy[:,1], c=r_abs, s=8)
    ax3.set_title("(C) Conservation residual |div flux|")
    ax3.set_aspect("equal")
    ax3.invert_yaxis()
    plt.colorbar(scC, ax=ax3, fraction=0.046, pad=0.04)

    # (D) overlay + boundary proxy (no labels required)
    ax4 = fig.add_subplot(2, 2, 4)
    scD = ax4.scatter(xy[:,0], xy[:,1], c=r_abs, s=8)
    ax4.set_title("(D) Residual overlay with boundary proxy")
    ax4.set_aspect("equal")
    ax4.invert_yaxis()

    # Simple boundary proxy: mark convex hull points (visual cue)
    # This is NOT a biological label—just a structural boundary indicator.
    from scipy.spatial import ConvexHull
    hull = ConvexHull(xy)
    hull_pts = xy[hull.vertices]
    ax4.plot(np.r_[hull_pts[:,0], hull_pts[0,0]], np.r_[hull_pts[:,1], hull_pts[0,1]], linewidth=1.0)
    plt.colorbar(scD, ax=ax4, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig("real_world_conservation_residual.png", dpi=300)
    plt.close(fig)

    print("Saved: real_world_conservation_residual.png")

if __name__ == "__main__":
    main()
