import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull, Delaunay
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

# Optional dependencies (will fallback if not installed)
try:
    import alphashape
    from shapely.geometry import Point
    HAS_ALPHA = True
except Exception:
    HAS_ALPHA = False

# --------------------------------------------------
# 1) Boundary Proximity Index (BPI)
# --------------------------------------------------

def boundary_distance(xy, alpha=0.01):
    """
    Distance to tissue boundary using alpha-shape if available.
    Falls back to convex hull boundary distances.
    """
    xy = np.asarray(xy, dtype=float)

    # Alpha-shape (preferred)
    if HAS_ALPHA:
        points = [Point(p) for p in xy]
        try:
            boundary = alphashape.alphashape(xy, alpha)
            if boundary.geom_type == "Polygon":
                return np.array([p.distance(boundary.exterior) for p in points], dtype=float)
        except Exception:
            pass  # fallback below

        # alpha-shape failed -> convex hull using shapely via alphashape
        try:
            hull = ConvexHull(xy)
            hull_pts = xy[hull.vertices]
            hull_geom = alphashape.alphashape(hull_pts, 0)
            return np.array([Point(p).distance(hull_geom.exterior) for p in xy], dtype=float)
        except Exception:
            pass

    # Convex hull fallback (no shapely needed)
    hull = ConvexHull(xy)
    hull_pts = xy[hull.vertices]
    # Compute distance to hull edges
    # Simple approximation: distance to nearest hull vertex (fast, adequate for proxy)
    # If you want exact distance-to-edge, ask and I’ll give the exact formula.
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1).fit(hull_pts)
    d, _ = nn.kneighbors(xy)
    return d.reshape(-1)

# --------------------------------------------------
# 2) Local Density Index (LDI)
# --------------------------------------------------

def local_density(xy, k=20, eps=1e-8):
    """
    Inverse mean kNN distance.
    """
    xy = np.asarray(xy, dtype=float)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(xy)
    dists, _ = nbrs.kneighbors(xy)
    mean_knn_dist = dists[:, 1:].mean(axis=1)
    return 1.0 / (mean_knn_dist + eps)

# --------------------------------------------------
# 3) Debug-friendly: load MERFISH cortex if xy not provided
# --------------------------------------------------

def load_merfish_xy():
    import squidpy as sq
    import scanpy as sc

    adata = sq.datasets.merfish()  # MERFISH dataset shipped with squidpy
    # coordinate key can be "spatial" or "X_spatial"
    if "spatial" in adata.obsm:
        xy = np.asarray(adata.obsm["spatial"], dtype=float)
    elif "X_spatial" in adata.obsm:
        xy = np.asarray(adata.obsm["X_spatial"], dtype=float)
    else:
        raise KeyError("MERFISH dataset has no adata.obsm['spatial'] or ['X_spatial'].")

    # We'll also compute PCA so we can make a placeholder residual if needed
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=min(2000, adata.n_vars), subset=True)
    sc.pp.pca(adata, n_comps=5)

    u = adata.obsm["X_pca"][:, 0].astype(float)  # scalar potential for placeholder
    return xy, u

# --------------------------------------------------
# 4) Placeholder residual (only if you don’t have residual_tensor)
# --------------------------------------------------

def placeholder_residual_from_pc1(xy, u):
    """
    Create a *placeholder* conservation-residual-like field:
    build Delaunay edges, flux = u_j - u_i, residual = |div flux|.
    This is NOT your method—only for making Panel E run end-to-end.
    Replace with your model residuals when available.
    """
    tri = Delaunay(xy)
    edges = set()
    for simplex in tri.simplices:
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            i, j = int(simplex[a]), int(simplex[b])
            edges.add((min(i, j), max(i, j)))
    edges = np.array(list(edges), dtype=int)
    i = edges[:, 0]
    j = edges[:, 1]
    f_ij = u[j] - u[i]

    div = np.zeros(xy.shape[0], dtype=float)
    div[i] += f_ij
    div[j] -= f_ij
    return div  # signed; later we take abs()

# --------------------------------------------------
# 5) Helper: binned median + IQR
# --------------------------------------------------

def binned_summary(x, y, bins=20):
    qs = np.quantile(x, np.linspace(0, 1, bins + 1))
    xm, ym, ylo, yhi = [], [], [], []
    for i in range(bins):
        mask = (x >= qs[i]) & (x < qs[i + 1])
        if mask.sum() < 20:
            continue
        xm.append(x[mask].mean())
        ym.append(np.median(y[mask]))
        ylo.append(np.percentile(y[mask], 25))
        yhi.append(np.percentile(y[mask], 75))
    return np.array(xm), np.array(ym), np.array(ylo), np.array(yhi)

# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    # --- Provide xy / residual_tensor if you have them ---
    # Example:
    # xy = ...
    # residual_tensor = ...

    # If xy not defined, load MERFISH + PC1 for placeholder
    if "xy" not in globals():
        print("xy not found -> loading MERFISH (squidpy.datasets.merfish)")
        xy, u_pc1 = load_merfish_xy()
    else:
        u_pc1 = None

    # If residual_tensor not defined, create a placeholder residual
    if "residual_tensor" in globals():
        # PyTorch tensor -> numpy
        r = residual_tensor.detach().cpu().numpy().reshape(-1)
        print("Using residual_tensor from globals().")
    else:
        print("residual_tensor not found -> using placeholder residual from PC1-gradient divergence.")
        r = placeholder_residual_from_pc1(xy, u_pc1).reshape(-1)

    # --- Sanity checks ---
    assert r.ndim == 1, "Residual vector must be 1D"
    assert len(r) == len(xy), "Residual vector must match number of spatial points"

    # --- Compute diagnostics ---
    eps = 1e-8
    residual = np.abs(r) + eps
    log_residual = np.log(residual)

    bpi = boundary_distance(xy)
    ldi = local_density(xy)

    bpi_norm = bpi / (np.max(bpi) + eps)
    ldi_norm = (ldi - ldi.mean()) / (ldi.std() + eps)

    rho_bpi, p_bpi = spearmanr(log_residual, bpi_norm)
    rho_ldi, p_ldi = spearmanr(log_residual, ldi_norm)

    print(f"Residual vs Boundary Distance: Spearman ρ = {rho_bpi:.3f}, p = {p_bpi:.2e}")
    print(f"Residual vs Local Density:     Spearman ρ = {rho_ldi:.3f}, p = {p_ldi:.2e}")

    # --- Plot Panel E ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # (E1) vs boundary distance
    ax = axes[0]
    ax.scatter(bpi_norm, log_residual, s=4, alpha=0.2)
    xm, ym, ylo, yhi = binned_summary(bpi_norm, log_residual)
    ax.plot(xm, ym)
    ax.fill_between(xm, ylo, yhi, alpha=0.3)
    ax.set_xlabel("Normalized distance to boundary")
    ax.set_ylabel("log |conservation residual|")
    ax.set_title("(E1) Boundary proximity")
    ax.text(0.05, 0.95, f"Spearman ρ = {rho_bpi:.2f}\np = {p_bpi:.1e}",
            transform=ax.transAxes, va="top")

    # (E2) vs local density
    ax = axes[1]
    ax.scatter(ldi_norm, log_residual, s=4, alpha=0.2)
    xm, ym, ylo, yhi = binned_summary(ldi_norm, log_residual)
    ax.plot(xm, ym)
    ax.fill_between(xm, ylo, yhi, alpha=0.3)
    ax.set_xlabel("Local density (z-score)")
    ax.set_ylabel("log |conservation residual|")
    ax.set_title("(E2) Local crowding")
    ax.text(0.05, 0.95, f"Spearman ρ = {rho_ldi:.2f}\np = {p_ldi:.1e}",
            transform=ax.transAxes, va="top")

    plt.tight_layout()
    out = "merfish_residual_correlation_panel.png"
    plt.savefig(out, dpi=300)
    plt.show()
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
