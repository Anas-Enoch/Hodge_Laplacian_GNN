import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.spatial import Delaunay
    SCIPY_OK = True
except Exception as e:
    SCIPY_OK = False
    _SCIPY_ERR = e

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def dedup_xy(xy, *arrays):
    """
    Deduplicate identical coordinate rows in xy.
    Keeps first occurrence. Applies same filtering to arrays.
    """
    xy = np.asarray(xy, dtype=float)
    key = np.round(xy, decimals=6)  # helps near-duplicates
    _, idx = np.unique(key, axis=0, return_index=True)
    idx = np.sort(idx)
    out = [xy[idx]]
    for a in arrays:
        if a is None:
            out.append(None)
        else:
            a = np.asarray(a).reshape(-1)
            out.append(a[idx])
    return out

def build_knn_edges(xy, k=6):
    """
    Pure numpy kNN (O(N^2), OK up to ~20k; for larger use sklearn/annoy/faiss).
    """
    xy = np.asarray(xy, float)
    N = xy.shape[0]
    # squared distances
    d2 = np.sum((xy[:, None, :] - xy[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    nn = np.argpartition(d2, kth=k, axis=1)[:, :k]
    edges = set()
    for i in range(N):
        for j in nn[i]:
            edges.add(tuple(sorted((i, int(j)))))
    return np.array(list(edges), dtype=int)

def build_delaunay_edges(xy):
    """
    Returns unique undirected edges from Delaunay triangulation.
    Robust to precision issues with QJ.
    """
    if not SCIPY_OK:
        raise ImportError(f"scipy not available: {_SCIPY_ERR}")

    tri = Delaunay(xy, qhull_options="QJ")  # <-- critical robustness
    simplices = tri.simplices
    edges = set()
    for a, b, c in simplices:
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((c, a))))
    return np.array(list(edges), dtype=int)

def scatter_heat(ax, xy, val, title, cbar=True, s=6):
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=val, s=s)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    if cbar:
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

def plot_edges(ax, xy, edges, lw=0.15, alpha=0.35):
    ax.scatter(xy[:, 0], xy[:, 1], s=2)
    for i, j in edges:
        ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], linewidth=lw, alpha=alpha)
    ax.set_title("(B) Mesh (edges)")
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])

def energy_fractions(grad_res, rot_res, harm_res, eps=1e-12):
    Eg = float(np.sum(grad_res**2))
    Er = float(np.sum(rot_res**2))
    Eh = float(np.sum(harm_res**2))
    Et = Eg + Er + Eh + eps
    return np.array([Eg/Et, Er/Et, Eh/Et]), np.array([Eg, Er, Eh, Et])

# ------------------------------------------------------------
# MAIN FIGURE
# ------------------------------------------------------------
def make_transport_phenotyping_figure(
    xy, grad_res, rot_res, harm_res, counts=None,
    out_pdf="fig_merfish_transport_phenotyping.pdf",
    mesh="delaunay", knn_k=6
):
    # ---- sanitize / validate
    xy = np.asarray(xy, dtype=float)
    grad_res = np.asarray(grad_res, dtype=float).reshape(-1)
    rot_res  = np.asarray(rot_res, dtype=float).reshape(-1)
    harm_res = np.asarray(harm_res, dtype=float).reshape(-1)
    if counts is not None:
        counts = np.asarray(counts, dtype=float).reshape(-1)

    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy must be (N,2). Got {xy.shape}")

    N = xy.shape[0]
    for name, arr in [("grad_res", grad_res), ("rot_res", rot_res), ("harm_res", harm_res)]:
        if arr.shape[0] != N:
            raise ValueError(f"{name} length {arr.shape[0]} != N={N}")
    if counts is not None and counts.shape[0] != N:
        raise ValueError(f"counts length {counts.shape[0]} != N={N}")

    if np.any(~np.isfinite(xy)):
        raise ValueError("xy contains NaN/Inf")
    for name, arr in [("grad_res", grad_res), ("rot_res", rot_res), ("harm_res", harm_res)]:
        if np.any(~np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN/Inf")

    # ---- deduplicate coordinates (common in MERFISH/Visium exports)
    xy, grad_res, rot_res, harm_res, counts = dedup_xy(xy, grad_res, rot_res, harm_res, counts)
    N = xy.shape[0]

    # ---- build mesh edges
    edges = None
    if mesh.lower() == "delaunay":
        try:
            edges = build_delaunay_edges(xy)
        except Exception as e:
            print(f"[WARN] Delaunay failed ({type(e).__name__}: {e}). Falling back to kNN mesh.")
            edges = build_knn_edges(xy, k=knn_k)
    else:
        edges = build_knn_edges(xy, k=knn_k)

    fracs, energies = energy_fractions(grad_res, rot_res, harm_res)

    fig = plt.figure(figsize=(10.5, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8])

    axA = fig.add_subplot(gs[0, 0])
    if counts is None:
        scatter_heat(axA, xy, np.ones(N), "(A) Spatial coordinates (cells)", cbar=False, s=6)
    else:
        scatter_heat(axA, xy, counts, "(A) Spatial coordinates (total counts)", cbar=True, s=6)

    axB = fig.add_subplot(gs[0, 1])
    plot_edges(axB, xy, edges)

    axC = fig.add_subplot(gs[1, 0])
    scatter_heat(axC, xy, grad_res, r"(C) Gradient residual $\|\mathrm{d}\alpha\|$")

    axD = fig.add_subplot(gs[1, 1])
    scatter_heat(axD, xy, rot_res, r"(D) Rotational residual $\|\delta\beta\|$")

    axE = fig.add_subplot(gs[2, 0])
    scatter_heat(axE, xy, harm_res, r"(E) Harmonic component $\|\gamma\|$")

    axF = fig.add_subplot(gs[2, 1])
    labels = ["Gradient", "Rotational", "Harmonic"]
    axF.bar(labels, fracs)
    axF.set_ylim(0, 1.0)
    axF.set_title("(F) Fraction of residual energy")
    axF.set_ylabel("Energy fraction")
    axF.text(
        0.02, 0.98,
        f"E_total = {energies[3]:.3e}\nE_g = {energies[0]:.3e}\nE_r = {energies[1]:.3e}\nE_h = {energies[2]:.3e}",
        transform=axF.transAxes, va="top"
    )

    fig.suptitle("Transport phenotyping on MERFISH cortex", y=0.995, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_pdf}")

# ------------------------------------------------------------
# CALL IT (YOU MUST UNCOMMENT AND PROVIDE REAL ARRAYS)
# ------------------------------------------------------------
if __name__ == "__main__":
    import os
    print("CWD =", os.getcwd())

    # ---- TODO: uncomment and adapt to your real objects
    # Example using AnnData (common for MERFISH/Visium):
    # xy = adata.obsm["spatial"].astype(float)
    # counts = np.asarray(adata.obs["total_counts"]).astype(float)
    # grad_res = ...
    # rot_res  = ...
    # harm_res = ...

    # TEMP sanity-check (delete later): makes sure the PDF pipeline works
    xy = np.random.rand(2000, 2) * 1000
    counts = np.random.gamma(5, 50, size=2000)
    grad_res = np.abs(np.random.randn(2000))
    rot_res  = np.abs(np.random.randn(2000))
    harm_res = np.abs(np.random.randn(2000))

    make_transport_phenotyping_figure(
        xy, grad_res, rot_res, harm_res,
        counts=counts,
        out_pdf="fig_merfish_transport_phenotyping.pdf",
        mesh="delaunay",   # will fall back to kNN if needed
        knn_k=6
    )

    print("Wrote fig_merfish_transport_phenotyping.pdf in", os.getcwd())
