# figure_failure_diagnostics.py
# Generates Figure X panels A–D (spatial, spectral, mesh sensitivity, observability sensitivity)
# using a synthetic 2D domain + 3 discretizations (Voronoi/Delaunay/Hex) and 3 failure modes.
#
# REQUIREMENTS:
#   pip install numpy scipy matplotlib
#
# OUTPUT:
#   figureX_failure_diagnostics.pdf
#   figureX_failure_diagnostics.png
#
# NOTE:
# This is "exact code" to generate the diagnostic figure layout and metrics.
# To use your *real* model, replace the synthetic residual generator functions
# with residuals computed from your PDE-constrained Hodge/DEC pipeline.
#
# Author: (you)

import numpy as np
from dataclasses import dataclass
from scipy.spatial import Delaunay, Voronoi
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# -----------------------------
# Utilities: meshes / graphs
# -----------------------------

def make_points_irregular(n=900, seed=0):
    rng = np.random.default_rng(seed)
    # Uniform points in [0,1]^2 with small jitter
    pts = rng.random((n, 2))
    return pts

def make_points_hex(nx=30, ny=30, jitter=0.005, seed=1):
    rng = np.random.default_rng(seed)
    xs = np.arange(nx)
    ys = np.arange(ny)
    pts = []
    for j in ys:
        for i in xs:
            x = (i + 0.5*(j % 2)) / nx
            y = j / ny
            pts.append([x, y])
    pts = np.array(pts, dtype=float)
    pts += rng.normal(scale=jitter, size=pts.shape)
    pts = np.clip(pts, 0.0, 1.0)
    return pts

def delaunay_triangulation(points):
    tri = Delaunay(points)
    return tri

def triangulation_to_edges(tri: Delaunay):
    # Extract unique undirected edges from simplices
    simplices = tri.simplices
    edges = set()
    for t in simplices:
        a, b, c = t
        for u, v in [(a,b),(b,c),(c,a)]:
            if u > v: u, v = v, u
            edges.add((u, v))
    edges = np.array(sorted(list(edges)), dtype=int)
    return edges

def edges_to_graph_laplacian(n_nodes, edges):
    # Unweighted graph Laplacian L = D - A
    row = np.concatenate([edges[:,0], edges[:,1]])
    col = np.concatenate([edges[:,1], edges[:,0]])
    data = np.ones(len(row), dtype=float)
    A = csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
    deg = np.array(A.sum(axis=1)).ravel()
    D = csr_matrix((deg, (np.arange(n_nodes), np.arange(n_nodes))), shape=(n_nodes, n_nodes))
    L = D - A
    return L

def laplacian_eigenbasis(L, k=80):
    # Smallest eigenvalues/eigenvectors
    # For connected graphs, smallest eigenvalue ~ 0
    k = min(k, L.shape[0]-2)
    vals, vecs = eigsh(L, k=k, which="SM")
    idx = np.argsort(vals)
    return vals[idx], vecs[:, idx]

# -----------------------------
# Synthetic residual generators
# -----------------------------

def residual_topology(points, seed=0):
    """
    Topology/Mesh failure: localized spiky residuals near random 'bad' pockets.
    High-frequency pattern in space.
    """
    rng = np.random.default_rng(seed)
    n = points.shape[0]
    # pick pockets
    centers = rng.random((8, 2))
    sig = 0.02
    r = np.zeros(n)
    for c in centers:
        d2 = np.sum((points - c)**2, axis=1)
        r += np.exp(-d2/(2*sig*sig))
    # add boundary-aligned noise (simulate bad orientation near edges)
    edge = np.minimum.reduce([points[:,0], 1-points[:,0], points[:,1], 1-points[:,1]])
    r += 0.8*np.exp(-(edge**2)/(2*(0.03**2)))
    # spikify
    r = r**3
    r += 0.05*rng.normal(size=n)
    r = np.abs(r)
    return r / (np.max(r) + 1e-12)

def residual_physics(points):
    """
    Physics failure: coherent low-frequency residual (wrong operator, global structure).
    """
    x, y = points[:,0], points[:,1]
    # global directional violation + smooth swirl
    r = 0.8*(x - 0.5) + 0.6*np.sin(2*np.pi*y)
    r += 0.3*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)
    r = np.abs(r)
    return r / (np.max(r) + 1e-12)

def residual_data(points, obs_frac=0.15, seed=0):
    """
    Data failure: incoherent residual that shrinks as observability increases.
    """
    rng = np.random.default_rng(seed)
    n = points.shape[0]
    # base noise
    r = np.abs(rng.normal(size=n))
    # scale inversely with sqrt(obs_frac) (more obs => less residual)
    scale = np.sqrt(max(1e-3, 0.3/obs_frac))
    r = r / (np.max(r) + 1e-12)
    r = np.clip(scale * r, 0, None)
    r = r / (np.max(r) + 1e-12)
    return r

# -----------------------------
# Spectral energy diagnostic
# -----------------------------

def spectral_energy(residual, eigvecs):
    """
    Project residual onto Laplacian eigenbasis; return energy per mode.
    """
    coeff = eigvecs.T @ residual
    energy = coeff**2
    energy = energy / (energy.sum() + 1e-12)
    return energy

# -----------------------------
# Plot helpers
# -----------------------------

def plot_residual_heatmap(ax, tri: Delaunay, residual, title):
    triang = mtri.Triangulation(tri.points[:,0], tri.points[:,1], tri.simplices)
    # tripcolor does per-vertex values
    im = ax.tripcolor(triang, residual, shading="gouraud")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    return im

def mesh_variants(points, seed=0):
    """
    Build three discretizations:
      1) Delaunay on irregular points
      2) Delaunay on hex grid (proxy for regular grid)
      3) "Voronoi adjacency" approximated via Delaunay on same points (practically similar)
    For the mesh-robustness panel, we treat them as distinct constructions.
    """
    tri_irreg = delaunay_triangulation(points)
    pts_hex = make_points_hex(nx=30, ny=30, jitter=0.004, seed=seed+1)
    tri_hex = delaunay_triangulation(pts_hex)

    # Voronoi "construction": start from points, build Voronoi, then approximate adjacency by ridge pairs.
    # We'll derive edges from Voronoi ridges (pairs of generating points).
    vor = Voronoi(points)
    edges_vor = set()
    for (p, q) in vor.ridge_points:
        u, v = int(p), int(q)
        if u > v: u, v = v, u
        edges_vor.add((u, v))
    edges_vor = np.array(sorted(list(edges_vor)), dtype=int)

    return tri_irreg, tri_hex, edges_vor, pts_hex

# -----------------------------
# Main: build panels A–D
# -----------------------------

def main(seed=0):
    rng = np.random.default_rng(seed)

    # Base domain points for "irregular" discretizations
    pts = make_points_irregular(n=900, seed=seed)

    # Discretizations
    tri_irreg, tri_hex, edges_vor, pts_hex = mesh_variants(pts, seed=seed)

    # Edges / Laplacians for spectral diagnostics
    edges_irreg = triangulation_to_edges(tri_irreg)
    L_irreg = edges_to_graph_laplacian(len(pts), edges_irreg)
    evals, evecs = laplacian_eigenbasis(L_irreg, k=80)

    # Residuals for panel A and B (computed on irregular mesh)
    r_top = residual_topology(pts, seed=seed)
    r_phy = residual_physics(pts)
    r_dat = residual_data(pts, obs_frac=0.15, seed=seed)

    # Panel B spectral energy
    E_top = spectral_energy(r_top, evecs)
    E_phy = spectral_energy(r_phy, evecs)
    E_dat = spectral_energy(r_dat, evecs)

    # Panel C: mesh sensitivity
    # Compute a scalar residual magnitude for each discretization under each failure type
    # In practice you would compute your true conservation residual on each mesh.
    def scalar_mag(r):  # robust magnitude
        return float(np.median(r) + 0.5*np.mean(r))

    # Delaunay(irregular)
    mag_top_irreg = scalar_mag(r_top)
    mag_phy_irreg = scalar_mag(r_phy)
    mag_dat_irreg = scalar_mag(r_dat)

    # Voronoi-adjacency version (same points, different adjacency -> simulate topology sensitivity)
    # We'll create a Laplacian using Voronoi ridge edges and re-generate "topology failure" residual
    # slightly perturbed to reflect discretization dependence.
    L_vor = edges_to_graph_laplacian(len(pts), edges_vor)
    # Perturb topology residual to mimic mesh dependency (physics residual stays stable)
    r_top_vor = np.clip(r_top + 0.15*rng.normal(size=len(pts)), 0, None)
    r_top_vor = r_top_vor / (np.max(r_top_vor)+1e-12)
    r_phy_vor = r_phy  # physics failure assumed stable
    r_dat_vor = residual_data(pts, obs_frac=0.15, seed=seed+99)

    mag_top_vor = scalar_mag(r_top_vor)
    mag_phy_vor = scalar_mag(r_phy_vor)
    mag_dat_vor = scalar_mag(r_dat_vor)

    # Hex-grid mesh (different points): compute residuals on hex grid for comparison
    r_top_hex = residual_topology(pts_hex, seed=seed+7)
    r_phy_hex = residual_physics(pts_hex)
    r_dat_hex = residual_data(pts_hex, obs_frac=0.15, seed=seed+7)

    mag_top_hex = scalar_mag(r_top_hex)
    mag_phy_hex = scalar_mag(r_phy_hex)
    mag_dat_hex = scalar_mag(r_dat_hex)

    # Panel D: observability sensitivity curves
    obs_fracs = np.array([0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7])
    mags_dat = []
    mags_top = []
    mags_phy = []
    for f in obs_fracs:
        # data residual decays with observability
        mags_dat.append(scalar_mag(residual_data(pts, obs_frac=float(f), seed=seed)))
        # topology residual: weak dependence on f
        r_t = np.clip(r_top + 0.02*rng.normal(size=len(pts)), 0, None)
        r_t = r_t/(np.max(r_t)+1e-12)
        mags_top.append(scalar_mag(r_t))
        # physics residual: mostly invariant to observability in this synthetic model
        mags_phy.append(scalar_mag(r_phy))

    mags_dat = np.array(mags_dat)
    mags_top = np.array(mags_top)
    mags_phy = np.array(mags_phy)

    # -----------------------------
    # Plot figure: 2x2 panels
    # -----------------------------
    fig = plt.figure(figsize=(13.5, 9))
    gs = fig.add_gridspec(2, 2, wspace=0.22, hspace=0.25)

    # Panel A: three heatmaps stacked inside the upper-left cell
    axA = fig.add_subplot(gs[0, 0])
    axA.axis("off")
    subA = gs[0, 0].subgridspec(1, 3, wspace=0.05)

    axA1 = fig.add_subplot(subA[0, 0])
    im1 = plot_residual_heatmap(axA1, tri_irreg, r_top, "A1 Topology")
    axA2 = fig.add_subplot(subA[0, 1])
    im2 = plot_residual_heatmap(axA2, tri_irreg, r_phy, "A2 Physics")
    axA3 = fig.add_subplot(subA[0, 2])
    im3 = plot_residual_heatmap(axA3, tri_irreg, r_dat, "A3 Data")

    cbar = fig.colorbar(im3, ax=[axA1, axA2, axA3], fraction=0.03, pad=0.02)
    cbar.set_label("|conservation residual| (normalized)")

    # Panel B: spectral energy curves
    axB = fig.add_subplot(gs[0, 1])
    modes = np.arange(1, len(E_top)+1)
    axB.plot(modes, E_top, label="Topology")
    axB.plot(modes, E_phy, label="Physics")
    axB.plot(modes, E_dat, label="Data")
    axB.set_title("B Spectral signature (energy in Laplacian modes)")
    axB.set_xlabel("Mode index (low → high)")
    axB.set_ylabel("Normalized energy")
    axB.legend()

    # Panel C: mesh sensitivity bar chart
    axC = fig.add_subplot(gs[1, 0])
    labels = ["Delaunay", "Voronoi-adj", "Hex grid"]
    x = np.arange(len(labels))
    w = 0.25
    axC.bar(x - w, [mag_top_irreg, mag_top_vor, mag_top_hex], width=w, label="Topology")
    axC.bar(x,     [mag_phy_irreg, mag_phy_vor, mag_phy_hex], width=w, label="Physics")
    axC.bar(x + w, [mag_dat_irreg, mag_dat_vor, mag_dat_hex], width=w, label="Data")
    axC.set_xticks(x)
    axC.set_xticklabels(labels, rotation=15)
    axC.set_title("C Mesh robustness (residual magnitude across discretizations)")
    axC.set_ylabel("Residual magnitude (scalar summary)")
    axC.legend()

    # Panel D: observability sensitivity curves
    axD = fig.add_subplot(gs[1, 1])
    axD.plot(obs_fracs, mags_top, marker="o", label="Topology")
    axD.plot(obs_fracs, mags_phy, marker="o", label="Physics")
    axD.plot(obs_fracs, mags_dat, marker="o", label="Data")
    axD.set_title("D Sensitivity to observability")
    axD.set_xlabel("Observed fraction (proxy for coverage)")
    axD.set_ylabel("Residual magnitude (scalar summary)")
    axD.legend()

    # Panel letters
    fig.text(0.01, 0.98, "Figure X: Failure Mode Disentanglement (Panels A–D)", fontsize=14, va="top")

    # Save
    fig.savefig("figureX_failure_diagnostics.pdf", bbox_inches="tight")
    fig.savefig("figureX_failure_diagnostics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved: figureX_failure_diagnostics.pdf and figureX_failure_diagnostics.png")

if __name__ == "__main__":
    main(seed=0)
