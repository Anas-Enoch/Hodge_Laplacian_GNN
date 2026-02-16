import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors

# --------------------------------------------------
# 1. Build local neighborhood graph (mesh-free, safe)
# --------------------------------------------------

def build_edges(xy):
    """
    Use Delaunay triangulation to define local adjacency.
    """
    tri = Delaunay(xy)
    edges = set()
    for simplex in tri.simplices:
        for a, b in [(0,1), (1,2), (2,0)]:
            i, j = simplex[a], simplex[b]
            edges.add((min(i,j), max(i,j)))
    return np.array(list(edges), dtype=int)

# --------------------------------------------------
# 2. Generator probes
# --------------------------------------------------

def gradient_response(xy, r, edges):
    """
    Gradient-like (compressive) response:
    magnitude of local residual differences.
    """
    i, j = edges[:,0], edges[:,1]
    return np.mean(np.abs(r[j] - r[i]))

def rotational_response(xy, r, edges):
    """
    Rotation-like (divergence-free) proxy:
    signed residual circulation around triangles.
    """
    tri = Delaunay(xy)
    circ = []
    for simplex in tri.simplices:
        vals = r[simplex]
        circ.append(vals[0] + vals[1] + vals[2])
    return np.mean(np.abs(circ))

def harmonic_response(xy, r, edges):
    """
    Harmonic/topological proxy:
    residual variance unexplained by local averaging.
    """
    nbrs = NearestNeighbors(n_neighbors=10).fit(xy)
    _, idx = nbrs.kneighbors(xy)
    local_mean = np.array([r[neigh].mean() for neigh in idx])
    return np.mean(np.abs(r - local_mean))

# --------------------------------------------------
# 3. Compute generator spectrum
# --------------------------------------------------

def generator_spectrum(xy, r):
    edges = build_edges(xy)

    G = gradient_response(xy, r, edges)
    R = rotational_response(xy, r, edges)
    H = harmonic_response(xy, r, edges)

    spec = np.array([G, R, H])
    return spec / (np.linalg.norm(spec) + 1e-8)

# --------------------------------------------------
# 4. Mesh perturbation (topology sensitivity test)
# --------------------------------------------------

def jitter_xy(xy, scale=0.02):
    return xy + scale * np.random.randn(*xy.shape)

# --------------------------------------------------
# 5. Generate Figure
# --------------------------------------------------

def plot_lie_diagnostic(xy, r):
    base = generator_spectrum(xy, r)
    perturbed = generator_spectrum(jitter_xy(xy), r)

    labels = ["Gradient", "Rotational", "Harmonic"]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(8,4), sharey=True)

    # (B) Physics-induced (invariant)
    axes[0].bar(x, base)
    axes[0].set_title("(B) Generator response")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Normalized response")

    # (C) Topology sensitivity
    axes[1].bar(x - 0.15, base, width=0.3, label="Original")
    axes[1].bar(x + 0.15, perturbed, width=0.3, label="Perturbed mesh")
    axes[1].set_title("(C) Mesh perturbation")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("lie_algebra_diagnostic.png", dpi=300)
    plt.show()
# --------------------------------------------------
# USAGE
# --------------------------------------------------
# xy = ...
# r  = ...
# plot_lie_diagnostic(xy, r)
