import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors

# -------------------------------
# Synthetic test data (DEBUG)
# -------------------------------
np.random.seed(0)
N = 300
xy = np.random.rand(N, 2)
r = np.sin(6 * xy[:, 0]) + 0.3 * np.random.randn(N)

# -------------------------------
# Core functions
# -------------------------------
def build_edges(xy):
    tri = Delaunay(xy)
    edges = set()
    for simplex in tri.simplices:
        for a, b in [(0,1), (1,2), (2,0)]:
            i, j = simplex[a], simplex[b]
            edges.add((min(i,j), max(i,j)))
    return np.array(list(edges))

def gradient_response(xy, r, edges):
    i, j = edges[:,0], edges[:,1]
    return np.mean(np.abs(r[j] - r[i]))

def rotational_response(xy, r):
    tri = Delaunay(xy)
    circ = []
    for simplex in tri.simplices:
        circ.append(np.sum(r[simplex]))
    return np.mean(np.abs(circ))

def harmonic_response(xy, r):
    nbrs = NearestNeighbors(n_neighbors=10).fit(xy)
    _, idx = nbrs.kneighbors(xy)
    local_mean = np.array([r[n].mean() for n in idx])
    return np.mean(np.abs(r - local_mean))

def generator_spectrum(xy, r):
    edges = build_edges(xy)
    G = gradient_response(xy, r, edges)
    R = rotational_response(xy, r)
    H = harmonic_response(xy, r)
    spec = np.array([G, R, H])
    return spec / (np.linalg.norm(spec) + 1e-8)

# -------------------------------
# Plot
# -------------------------------
base = generator_spectrum(xy, r)
pert = generator_spectrum(xy + 0.02*np.random.randn(*xy.shape), r)

labels = ["Gradient", "Rotational", "Harmonic"]
x = np.arange(3)

plt.figure(figsize=(6,4))
plt.bar(x-0.15, base, width=0.3, label="Original")
plt.bar(x+0.15, pert, width=0.3, label="Perturbed")
plt.xticks(x, labels)
plt.ylabel("Normalized response")
plt.title("Lie-algebra diagnostic (DEBUG)")
plt.legend()
plt.tight_layout()
plt.show()
