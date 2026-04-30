#!/usr/bin/env python3

"""
Step 04 — Visium Hodge Decomposition (Edge Projection)

Input:
    *_edges_wedge.csv
    *_spots_regions.csv

Output:
    *_edges_hodge.csv

Computes:
    flux_exact (gradient)
    flux_coexact (residual rotational)
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


# ── Main ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges_csv", type=Path, required=True)
    parser.add_argument("--spots_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    args = parser.parse_args()

    edges = pd.read_csv(args.edges_csv)
    spots = pd.read_csv(args.spots_csv)

    n_nodes = len(spots)
    n_edges = len(edges)

    print(f"[nodes] {n_nodes}  [edges] {n_edges}")

    # ── Build incidence matrix B (nodes × edges) ─────────

    row = []
    col = []
    data = []

    for k, e in edges.iterrows():
        i = int(e["i"])
        j = int(e["j"])

        # orientation: i → j
        row.extend([i, j])
        col.extend([k, k])
        data.extend([1, -1])

    B = coo_matrix((data, (row, col)), shape=(n_nodes, n_edges)).tocsr()

    # ── Edge flux vector ────────────────────────────────

    f = edges["flux_wedge"].values

    # ── Solve for potential φ (node scalar field) ───────

    # Solve: B B^T φ = B f

    L = B @ B.T  # graph Laplacian
    rhs = B @ f

    # Regularization (important)
    L = L + 1e-6 * coo_matrix(np.eye(n_nodes))

    print("[solve] Laplacian system")

    phi = spsolve(L, rhs)

    # ── Exact component (gradient) ─────────────────────

    f_exact = B.T @ phi

    # ── Coexact = residual ─────────────────────────────

    f_coexact = f - f_exact

    # ── Diagnostics ───────────────────────────────────

    print("[norms]")
    print(f"||f||       = {np.linalg.norm(f):.3f}")
    print(f"||exact||   = {np.linalg.norm(f_exact):.3f}")
    print(f"||coexact|| = {np.linalg.norm(f_coexact):.3f}")

    # sanity: orthogonality
    dot = np.dot(f_exact, f_coexact)
    print(f"[orthogonality] exact·coexact = {dot:.6f}")

    # ── Save ──────────────────────────────────────────

    edges["flux_exact"] = f_exact
    edges["flux_coexact"] = f_coexact

    edges.to_csv(args.output_csv, index=False)

    print(f"[done] → {args.output_csv}")


if __name__ == "__main__":
    main()
