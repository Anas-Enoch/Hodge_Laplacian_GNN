#!/usr/bin/env python3

"""
Step 08 — Interface-Restricted Zeta Diagnostic

Input:
    *_spots_coexact_energy.csv
    *_edges_hodge.csv

Output:
    *_zeta_interface.csv

Computes:
    Z_global
    Z_interface
    ratio = Z_interface / Z_global
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh


def build_laplacian(edges: pd.DataFrame, n_nodes: int):
    rows = []
    cols = []
    data = []

    for _, e in edges.iterrows():
        i = int(e["i"])
        j = int(e["j"])

        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([1.0, 1.0])

    A = coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    L = diags(deg) - A
    return L


def compute_zeta(signal: np.ndarray, L, k: int = 50) -> tuple[float, float, int]:
    """
    Compute normalized Zeta spectral concentration statistic.

    Z = [Σ_k α_k λ_k^{-1}] / [Σ_k α_k]

    where α_k = <signal, φ_k>^2 are spectral projection coefficients
    onto the k smallest nonzero eigenmodes of L.

    Normalization by Σ_k α_k ensures Z is scale-invariant and comparable
    across sections of different size and signal magnitude.

    NOTE — truncation: only the k smallest nonzero eigenmodes are used
    (default k=50). This is an approximation: the full spectrum would
    require dense eigendecomposition (O(N^3)). The truncated form reliably
    captures low-frequency structure but underestimates the denominator
    Σ_k α_k when high-frequency modes carry substantial signal. Results
    should not be compared numerically to the full-spectrum normalized
    Z(s=1) computed in the main TNBC pipeline (GSE210616), which uses
    complete eigendecomposition. Hypothesis tests (null model comparison)
    are unaffected because the same truncation is applied to observed and
    permuted signals.

    Returns
    -------
    z_normalized : float
        Normalized Zeta. Z > 1 indicates low-frequency concentration.
    alpha_sum    : float
        Total spectral mass captured (denominator). Values near total
        signal energy confirm truncation captures most variance.
    k_used       : int
        Number of nonzero eigenmodes actually used.
    """
    n = L.shape[0]
    k_eff = min(k, n - 2)

    if k_eff < 2:
        return np.nan, np.nan, 0

    vals, vecs = eigsh(L, k=k_eff, which="SM")

    numerator   = 0.0
    alpha_sum   = 0.0
    k_used      = 0

    for lam, u in zip(vals, vecs.T):
        if lam < 1e-8:          # skip zero modes (harmonic / constant)
            continue
        proj       = float(np.dot(signal, u))
        alpha_k    = proj * proj
        numerator  += alpha_k / lam
        alpha_sum  += alpha_k
        k_used     += 1

    if alpha_sum < 1e-12:
        return 0.0, 0.0, k_used

    z_normalized = numerator / alpha_sum
    return float(z_normalized), float(alpha_sum), k_used


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spots_csv", type=Path, required=True)
    parser.add_argument("--edges_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, default=None)
    parser.add_argument("--k-eigs", type=int, default=50)
    args = parser.parse_args()

    spots = pd.read_csv(args.spots_csv)
    edges = pd.read_csv(args.edges_csv)

    required_spots = {"coexact_energy", "region"}
    required_edges = {"i", "j"}

    if not required_spots <= set(spots.columns):
        raise ValueError(f"spots missing columns: {required_spots - set(spots.columns)}")

    if not required_edges <= set(edges.columns):
        raise ValueError(f"edges missing columns: {required_edges - set(edges.columns)}")

    sample = args.spots_csv.name.replace("_spots_coexact_energy.csv", "")

    n_nodes = len(spots)
    L = build_laplacian(edges, n_nodes)

    signal_global = spots["coexact_energy"].to_numpy(float)

    interface_mask = (spots["region"] == "interface").to_numpy()
    signal_interface = signal_global * interface_mask

    Z_global,    alpha_sum_global,    k_global    = compute_zeta(signal_global,    L, k=args.k_eigs)
    Z_interface, alpha_sum_interface, k_interface = compute_zeta(signal_interface, L, k=args.k_eigs)

    ratio = Z_interface / (Z_global + 1e-12)

    out = pd.DataFrame({
        "sample":                  [sample],
        "n_nodes":                 [n_nodes],
        "n_interface":             [int(interface_mask.sum())],
        "Z_global":                [Z_global],
        "Z_interface":             [Z_interface],
        "Z_interface_over_global": [ratio],
        # diagnostic columns — document truncation quality
        "alpha_sum_global":        [alpha_sum_global],
        "alpha_sum_interface":     [alpha_sum_interface],
        "k_eigenmodes_used":       [k_global],
        "k_requested":             [args.k_eigs],
        "normalization":           ["normalized_alpha_sum"],
        "truncation_note":         [
            "k=50 truncated approximation; not numerically comparable "
            "to full-spectrum Z(s=1) in GSE210616 pipeline"
        ],
    })

    if args.output_csv is None:
        output_csv = args.spots_csv.parent / f"{sample}_zeta_interface.csv"
    else:
        output_csv = args.output_csv

    out.to_csv(output_csv, index=False)

    print(f"[sample] {sample}")
    print(f"[nodes] {n_nodes}")
    print(f"[interface nodes] {int(interface_mask.sum())}")
    print(f"Z_global (normalized)    = {Z_global:.6f}")
    print(f"Z_interface (normalized) = {Z_interface:.6f}")
    print(f"Z_interface/Z_global     = {ratio:.6f}")
    print(f"alpha_sum_global         = {alpha_sum_global:.4e}  (spectral mass captured)")
    print(f"k_eigenmodes_used        = {k_global}")
    print(f"[note] Truncated k={args.k_eigs} approximation — see normalization column")
    print(f"[done] → {output_csv}")


if __name__ == "__main__":
    main()
