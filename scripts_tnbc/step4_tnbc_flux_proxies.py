from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def compute_flux(a_i, a_j, b_i, b_j, length):
    """
    Residualized antisymmetric wedge-product proxy

    f_ij = (a_i * b_j - a_j * b_i) / length
    """
    eps = 1e-8
    return (a_i * b_j - a_j * b_i) / (length + eps)


def build_node_laplacian(B1: sparse.csr_matrix) -> sparse.csr_matrix:
    """
    L0 = B1 B1^T
    """
    return (B1 @ B1.T).tocsr()


def smooth_field(field: np.ndarray, L0: sparse.csr_matrix, lam: float = 1.0) -> np.ndarray:
    """
    Solve graph-diffusive smoothing:
        (I + lam * L0) u_smooth = u
    """
    n = L0.shape[0]
    A = sparse.eye(n, format="csr") + lam * L0
    u_smooth = spsolve(A, field)
    return np.asarray(u_smooth).ravel()


def residualize_field(field: np.ndarray, L0: sparse.csr_matrix, lam: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        smooth field
        residual field = field - smooth
    """
    smooth = smooth_field(field, L0, lam=lam)
    resid = field - smooth
    return smooth, resid


def summarize_vector(name: str, x: np.ndarray) -> None:
    print(f"{name:28s} mean={np.mean(x): .6e}  std={np.std(x): .6e}  min={np.min(x): .6e}  max={np.max(x): .6e}")


def main():
    parser = argparse.ArgumentParser(
        description="Step 4 TNBC: build residualized edge flux proxies."
    )
    parser.add_argument("--sample_id", required=True)
    parser.add_argument("--statsdir", default="stats")
    parser.add_argument(
        "--lambda_smooth",
        type=float,
        default=1.0,
        help="Graph smoothing strength for residualization.",
    )
    args = parser.parse_args()

    sample = args.sample_id
    statsdir = Path(args.statsdir)
    lam = float(args.lambda_smooth)

    nodes_path = require_file(statsdir / f"{sample}_step3_nodes.csv")
    edges_path = require_file(statsdir / f"{sample}_step3_edges.csv")
    B1_path = require_file(statsdir / f"{sample}_step3_B1.npz")

    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)
    B1 = sparse.load_npz(B1_path).tocsr()

    print("==============================================")
    print("STEP 4: Residualized edge flux proxies")
    print("Sample:", sample)
    print("Nodes :", len(nodes))
    print("Edges :", len(edges))
    print("lambda_smooth:", lam)
    print("==============================================")

    # Raw fields
    tumor_raw = nodes["tumor_score"].to_numpy(dtype=float)
    stroma_raw = nodes["stroma_score"].to_numpy(dtype=float)
    immune_raw = nodes["immune_score"].to_numpy(dtype=float)

    # Build node Laplacian
    L0 = build_node_laplacian(B1)

    # Smooth + residual fields
    tumor_smooth, tumor_res = residualize_field(tumor_raw, L0, lam=lam)
    stroma_smooth, stroma_res = residualize_field(stroma_raw, L0, lam=lam)
    immune_smooth, immune_res = residualize_field(immune_raw, L0, lam=lam)

    print("\nField summaries")
    print("----------------------------------------------")
    summarize_vector("tumor_raw", tumor_raw)
    summarize_vector("tumor_smooth", tumor_smooth)
    summarize_vector("tumor_residual", tumor_res)
    summarize_vector("stroma_raw", stroma_raw)
    summarize_vector("stroma_smooth", stroma_smooth)
    summarize_vector("stroma_residual", stroma_res)
    summarize_vector("immune_raw", immune_raw)
    summarize_vector("immune_smooth", immune_smooth)
    summarize_vector("immune_residual", immune_res)

    # Save residualized node fields
    nodes_out = nodes.copy()
    nodes_out["tumor_smooth"] = tumor_smooth
    nodes_out["tumor_residual"] = tumor_res
    nodes_out["stroma_smooth"] = stroma_smooth
    nodes_out["stroma_residual"] = stroma_res
    nodes_out["immune_smooth"] = immune_smooth
    nodes_out["immune_residual"] = immune_res

    node_out_path = statsdir / f"{sample}_step4_node_residualized_fields.csv"
    nodes_out.to_csv(node_out_path, index=False)

    # Build residualized wedge fluxes on edges
    flux_ti = []
    flux_ts = []
    flux_is = []

    for _, e in edges.iterrows():
        i = int(e["tail"])
        j = int(e["head"])
        L = float(e["length"])

        flux_ti.append(
            compute_flux(
                tumor_res[i], tumor_res[j],
                immune_res[i], immune_res[j],
                L,
            )
        )

        flux_ts.append(
            compute_flux(
                tumor_res[i], tumor_res[j],
                stroma_res[i], stroma_res[j],
                L,
            )
        )

        flux_is.append(
            compute_flux(
                immune_res[i], immune_res[j],
                stroma_res[i], stroma_res[j],
                L,
            )
        )

    edges_out = edges.copy()
    edges_out["flux_tumor_immune"] = flux_ti
    edges_out["flux_tumor_stroma"] = flux_ts
    edges_out["flux_immune_stroma"] = flux_is

    # Keep raw edge-region metadata for later tests
    edge_out_path = statsdir / f"{sample}_step4_edge_fluxes.csv"
    edges_out.to_csv(edge_out_path, index=False)

    print("\nResidualized flux summary")
    print("----------------------------------------------")
    summarize_vector("flux_tumor_immune", np.asarray(flux_ti))
    summarize_vector("flux_tumor_stroma", np.asarray(flux_ts))
    summarize_vector("flux_immune_stroma", np.asarray(flux_is))

    print("\nSaved:")
    print(node_out_path)
    print(edge_out_path)


if __name__ == "__main__":
    main()
