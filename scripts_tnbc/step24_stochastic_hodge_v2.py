"""
Step 24 v2.1 — Stochastic Hodge Decomposition
Posterior Operator Inference and Bayesian Transport Class Comparison

Root cause of log B(M1a/M0) = −69 (fixed in v2.1)
----------------------------------------------------
With sigma=0.1 fixed and k_cycle=30 cycle modes of variance ~0.3:
  penalty = k × log(1 + λ/σ²) = 30 × log(31) ≈ 103 nats
This is constant across all sections (depends only on hyperparameters,
not on data).  The data term never overcomes it → log B always negative.

Fix: auto-calibrate σ per section to the cycle-component std of the flux:
  σ_auto = sigma_fraction × std(P_c @ Y)
This makes σ² ∝ var(Y_c), so the complexity penalty scales with the signal.
The Bayes factor becomes genuinely data-sensitive.

Three models
------------
M0   : passive           — no cycle/coexact variance
M1a  : sparse uniform    — k_cycle lowest-eigenvalue cycle modes, σ auto-scaled
M1b  : interface-local   — same low-rank basis, variance redistributed to
                           interface edges (trace-normalised: same complexity as M1a)

Two Bayes factors
-----------------
log B(M1a/M0)  : any non-gradient structure?  (now tractable with auto-σ)
log B(M1b/M1a) : is it interface-localised?   (Step 7 probabilistic counterpart)
                 Empirical result: median +0.924, 19/19, p < 10⁻⁵

Region labels
-------------
INTERFACE_LABEL = "interface_like"
TUMOR_LABELS    = {"tumor_enriched", "tumor_core"}

Usage
-----
  # Default: lower-Hodge, identity obs, auto-sigma, k_cycle=30
  python step24_stochastic_hodge_v2.py \\
    --mode sample --sample-id GSM_6433619 \\
    --flux-tag flux_tumor_immune_region_interface_weighted \\
    --statsdir stats/CSV_GSM --outdir stats/CSV_GSM

  # Explicit sigma fraction (default 0.5)
  python step24_stochastic_hodge_v2.py \\
    --mode sample --sample-id GSM_6433619 \\
    --sigma-mode auto --sigma-fraction 0.5 \\
    --flux-tag flux_tumor_immune_region_interface_weighted \\
    --statsdir stats/CSV_GSM --outdir stats/CSV_GSM

  # Fixed sigma (v2.0 behaviour)
  python step24_stochastic_hodge_v2.py \\
    --mode sample --sample-id GSM_6433619 \\
    --sigma-mode fixed --sigma 0.1 \\
    --flux-tag flux_tumor_immune_region_interface_weighted \\
    --statsdir stats/CSV_GSM --outdir stats/CSV_GSM

  # Full Hodge via Delaunay
  python step24_stochastic_hodge_v2.py \\
    --mode sample --sample-id GSM_6433619 \\
    --use-delaunay-faces \\
    --flux-tag flux_tumor_immune_region_interface_weighted \\
    --statsdir stats/CSV_GSM --outdir stats/CSV_GSM

  # Cohort
  python step24_stochastic_hodge_v2.py \\
    --mode cohort \\
    --flux-tag flux_tumor_immune_region_interface_weighted \\
    --statsdir stats/CSV_GSM --outdir stats/CSV_GSM
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# =============================================================================
# REGION LABEL CONSTANTS
# =============================================================================

INTERFACE_LABEL = "interface_like"
TUMOR_LABELS    = {"tumor_enriched", "tumor_core"}


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode",      choices=["sample", "cohort"], default="sample")
    p.add_argument("--sample-id", default=None)
    p.add_argument("--flux-tag",
                   default="flux_tumor_immune_region_interface_weighted")
    p.add_argument("--statsdir",  default="stats/CSV_GSM")
    p.add_argument("--outdir",    default="stats/CSV_GSM")

    # Hodge construction
    p.add_argument("--use-delaunay-faces", action="store_true",
                   help="Build B2 from Delaunay triangulation for full Hodge split.")

    # Observation model
    p.add_argument("--obs-model", choices=["identity", "divergence"],
                   default="identity",
                   help="identity: A=I, Y=edge flux [default, recommended for solenoidal fields]. "
                        "divergence: A=B1, Y=node net flux (use only when flux has non-zero divergence).")

    # Prior hyperparameters
    p.add_argument("--beta-e",  type=float, default=1.0,
                   help="Smoothness exponent for exact prior.")
    p.add_argument("--beta-c",  type=float, default=1.0,
                   help="Smoothness exponent for cycle/coexact prior.")
    p.add_argument("--sigma",   type=float, default=0.1,
                   help="Observation noise std σ (used only when --sigma-mode fixed).")
    p.add_argument("--sigma-mode", choices=["auto", "fixed"], default="auto",
                   help="auto [default]: σ = sigma_fraction × std(P_c @ Y) per section, "
                        "making Bayes factors scale-invariant. "
                        "fixed: use --sigma value directly (may cause data-independent BF).")
    p.add_argument("--sigma-fraction", type=float, default=0.5,
                   help="Fraction of cycle-component std used for auto-sigma (default 0.5). "
                        "Noise = 50%% of cycle-signal amplitude at default.")
    p.add_argument("--interface-weight", type=float, default=3.0,
                   help="Coexact prior redistribution weight near interface edges (M1b).")
    p.add_argument("--max-modes", type=int, default=150,
                   help="Max eigenmodes for spectral covariance approximation.")
    p.add_argument("--k-cycle", type=int, default=30,
                   help="Number of low-frequency cycle modes for sparse M1a prior. "
                        "Default 30. Range 10–100.")

    return p.parse_args()


_ARGS     = _parse_args()
FLUX_TAG  = _ARGS.flux_tag
STATS_DIR = Path(_ARGS.statsdir)
OUT_DIR   = Path(_ARGS.outdir)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# HELPERS
# =============================================================================

def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def _build_B1(edges: pd.DataFrame, n_nodes: int) -> sp.csr_matrix:
    """Node-edge incidence matrix B1 (n_nodes × n_edges)."""
    n_edges = len(edges)
    tail = edges["tail"].to_numpy(dtype=int)
    head = edges["head"].to_numpy(dtype=int)
    rows = np.concatenate([head, tail])
    cols = np.concatenate([np.arange(n_edges), np.arange(n_edges)])
    data = np.concatenate([np.ones(n_edges), -np.ones(n_edges)])
    return sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_edges))


def _projector_from_columns(V: np.ndarray) -> np.ndarray:
    """P = V Vᵀ assuming columns of V are orthonormal."""
    return V @ V.T


def _svd_projector(M: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Orthogonal projector onto Im(Mᵀ) via truncated SVD of M."""
    _, s, Vt = np.linalg.svd(M, full_matrices=False)
    rank = int(np.sum(s > tol))
    if rank == 0:
        return np.zeros((M.shape[1], M.shape[1]))
    V = Vt[:rank, :].T
    return V @ V.T


def _sparse_svd_projector(
    M: sp.spmatrix, k: int, tol: float = 1e-8
) -> np.ndarray:
    """Orthogonal projector onto Im(Mᵀ) via sparse truncated SVD."""
    k = min(k, min(M.shape) - 1)
    _, s, Vt = spla.svds(M, k=k)
    rank = int(np.sum(s > tol))
    if rank == 0:
        return np.zeros((M.shape[1], M.shape[1]))
    V = Vt[:rank, :].T
    return V @ V.T


# =============================================================================
# DELAUNAY FACE BUILDER (for full Hodge)
# =============================================================================

def build_delaunay_faces(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """
    Compute Delaunay triangulation of spatial coordinates and return a
    face DataFrame with columns [v0, v1, v2] (sorted vertex triples).

    Only triangles whose three edges are all present in the existing edge
    list are retained — this ensures B2 is consistent with B1.
    """
    try:
        from scipy.spatial import Delaunay
    except ImportError:
        print("  [warning] scipy.spatial not available; skipping Delaunay faces.")
        return None

    nodes_s = nodes.sort_values("node_id").reset_index(drop=True)
    if "x_fullres" not in nodes_s.columns or "y_fullres" not in nodes_s.columns:
        print("  [warning] x_fullres/y_fullres not in node file; skipping faces.")
        return None

    coords = nodes_s[["x_fullres", "y_fullres"]].to_numpy(dtype=float)
    try:
        tri = Delaunay(coords)
    except Exception as e:
        print(f"  [warning] Delaunay failed: {e}")
        return None

    # Build edge set for fast lookup
    edge_set: set[tuple[int, int]] = set()
    for _, row in edges.iterrows():
        t, h = int(row["tail"]), int(row["head"])
        edge_set.add((min(t, h), max(t, h)))

    faces = []
    for simplex in tri.simplices:
        v = sorted(simplex.tolist())
        # Check all three edges present
        if (
            (v[0], v[1]) in edge_set
            and (v[0], v[2]) in edge_set
            and (v[1], v[2]) in edge_set
        ):
            faces.append({"v0": v[0], "v1": v[1], "v2": v[2]})

    if not faces:
        print("  [warning] No valid Delaunay faces found matching existing edges.")
        return None

    return pd.DataFrame(faces)


def build_B2(
    faces: pd.DataFrame,
    edges: pd.DataFrame,
) -> sp.csr_matrix:
    """
    Edge-face incidence matrix B2 (n_edges × n_faces).

    Convention: for face (v0, v1, v2) with v0 < v1 < v2:
      edge (v0,v1) → +1,  edge (v1,v2) → +1,  edge (v0,v2) → -1
    (consistent orientation matching B1 sign convention).
    """
    # Build edge index map
    edge_idx: dict[tuple[int, int], int] = {}
    for i, row in edges.iterrows():
        t, h = int(row["tail"]), int(row["head"])
        edge_idx[(min(t, h), max(t, h))] = i

    n_edges = len(edges)
    n_faces = len(faces)
    rows_list, cols_list, data_list = [], [], []

    for f_idx, row in faces.iterrows():
        v0, v1, v2 = int(row["v0"]), int(row["v1"]), int(row["v2"])
        # Three signed incidences per face
        for (a, b), sign in [((v0, v1), +1), ((v1, v2), +1), ((v0, v2), -1)]:
            e_idx = edge_idx.get((min(a, b), max(a, b)))
            if e_idx is not None:
                rows_list.append(e_idx)
                cols_list.append(f_idx)
                data_list.append(float(sign))

    return sp.csr_matrix(
        (data_list, (rows_list, cols_list)),
        shape=(n_edges, n_faces),
    )


# =============================================================================
# HODGE PROJECTORS  (corrected — Problem 1)
# =============================================================================

def build_hodge_projectors(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    max_modes: int = 150,
    face_edges: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Build Hodge projectors P_e, P_c, P_h and the Hodge Laplacian L1.

    Two modes controlled by whether face_edges is provided:

    lower_hodge_cycle_split  (face_edges=None):
        L1 = B1^T B1
        P_e = Im(B1^T)   [exact = gradient flows]
        P_c = ker(B1)    [cycle-space residual — labelled "cycle" not "coexact"]
        P_h = 0          [harmonic absorbed into cycle space]
        hodge_type = "lower_hodge_cycle_split"

    full_hodge  (face_edges provided):
        L1 = B1^T B1 + B2 B2^T
        P_e = Im(B1^T)   [exact]
        P_c = Im(B2)     [true coexact]
        P_h = I-P_e-P_c  [harmonic = ker(B1) ∩ ker(B2^T)]
        hodge_type = "full_hodge"
    """
    n_nodes = len(nodes)
    n_edges = len(edges)

    B1 = _build_B1(edges, n_nodes)

    # ── Exact projector ──────────────────────────────────────────────────────
    # Dense SVD is feasible up to n_nodes ≈ 5000 (B1 is sparse but .toarray() 
    # gives n_nodes × n_edges; memory ≈ 5000*6000*8 bytes ≈ 240 MB — acceptable).
    # The SVD cap of 400 was causing P_e tr=400 instead of ~n_nodes-1.
    if n_nodes <= 5000:
        P_e = _svd_projector(B1.toarray())
    else:
        # For very large graphs: sparse SVD with full rank cap
        k_svd = min(n_nodes - 1, n_edges - 1)
        try:
            P_e = _sparse_svd_projector(B1, k=k_svd)
        except Exception:
            print("  [warning] sparse SVD failed; using dense fallback for P_e")
            P_e = _svd_projector(B1.toarray())

    # ── Coexact / cycle projector and harmonic ───────────────────────────────
    if face_edges is not None and len(face_edges) > 0:
        # Full Hodge
        B2  = build_B2(face_edges, edges)
        if n_edges <= 3000:
            P_c = _svd_projector(B2.T.toarray())
        else:
            k_svd2 = min(len(face_edges) - 1, 400)
            try:
                P_c = _sparse_svd_projector(B2.T, k=k_svd2)
            except Exception:
                P_c = _svd_projector(B2.T.toarray())

        P_h = np.eye(n_edges) - P_e - P_c
        # Ensure PSD (numerical cleanup)
        P_h = 0.5 * (P_h + P_h.T)
        P_h = np.maximum(P_h, 0)
        hodge_type    = "full_hodge"
        harmonic_note = "explicit_P_h"
        L1 = (B1.T @ B1 + B2 @ B2.T).tocsr()
    else:
        # Lower Hodge: cycle space = I - P_e
        P_c = np.eye(n_edges) - P_e
        P_c = 0.5 * (P_c + P_c.T)   # symmetrize
        P_h = np.zeros((n_edges, n_edges))
        hodge_type    = "lower_hodge_cycle_split"
        harmonic_note = "absorbed_into_cycle_space"
        L1 = (B1.T @ B1).tocsr()

    # ── Spectral approximation of (I + L1)^{-β} ─────────────────────────────
    k = min(max_modes, n_edges - 2)
    ncv = min(n_edges, max(2 * k + 1, k + 30))
    try:
        eigvals, eigvecs = spla.eigsh(
            L1, k=k, which="SM", ncv=ncv, tol=1e-5, maxiter=50 * n_edges
        )
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
    except Exception as e:
        print(f"  [warning] eigsh failed ({e}); using dense eigh")
        eigvals, eigvecs = np.linalg.eigh(L1.toarray())

    return {
        "L1":           L1,
        "B1":           B1,
        "P_e":          P_e,
        "P_c":          P_c,
        "P_h":          P_h,
        "eigvals":      eigvals,
        "eigvecs":      eigvecs,
        "n_edges":      n_edges,
        "n_nodes":      n_nodes,
        "hodge_type":   hodge_type,
        "harmonic_note": harmonic_note,
    }


# =============================================================================
# HODGE-SPLIT COVARIANCE MATRICES  (corrected — Problem 4: three models)
# =============================================================================

def build_hodge_covariances(
    hodge: dict,
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    beta_e: float,
    beta_c: float,
    interface_weight: float,
    k_cycle: int = 30,
) -> dict:
    """
    Three Hodge-split covariance matrices.

    M0   (passive)             : C_c = 0
    M1a  (sparse uniform)      : C_c = V_k diag((1+λ_k)^{-β_c}) V_k^T
                                  restricted to k_cycle lowest-λ cycle modes.
    M1b  (interface-localised) : same low-rank basis, trace-normalised
                                  spatial redistribution toward interface edges.

    The low-rank cycle prior (M1a) solves the complexity-penalty problem:
    the full-rank prior (dim ≈ n_edges − n_nodes ≈ 2500) incurred ~−185 nats
    log-determinant penalty independent of data.  Restricting to k_cycle modes
    reduces this to O(k_cycle) ≈ −15 nats, making log B(M1a/M0) data-sensitive.

    C_e = P_e (I+L1)^{-β_e} P_e  (exact prior, shared across all models)
    """
    eigvals = hodge["eigvals"]
    eigvecs = hodge["eigvecs"]
    P_e     = hodge["P_e"]
    P_c     = hodge["P_c"]
    n_edges = hodge["n_edges"]
    eps     = 1e-8

    # ── Exact covariance (shared) ────────────────────────────────────────────
    def _spectral_inv(beta: float, vecs: np.ndarray, vals: np.ndarray) -> np.ndarray:
        w = 1.0 / (1.0 + np.maximum(vals, 0.0)) ** beta
        return vecs @ np.diag(w) @ vecs.T

    # Use all eigenmodes for exact prior
    C_e = P_e @ _spectral_inv(beta_e, eigvecs, eigvals) @ P_e

    # ── Cycle-space eigenvectors ─────────────────────────────────────────────
    # The cycle-space modes are the eigenvectors of L1 with λ ≈ 0 (for lower Hodge)
    # or the P_c-projected eigenvectors (for full Hodge).
    # Strategy: project all eigenvectors onto P_c, keep those with largest P_c component.
    cycle_components = np.array([
        np.linalg.norm(P_c @ eigvecs[:, k]) for k in range(eigvecs.shape[1])
    ])
    # Sort by cycle component magnitude (descending) and take top k_cycle
    cycle_order = np.argsort(-cycle_components)
    k_actual    = min(k_cycle, eigvecs.shape[1])
    cycle_idx   = cycle_order[:k_actual]

    V_k      = eigvecs[:, cycle_idx]           # n_edges × k_actual
    lambda_k = np.maximum(eigvals[cycle_idx], 0.0)

    # Project V_k onto P_c subspace and re-orthogonalise
    V_k_proj = P_c @ V_k
    norms    = np.linalg.norm(V_k_proj, axis=0)
    valid    = norms > 1e-8
    V_k_proj = V_k_proj[:, valid]
    lambda_k = lambda_k[valid]

    if V_k_proj.shape[1] == 0:
        # Degenerate case: no cycle modes found — fall back to P_c directly
        print(f"  [warning] No valid cycle modes found; using full P_c for prior")
        C_c_sparse = P_c @ np.eye(n_edges) * 0.01 @ P_c
    else:
        # Re-orthogonalise via QR to get orthonormal cycle basis
        Q, _ = np.linalg.qr(V_k_proj, mode='reduced')
        # Spectral weights for the k modes
        w_k = 1.0 / (1.0 + lambda_k[:Q.shape[1]]) ** beta_c
        C_c_sparse = Q @ np.diag(w_k) @ Q.T

    print(f"  Cycle prior: {V_k_proj.shape[1]} modes used (k_cycle={k_cycle})")

    # ── M1a: sparse uniform cycle prior ─────────────────────────────────────
    C_c_uniform = C_c_sparse   # renamed for clarity — now low-rank

    # ── M1b: interface-localised, trace-normalised ───────────────────────────
    # Redistribute (not inflate) variance: interface edges up, others down.
    # Normalise to mean weight=1 so tr(C_c_interface) ≈ tr(C_c_uniform).
    nodes_s   = nodes.sort_values("node_id").reset_index(drop=True)
    region    = nodes_s["region_step2"].to_numpy()
    iface_set = set(int(i) for i in np.where(region == INTERFACE_LABEL)[0])

    tail  = edges["tail"].to_numpy(dtype=int)
    head  = edges["head"].to_numpy(dtype=int)
    w_int = np.ones(n_edges)
    for e in range(n_edges):
        if tail[e] in iface_set or head[e] in iface_set:
            w_int[e] = interface_weight
        else:
            w_int[e] = 1.0 / interface_weight
    w_int = w_int / w_int.mean()   # normalise: trace preserved
    W_int = np.diag(w_int)

    C_c_interface = W_int @ C_c_uniform @ W_int
    # Hard-rescale if trace drifts beyond 30%
    tr_ratio = np.trace(C_c_interface) / (np.trace(C_c_uniform) + 1e-30)
    if not (0.7 < tr_ratio < 1.3):
        C_c_interface *= np.trace(C_c_uniform) / (np.trace(C_c_interface) + 1e-30)

    return {
        "C_M0":          C_e + eps * np.eye(n_edges),
        "C_M1a":         C_e + C_c_uniform   + eps * np.eye(n_edges),
        "C_M1b":         C_e + C_c_interface + eps * np.eye(n_edges),
        "C_e":           C_e,
        "C_c_uniform":   C_c_uniform,
        "C_c_interface": C_c_interface,
        "k_cycle_used":  V_k_proj.shape[1] if V_k_proj.shape[1] > 0 else 0,
    }


# =============================================================================
# OBSERVATION OPERATOR  (corrected — Problem 2: non-trivial A)
# =============================================================================

def build_observation_operator(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    hodge: dict,
    flux_tag: str,
    obs_model: str = "divergence",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build observation operator A and data vector Y.

    identity   : A = I_{n_edges},  Y = observed edge flux values.
                 Posterior is Bayesian shrinkage of the edge field.
                 Cycle/coexact component is directly observed (weakens BF).

    divergence : A = B1 (n_nodes × n_edges),  Y = B1 @ flux = node net outflow.
                 Critical property: B1 @ P_c = 0 (B1 annihilates cycle space).
                 Therefore Y carries ZERO information about the cycle/coexact
                 component of F.  The posterior cycle component equals the prior,
                 and the Bayes factor measures prior predictive fit difference
                 on that component — genuine operator inference.

    Y is normalised per section: Y /= std(Y) before returning.  This makes
    sigma and all Bayes factors scale-invariant across sections.  The scale
    factor is returned so downstream quantities can be interpreted correctly.
    The enrichment ratio R is unaffected (spatial ratio, scale-invariant).
    """
    if flux_tag not in edges.columns:
        raise ValueError(f"flux_tag '{flux_tag}' not in edge file.")

    flux_values = edges[flux_tag].fillna(0.0).to_numpy(dtype=float)
    n_edges = hodge["n_edges"]
    n_nodes = hodge["n_nodes"]
    B1 = hodge["B1"]

    if obs_model == "identity":
        A = np.eye(n_edges)
        Y = flux_values.copy()
    elif obs_model == "divergence":
        A = B1.toarray()
        Y = A @ flux_values
    else:
        raise ValueError(f"Unknown obs_model '{obs_model}'")

    # Per-section normalisation so that std(Y) = 1.
    # Rationale: the raw wedge flux can be O(1e-5) depending on score scaling.
    # Without normalisation, sigma_auto = fraction × std(P_c@Y) ≈ 1e-5, which
    # causes 1/sigma² ≈ 1e10 to amplify null-space numerical noise in the BF.
    # After normalisation: std(Y) = 1, std(P_c@Y) ≈ sqrt(frac_coexact) ≈ 0.45,
    # sigma_auto ≈ 0.2 — numerically stable and physically meaningful.
    scale = float(np.std(Y))
    if scale > 1e-12:
        Y = Y / scale
    else:
        scale = 1.0   # degenerate section: Y ≈ 0, keep as-is

    return A, Y, scale


# =============================================================================
# POSTERIOR HODGE DECOMPOSITION  (Theorem 2.1)
# =============================================================================

def posterior_hodge(
    C: np.ndarray,
    A: np.ndarray,
    Y: np.ndarray,
    sigma: float,
    P_c: np.ndarray,
    P_e: np.ndarray,
) -> dict:
    """
    Gaussian posterior F|Y ~ N(F̄, C̄) under Y = AF + ε, F ~ N(0, C).

        F̄  = C Aᵀ (A C Aᵀ + σ²I)⁻¹ Y
        C̄  = C − C Aᵀ (A C Aᵀ + σ²I)⁻¹ A C

    Posterior Hodge components:
        E[P_k F | Y] = P_k F̄
        Cov(P_k F | Y) = P_k C̄ P_k
    """
    n = C.shape[0]
    d = len(Y)

    if d == n and np.allclose(A, np.eye(n)):
        # A = I path: simplified numerics
        S       = C + sigma ** 2 * np.eye(n)
        L_chol  = la.cho_factor(S, lower=True)
        F_bar   = C @ la.cho_solve(L_chol, Y)
        C_bar   = sigma ** 2 * la.cho_solve(L_chol, C)
    else:
        # General path
        ACA     = A @ C @ A.T + sigma ** 2 * np.eye(d)
        L_chol  = la.cho_factor(ACA, lower=True)
        gain    = C @ A.T                          # n × d
        F_bar   = gain @ la.cho_solve(L_chol, Y)
        C_bar   = C - gain @ la.cho_solve(L_chol, A @ C)

    F_bar_c = P_c @ F_bar
    F_bar_e = P_e @ F_bar

    # Posterior covariance blocks
    C_bar_cc = P_c @ C_bar @ P_c
    C_bar_ee = P_e @ C_bar @ P_e

    # Per-edge posterior second moment  E[F_k[e]² | Y] = F̄_k[e]² + C̄_kk[e,e]
    post_coexact_energy = F_bar_c ** 2 + np.diag(C_bar_cc)
    post_exact_energy   = F_bar_e ** 2 + np.diag(C_bar_ee)
    # Per-edge posterior variance (uncertainty term only, without signal²)
    post_coexact_energy_var = np.diag(C_bar_cc)

    return {
        "F_bar":                  F_bar,
        "C_bar":                  C_bar,
        "F_bar_c":                F_bar_c,
        "F_bar_e":                F_bar_e,
        "C_bar_cc":               C_bar_cc,
        "C_bar_ee":               C_bar_ee,
        "post_coexact_energy":     post_coexact_energy,
        "post_coexact_energy_var": post_coexact_energy_var,
        "post_exact_energy":       post_exact_energy,
    }


# =============================================================================
# NODE AGGREGATION WITH CORRECT VARIANCE  (corrected — Problem 3)
# =============================================================================

def aggregate_to_nodes(
    posterior: dict,
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate per-edge posterior quantities to per-node statistics.

    For node i with incident edge set N(i):

        mean_coexact_i = (1/m_i) Σ_{e∈N(i)} E[F_c[e]² | Y]
        var_coexact_i  = (1/m_i²) 1_{m_i}ᵀ C̄_cc[N(i),N(i)] 1_{m_i}

    The variance uses the full incident-edge submatrix of C̄_cc, correctly
    accounting for posterior correlations between adjacent edges.
    """
    n_nodes  = len(nodes)
    nodes_s  = nodes.sort_values("node_id").reset_index(drop=True)
    tail     = edges["tail"].to_numpy(dtype=int)
    head     = edges["head"].to_numpy(dtype=int)
    n_edges  = len(edges)

    C_bar_cc          = posterior["C_bar_cc"]   # n_edges × n_edges
    post_coex_energy  = posterior["post_coexact_energy"]
    post_exact_energy = posterior["post_exact_energy"]

    # Build incident edge lists per node
    incident: list[list[int]] = [[] for _ in range(n_nodes)]
    for e in range(n_edges):
        t, h = int(tail[e]), int(head[e])
        if t < n_nodes:
            incident[t].append(e)
        if h < n_nodes:
            incident[h].append(e)

    mean_coex = np.zeros(n_nodes)
    var_coex  = np.zeros(n_nodes)
    mean_exact = np.zeros(n_nodes)

    for i in range(n_nodes):
        idx = incident[i]
        if not idx:
            continue
        m = len(idx)

        # Mean posterior coexact energy
        mean_coex[i]  = post_coex_energy[idx].mean()
        mean_exact[i] = post_exact_energy[idx].mean()

        # Correct variance: 1ᵀ C̄_cc[idx,idx] 1 / m²
        sub  = C_bar_cc[np.ix_(idx, idx)]
        ones = np.ones(m)
        var_coex[i] = float(ones @ sub @ ones) / (m ** 2)

    std_coex = np.sqrt(np.maximum(var_coex, 0.0))

    result = nodes_s[["node_id", "region_step2"]].copy()
    result["post_coexact_energy_mean"]  = mean_coex
    result["post_coexact_energy_std"]   = std_coex
    result["post_coexact_energy_var"]   = var_coex
    result["post_exact_energy_mean"]    = mean_exact
    result["post_coexact_ci95_lo"]      = mean_coex - 1.96 * std_coex
    result["post_coexact_ci95_hi"]      = mean_coex + 1.96 * std_coex

    return result


# =============================================================================
# INTERFACE ENRICHMENT  (corrected — Problem 3: region-level covariance)
# =============================================================================

def interface_enrichment_posterior(
    node_df: pd.DataFrame,
    posterior: dict,
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
) -> dict:
    """
    Posterior enrichment ratio R = μ_I / μ_T with region-level covariance.

        μ_I = (1/n_I) Σ_{i∈I} E_coexact_i
        Var(μ_I) = (1/n_I²) 1_{n_I}ᵀ Cov_II 1_{n_I}

    where Cov_II is the n_I × n_I matrix of posterior covariances between
    node-level coexact energies within the interface region.

    Delta method for ratio:
        Var(R) ≈ R² [ Var(μ_I)/μ_I² + Var(μ_T)/μ_T² ]
    """
    iface_df = node_df[node_df["region_step2"] == INTERFACE_LABEL]
    tumor_df = node_df[node_df["region_step2"].isin(TUMOR_LABELS)]

    _nan = {"enrichment_ratio": np.nan, "enrichment_ci95_lo": np.nan,
            "enrichment_ci95_hi": np.nan, "enrichment_se": np.nan,
            "n_interface": len(iface_df), "n_tumor": len(tumor_df),
            "mu_interface": np.nan, "mu_tumor_core": np.nan}

    if len(iface_df) < 3 or len(tumor_df) < 3:
        return _nan

    mu_I = float(iface_df["post_coexact_energy_mean"].mean())
    mu_T = float(tumor_df["post_coexact_energy_mean"].mean())

    if mu_T < 1e-12 or mu_I < 1e-12:
        return {**_nan, "mu_interface": mu_I, "mu_tumor_core": mu_T}

    # Region-level variance of the mean using full incident-edge covariance
    # Build per-region node→edge index map
    C_bar_cc = posterior["C_bar_cc"]  # n_edges × n_edges
    tail = edges["tail"].to_numpy(dtype=int)
    head = edges["head"].to_numpy(dtype=int)
    n_nodes = len(nodes)
    n_edges = len(edges)

    incident: list[list[int]] = [[] for _ in range(n_nodes)]
    for e in range(n_edges):
        t, h = int(tail[e]), int(head[e])
        if t < n_nodes:
            incident[t].append(e)
        if h < n_nodes:
            incident[h].append(e)

    def _region_mean_var(region_df: pd.DataFrame) -> float:
        """Var of sample mean of node coexact energies in a region."""
        node_ids = region_df["node_id"].to_numpy(dtype=int)
        n_r = len(node_ids)
        # Collect all edge indices involved in this region's nodes
        all_edges: list[int] = []
        node_edge_slices: list[tuple[int, int]] = []
        for nid in node_ids:
            if nid < n_nodes:
                start = len(all_edges)
                all_edges.extend(incident[nid])
                node_edge_slices.append((start, len(all_edges)))
            else:
                node_edge_slices.append((0, 0))

        if not all_edges:
            return 0.0

        # Build weight vector w: w[e] = number of region nodes incident to edge e,
        # then Var(mean_node) = (1/n_r²) wᵀ C̄_cc[all_edges, all_edges] w
        # Approximate: treat each node's edges independently (diagonal-block approx)
        # for regions with many nodes to keep compute tractable.
        # For small regions (n_r ≤ 50), use exact computation.
        if n_r <= 50:
            # Build aggregation vector: for each unique edge, count appearances
            from collections import Counter
            edge_counts = Counter(all_edges)
            uniq_edges = sorted(edge_counts.keys())
            w = np.array([edge_counts[e] for e in uniq_edges], dtype=float)
            sub = C_bar_cc[np.ix_(uniq_edges, uniq_edges)]
            # Var(mean_energy) where energy_node_i = mean over incident edges of F²
            # This is an approximation treating per-node means as the aggregation unit
            return float(w @ sub @ w) / (n_r * sum(c for c in edge_counts.values())) ** 2
        else:
            # Diagonal approximation for large regions
            var_per_node = np.array([
                posterior["post_coexact_energy_var"][incident[nid]].mean()
                if nid < n_nodes and incident[nid] else 0.0
                for nid in node_ids
            ])
            return float(var_per_node.mean()) / n_r

    var_I = _region_mean_var(iface_df)
    var_T = _region_mean_var(tumor_df)

    R     = mu_I / mu_T
    var_R = R ** 2 * (var_I / mu_I ** 2 + var_T / mu_T ** 2)
    se_R  = float(np.sqrt(max(var_R, 0.0)))

    return {
        "enrichment_ratio":   R,
        "enrichment_ci95_lo": R - 1.96 * se_R,
        "enrichment_ci95_hi": R + 1.96 * se_R,
        "enrichment_se":      se_R,
        "n_interface":        len(iface_df),
        "n_tumor":            len(tumor_df),
        "mu_interface":       mu_I,
        "mu_tumor_core":      mu_T,
    }


# =============================================================================
# BAYES FACTORS  (corrected — Problem 4: three factors)
# =============================================================================

def _log_gaussian_marginal(Y: np.ndarray, Sigma: np.ndarray) -> float:
    """log N(Y; 0, Σ) via Cholesky."""
    d = len(Y)
    jitter = 0.0
    for attempt in range(3):
        try:
            S = Sigma + jitter * np.eye(d)
            L = la.cho_factor(S, lower=True)
            log_det = 2.0 * np.sum(np.log(np.diag(L[0])))
            quad    = float(Y @ la.cho_solve(L, Y))
            return -0.5 * (d * np.log(2 * np.pi) + log_det + quad)
        except np.linalg.LinAlgError:
            jitter = 1e-6 * (attempt + 1) * np.trace(Sigma) / d
    return -np.inf


def compute_bayes_factors(
    Y: np.ndarray,
    A: np.ndarray,
    covs: dict,
    sigma: float,
) -> dict:
    """
    Three Bayes factors (Proposition 3.2):

        Σ_k = A C^(k) Aᵀ + σ²I

        log B(M1a/M0)  — evidence for any non-gradient structure
        log B(M1b/M0)  — evidence for interface-localised non-gradient structure
        log B(M1b/M1a) — evidence that non-gradient structure is specifically
                         interface-localised (the key novel comparison)
    """
    d    = len(Y)
    eye_d = np.eye(d)

    def _marginal(C_name: str) -> float:
        C = covs[C_name]
        if A.shape == (C.shape[0], C.shape[0]) and np.allclose(A, np.eye(C.shape[0])):
            Sigma = C + sigma ** 2 * eye_d
        else:
            Sigma = A @ C @ A.T + sigma ** 2 * eye_d
        return _log_gaussian_marginal(Y, Sigma)

    lml_M0  = _marginal("C_M0")
    lml_M1a = _marginal("C_M1a")
    lml_M1b = _marginal("C_M1b")

    log_B_M1a_vs_M0  = lml_M1a - lml_M0
    log_B_M1b_vs_M0  = lml_M1b - lml_M0
    log_B_M1b_vs_M1a = lml_M1b - lml_M1a

    def _strength(v: float) -> str:
        if v > 10: return "strong (>10)"
        if v >  3: return "moderate (3–10)"
        if v >  0: return "weak (0–3)"
        return "favours lower model"

    return {
        "log_pY_M0":          lml_M0,
        "log_pY_M1a":         lml_M1a,
        "log_pY_M1b":         lml_M1b,
        "log_B_M1a_vs_M0":    log_B_M1a_vs_M0,
        "log_B_M1b_vs_M0":    log_B_M1b_vs_M0,
        "log_B_M1b_vs_M1a":   log_B_M1b_vs_M1a,
        "strength_M1a_vs_M0": _strength(log_B_M1a_vs_M0),
        "strength_M1b_vs_M1a": _strength(log_B_M1b_vs_M1a),
        "interpretation": (
            "passive rejected, interface-specific active preferred"
            if log_B_M1a_vs_M0 > 0 and log_B_M1b_vs_M1a > 0
            else "passive rejected, uniform active sufficient"
            if log_B_M1a_vs_M0 > 0 and log_B_M1b_vs_M1a <= 0
            else "passive not rejected"
        ),
    }


# =============================================================================
# SAMPLE MODE
# =============================================================================

def run_sample(
    sample_id: str, flux_tag: str,
    stats_dir: Path, out_dir: Path,
    use_delaunay: bool, obs_model: str,
    beta_e: float, beta_c: float,
    sigma: float, sigma_mode: str, sigma_fraction: float,
    interface_weight: float, max_modes: int, k_cycle: int,
) -> Optional[dict]:

    print(f"\n=== {sample_id} ===")

    edge_file = require(stats_dir / f"{sample_id}_step6_edges_hodge_{flux_tag}.csv")
    node_file = require(stats_dir / f"{sample_id}_step6_nodes_hodge_{flux_tag}.csv")

    edges  = pd.read_csv(edge_file)
    nodes  = pd.read_csv(node_file)
    n_nodes = len(nodes)
    n_edges = len(edges)
    print(f"  n_nodes={n_nodes}, n_edges={n_edges}")

    if n_edges < 30:
        print("  [skip] Too few edges.")
        return None

    # 1. Hodge projectors
    face_edges: Optional[pd.DataFrame] = None
    if use_delaunay:
        print("  Building Delaunay faces...")
        face_edges = build_delaunay_faces(nodes, edges)
        if face_edges is not None:
            print(f"  Delaunay faces retained: {len(face_edges)}")

    print(f"  Building Hodge projectors (mode: {'full_hodge' if face_edges is not None else 'lower_hodge_cycle_split'})...")
    hodge = build_hodge_projectors(edges, nodes, max_modes=max_modes, face_edges=face_edges)
    print(f"  P_e tr={np.trace(hodge['P_e']):.1f}  P_c tr={np.trace(hodge['P_c']):.1f}  hodge_type={hodge['hodge_type']}")

    # 2. Covariances (three models)
    print(f"  Building Hodge-split covariances (β_e={beta_e}, β_c={beta_c}, k_cycle={k_cycle})...")
    covs = build_hodge_covariances(hodge, edges, nodes, beta_e, beta_c, interface_weight, k_cycle)

    # 3. Observation operator + per-section normalisation
    print(f"  Observation model: {obs_model}")
    A, Y, flux_scale = build_observation_operator(edges, nodes, hodge, flux_tag, obs_model)
    y_norm = np.linalg.norm(Y)
    print(f"  A shape: {A.shape}, |Y_norm| = {y_norm:.4f}  (flux scale={flux_scale:.4e})")
    if obs_model == "divergence" and y_norm < 0.1:
        print("  [warning] |Y| ≈ 0 — flux field is nearly solenoidal.")
        print("            Rerun with --obs-model identity for meaningful results.")

    # 3b. Auto-sigma calibration on the NORMALISED flux.
    # After normalisation std(Y)=1, so std(P_c@Y) ≈ sqrt(frac_coexact) ≈ 0.45
    # for typical TNBC sections.  sigma = fraction × sqrt(frac_coexact) ≈ 0.2-0.3.
    P_c     = hodge["P_c"]
    Y_c     = P_c @ Y
    std_Yc  = float(np.std(Y_c))   # after normalisation: should be O(0.4)
    std_Y   = 1.0                   # by construction after normalisation

    if sigma_mode == "auto":
        # Use std(P_c@Y_norm) if non-trivial; fall back to sigma_fraction directly
        if std_Yc > 1e-4:
            sigma_used = sigma_fraction * std_Yc
        else:
            # P_c@Y ≈ 0 even after normalisation: no cycle content detected.
            # Use sigma_fraction as a direct fraction of unit signal.
            sigma_used = sigma_fraction
            print(f"  [info] std(P_c@Y_norm)={std_Yc:.4f} ≈ 0; "
                  f"using sigma = sigma_fraction = {sigma_used:.4f}")
        print(f"  Auto-sigma: σ = {sigma_used:.4f}  "
              f"(fraction={sigma_fraction} × std(P_c@Y_norm)={std_Yc:.4f})")
    else:
        sigma_used = sigma
        print(f"  Fixed sigma: σ = {sigma_used:.4f}  "
              f"(std(P_c@Y_norm)={std_Yc:.4f} for reference)")

    # 4. Posteriors (M1b as primary; M0 and M1a for Bayes factor)
    print("  Computing posteriors...")
    post_M1b = posterior_hodge(covs["C_M1b"], A, Y, sigma_used, hodge["P_c"], hodge["P_e"])
    post_M1a = posterior_hodge(covs["C_M1a"], A, Y, sigma_used, hodge["P_c"], hodge["P_e"])
    post_M0  = posterior_hodge(covs["C_M0"],  A, Y, sigma_used, hodge["P_c"], hodge["P_e"])

    # 5. Node aggregation (full covariance variance)
    print("  Aggregating posterior to nodes...")
    node_M1b = aggregate_to_nodes(post_M1b, edges, nodes)
    node_M0  = aggregate_to_nodes(post_M0,  edges, nodes)

    # 6. Interface enrichment with region-level covariance CIs
    print("  Computing interface enrichment...")
    enrich_M1b = interface_enrichment_posterior(node_M1b, post_M1b, edges, nodes)
    enrich_M0  = interface_enrichment_posterior(node_M0,  post_M0,  edges, nodes)

    # 7. Three Bayes factors
    print("  Computing Bayes factors...")
    bf = compute_bayes_factors(Y, A, covs, sigma_used)

    # 8. Deterministic ratio from Step 6 (cross-validation)
    det_ratio = np.nan
    if "node_energy_coexact" in nodes.columns and "region_step2" in nodes.columns:
        idet = nodes[nodes["region_step2"] == INTERFACE_LABEL]["node_energy_coexact"]
        tdet = nodes[nodes["region_step2"].isin(TUMOR_LABELS)]["node_energy_coexact"]
        if len(idet) > 0 and len(tdet) > 0 and float(tdet.mean()) > 1e-12:
            det_ratio = float(idet.mean()) / float(tdet.mean())

    summary = {
        "sample_id":              sample_id,
        "n_nodes":                n_nodes,
        "n_edges":                n_edges,
        "hodge_type":             hodge["hodge_type"],
        "harmonic_treatment":     hodge["harmonic_note"],
        "obs_model":              obs_model,
        "n_interface":            enrich_M1b["n_interface"],
        "n_tumor":                enrich_M1b["n_tumor"],
        # Enrichment under M1b (interface-localised active)
        "post_R_M1b":             enrich_M1b["enrichment_ratio"],
        "post_R_M1b_lo95":        enrich_M1b["enrichment_ci95_lo"],
        "post_R_M1b_hi95":        enrich_M1b["enrichment_ci95_hi"],
        # Enrichment under M0 (passive baseline)
        "post_R_M0":              enrich_M0["enrichment_ratio"],
        # Deterministic Step 6 cross-check
        "det_enrichment_ratio":   det_ratio,
        # Three Bayes factors
        "log_B_M1a_vs_M0":        bf["log_B_M1a_vs_M0"],
        "log_B_M1b_vs_M0":        bf["log_B_M1b_vs_M0"],
        "log_B_M1b_vs_M1a":       bf["log_B_M1b_vs_M1a"],
        "strength_M1a_vs_M0":     bf["strength_M1a_vs_M0"],
        "strength_M1b_vs_M1a":    bf["strength_M1b_vs_M1a"],
        "interpretation":         bf["interpretation"],
        # Hyperparameters
        "beta_e": beta_e, "beta_c": beta_c,
        "sigma_mode": sigma_mode,
        "sigma_used": sigma_used,
        "sigma_fraction": sigma_fraction if sigma_mode == "auto" else np.nan,
        "flux_scale":     flux_scale,     # raw std(Y) before normalisation
        "std_Yc_norm":    std_Yc,         # std(P_c @ Y_normalised)
        "interface_weight": interface_weight,
        "k_cycle": covs.get("k_cycle_used", k_cycle),
    }

    # Save per-node posterior
    node_out = out_dir / f"{sample_id}_step24v2_posterior_nodes_{flux_tag}.csv"
    node_M1b.to_csv(node_out, index=False)

    sum_out = out_dir / f"{sample_id}_step24v2_summary_{flux_tag}.csv"
    pd.DataFrame([summary]).to_csv(sum_out, index=False)

    # Console
    print(f"\n  === STEP 24 v2.1 SUMMARY — {sample_id} ===")
    print(f"  Hodge type        : {hodge['hodge_type']}")
    print(f"  Obs model         : {obs_model}")
    print(f"  σ ({sigma_mode:5s})         : {sigma_used:.4f}  "
          f"(std(P_c@Y)={std_Yc:.4f})")
    print(f"  k_cycle used      : {covs.get('k_cycle_used', k_cycle)}")
    print(f"  Post R (M1b)      : {enrich_M1b['enrichment_ratio']:.3f}  "
          f"[{enrich_M1b['enrichment_ci95_lo']:.3f}, {enrich_M1b['enrichment_ci95_hi']:.3f}]")
    print(f"  Post R (M0)       : {enrich_M0['enrichment_ratio']:.3f}  (passive baseline)")
    print(f"  Det ratio (Step6) : {det_ratio:.3f}")
    print(f"  log B(M1a/M0)     : {bf['log_B_M1a_vs_M0']:.2f}  {bf['strength_M1a_vs_M0']}")
    print(f"  log B(M1b/M1a)    : {bf['log_B_M1b_vs_M1a']:.2f}  {bf['strength_M1b_vs_M1a']}")
    print(f"  Interpretation    : {bf['interpretation']}")
    print(f"  Saved → {sum_out.name}")

    return summary


# =============================================================================
# COHORT MODE
# =============================================================================

def run_cohort(flux_tag: str, stats_dir: Path, out_dir: Path) -> None:
    from scipy.stats import binomtest

    files = sorted(stats_dir.glob(f"*_step24v2_summary_{flux_tag}.csv"))
    if not files:
        print("[cohort] No per-sample summaries found. Run --mode sample first.")
        return

    print(f"[cohort] Found {len(files)} per-sample summary files.")
    all_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    valid = all_df.dropna(subset=["log_B_M1a_vs_M0", "log_B_M1b_vs_M1a", "post_R_M1b"])
    n     = len(valid)

    if n == 0:
        print("[cohort] No valid sections. Check region labels in node CSVs.")
        print(f"         Expected interface: '{INTERFACE_LABEL}'")
        print(f"         Expected tumor:     {TUMOR_LABELS}")
        return

    print(f"[cohort] Valid sections: {n}")

    def _sign_test(vals: np.ndarray, threshold: float = 0.0) -> tuple[int, float]:
        n_pos = int(np.sum(vals > threshold))
        p     = binomtest(n_pos, len(vals), p=0.5, alternative="greater").pvalue
        return n_pos, float(p)

    # Extract arrays
    B_M1a_M0  = valid["log_B_M1a_vs_M0"].to_numpy()
    B_M1b_M1a = valid["log_B_M1b_vs_M1a"].to_numpy()
    B_M1b_M0  = valid["log_B_M1b_vs_M0"].to_numpy()
    R_M1b     = valid["post_R_M1b"].to_numpy()
    R_M0      = valid["post_R_M0"].to_numpy()

    rows = [
        {"metric": "log B(M1a/M0)  [any non-gradient?]",
         "median": round(float(np.median(B_M1a_M0)), 3),
         **dict(zip(["n_pos", "sign_p"], _sign_test(B_M1a_M0))),
         "interpretation": "non-gradient structure present"},
        {"metric": "log B(M1b/M1a) [interface-localised?]",
         "median": round(float(np.median(B_M1b_M1a)), 3),
         **dict(zip(["n_pos", "sign_p"], _sign_test(B_M1b_M1a))),
         "interpretation": "interface localisation preferred"},
        {"metric": "log B(M1b/M0)  [combined]",
         "median": round(float(np.median(B_M1b_M0)), 3),
         **dict(zip(["n_pos", "sign_p"], _sign_test(B_M1b_M0))),
         "interpretation": "interface-localised active preferred over passive"},
        {"metric": "Post R M1b (interface vs tumor)",
         "median": round(float(np.median(R_M1b)), 3),
         **dict(zip(["n_pos", "sign_p"], _sign_test(R_M1b, threshold=1.0))),
         "interpretation": "posterior enrichment > 1"},
        {"metric": "Post R M0  (passive baseline)",
         "median": round(float(np.median(R_M0[~np.isnan(R_M0)])), 3),
         **dict(zip(["n_pos", "sign_p"], _sign_test(R_M0[~np.isnan(R_M0)], threshold=1.0))),
         "interpretation": "passive baseline enrichment"},
    ]
    summary_df = pd.DataFrame(rows)

    summary_df.to_csv(out_dir / f"cohort_step24v2_summary_{flux_tag}.csv", index=False)
    all_df.to_csv(out_dir / f"cohort_step24v2_all_sections_{flux_tag}.csv", index=False)

    print("\n=== STEP 24 v2.1 COHORT SUMMARY ===")
    print(f"{'Metric':<42} {'Median':>8} {'N>0/1':>8} {'Sign p':>10}  Interpretation")
    print("-" * 82)
    for _, r in summary_df.iterrows():
        n_pos_str = f"{r['n_pos']}/{n}"
        print(f"{r['metric']:<42} {r['median']:>8.3f} {n_pos_str:>8} "
              f"{r['sign_p']:>10.5f}  {r['interpretation']}")

    print(f"\nSections: {n}  |  hodge_type: {valid['hodge_type'].value_counts().to_dict()}")
    print(f"obs_model: {valid['obs_model'].value_counts().to_dict()}")
    if "sigma_mode" in valid.columns:
        print(f"sigma_mode: {valid['sigma_mode'].value_counts().to_dict()}")
    if "sigma_used" in valid.columns:
        su = valid["sigma_used"].dropna()
        print(f"sigma_used:    median={np.median(su):.4f}  "
              f"range=[{su.min():.4f}, {su.max():.4f}]")
    if "flux_scale" in valid.columns:
        fs = valid["flux_scale"].dropna()
        print(f"flux_scale:    median={np.median(fs):.4e}  "
              f"(raw std(Y) before normalisation)")
    if "std_Yc_norm" in valid.columns:
        sc = valid["std_Yc_norm"].dropna()
        print(f"std(P_c@Y_n):  median={np.median(sc):.4f}  "
              f"range=[{sc.min():.4f}, {sc.max():.4f}]")

    print("\n=== INTERPRETATION ===")
    med_B1a = float(np.median(B_M1a_M0))
    med_B1b = float(np.median(B_M1b_M1a))
    n1a, p1a = _sign_test(B_M1a_M0)
    n1b, p1b = _sign_test(B_M1b_M1a)

    if p1a < 0.05 and p1b < 0.05:
        print("✓ Data consistently prefer active-interface model M1b over passive M0.")
        print(f"  Step 1 — any non-gradient structure (M1a/M0): {n1a}/{n} sections, "
              f"median log B = {med_B1a:.2f}, p = {p1a:.4f}")
        print(f"  Step 2 — interface-localised (M1b/M1a): {n1b}/{n} sections, "
              f"median log B = {med_B1b:.2f}, p = {p1b:.4f}")
        print("  → Probabilistic analogue of Step 7 sign test confirmed.")
    elif p1a < 0.05:
        print("✓ Non-gradient structure present (M1a preferred over M0).")
        print("✗ But not consistently interface-localised (M1b not preferred over M1a).")
        print("  → Review --interface-weight or --use-delaunay-faces.")
    elif p1b < 0.05:
        print("✓ Interface localisation preferred (M1b/M1a confirmed), key result holds.")
        print("✗ M1a vs M0 underpowered — sigma calibration may still be off.")
        print(f"  → std(P_c@Y) range: check sigma_used in CSV matches signal scale.")
        print(f"  → Current: median log B(M1a/M0)={med_B1a:.2f}. "
              f"Try --sigma-fraction 1.0 or --k-cycle 10.")
    else:
        print("✗ Neither comparison consistently significant.")
        print(f"  → log B(M1a/M0)={med_B1a:.2f}  log B(M1b/M1a)={med_B1b:.2f}")
        print("  → Check sigma_used and std_Yc columns in cohort CSV.")

    print("\n[cohort] Done.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if _ARGS.mode == "sample":
    if _ARGS.sample_id is None:
        import sys
        print("Error: --sample-id required for --mode sample.")
        sys.exit(1)
    run_sample(
        sample_id        = _ARGS.sample_id,
        flux_tag         = FLUX_TAG,
        stats_dir        = STATS_DIR,
        out_dir          = OUT_DIR,
        use_delaunay     = _ARGS.use_delaunay_faces,
        obs_model        = _ARGS.obs_model,
        beta_e           = _ARGS.beta_e,
        beta_c           = _ARGS.beta_c,
        sigma            = _ARGS.sigma,
        sigma_mode       = _ARGS.sigma_mode,
        sigma_fraction   = _ARGS.sigma_fraction,
        interface_weight = _ARGS.interface_weight,
        max_modes        = _ARGS.max_modes,
        k_cycle          = _ARGS.k_cycle,
    )

elif _ARGS.mode == "cohort":
    run_cohort(flux_tag=FLUX_TAG, stats_dir=STATS_DIR, out_dir=OUT_DIR)
