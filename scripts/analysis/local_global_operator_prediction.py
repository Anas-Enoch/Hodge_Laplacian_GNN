#!/usr/bin/env python3
"""
step23b_local_vs_global_prediction.py  — Part C: Local vs Global Predictability
=================================================================================
Tests whether the coexact energy field is predictable from LOCAL neighborhood
structure but NOT from GLOBAL spectral modes.

This is the operationalisation of "constraint-driven local interactions":
  - Local rules govern  → local prediction should be accurate
  - No global template  → global (spectral) prediction should NOT add information
                           beyond what local structure already provides

If both local AND global predictions are strong, the field is globally periodic.
If only local prediction is strong, the field is locally structured but globally
non-periodic — the aperiodic candidate signature.
If neither, the field is random.

Three prediction models compared per section
--------------------------------------------
  M_LOCAL    k-hop neighborhood mean (k=1,2,3)
             Uses only the node's immediate spatial neighborhood.
             Captures local matching rules / constraint satisfaction.

  M_GLOBAL   Projection onto the top-k eigenmodes of L, then reconstruction.
             Uses only global spectral structure.
             Captures any periodic or long-range organised pattern.

  M_NULL     Global mean of u (constant predictor).
             Baseline: R² > 0 means the model adds information.

Prediction target: log(1 + coexact_energy) at each node.

Metric: leave-one-out R² (Spearman rho also reported as rank-based alternative).

Aperiodic prediction
--------------------
  R²(M_LOCAL)  >  R²(M_GLOBAL)   in the majority of sections
  R²(M_LOCAL)  significantly > 0 (structure is locally predictable)
  R²(M_GLOBAL) not significantly > R²(M_NULL) after controlling for energy

This pattern cannot arise from periodic structure (global would dominate)
or random noise (neither would predict), and is the minimal signature of
local-constraint-driven, non-periodic organisation.

Output
------
  {sid}_local_vs_global.csv       per-section R² for each model
  cohort_local_vs_global.csv      cohort-level summary with sign tests

Usage
-----
  python step23b_local_vs_global_prediction.py \\
      --statsdir Results_TNBC_rebuild_gse278936 \\
      --k-hops 1 2 3 --k-spectral 10 --seed 123
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.stats import spearmanr, binomtest


# ── Graph utilities ────────────────────────────────────────────────────────

def build_adjacency(edges: pd.DataFrame, n: int) -> sp.csr_matrix:
    i = edges["i"].values.astype(int)
    j = edges["j"].values.astype(int)
    w = np.abs(edges["flux_coexact"].values.astype(float))
    A = sp.coo_matrix((w, (i, j)), shape=(n, n)).tocsr()
    return (A + A.T).tocsr()


def build_laplacian(A: sp.csr_matrix) -> sp.csr_matrix:
    D = sp.diags(np.array(A.sum(axis=1)).ravel())
    return (D - A).tocsr()


def k_hop_mean(u: np.ndarray, A: sp.csr_matrix, k: int) -> np.ndarray:
    """
    For each node i: predict u_i as weighted mean of u over k-hop neighbors.
    Uses the adjacency matrix raised to the k-th power (sum of paths).
    Excludes the node itself (leave-one-out spirit for local prediction).
    """
    Ak = A.copy().astype(float)
    for _ in range(k - 1):
        Ak = Ak @ A
    # Zero out diagonal (exclude self)
    Ak = Ak.tolil()
    Ak.setdiag(0)
    Ak = Ak.tocsr()
    row_sum = np.array(Ak.sum(axis=1)).ravel()
    pred = np.array(Ak @ u).ravel() / np.maximum(row_sum, 1e-12)
    # For isolated nodes (no k-hop neighbors), predict global mean
    isolated = row_sum < 1e-12
    pred[isolated] = u.mean()
    return pred


def spectral_reconstruction(u: np.ndarray, L: sp.csr_matrix,
                             k: int) -> np.ndarray:
    """
    Reconstruct u using only the top-k eigenmodes of L (low-frequency global modes).
    This is the global spectral predictor: if the field is periodic,
    this reconstruction should be accurate.
    """
    n  = L.shape[0]
    k  = min(k, n - 2)
    if k < 1:
        return np.full(n, u.mean())
    try:
        vals, vecs = eigsh(L, k=k, which="SM", tol=1e-6)
    except Exception:
        return np.full(n, u.mean())
    # Project and reconstruct
    coeffs = vecs.T @ u          # shape (k,)
    recon  = vecs @ coeffs       # shape (n,)
    return recon


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot < 1e-12:
        return np.nan
    return float(1 - ss_res / ss_tot)


# ── Per-section computation ────────────────────────────────────────────────

def process_sample(sid: str, statsdir: Path, outdir: Path,
                   k_hops: list[int], k_spectral: int,
                   seed: int) -> dict | None:
    spots_path = statsdir / f"{sid}_spots_coexact_energy.csv"
    edges_path = statsdir / f"{sid}_edges_hodge.csv"
    if not spots_path.exists() or not edges_path.exists():
        print(f"  [{sid}] SKIP — missing inputs")
        return None

    spots = pd.read_csv(spots_path)
    edges = pd.read_csv(edges_path)
    n     = len(spots)
    u_raw = spots["coexact_energy"].values.astype(float)
    u     = np.log1p(np.clip(u_raw, 0, None))   # log-transform for regression

    A  = build_adjacency(edges, n)
    L  = build_laplacian(A)

    # ── Null: constant global mean ─────────────────────────────────────────
    pred_null  = np.full(n, u.mean())
    r2_null    = r_squared(u, pred_null)   # should be 0 by definition

    # ── Global spectral predictor ──────────────────────────────────────────
    pred_global = spectral_reconstruction(u, L, k_spectral)
    r2_global   = r_squared(u, pred_global)
    rho_global  = float(spearmanr(u, pred_global).statistic)

    # ── Local predictors (multiple hop distances) ──────────────────────────
    local_results = {}
    best_r2_local = -np.inf
    for k in k_hops:
        pred_local    = k_hop_mean(u, A, k)
        r2_local      = r_squared(u, pred_local)
        rho_local     = float(spearmanr(u, pred_local).statistic)
        local_results[k] = dict(r2=r2_local, rho=rho_local)
        best_r2_local = max(best_r2_local, r2_local)

    # ── Key comparison: local > global? ───────────────────────────────────
    local_dominates = int(best_r2_local > r2_global)
    r2_gap          = float(best_r2_local - r2_global)

    # ── Also test on exact-energy for comparison ───────────────────────────
    r2_global_exact = np.nan
    if "flux_exact" in edges.columns:
        i_idx = edges["i"].values.astype(int)
        j_idx = edges["j"].values.astype(int)
        deg   = np.maximum(
            np.bincount(i_idx, minlength=n) + np.bincount(j_idx, minlength=n), 1
        ).astype(float)
        ex  = np.bincount(i_idx, weights=edges["flux_exact"].values**2,
                          minlength=n) / deg
        ex_log = np.log1p(np.clip(ex, 0, None))
        pred_global_ex = spectral_reconstruction(ex_log, L, k_spectral)
        r2_global_exact = r_squared(ex_log, pred_global_ex)

    row = dict(
        sample_id              = sid,
        n_nodes                = n,
        r2_null                = r2_null,
        r2_global_spectral     = r2_global,
        rho_global_spectral    = rho_global,
        best_r2_local          = best_r2_local,
        r2_gap_local_minus_global = r2_gap,
        local_dominates_global = local_dominates,
        r2_global_exact        = r2_global_exact,
        k_spectral_modes_used  = k_spectral,
    )

    # Add per-hop local results
    for k in k_hops:
        row[f"r2_local_{k}hop"]  = local_results[k]["r2"]
        row[f"rho_local_{k}hop"] = local_results[k]["rho"]

    pd.DataFrame([row]).to_csv(outdir / f"{sid}_local_vs_global.csv", index=False)
    print(f"  [{sid}] local_best={best_r2_local:.3f} | "
          f"global={r2_global:.3f} | gap={r2_gap:+.3f} | "
          f"local_dominates={bool(local_dominates)}")
    return row


def main():
    ap = argparse.ArgumentParser(
        description="Step 23b: Local vs global predictability test")
    ap.add_argument("--statsdir",    type=Path, default=Path("Results_TNBC_rebuild_gse278936"))
    ap.add_argument("--outdir",      type=Path, default=None)
    ap.add_argument("--sample-id",   type=str,  default=None)
    ap.add_argument("--k-hops",      type=int,  nargs="+", default=[1, 2, 3])
    ap.add_argument("--k-spectral",  type=int,  default=10,
                    help="Number of global eigenmodes for spectral reconstruction")
    ap.add_argument("--seed",        type=int,  default=123)
    args = ap.parse_args()

    outdir = args.outdir or args.statsdir
    outdir.mkdir(parents=True, exist_ok=True)

    sids = ([args.sample_id] if args.sample_id else sorted([
        p.name.replace("_spots_coexact_energy.csv", "")
        for p in args.statsdir.glob("*_spots_coexact_energy.csv")
    ]))

    rows = []
    for sid in sids:
        print(f"\n── {sid} ──")
        r = process_sample(sid, args.statsdir, outdir,
                           args.k_hops, args.k_spectral, args.seed)
        if r:
            rows.append(r)

    if not rows:
        print("No results."); return

    df  = pd.DataFrame(rows)
    out = outdir / "cohort_local_vs_global.csv"
    df.to_csv(out, index=False)

    n_local_dom = int(df["local_dominates_global"].sum())
    n_tot       = len(df)
    bt          = binomtest(n_local_dom, n_tot, 0.5, alternative="greater")

    print("\n=== LOCAL vs GLOBAL PREDICTABILITY COHORT SUMMARY (PART C) ===")
    print(f"Local R² > Global R²: {n_local_dom}/{n_tot} sections")
    print(f"Sign test p          = {bt.pvalue:.4g}")
    print(f"\nMedian best local R²      : {df['best_r2_local'].median():.4f}")
    print(f"Median global spectral R² : {df['r2_global_spectral'].median():.4f}")
    print(f"Median R² gap (local-global): {df['r2_gap_local_minus_global'].median():+.4f}")
    if df["r2_global_exact"].notna().any():
        print(f"\nMedian global R² (exact field) : "
              f"{df['r2_global_exact'].median():.4f}")
        print("Prediction: exact should be BETTER globally predictable than coexact.")
    print(f"\nINTERPRETATION:")
    print(f"  local_dominates_global = True  → locally structured, globally non-periodic")
    print(f"  local_dominates_global = False → globally periodic OR random")
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
