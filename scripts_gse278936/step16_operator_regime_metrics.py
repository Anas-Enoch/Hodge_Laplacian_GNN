#!/usr/bin/env python3
"""
step16_operator_regime_metrics.py — Operator Regime Metrics
============================================================
Computes per-regime operator-level summaries for each sample:

  exact_energy       : Σ f_exact(i,j)² / deg_i  (gradient component)
  coexact_energy     : Σ f_coexact(i,j)² / deg_i (non-gradient component)
  coexact_exact_ratio: coexact_energy / (exact_energy + ε)
  graph_curvature    : |Lu|  (graph Laplacian applied to coexact energy field)
  bilaplacian_mag    : |L²u| (higher-order stabilizing/damping proxy)
  nonlin_grad_energy : Σ (u_i - u_j)² / deg_i  (nonlinear gradient proxy)

These metrics operationalize the two-regime hypothesis:
  bulk_like   → low coexact/exact ratio, low curvature, low KS-like activity
  interface_like → high coexact/exact ratio, high curvature, high KS-like activity

SAFETY STATEMENT:
-----------------
The graph curvature and bi-Laplacian magnitudes are graph-operator analogies
used to characterize the non-gradient component of the coexact field. They
do not claim that tumor tissue follows Euler–Bernoulli or any other physical PDE.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def build_laplacian(edges: pd.DataFrame, n_nodes: int) -> sp.csr_matrix:
    i  = edges["i"].values.astype(int)
    j  = edges["j"].values.astype(int)
    w  = np.abs(edges["flux_coexact"].values.astype(float))
    A  = sp.coo_matrix((w, (i, j)), shape=(n_nodes, n_nodes))
    A  = (A + A.T).tocsr()
    d  = np.array(A.sum(axis=1)).ravel()
    D  = sp.diags(d)
    return (D - A).tocsr()


def regime_metrics(spots: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    n = len(spots)
    u = spots["coexact_energy"].values.astype(float)

    # Exact energy (from edges if available, else zero)
    exact_col = "flux_exact" if "flux_exact" in edges.columns else None

    # Build weighted Laplacian
    L = build_laplacian(edges, n)

    # Graph curvature |Lu|
    Lu  = L @ u
    L2u = L @ Lu

    # Nonlinear gradient energy per node
    i_idx = edges["i"].values.astype(int)
    j_idx = edges["j"].values.astype(int)
    diff2 = (u[i_idx] - u[j_idx]) ** 2
    deg   = np.maximum(np.bincount(i_idx, minlength=n) +
                       np.bincount(j_idx, minlength=n), 1).astype(float)
    nonlin_grad = np.bincount(i_idx, weights=diff2, minlength=n) / deg

    # Exact energy from edges if available
    if exact_col:
        ex = edges[exact_col].values.astype(float)
        exact_e = np.bincount(i_idx, weights=ex**2, minlength=n) / deg
        coexact_exact_ratio = u / (exact_e + 1e-12)
    else:
        # Do NOT divide by near-zero: ratio is undefined without exact energy.
        # Set to NaN throughout; CDIS and downstream analyses must handle this.
        exact_e = np.full(n, np.nan)
        coexact_exact_ratio = np.full(n, np.nan)

    spots = spots.copy()
    spots["exact_energy_node"]    = exact_e
    spots["graph_curvature"]      = np.abs(Lu)
    spots["bilaplacian_mag"]      = np.abs(L2u)
    spots["nonlin_grad_energy"]   = nonlin_grad
    spots["coexact_exact_ratio"]  = coexact_exact_ratio
    spots["exact_energy_available"] = exact_col is not None

    # Aggregate by regime
    agg_cols = ["coexact_energy", "exact_energy_node", "coexact_exact_ratio",
                "graph_curvature", "bilaplacian_mag", "nonlin_grad_energy"]
    regime_col = "regime" if "regime" in spots.columns else "region"
    summary = (
        spots.groupby(regime_col)[agg_cols]
        .agg(["median", "mean", "std"])
        .round(6)
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    return summary.reset_index()


def process_sample(sid: str, statsdir: Path, outdir: Path) -> pd.DataFrame | None:
    spots_path  = statsdir / f"{sid}_spots_coexact_energy.csv"
    edges_path  = statsdir / f"{sid}_edges_hodge.csv"
    regime_path = statsdir / f"{sid}_regime_assignment.csv"

    if not spots_path.exists() or not edges_path.exists():
        print(f"  [{sid}] SKIP — missing input files")
        return None

    spots = pd.read_csv(spots_path)
    edges = pd.read_csv(edges_path)

    if regime_path.exists():
        reg = pd.read_csv(regime_path)[["node_id", "regime"]]
        if "node_id" not in spots.columns:
            spots = spots.reset_index().rename(columns={"index": "node_id"})
        spots = spots.merge(reg, on="node_id", how="left")

    if not {"i", "j", "flux_coexact"} <= set(edges.columns):
        print(f"  [{sid}] SKIP — edges missing flux_coexact")
        return None

    summary = regime_metrics(spots, edges)
    summary.insert(0, "sample_id", sid)

    out_path = outdir / f"{sid}_operator_regime_metrics.csv"
    summary.to_csv(out_path, index=False)
    print(f"  [{sid}] wrote {out_path}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Step 16: Operator regime metrics")
    ap.add_argument("--statsdir", type=Path, default=Path("results_gse278936"))
    ap.add_argument("--outdir",   type=Path, default=None)
    ap.add_argument("--sample-id", type=str, default=None)
    args = ap.parse_args()

    outdir = args.outdir or args.statsdir
    outdir.mkdir(parents=True, exist_ok=True)

    if args.sample_id:
        sample_ids = [args.sample_id]
    else:
        sample_ids = sorted([
            p.name.replace("_spots_coexact_energy.csv", "")
            for p in args.statsdir.glob("*_spots_coexact_energy.csv")
        ])

    all_rows = []
    for sid in sample_ids:
        print(f"\n── {sid} ──")
        r = process_sample(sid, args.statsdir, outdir)
        if r is not None:
            all_rows.append(r)

    if all_rows:
        cohort = pd.concat(all_rows, ignore_index=True)
        out_cohort = outdir / "cohort_operator_regime_summary.csv"
        cohort.to_csv(out_cohort, index=False)
        print(f"\nCohort summary → {out_cohort}")


if __name__ == "__main__":
    main()
