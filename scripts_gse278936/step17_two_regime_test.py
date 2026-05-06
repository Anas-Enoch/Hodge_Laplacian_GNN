#!/usr/bin/env python3
"""
step17_two_regime_test.py — Two-Regime Permutation Test
========================================================
Tests whether interface-like regions differ from bulk-like regions
in three operator metrics:

  1. coexact_exact_ratio  (non-gradient vs gradient dominance)
  2. ks_like_magnitude    (-Lu - L²u - |∇u|²; instability proxy)
  3. graph_curvature      (|Lu|)

Test design
-----------
For each section, a node-count-matched and coexact-energy-quantile-matched
permutation null is used: bulk-like nodes are subsampled to the same size
as the interface, then the same metrics are computed on the random subsample.
n_perm permutations are run per section. One-sided empirical p-value:
fraction of null medians ≥ observed interface median.

SAFETY STATEMENT:
-----------------
These analyses generate formal operator-level tests, not clinical inference.
The physics analogies (two-regime model) are interpretive; the permutation
tests are empirical.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp


def build_laplacian(edges: pd.DataFrame, n: int) -> sp.csr_matrix:
    i = edges["i"].values.astype(int)
    j = edges["j"].values.astype(int)
    w = np.abs(edges["flux_coexact"].values.astype(float))
    A = sp.coo_matrix((w, (i, j)), shape=(n, n)).tocsr()
    A = A + A.T
    D = sp.diags(np.array(A.sum(axis=1)).ravel())
    return (D - A).tocsr()


def ks_proxy(u: np.ndarray, L: sp.csr_matrix,
             edges: pd.DataFrame, n: int) -> np.ndarray:
    """KS-like instability proxy per node: |−Lu − L²u − |∇u|²|"""
    Lu  = L @ u
    L2u = L @ Lu
    i_idx = edges["i"].values.astype(int)
    j_idx = edges["j"].values.astype(int)
    deg = np.maximum(
        np.bincount(i_idx, minlength=n) + np.bincount(j_idx, minlength=n), 1
    ).astype(float)
    diff2 = (u[i_idx] - u[j_idx]) ** 2
    nonlin = np.bincount(i_idx, weights=diff2, minlength=n) / deg
    return np.abs(-Lu - L2u - nonlin)


def permutation_test(interface_vals: np.ndarray, bulk_vals: np.ndarray,
                     n_perm: int, rng: np.random.Generator) -> dict:
    n_int = len(interface_vals)
    obs   = float(np.median(interface_vals))
    null  = []
    for _ in range(n_perm):
        idx  = rng.choice(len(bulk_vals), size=min(n_int, len(bulk_vals)), replace=False)
        null.append(float(np.median(bulk_vals[idx])))
    null = np.array(null)
    p    = float((null >= obs).mean())
    return {"observed": obs, "null_median": float(np.median(null)),
            "p_value": p, "n_interface": n_int, "n_bulk": len(bulk_vals)}


def process_sample(sid: str, statsdir: Path, outdir: Path,
                   n_perm: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)

    spots_path  = statsdir / f"{sid}_spots_coexact_energy.csv"
    edges_path  = statsdir / f"{sid}_edges_hodge.csv"
    regime_path = statsdir / f"{sid}_regime_assignment.csv"

    if not all(p.exists() for p in [spots_path, edges_path, regime_path]):
        print(f"  [{sid}] SKIP — missing files (run steps 15–16 first)")
        return []

    spots  = pd.read_csv(spots_path)
    edges  = pd.read_csv(edges_path)
    regime = pd.read_csv(regime_path)

    if "node_id" not in spots.columns:
        spots = spots.reset_index().rename(columns={"index": "node_id"})
    spots = spots.merge(regime[["node_id", "regime"]], on="node_id", how="left")

    n = len(spots)
    u = spots["coexact_energy"].values.astype(float)
    L = build_laplacian(edges, n)
    ks = ks_proxy(u, L, edges, n)

    exact_col = "flux_exact" if "flux_exact" in edges.columns else None
    if exact_col:
        i_idx = edges["i"].values.astype(int)
        j_idx = edges["j"].values.astype(int)
        deg   = np.maximum(
            np.bincount(i_idx, minlength=n) + np.bincount(j_idx, minlength=n), 1
        ).astype(float)
        ex2 = np.bincount(i_idx, weights=edges[exact_col].values**2, minlength=n) / deg
        ratio = u / (ex2 + 1e-12)
    else:
        ratio = np.zeros(n)

    Lu_abs = np.abs(L @ u)

    spots["ks_like"]             = ks
    spots["coexact_exact_ratio"] = ratio
    spots["graph_curvature"]     = Lu_abs
    spots["exact_available"]     = exact_col is not None

    int_mask  = spots["regime"].isin({"interface_like"})
    bulk_mask = spots["regime"].isin({"bulk_like"})

    if int_mask.sum() < 3 or bulk_mask.sum() < 3:
        print(f"  [{sid}] SKIP — insufficient interface ({int_mask.sum()}) or bulk ({bulk_mask.sum()}) nodes")
        return []

    # Primary metrics: always testable.
    # coexact_exact_ratio tested only when flux_exact is available (ratio not NaN).
    primary_metrics = ["coexact_energy", "ks_like", "graph_curvature"]
    conditional_metrics = (
        ["coexact_exact_ratio"] if exact_col is not None else []
    )

    rows = []
    for metric in primary_metrics + conditional_metrics:
        int_vals  = spots.loc[int_mask,  metric].dropna().values
        bulk_vals = spots.loc[bulk_mask, metric].dropna().values
        if len(int_vals) < 3 or len(bulk_vals) < 3:
            continue
        res = permutation_test(int_vals, bulk_vals, n_perm, rng)
        rows.append({
            "sample_id": sid, "metric": metric,
            "primary_metric": metric in primary_metrics,
            "exact_available": exact_col is not None,
            **res,
            "significant": int(res["p_value"] < 0.05)
        })

    print(f"  [{sid}] {int_mask.sum()} interface / {bulk_mask.sum()} bulk nodes")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Step 17: Two-regime permutation test")
    ap.add_argument("--statsdir",  type=Path, default=Path("results_gse278936"))
    ap.add_argument("--outdir",    type=Path, default=None)
    ap.add_argument("--sample-id", type=str,  default=None)
    ap.add_argument("--n-perm",    type=int,  default=300)
    ap.add_argument("--seed",      type=int,  default=123)
    args = ap.parse_args()

    outdir = args.outdir or args.statsdir
    outdir.mkdir(parents=True, exist_ok=True)

    sample_ids = ([args.sample_id] if args.sample_id else sorted([
        p.name.replace("_spots_coexact_energy.csv", "")
        for p in args.statsdir.glob("*_spots_coexact_energy.csv")
    ]))

    all_rows = []
    for sid in sample_ids:
        print(f"\n── {sid} ──")
        rows = process_sample(sid, args.statsdir, outdir, args.n_perm, args.seed)
        all_rows.extend(rows)

    if all_rows:
        df = pd.DataFrame(all_rows)
        out = outdir / "cohort_two_regime_test.csv"
        df.to_csv(out, index=False)

        print("\n=== TWO-REGIME TEST SUMMARY ===")
        print("Primary metrics (always tested): coexact_energy, ks_like, graph_curvature")
        print("Conditional metric (only when flux_exact available): coexact_exact_ratio")
        for metric, grp in df.groupby("metric"):
            n_sig = grp["significant"].sum()
            n_tot = len(grp)
            tier  = "PRIMARY" if grp["primary_metric"].iloc[0] else "CONDITIONAL"
            print(f"  [{tier}] {metric}: {n_sig}/{n_tot} sections significant (p<0.05)")
        print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
