#!/usr/bin/env python3
"""
step18_bulk_equilibrium_null.py — Bulk Equilibrium Null Model
=============================================================
Treats tumor-core / bulk-like nodes as the baseline near-equilibrium
statistical regime. Tests whether interface nodes significantly exceed
bulk-matched null distributions in coexact energy, spectral Zeta
concentration, and KS-like instability.

Near-equilibrium analogy: the tumor bulk is gradient-compatible and
spectrally flat relative to the interface — consistent with a statistically
diffuse, non-structured interaction field analogous to a near-Maxwell–
Boltzmann-like regime in operator terms.

SAFETY STATEMENT:
-----------------
The Maxwell–Boltzmann reference is an operator-level analogy. This script
does not claim that tumor cells literally obey Maxwell–Boltzmann statistics.
The test simply checks whether bulk nodes behave like a null distribution
for the operator metrics measured at interface nodes.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import binomtest


def build_laplacian(edges, n):
    i = edges["i"].values.astype(int)
    j = edges["j"].values.astype(int)
    w = np.abs(edges["flux_coexact"].values.astype(float))
    A = sp.coo_matrix((w, (i, j)), shape=(n, n)).tocsr()
    A = A + A.T
    D = sp.diags(np.array(A.sum(axis=1)).ravel())
    return (D - A).tocsr()


def compute_metrics(u, L, edges, n):
    Lu  = L @ u
    L2u = L @ Lu
    i_idx = edges["i"].values.astype(int)
    j_idx = edges["j"].values.astype(int)
    deg = np.maximum(
        np.bincount(i_idx, minlength=n) + np.bincount(j_idx, minlength=n), 1
    ).astype(float)
    diff2 = (u[i_idx] - u[j_idx]) ** 2
    nonlin = np.bincount(i_idx, weights=diff2, minlength=n) / deg
    ks = np.abs(-Lu - L2u - nonlin)

    # Truncated Zeta Z = [Σαk/λk] / [Σαk], k=50 smallest nonzero modes
    try:
        from scipy.sparse.linalg import eigsh
        k = min(50, n - 2)
        if k > 1:
            vals, vecs = eigsh(L, k=k, which="SM")
            signal = np.log1p(np.clip(u, 0, None))
            num = 0.0; denom = 0.0
            for lam, phi in zip(vals, vecs.T):
                if lam < 1e-8:
                    continue
                alpha = float(np.dot(signal, phi)) ** 2
                num   += alpha / lam
                denom += alpha
            zeta = float(num / denom) if denom > 1e-12 else 0.0
        else:
            zeta = np.nan
    except Exception:
        zeta = np.nan

    return u, ks, zeta


def process_sample(sid, statsdir, outdir, n_perm, seed):
    rng = np.random.default_rng(seed)

    spots_path  = statsdir / f"{sid}_spots_coexact_energy.csv"
    edges_path  = statsdir / f"{sid}_edges_hodge.csv"
    regime_path = statsdir / f"{sid}_regime_assignment.csv"

    if not all(p.exists() for p in [spots_path, edges_path, regime_path]):
        return None

    spots  = pd.read_csv(spots_path)
    edges  = pd.read_csv(edges_path)
    regime = pd.read_csv(regime_path)
    if "node_id" not in spots.columns:
        spots = spots.reset_index().rename(columns={"index": "node_id"})
    spots = spots.merge(regime[["node_id", "regime"]], on="node_id", how="left")

    n  = len(spots)
    u  = spots["coexact_energy"].values.astype(float)
    L  = build_laplacian(edges, n)
    _, ks, _ = compute_metrics(u, L, edges, n)
    spots["ks_like"] = ks

    int_mask  = spots["regime"].isin({"interface_like"})
    bulk_mask = spots["regime"].isin({"bulk_like"})

    if int_mask.sum() < 3 or bulk_mask.sum() < 3:
        return None

    n_int = int(int_mask.sum())
    rows  = []

    for metric, vals in [("coexact_energy", u), ("ks_like", ks)]:
        obs_int  = float(np.median(vals[int_mask]))
        obs_bulk = float(np.median(vals[bulk_mask]))

        null = []
        for _ in range(n_perm):
            idx = rng.choice(np.where(bulk_mask)[0], size=n_int, replace=True)
            null.append(float(np.median(vals[idx])))

        null = np.array(null)
        p    = float((null >= obs_int).mean())
        fold = obs_int / (obs_bulk + 1e-12)

        rows.append({
            "sample_id":       sid,
            "metric":          metric,
            "interface_median": obs_int,
            "bulk_median":      obs_bulk,
            "fold_interface_vs_bulk": fold,
            "p_bulk_matched_null": p,
            "significant":     int(p < 0.05),
            "n_interface":     n_int,
            "n_bulk":          int(bulk_mask.sum()),
        })

    print(f"  [{sid}] done (n_int={n_int}, n_bulk={int(bulk_mask.sum())})")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Step 18: Bulk equilibrium null model")
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
        if rows:
            all_rows.extend(rows)

    if all_rows:
        df  = pd.DataFrame(all_rows)
        out = outdir / "cohort_bulk_vs_interface_null.csv"
        df.to_csv(out, index=False)

        print("\n=== BULK EQUILIBRIUM NULL SUMMARY ===")
        for metric, grp in df.groupby("metric"):
            n_sig  = grp["significant"].sum()
            n_tot  = len(grp)
            med_fold = grp["fold_interface_vs_bulk"].median()
            bt = binomtest(int(n_sig), int(n_tot), 0.5, alternative="greater")
            print(f"  {metric}: {n_sig}/{n_tot} significant | "
                  f"median fold = {med_fold:.3f} | sign p = {bt.pvalue:.4g}")

        print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
