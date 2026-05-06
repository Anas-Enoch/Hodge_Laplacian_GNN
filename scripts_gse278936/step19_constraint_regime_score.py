#!/usr/bin/env python3
"""
step19_constraint_regime_score.py — Constraint-Dominated Interface Score
=========================================================================
Combines three operator metrics into a Constraint-Dominated Interface Score
(CDIS) per node using ROBUST z-scores (median-centered, IQR-scaled):

  CDIS = rz(coexact_exact_ratio_or_coexact_energy)
        + rz(|L²u|)
        + rz(nonlin_grad_energy)

where rz(x) = (x − median(x)) / (IQR(x) / 1.349)
(robust z-score: IQR/1.349 ≈ σ for a normal distribution)

WHY ROBUST Z-SCORES:
--------------------
The coexact_exact_ratio and bilaplacian_mag distributions are
extremely right-skewed (mean/median ratio up to 10^10 due to
outlier nodes). Standard z-scores using mean/std produce negative
z-scores for all nodes below the (outlier-inflated) mean, including
genuine interface nodes. Robust z-scores are insensitive to these
outliers and correctly reflect whether interface nodes are above the
section-wide median — which is the meaningful criterion here.

CDIS formula decision:
  Full: rz(coexact_exact_ratio) + rz(|L²u|) + rz(nonlin_grad)
        when flux_exact is available
  Reduced: rz(coexact_energy) + rz(|L²u|) + rz(nonlin_grad)
        when flux_exact is unavailable (set coexact/exact ratio to NaN
        rather than dividing by near-zero)

SAFETY STATEMENT:
-----------------
The CDIS is an operator-derived composite score. It is not a clinical
biomarker and does not predict treatment response. It quantifies the
degree to which a node's operator profile is consistent with the
non-gradient, constraint-dominated regime described in the manuscript.
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


def robust_zscore(x: np.ndarray) -> np.ndarray:
    """Median-centered, IQR-scaled robust z-score.
    Uses IQR/1.349 as the robust scale estimate (= σ for Gaussian).
    Handles constant arrays by returning zeros.
    """
    med = np.nanmedian(x)
    q75, q25 = np.nanpercentile(x, [75, 25])
    iqr = q75 - q25
    scale = iqr / 1.349 if iqr > 0 else 1.0
    return (x - med) / scale


def compute_cdis(u: np.ndarray, L: sp.csr_matrix,
                 edges: pd.DataFrame, n: int,
                 exact_energy: np.ndarray | None = None) -> tuple[np.ndarray, str]:
    """
    Returns (cdis, cdis_formula_used) using robust z-scores.

    Full formula (when exact_energy is available):
      CDIS = rz(coexact_exact_ratio) + rz(|L²u|) + rz(nonlin_grad_energy)

    Reduced formula (when exact_energy unavailable or all-NaN):
      CDIS = rz(coexact_energy) + rz(|L²u|) + rz(nonlin_grad_energy)
    """
    Lu  = L @ u
    L2u = L @ Lu

    i_idx = edges["i"].values.astype(int)
    j_idx = edges["j"].values.astype(int)
    deg   = np.maximum(
        np.bincount(i_idx, minlength=n) + np.bincount(j_idx, minlength=n), 1
    ).astype(float)
    diff2  = (u[i_idx] - u[j_idx]) ** 2
    nonlin = np.bincount(i_idx, weights=diff2, minlength=n) / deg

    use_ratio = (exact_energy is not None and
                 not np.all(np.isnan(exact_energy)))

    if use_ratio:
        coexact_exact_ratio = u / (exact_energy + 1e-12)
        cdis = (robust_zscore(coexact_exact_ratio) +
                robust_zscore(np.abs(L2u)) +
                robust_zscore(nonlin))
        formula = ("rz(coexact_exact_ratio) + rz(|L2u|) + rz(nonlin_grad)"
                   " [robust z-score: median-centered, IQR-scaled]")
    else:
        cdis = (robust_zscore(u) +
                robust_zscore(np.abs(L2u)) +
                robust_zscore(nonlin))
        formula = ("rz(coexact_energy) + rz(|L2u|) + rz(nonlin_grad)"
                   " [ratio unavailable; robust z-score: median-centered, IQR-scaled]")

    return cdis, formula


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

    n = len(spots)
    u = spots["coexact_energy"].values.astype(float)
    L = build_laplacian(edges, n)

    if "flux_exact" in edges.columns:
        i_idx = edges["i"].values.astype(int)
        j_idx = edges["j"].values.astype(int)
        deg   = np.maximum(
            np.bincount(i_idx, minlength=n) + np.bincount(j_idx, minlength=n), 1
        ).astype(float)
        exact_e = np.bincount(
            i_idx, weights=edges["flux_exact"].values**2, minlength=n
        ) / deg
    else:
        exact_e = None

    cdis, formula = compute_cdis(u, L, edges, n, exact_e)
    spots["CDIS"] = cdis
    spots["cdis_formula"] = formula

    out_cols = ["node_id", "regime", "coexact_energy", "CDIS", "cdis_formula"]
    out_cols = [c for c in out_cols if c in spots.columns]
    spots[out_cols].to_csv(outdir / f"{sid}_constraint_score.csv", index=False)

    int_mask  = spots["regime"].isin({"interface_like"})
    bulk_mask = spots["regime"].isin({"bulk_like"})

    if int_mask.sum() < 3 or bulk_mask.sum() < 3:
        return {"sample_id": sid, "status": "insufficient_nodes",
                "n_interface": int(int_mask.sum()), "n_bulk": int(bulk_mask.sum())}

    obs       = float(np.median(cdis[int_mask]))
    bulk_cdis = cdis[bulk_mask]
    null      = [float(np.median(
                    rng.choice(bulk_cdis, size=int(int_mask.sum()), replace=True)
                )) for _ in range(n_perm)]
    p = float((np.array(null) >= obs).mean())

    gap = obs - float(np.median(bulk_cdis))
    print(f"  [{sid}] CDIS interface={obs:.4f} | bulk={float(np.median(bulk_cdis)):.4f}"
          f" | gap={gap:+.4f} | p={p:.4g}")

    return {
        "sample_id":             sid,
        "status":                "ok",
        "n_interface":           int(int_mask.sum()),
        "n_bulk":                int(bulk_mask.sum()),
        "interface_CDIS_median": obs,
        "bulk_CDIS_median":      float(np.median(bulk_cdis)),
        "cdis_gap":              gap,
        "p_value":               p,
        "significant":           int(p < 0.05),
        "cdis_formula":          formula,
    }


def main():
    ap = argparse.ArgumentParser(description="Step 19: Constraint-Dominated Interface Score (robust)")
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

    rows = []
    for sid in sample_ids:
        print(f"\n── {sid} ──")
        r = process_sample(sid, args.statsdir, outdir, args.n_perm, args.seed)
        if r:
            rows.append(r)

    if rows:
        df  = pd.DataFrame(rows)
        out = outdir / "cohort_constraint_score_summary.csv"
        df.to_csv(out, index=False)

        ok = df[df["status"] == "ok"]
        if len(ok):
            n_sig = int(ok["significant"].sum())
            n_tot = len(ok)
            bt    = binomtest(n_sig, n_tot, 0.5, alternative="greater")
            print(f"\n=== CDIS COHORT SUMMARY ===")
            print(f"  {n_sig}/{n_tot} sections: interface CDIS > bulk null (p<0.05)")
            print(f"  Sign test p = {bt.pvalue:.4g}")
            print(f"  Median interface CDIS = {ok['interface_CDIS_median'].median():.4f}")
            print(f"  Median bulk CDIS      = {ok['bulk_CDIS_median'].median():.4f}")
            print(f"  Median CDIS gap (interface - bulk) = "
                  f"{ok['cdis_gap'].median():.4f}")
            print(f"  CDIS formula: {ok['cdis_formula'].iloc[0]}")
            print(f"\nINTERPRETATION: Robust CDIS uses median-centered IQR-scaled")
            print(f"z-scores to prevent outlier inflation. Both interface and bulk")
            print(f"CDIS may be negative (absolute values are section-relative).")
            print(f"The test is: interface > bulk, not interface > 0.")
        print(f"\n[done] {out}")

if __name__ == "__main__":
    main()
