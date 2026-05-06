#!/usr/bin/env python3
"""
step23c_spatial_autocorrelation.py  — Part A (structure confirmation)
======================================================================
Tests whether the coexact energy field is spatially structured —
not random — by characterising its spatial autocorrelation function
and distinguishing it from random and periodic alternatives.

Part A is already empirically established (40/40 TNBC, p < 1e-12).
This script characterises the *form* of the structure, which is the
additional claim: locally correlated but without long-range periodicity.

Structure types discriminated by spatial autocorrelation shape
--------------------------------------------------------------
  RANDOM     ACF decays immediately (Moran's I ≈ 0 at all lags)
  PERIODIC   ACF oscillates with a characteristic period
             (negative values at lag = half-period)
  APERIODIC  ACF decays monotonically from positive at lag 1
             to near-zero at long range, WITHOUT oscillation.
             This means: local positive correlation, no long-range
             periodicity, consistent with constraint-driven local structure.

Method
------
  1. Compute graph-distance-based spatial autocorrelation:
       ACF(d) = Moran's I computed only among node pairs at graph distance d
     for d = 1, 2, 3, 4, 5 hops.
  2. Fit an exponential decay model: ACF(d) ≈ A exp(-d/tau)
     tau = characteristic correlation length (hops)
  3. Test for oscillation: does ACF change sign at any lag?
     Periodic → yes; Aperiodic → no.
  4. Compare coexact ACF shape against:
     (a) Label-permutation null (random baseline)
     (b) Exact-energy ACF (gradient field, should be smoother / longer-range)

Output
------
  {sid}_autocorrelation.csv       per-section ACF values and shape metrics
  cohort_autocorrelation.csv      cohort-level summary

Usage
-----
  python step23c_spatial_autocorrelation.py \\
      --statsdir Results_TNBC_rebuild_gse278936 \\
      --max-hops 5 --n-perm 100 --seed 123
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.stats import binomtest
from scipy.optimize import curve_fit


# ── Graph utilities ────────────────────────────────────────────────────────

def build_adjacency(edges: pd.DataFrame, n: int) -> sp.csr_matrix:
    i = edges["i"].values.astype(int)
    j = edges["j"].values.astype(int)
    A = sp.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    return (A + A.T).tocsr()


# ── Moran's I at a given graph distance ────────────────────────────────────

def morans_i_at_distance(u: np.ndarray,
                          dist_matrix: np.ndarray,
                          d: int,
                          max_pairs: int = 5000,
                          rng: np.random.Generator | None = None) -> float:
    """
    Compute Moran's I using only node pairs exactly d hops apart.
    Subsamples pairs if more than max_pairs to keep runtime bounded.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    rows, cols = np.where(dist_matrix == d)
    # Keep unique pairs (i < j)
    mask = rows < cols
    rows, cols = rows[mask], cols[mask]
    if len(rows) == 0:
        return np.nan
    if len(rows) > max_pairs:
        idx = rng.choice(len(rows), max_pairs, replace=False)
        rows, cols = rows[idx], cols[idx]

    u_bar = u.mean()
    denom = np.sum((u - u_bar) ** 2)
    if denom < 1e-12:
        return np.nan
    numer = np.sum((u[rows] - u_bar) * (u[cols] - u_bar))
    return float(numer / denom * len(u) / len(rows))


def exponential_decay(d: np.ndarray, A: float, tau: float) -> np.ndarray:
    return A * np.exp(-d / tau)


def fit_acf(lags: np.ndarray, acf_vals: np.ndarray) -> dict:
    """Fit ACF(d) = A exp(-d/tau). Returns A, tau, R² of fit."""
    valid = ~np.isnan(acf_vals)
    if valid.sum() < 3:
        return dict(decay_A=np.nan, decay_tau=np.nan, decay_r2=np.nan)
    try:
        popt, _ = curve_fit(exponential_decay, lags[valid], acf_vals[valid],
                            p0=[1.0, 2.0], maxfev=1000,
                            bounds=([0, 0.1], [10, 50]))
        A, tau = popt
        pred   = exponential_decay(lags[valid], A, tau)
        ss_res = np.sum((acf_vals[valid] - pred) ** 2)
        ss_tot = np.sum((acf_vals[valid] - acf_vals[valid].mean()) ** 2)
        r2     = float(1 - ss_res / (ss_tot + 1e-12))
        return dict(decay_A=float(A), decay_tau=float(tau), decay_r2=r2)
    except Exception:
        return dict(decay_A=np.nan, decay_tau=np.nan, decay_r2=np.nan)


# ── Per-section computation ────────────────────────────────────────────────

def process_sample(sid: str, statsdir: Path, outdir: Path,
                   max_hops: int, n_perm: int, seed: int) -> dict | None:
    rng = np.random.default_rng(seed)

    spots_path = statsdir / f"{sid}_spots_coexact_energy.csv"
    edges_path = statsdir / f"{sid}_edges_hodge.csv"
    if not spots_path.exists() or not edges_path.exists():
        print(f"  [{sid}] SKIP"); return None

    spots = pd.read_csv(spots_path)
    edges = pd.read_csv(edges_path)
    n     = len(spots)
    u     = spots["coexact_energy"].values.astype(float)
    A     = build_adjacency(edges, n)

    # Compute graph distances (cap at max_hops to avoid O(n²) cost)
    # Use sparse BFS up to max_hops
    dist_matrix = np.full((n, n), max_hops + 1, dtype=np.int32)
    np.fill_diagonal(dist_matrix, 0)
    current = A.copy().astype(bool)
    hop_mat = A.toarray().astype(np.int32)
    for d in range(1, max_hops + 1):
        rows, cols = np.where((hop_mat > 0) & (dist_matrix > d))
        if len(rows):
            dist_matrix[rows, cols] = d
        if d < max_hops:
            hop_mat = np.minimum(hop_mat @ A.toarray(), 1)

    lags = np.arange(1, max_hops + 1)

    # ── Observed ACF ──────────────────────────────────────────────────────
    acf_obs = np.array([morans_i_at_distance(u, dist_matrix, d, rng=rng)
                        for d in lags])

    fit      = fit_acf(lags, acf_obs)
    oscillates = int(np.any(acf_obs[~np.isnan(acf_obs)] < 0))
    monotone   = int(all(
        acf_obs[i] >= acf_obs[i+1]
        for i in range(len(acf_obs)-1)
        if not (np.isnan(acf_obs[i]) or np.isnan(acf_obs[i+1]))
    ))

    # ACF shape classification
    if oscillates:
        shape = "PERIODIC_CANDIDATE"
    elif float(np.nanmean(np.abs(acf_obs))) < 0.02:
        shape = "RANDOM"
    elif monotone:
        shape = "APERIODIC_CANDIDATE"
    else:
        shape = "MIXED"

    # ── Null ACF (permutation) ────────────────────────────────────────────
    null_acf_lag1 = []
    for _ in range(n_perm):
        u_p = rng.permutation(u)
        v   = morans_i_at_distance(u_p, dist_matrix, 1, rng=rng)
        if not np.isnan(v):
            null_acf_lag1.append(v)
    null_acf_lag1 = np.array(null_acf_lag1)
    p_acf1_gt_null = float((null_acf_lag1 <= acf_obs[0]).mean()) \
                     if (len(null_acf_lag1) and not np.isnan(acf_obs[0])) else np.nan

    # ── Exact-energy ACF for comparison ──────────────────────────────────
    acf_exact_lag1 = np.nan
    if "flux_exact" in edges.columns:
        i_idx = edges["i"].values.astype(int)
        j_idx = edges["j"].values.astype(int)
        deg   = np.maximum(
            np.bincount(i_idx, minlength=n) + np.bincount(j_idx, minlength=n), 1
        ).astype(float)
        ex = np.bincount(i_idx, weights=edges["flux_exact"].values**2,
                         minlength=n) / deg
        acf_exact_lag1 = morans_i_at_distance(ex, dist_matrix, 1, rng=rng)

    row = dict(
        sample_id       = sid,
        n_nodes         = n,
        acf_shape       = shape,
        acf_oscillates  = oscillates,
        acf_monotone    = monotone,
        **{f"acf_lag{d}": float(acf_obs[d-1]) for d in lags},
        **fit,
        null_acf_lag1_median   = float(np.nanmedian(null_acf_lag1)),
        p_acf_lag1_gt_null     = p_acf1_gt_null,
        acf_exact_lag1         = acf_exact_lag1,
        coexact_acf1_gt_exact  = int(
            not np.isnan(acf_obs[0]) and not np.isnan(acf_exact_lag1)
            and acf_obs[0] > acf_exact_lag1   # coexact more locally correlated?
        ),
    )

    pd.DataFrame([row]).to_csv(outdir / f"{sid}_autocorrelation.csv", index=False)
    acf_str = " ".join([f"{v:.3f}" if not np.isnan(v) else "nan"
                        for v in acf_obs])
    print(f"  [{sid}] {shape:25s} | ACF: [{acf_str}] "
          f"| tau={fit.get('decay_tau', np.nan):.2f}")
    return row


def main():
    ap = argparse.ArgumentParser(
        description="Step 23c: Spatial autocorrelation test")
    ap.add_argument("--statsdir",  type=Path, default=Path("Results_TNBC_rebuild_gse278936"))
    ap.add_argument("--outdir",    type=Path, default=None)
    ap.add_argument("--sample-id", type=str,  default=None)
    ap.add_argument("--max-hops",  type=int,  default=5,
                    help="Maximum graph hop distance for ACF computation")
    ap.add_argument("--n-perm",    type=int,  default=100)
    ap.add_argument("--seed",      type=int,  default=123)
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
                           args.max_hops, args.n_perm, args.seed)
        if r:
            rows.append(r)

    if not rows:
        print("No results."); return

    df  = pd.DataFrame(rows)
    out = outdir / "cohort_autocorrelation.csv"
    df.to_csv(out, index=False)

    print("\n=== SPATIAL AUTOCORRELATION COHORT SUMMARY (PART A — structure form) ===")
    print(df["acf_shape"].value_counts().to_string())
    print(f"\nMedian ACF lag-1 (local correlation): "
          f"{df['acf_lag1'].median():.4f}")
    print(f"Median decay tau (correlation length): "
          f"{df['decay_tau'].median():.2f} hops")
    n_aperiodic = int((df["acf_shape"] == "APERIODIC_CANDIDATE").sum())
    n_tot       = len(df)
    bt          = binomtest(n_aperiodic, n_tot, 0.5, alternative="greater")
    print(f"APERIODIC_CANDIDATE: {n_aperiodic}/{n_tot} (sign test p = {bt.pvalue:.4g})")
    print(f"\nINTERPRETATION:")
    print(f"  APERIODIC_CANDIDATE = monotone decay, no oscillation, positive lag-1")
    print(f"  PERIODIC_CANDIDATE  = sign change in ACF (long-range periodicity)")
    print(f"  RANDOM              = ACF near zero at all lags")
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
