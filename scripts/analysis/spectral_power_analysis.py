#!/usr/bin/env python3
"""
step23a_power_spectrum_test.py  — Part B: Power Spectrum Shape Test
====================================================================
Tests whether the coexact energy field has a power spectrum consistent
with aperiodic structure: continuous, no dominant frequency peaks,
but NOT flat (not white noise).

Three discriminable spectral regimes:
  PERIODIC         discrete dominant peaks — high peak_ratio, low entropy, high Gini
  RANDOM           flat spectrum          — peak_ratio ≈ 1, max entropy, Gini ≈ 0
  APERIODIC_CAND   structured, no peaks   — intermediate entropy, no single dominant
                                            mode, continuous but non-uniform

Method
------
  1. Compute k lowest eigenmodes of the node graph Laplacian L (graph Fourier).
  2. Project log(1 + E_coexact) onto each eigenmode → spectral coefficients alpha_k.
  3. Characterise the spectrum by:
       peak_ratio       = max(alpha_k) / mean(alpha_k)
       spectral_entropy = -sum(p_k log p_k),  p_k = alpha_k / sum(alpha_k)
       Gini coefficient of alpha_k distribution
       n_dominant_peaks = number of modes with alpha_k > 2 × mean
  4. Compare observed statistics against a label-permutation null (n_perm shuffles).
  5. Compare coexact spectrum against exact-energy spectrum:
       Aperiodic prediction: coexact is LESS peaked than exact.

Output
------
  {sid}_power_spectrum.csv       per-section spectral statistics
  cohort_power_spectrum.csv      cohort-level summary with classification counts

Usage
-----
  python step23a_power_spectrum_test.py \\
      --statsdir Results_TNBC_rebuild_gse278936 \\
      --n-perm 100 --k-eigs 50 --seed 123
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.stats import binomtest


# ── Graph utilities ────────────────────────────────────────────────────────

def build_laplacian(edges: pd.DataFrame, n: int) -> sp.csr_matrix:
    i = edges["i"].values.astype(int)
    j = edges["j"].values.astype(int)
    w = np.abs(edges["flux_coexact"].values.astype(float))
    A = sp.coo_matrix((w, (i, j)), shape=(n, n)).tocsr()
    A = A + A.T
    D = sp.diags(np.array(A.sum(axis=1)).ravel())
    return (D - A).tocsr()


def gini(x: np.ndarray) -> float:
    x = np.sort(np.abs(x.ravel()))
    n = len(x)
    if n == 0 or x.sum() < 1e-12:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * np.dot(idx, x) - (n + 1) * x.sum()) / (n * x.sum()))


def spectral_entropy(alpha: np.ndarray) -> float:
    s = alpha.sum()
    if s < 1e-12:
        return np.nan
    p = (alpha / s)
    p = p[p > 0]
    return float(-np.dot(p, np.log(p)))


# ── Main per-section computation ───────────────────────────────────────────

def compute_spectrum(u: np.ndarray, L: sp.csr_matrix, k: int) -> dict:
    """Graph Fourier power spectrum of signal u. Returns scalar statistics."""
    n   = L.shape[0]
    k   = min(k, n - 2)
    nan = {s: np.nan for s in ["peak_ratio", "spectral_entropy", "gini",
                                "n_dominant_peaks", "low_freq_frac",
                                "mid_freq_frac",   "high_freq_frac"]}
    if k < 4:
        return nan
    try:
        vals, vecs = eigsh(L, k=k, which="SM", tol=1e-6)
    except Exception:
        return nan

    order = np.argsort(vals)
    vals  = vals[order];  vecs = vecs[:, order]
    sig   = np.log1p(np.clip(u, 0, None))
    alpha = np.array([float(np.dot(sig, vecs[:, i])) ** 2 for i in range(k)])

    # Remove trivial zero-eigenvalue mode
    nz    = vals > 1e-8
    alpha = alpha[nz];  vals = vals[nz]
    if len(alpha) < 3:
        return nan

    peak_ratio = float(alpha.max() / (alpha.mean() + 1e-12))
    s_entropy  = spectral_entropy(alpha)
    g          = gini(alpha)
    n_dominant = int((alpha > 2 * alpha.mean()).sum())

    # Frequency band energy fractions
    cuts = np.percentile(vals, [33, 67])
    lo   = alpha[vals <= cuts[0]].sum()
    mid  = alpha[(vals > cuts[0]) & (vals <= cuts[1])].sum()
    hi   = alpha[vals > cuts[1]].sum()
    tot  = lo + mid + hi + 1e-12

    return dict(peak_ratio=peak_ratio, spectral_entropy=s_entropy,
                gini=g, n_dominant_peaks=n_dominant,
                low_freq_frac=float(lo/tot), mid_freq_frac=float(mid/tot),
                high_freq_frac=float(hi/tot))


def classify(peak_ratio: float, s_entropy: float,
             gini_val:  float, k: int) -> str:
    """Heuristic classification; all continuous metrics also reported."""
    if np.isnan(peak_ratio):
        return "UNKNOWN"
    max_h = np.log(max(k, 1))
    if peak_ratio > 5.0 and gini_val > 0.60:
        return "PERIODIC"
    if s_entropy > 0.85 * max_h and gini_val < 0.15:
        return "RANDOM"
    return "APERIODIC_CANDIDATE"


def process_sample(sid: str, statsdir: Path, outdir: Path,
                   n_perm: int, seed: int, k: int) -> dict | None:
    rng = np.random.default_rng(seed)

    spots_path = statsdir / f"{sid}_spots_coexact_energy.csv"
    edges_path = statsdir / f"{sid}_edges_hodge.csv"
    if not spots_path.exists() or not edges_path.exists():
        print(f"  [{sid}] SKIP — missing inputs")
        return None

    spots = pd.read_csv(spots_path)
    edges = pd.read_csv(edges_path)
    n     = len(spots)
    u     = spots["coexact_energy"].values.astype(float)
    L     = build_laplacian(edges, n)

    # Observed coexact spectrum
    obs = compute_spectrum(u, L, k)
    cls = classify(obs["peak_ratio"], obs["spectral_entropy"], obs["gini"], k)

    # Exact-energy spectrum (for comparison)
    exact_obs = {}
    if "flux_exact" in edges.columns:
        i_idx = edges["i"].values.astype(int)
        j_idx = edges["j"].values.astype(int)
        deg   = np.maximum(
            np.bincount(i_idx, minlength=n) + np.bincount(j_idx, minlength=n), 1
        ).astype(float)
        ex = np.bincount(i_idx, weights=edges["flux_exact"].values**2,
                         minlength=n) / deg
        exact_obs = compute_spectrum(ex, L, k)

    # Label-permutation null (reduced k for speed)
    k_null = min(20, k)
    null_peak  = []
    null_gini  = []
    null_entr  = []
    for _ in range(n_perm):
        u_p = rng.permutation(u)
        s   = compute_spectrum(u_p, L, k_null)
        null_peak.append(s["peak_ratio"])
        null_gini.append(s["gini"])
        null_entr.append(s["spectral_entropy"])

    null_peak = np.array([x for x in null_peak if not np.isnan(x)])
    null_gini = np.array([x for x in null_gini if not np.isnan(x)])
    null_entr = np.array([x for x in null_entr if not np.isnan(x)])

    # One-sided p: observed MORE structured than null
    p_peak = float((null_peak <= obs["peak_ratio"]).mean()) if len(null_peak) else np.nan
    p_gini = float((null_gini <= obs["gini"]).mean())       if len(null_gini) else np.nan
    # One-sided p: observed LESS entropy than null (more concentrated)
    p_entr = float((null_entr >= obs["spectral_entropy"]).mean()) if len(null_entr) else np.nan

    row = dict(
        sample_id   = sid,
        n_nodes     = n,
        classification = cls,
        **{f"coexact_{k2}": v for k2, v in obs.items()},
        null_peak_ratio_median = float(np.nanmedian(null_peak)),
        null_gini_median       = float(np.nanmedian(null_gini)),
        null_entropy_median    = float(np.nanmedian(null_entr)),
        p_peak_ratio_gt_null   = p_peak,
        p_gini_gt_null         = p_gini,
        p_entropy_lt_null      = p_entr,
        exact_peak_ratio       = exact_obs.get("peak_ratio",       np.nan),
        exact_gini             = exact_obs.get("gini",             np.nan),
        exact_spectral_entropy = exact_obs.get("spectral_entropy", np.nan),
        coexact_less_peaked_than_exact = int(
            obs["peak_ratio"] < exact_obs.get("peak_ratio", obs["peak_ratio"] + 1)
        ),
    )

    pd.DataFrame([row]).to_csv(outdir / f"{sid}_power_spectrum.csv", index=False)
    print(f"  [{sid}] {cls:22s} | peak={obs['peak_ratio']:.2f} "
          f"| gini={obs['gini']:.3f} | H={obs['spectral_entropy']:.3f} "
          f"| p_peak={p_peak:.3f}")
    return row


def main():
    ap = argparse.ArgumentParser(description="Step 23a: Power spectrum shape test")
    ap.add_argument("--statsdir",  type=Path, default=Path("Results_TNBC_rebuild_gse278936"))
    ap.add_argument("--outdir",    type=Path, default=None)
    ap.add_argument("--sample-id", type=str,  default=None)
    ap.add_argument("--n-perm",    type=int,  default=100)
    ap.add_argument("--seed",      type=int,  default=123)
    ap.add_argument("--k-eigs",    type=int,  default=50)
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
                           args.n_perm, args.seed, args.k_eigs)
        if r:
            rows.append(r)

    if not rows:
        print("No results."); return

    df  = pd.DataFrame(rows)
    out = outdir / "cohort_power_spectrum.csv"
    df.to_csv(out, index=False)

    print("\n=== POWER SPECTRUM COHORT SUMMARY (PART B) ===")
    print(df["classification"].value_counts().to_string())
    print(f"\nMedian coexact peak_ratio : {df['coexact_peak_ratio'].median():.3f}")
    print(f"Median coexact Gini       : {df['coexact_gini'].median():.3f}")
    print(f"Median coexact entropy    : {df['coexact_spectral_entropy'].median():.3f}")
    print(f"Median null peak_ratio    : {df['null_peak_ratio_median'].median():.3f}")
    if "coexact_less_peaked_than_exact" in df.columns:
        n_less = int(df["coexact_less_peaked_than_exact"].sum())
        n_tot  = len(df)
        bt = binomtest(n_less, n_tot, 0.5, alternative="greater")
        print(f"\nCoexact less peaked than exact: {n_less}/{n_tot} "
              f"(sign test p = {bt.pvalue:.4g})")
        print("Prediction: coexact should be LESS periodic than exact.")
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
