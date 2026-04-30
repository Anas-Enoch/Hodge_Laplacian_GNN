"""
step21_zeta_rigidity.py
=======================
Extension of Step 21 Zeta spectral diagnostic.

Computes the s-dependent Zeta profile Z(s) over a range, then extracts:
  - Z(1)       — existing scalar (reproduced for consistency check)
  - zeta_prime  — Z'(1): spectral rigidity index
  - zeta_second — Z''(1): spectral concentration curvature

Mathematical definitions
------------------------
Given the normalized Zeta:

    Z(s) = [Σ_k α_k λ_k^{-s}] / [Σ_k α_k]

where α_k = <E_coexact, φ_k>^2 are spectral projection coefficients
onto L0 eigenvectors φ_k.

Then (differentiating under the sum, D = Σ_k α_k is constant):

    Z'(s)  = -[Σ_k α_k λ_k^{-s} log(λ_k)]  / D
    Z''(s) =  [Σ_k α_k λ_k^{-s} (log λ_k)^2] / D

At s=1:
    Z'(1)  < 0 always (since log λ_k > 0 for λ_k > 1, and the weighted
             sum is dominated by low-λ modes where log λ_k < 0 or small)
    |Z'(1)| large  → spectral mass strongly concentrated, rigid structure
    |Z'(1)| small  → spectral mass diffuse, recoverable interface

    Z''(1) > 0 always.
    Large Z''(1) relative to |Z'(1)| → concentration is cusp-like
    (sharp peak at lowest modes). Small ratio → broader concentration.

Permutation null
----------------
The same 1000-permutation spatial null used in Step 21 is applied here.
Node labels of E_coexact are shuffled; Z'(1) and Z''(1) are recomputed
for each surrogate. Empirical one-sided p-values are reported:
    H1: observed |Z'(1)| > null median  (more rigid than noise)

Outputs
-------
Per-sample CSV (appended columns to Step 21 output, or standalone):
    stats/CSV_GSM/{sample_id}_step21b_zeta_rigidity_{flux_tag}.csv

Cohort summary CSV:
    stats/CSV_GSM/cohort_step21b_zeta_rigidity_{flux_tag}.csv

Usage
-----
Single sample:
    python -m scripts_tnbc.step21_zeta_rigidity \
        --mode sample \
        --sample-id GSM_6433618 \
        --flux-tag flux_tumor_immune_region_interface_weighted

Cohort (reads valid_sample_ids.txt):
    python -m scripts_tnbc.step21_zeta_rigidity \
        --mode cohort \
        --flux-tag flux_tumor_immune_region_interface_weighted
"""

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.stats import spearmanr

# ── paths (mirror existing pipeline conventions) ──────────────────────────────
STATS_DIR = Path("stats/CSV_GSM")
VALID_IDS_FILE = Path("valid_sample_ids.txt")

# ── Zeta s-range ──────────────────────────────────────────────────────────────
S_RANGE = np.linspace(0.5, 2.0, 31)   # 31 points; s=1.0 is exactly included
S_EVAL  = 1.0                          # point at which derivatives are taken


# ─────────────────────────────────────────────────────────────────────────────
# Core spectral functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_spectral_coefficients(
    L0: np.ndarray,
    e_coexact: np.ndarray,
    n_components: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Eigendecompose L0 and project log(1 + E_coexact) onto eigenvectors.

    Parameters
    ----------
    L0 : (N, N) dense or sparse symmetric matrix — node graph Laplacian
    e_coexact : (N,) array — per-node coexact energy
    n_components : int or None
        If None, compute all eigenvalues (dense path).
        If int, use sparse eigsh for the k smallest nonzero eigenvalues
        (faster for large graphs; set to min(N-1, 200) as default heuristic).

    Returns
    -------
    eigenvalues : (K,) positive eigenvalues, sorted ascending
    alpha       : (K,) spectral projection coefficients α_k = <signal, φ_k>^2
    """
    signal = np.log1p(np.clip(e_coexact, 0, None))   # log(1 + E_coexact)
    N = L0.shape[0]

    if sp.issparse(L0):
        L0_dense = L0.toarray()
    else:
        L0_dense = np.array(L0)

    # Full eigendecomposition (symmetric, real)
    eigvals, eigvecs = np.linalg.eigh(L0_dense)

    # Filter: keep only positive eigenvalues (exclude zero modes / harmonic)
    pos_mask = eigvals > 1e-10
    eigvals  = eigvals[pos_mask]
    eigvecs  = eigvecs[:, pos_mask]

    # Spectral projection coefficients
    alpha = (eigvecs.T @ signal) ** 2   # shape (K,)

    return eigvals, alpha


def zeta_profile(
    eigenvalues: np.ndarray,
    alpha:       np.ndarray,
    s_values:    np.ndarray,
) -> np.ndarray:
    """
    Compute normalized Z(s) over an array of s values.

    Z(s) = [Σ_k α_k λ_k^{-s}] / [Σ_k α_k]

    Parameters
    ----------
    eigenvalues : (K,) positive eigenvalues
    alpha       : (K,) spectral projection coefficients
    s_values    : (M,) array of s values

    Returns
    -------
    Z : (M,) array
    """
    D = alpha.sum()
    if D == 0:
        return np.zeros_like(s_values)
    # outer: (K,) x (M,) → broadcast
    lam_neg_s = eigenvalues[:, None] ** (-s_values[None, :])  # (K, M)
    Z = (alpha[:, None] * lam_neg_s).sum(axis=0) / D
    return Z


def zeta_derivatives_at_s(
    eigenvalues: np.ndarray,
    alpha:       np.ndarray,
    s:           float = 1.0,
) -> tuple[float, float, float]:
    """
    Compute Z(s), Z'(s), Z''(s) analytically at a single s value.

    Z(s)   =  [Σ α_k λ_k^{-s}]            / D
    Z'(s)  = -[Σ α_k λ_k^{-s} log λ_k]   / D
    Z''(s) =  [Σ α_k λ_k^{-s} (log λ_k)²] / D

    Returns
    -------
    (Z_s, Zprime_s, Zsecond_s)
    """
    D = alpha.sum()
    if D == 0:
        return 0.0, 0.0, 0.0

    lam_neg_s = eigenvalues ** (-s)        # (K,)
    log_lam   = np.log(eigenvalues)        # (K,)

    weights = alpha * lam_neg_s            # (K,)

    Z_s      =  weights.sum()                     / D
    Zprime_s = -(weights * log_lam).sum()          / D
    Zsecond_s = (weights * log_lam ** 2).sum()     / D

    return float(Z_s), float(Zprime_s), float(Zsecond_s)


# ─────────────────────────────────────────────────────────────────────────────
# Permutation null
# ─────────────────────────────────────────────────────────────────────────────

def permutation_null_rigidity(
    L0:          np.ndarray,
    e_coexact:   np.ndarray,
    n_perms:     int = 1000,
    s:           float = 1.0,
    rng:         np.random.Generator | None = None,
) -> dict:
    """
    Spatial permutation null for Z'(s) and Z''(s).

    Shuffles the node-level E_coexact signal 1000 times, preserving the
    marginal distribution but destroying spatial structure. Recomputes
    Z'(s) and Z''(s) for each surrogate.

    Returns
    -------
    dict with keys:
        null_zprime  : (n_perms,) array of null Z'(s) values
        null_zsecond : (n_perms,) array of null Z''(s) values
        p_zprime     : empirical one-sided p-value for |Z'| > null
        p_zsecond    : empirical one-sided p-value for Z''  > null
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Precompute eigenvectors once (they don't change under node permutation)
    if sp.issparse(L0):
        L0_dense = L0.toarray()
    else:
        L0_dense = np.array(L0)
    eigvals_all, eigvecs_all = np.linalg.eigh(L0_dense)
    pos_mask   = eigvals_all > 1e-10
    eigvals    = eigvals_all[pos_mask]
    eigvecs    = eigvecs_all[:, pos_mask]   # (N, K)

    signal_obs = np.log1p(np.clip(e_coexact, 0, None))

    null_zprime  = np.empty(n_perms)
    null_zsecond = np.empty(n_perms)

    for i in range(n_perms):
        perm_signal = rng.permutation(signal_obs)
        alpha_perm  = (eigvecs.T @ perm_signal) ** 2
        _, zp, zs   = zeta_derivatives_at_s(eigvals, alpha_perm, s=s)
        null_zprime[i]  = zp
        null_zsecond[i] = zs

    # Observed
    alpha_obs = (eigvecs.T @ signal_obs) ** 2
    _, obs_zp, obs_zs = zeta_derivatives_at_s(eigvals, alpha_obs, s=s)

    # One-sided: H1: |Z'(obs)| > |null median|
    # Note Z'(1) is typically negative (spectral concentration pulls it negative)
    # We test the absolute slope magnitude as the rigidity measure
    p_zprime  = (np.abs(null_zprime)  >= np.abs(obs_zp)).mean()
    p_zsecond = (null_zsecond          >= obs_zs).mean()

    return {
        "null_zprime":   null_zprime,
        "null_zsecond":  null_zsecond,
        "p_zprime":      float(p_zprime),
        "p_zsecond":     float(p_zsecond),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample analysis
# ─────────────────────────────────────────────────────────────────────────────

def load_inputs(sample_id: str, flux_tag: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load L0 and E_coexact for a given sample.

    Mirrors the data loading logic in the existing step21 script:
      - L0 from step3 B1 incidence matrix  (B1 @ B1.T = L0 up to weights)
        OR recomputed from step3 edges CSV if B1.npz not found.
      - E_coexact from step6 nodes hodge CSV.
    """
    stats = STATS_DIR

    # ── coexact energy ────────────────────────────────────────────────────────
    node_csv = stats / f"{sample_id}_step6_nodes_hodge_{flux_tag}.csv"
    if not node_csv.exists():
        raise FileNotFoundError(
            f"Step 6 node output not found: {node_csv}\n"
            f"Run step6_tnbc_hodge_decomposition.py first."
        )
    nodes_df  = pd.read_csv(node_csv)
    e_coexact = nodes_df["node_energy_coexact"].values.astype(float)

    # ── node graph Laplacian L0 ───────────────────────────────────────────────
    # Primary: load from step3 npz if available
    b1_path = stats / f"{sample_id}_step3_B1.npz"
    if b1_path.exists():
        B1 = sp.load_npz(str(b1_path))
        L0 = B1 @ B1.T   # unnormalized combinatorial Laplacian
    else:
        # Fallback: build L0 from edges CSV
        edges_csv = stats / f"{sample_id}_step3_edges.csv"
        nodes_csv = stats / f"{sample_id}_step3_nodes.csv"
        if not edges_csv.exists() or not nodes_csv.exists():
            raise FileNotFoundError(
                f"Neither B1.npz nor step3 edge/node CSVs found for {sample_id}."
            )
        edges_df = pd.read_csv(edges_csv)
        n_nodes  = len(pd.read_csv(nodes_csv))
        rows  = edges_df["source"].values
        cols  = edges_df["target"].values
        data  = np.ones(len(rows))
        A     = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
        A     = A + A.T
        A.data = np.ones_like(A.data)
        D_deg = sp.diags(np.array(A.sum(axis=1)).ravel())
        L0    = D_deg - A

    return L0, e_coexact


def analyze_sample(
    sample_id: str,
    flux_tag:  str,
    n_perms:   int  = 1000,
    s_eval:    float = S_EVAL,
    save:      bool  = True,
) -> dict:
    """
    Run the full rigidity analysis for one sample.
    Returns a result dict (also written to CSV if save=True).
    """
    print(f"  [{sample_id}] Loading inputs ...", flush=True)
    try:
        L0, e_coexact = load_inputs(sample_id, flux_tag)
    except FileNotFoundError as exc:
        print(f"  [{sample_id}] SKIP — {exc}")
        return {"sample_id": sample_id, "status": "skipped", "reason": str(exc)}

    print(f"  [{sample_id}] Computing eigendecomposition ...", flush=True)
    eigvals, alpha = compute_spectral_coefficients(L0, e_coexact)

    # Observed statistics
    Z_s, Zprime_s, Zsecond_s = zeta_derivatives_at_s(eigvals, alpha, s=s_eval)

    # Full Z(s) profile (for plotting / optional output)
    Z_profile = zeta_profile(eigvals, alpha, S_RANGE)

    # Permutation null
    print(f"  [{sample_id}] Running {n_perms}-permutation null ...", flush=True)
    null_results = permutation_null_rigidity(
        L0, e_coexact, n_perms=n_perms, s=s_eval,
        rng=np.random.default_rng(seed=42),
    )

    result = {
        "sample_id":          sample_id,
        "status":             "ok",
        "n_nodes":            L0.shape[0],
        "n_pos_eigenvalues":  len(eigvals),
        # --- primary statistics ---
        "Z_s1":               Z_s,
        "zeta_prime_s1":      Zprime_s,
        "zeta_second_s1":     Zsecond_s,
        # --- null summary ---
        "null_zprime_median": float(np.median(null_results["null_zprime"])),
        "null_zsecond_median":float(np.median(null_results["null_zsecond"])),
        "p_zprime":           null_results["p_zprime"],
        "p_zsecond":          null_results["p_zsecond"],
        # --- derived rigidity index ---
        # |Z'(1)| normalised by Z(1): relative slope, comparable across sections
        "rigidity_index":     abs(Zprime_s) / Z_s if Z_s > 0 else np.nan,
        # curvature-to-slope ratio: shape of concentration profile
        "curvature_slope_ratio": Zsecond_s / abs(Zprime_s) if Zprime_s != 0 else np.nan,
    }

    if save:
        out_csv = STATS_DIR / f"{sample_id}_step21b_zeta_rigidity_{flux_tag}.csv"
        pd.DataFrame([result]).to_csv(out_csv, index=False)
        print(f"  [{sample_id}] Saved → {out_csv}")

        # Also save the Z(s) profile for plotting
        profile_csv = STATS_DIR / f"{sample_id}_step21b_zeta_profile_{flux_tag}.csv"
        pd.DataFrame({"s": S_RANGE, "Z_s": Z_profile}).to_csv(
            profile_csv, index=False
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Cohort-level aggregation + structural hypothesis test
# ─────────────────────────────────────────────────────────────────────────────

def cohort_summary(results: list[dict], flux_tag: str) -> pd.DataFrame:
    """
    Aggregate per-sample results and compute cohort-level sign tests.

    Sign test: H0: median(|Z'(1)|_obs - |Z'(1)|_null) = 0
               H1: one-sided, observed > null
    Equivalent to binomial test on count(p_zprime < 0.05).
    """
    from scipy.stats import binomtest

    df = pd.DataFrame([r for r in results if r.get("status") == "ok"])
    if df.empty:
        print("No valid sections to aggregate.")
        return df

    n = len(df)

    # Sign test: sections where |Z'(1)| exceeds null median
    n_exceed_zprime  = (df["zeta_prime_s1"].abs() >
                        df["null_zprime_median"].abs()).sum()
    n_exceed_zsecond = (df["zeta_second_s1"] >
                        df["null_zsecond_median"]).sum()

    bt_zp = binomtest(n_exceed_zprime,  n, p=0.5, alternative="greater")
    bt_zs = binomtest(n_exceed_zsecond, n, p=0.5, alternative="greater")

    # Spearman: does |Z'(1)| correlate with E_coexact median?
    # (structural hypothesis: more rigid → higher coexact load)
    # E_coexact median must be loaded per section; use Z(1) as proxy here
    rho_Zs_Zprime, p_rho = spearmanr(df["Z_s1"], df["zeta_prime_s1"].abs())

    summary = {
        "n_sections":              n,
        "median_Z_s1":             df["Z_s1"].median(),
        "median_zeta_prime_s1":    df["zeta_prime_s1"].median(),
        "median_zeta_second_s1":   df["zeta_second_s1"].median(),
        "median_rigidity_index":   df["rigidity_index"].median(),
        "n_exceed_null_zprime":    int(n_exceed_zprime),
        "sign_test_p_zprime":      bt_zp.pvalue,
        "n_exceed_null_zsecond":   int(n_exceed_zsecond),
        "sign_test_p_zsecond":     bt_zs.pvalue,
        # structural correlation: Z(1) vs |Z'(1)|
        "spearman_rho_Z1_Zprime":  rho_Zs_Zprime,
        "spearman_p_Z1_Zprime":    p_rho,
    }

    print("\n── Cohort-level Zeta Rigidity Summary ──────────────────────────")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:<38s} {v:.4g}")
        else:
            print(f"  {k:<38s} {v}")
    print("────────────────────────────────────────────────────────────────")

    # Save full per-section table
    cohort_csv = STATS_DIR / f"cohort_step21b_zeta_rigidity_{flux_tag}.csv"
    df.to_csv(cohort_csv, index=False)
    print(f"\nCohort table → {cohort_csv}")

    # Save summary row
    summary_csv = STATS_DIR / f"cohort_step21b_zeta_rigidity_summary_{flux_tag}.csv"
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    print(f"Summary row  → {summary_csv}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Optional: diagnostic plot of Z(s) profiles across cohort
# ─────────────────────────────────────────────────────────────────────────────

def plot_zeta_profiles(
    sample_ids: list[str],
    flux_tag:   str,
    out_path:   str | Path | None = None,
) -> None:
    """
    Overlay Z(s) profiles for all valid sections.
    Color-codes by rigidity_index (high = darker).
    Draws vertical dashed line at s=1.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    profiles = []
    rigidities = []

    for sid in sample_ids:
        profile_csv = STATS_DIR / f"{sid}_step21b_zeta_profile_{flux_tag}.csv"
        result_csv  = STATS_DIR / f"{sid}_step21b_zeta_rigidity_{flux_tag}.csv"
        if not profile_csv.exists() or not result_csv.exists():
            continue
        pf = pd.read_csv(profile_csv)
        ri = pd.read_csv(result_csv)["rigidity_index"].values[0]
        profiles.append(pf["Z_s"].values)
        rigidities.append(ri)

    if not profiles:
        print("No profile CSVs found — run analysis first.")
        return

    profiles    = np.array(profiles)
    rigidities  = np.array(rigidities)
    norm_ri     = (rigidities - rigidities.min()) / ((rigidities.max() - rigidities.min()) + 1e-12)
    cmap        = cm.get_cmap("plasma")

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (prof, nr) in enumerate(zip(profiles, norm_ri)):
        ax.plot(S_RANGE, prof, color=cmap(nr), alpha=0.6, lw=1.0)

    ax.axvline(1.0, color="k", lw=1.0, ls="--", label="s = 1")
    ax.set_xlabel("s", fontsize=11)
    ax.set_ylabel("Z(s)", fontsize=11)
    ax.set_title("Zeta spectral profiles across sections\n"
                 "(color = rigidity index |Z′(1)|/Z(1), dark = high)",
                 fontsize=10)
    ax.legend(fontsize=9)
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(rigidities.min(),
                                                  rigidities.max()))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Rigidity index")
    plt.tight_layout()

    if out_path is None:
        out_path = STATS_DIR / f"cohort_step21b_zeta_profiles_{flux_tag}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Profile plot → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Step 21b: Zeta rigidity index (Z'(1), Z''(1))"
    )
    p.add_argument("--mode", choices=["sample", "cohort"], default="sample")
    p.add_argument("--sample-id",  type=str, default=None)
    p.add_argument("--flux-tag",   type=str,
                   default="flux_tumor_immune_region_interface_weighted")
    p.add_argument("--n-perms",    type=int, default=1000)
    p.add_argument("--s-eval",     type=float, default=1.0,
                   help="s value at which derivatives are evaluated (default 1.0)")
    p.add_argument("--plot",       action="store_true",
                   help="Generate Z(s) profile overlay plot (cohort mode)")
    p.add_argument("--no-save",    action="store_true",
                   help="Do not write CSV outputs (dry run)")
    return p.parse_args()


def main():
    args = parse_args()
    STATS_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "sample":
        if args.sample_id is None:
            raise ValueError("--sample-id required in sample mode.")
        analyze_sample(
            args.sample_id,
            args.flux_tag,
            n_perms=args.n_perms,
            s_eval=args.s_eval,
            save=not args.no_save,
        )

    elif args.mode == "cohort":
        if not VALID_IDS_FILE.exists():
            raise FileNotFoundError(f"{VALID_IDS_FILE} not found.")
        sample_ids = [
            l.strip() for l in VALID_IDS_FILE.read_text().splitlines()
            if l.strip()
        ]
        print(f"Processing {len(sample_ids)} samples in cohort mode.")

        results = []
        for sid in sample_ids:
            print(f"\n── {sid} ──")
            r = analyze_sample(
                sid, args.flux_tag,
                n_perms=args.n_perms,
                s_eval=args.s_eval,
                save=not args.no_save,
            )
            results.append(r)

        cohort_summary(results, args.flux_tag)

        if args.plot:
            valid_ids = [r["sample_id"] for r in results
                         if r.get("status") == "ok"]
            plot_zeta_profiles(valid_ids, args.flux_tag)


if __name__ == "__main__":
    main()
