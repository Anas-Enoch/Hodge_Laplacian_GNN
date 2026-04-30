"""
Step 21 — Zeta Function Spectral Diagnostic

Theoretical basis
-----------------
We ask whether the spatial distribution of coexact energy across tissue nodes
is concentrated in low-frequency (geometrically structured, large-scale)
eigenmodes of the spatial graph, or spread uniformly across high-frequency
(noise-like) modes.

The signal used is per-node coexact energy E_coexact_i (from Step 6 nodes CSV),
projected onto eigenvectors of the node graph Laplacian L0 = D - A.

WHY L0 NOT L1
-------------
Projecting the edge-level coexact flux onto L1 = B1^T B1 eigenvectors produces
zero projections by construction: the coexact subspace is orthogonal to the
exact subspace (image of B1^T). Using per-node coexact energy and L0 eigenvectors
gives a well-defined, interpretable spectral diagnostic: does coexact energy
live in spatially smooth large-scale patterns or high-frequency noise?

Primary statistics
------------------
  α_k  = <log(1+E_coexact), φ_k>²      spectral energy in mode k
  Z(s) = Σ_k α_k λ_k^{-s} / Σ_k α_k   zeta-weighted concentration (s=1, 2)
  Gini                                   unequality of spectral energy distribution
  f_low = fraction of energy in bottom half of modes

Null: 1000 spatial permutations of node labels (shuffle E_coexact across nodes).
Z(s) > null means coexact energy has large-scale spatial coherence.

Modes
-----
  --mode sample   Per-sample analysis
  --mode cohort   Aggregate all per-sample CSVs → manuscript summary table

Usage
-----
  python step21_zeta_spectral.py \\
    --mode sample --sample-id GSM_6433619 \\
    --flux-tag flux_tumor_immune_region_interface_weighted \\
    --statsdir stats/CSV_GSM --outdir stats/CSV_GSM

  python step21_zeta_spectral.py \\
    --mode cohort \\
    --flux-tag flux_tumor_immune_region_interface_weighted \\
    --statsdir stats/CSV_GSM --outdir stats/CSV_GSM
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.stats import binomtest


def _parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["sample", "cohort"], default="sample")
    p.add_argument("--sample-id", default=None)
    p.add_argument("--flux-tag", default="flux_tumor_immune_region_interface_weighted")
    p.add_argument("--statsdir", default="stats/CSV_GSM")
    p.add_argument("--outdir",   default="stats/CSV_GSM")
    p.add_argument("--max-modes", type=int, default=100)
    p.add_argument("--n-perm",    type=int, default=1000)
    p.add_argument("--seed",      type=int, default=42)
    return p.parse_args()

_ARGS     = _parse_args()
FLUX_TAG  = _ARGS.flux_tag
STATS_DIR = Path(_ARGS.statsdir)
OUT_DIR   = Path(_ARGS.outdir)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def require(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def build_L0(edges, n_nodes):
    """Node graph Laplacian L0 = D - A."""
    tail = edges["tail"].to_numpy(dtype=int)
    head = edges["head"].to_numpy(dtype=int)
    ne   = len(edges)
    rows = np.concatenate([tail, head, np.arange(n_nodes)])
    cols = np.concatenate([head, tail, np.arange(n_nodes)])
    deg  = np.bincount(np.concatenate([tail, head]), minlength=n_nodes).astype(float)
    data = np.concatenate([-np.ones(2*ne), deg])
    return sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))


def compute_eigenmodes(L0, k):
    n   = L0.shape[0]
    k   = min(k, n - 2)
    ncv = min(n, max(2*k + 1, k + 30))
    try:
        vals, vecs = spla.eigsh(L0, k=k, which="SM", ncv=ncv,
                                tol=1e-5, maxiter=30*n)
        idx = np.argsort(vals)
        return vals[idx], vecs[:, idx]
    except Exception as e:
        print(f"  [error] Eigendecomposition failed: {e}")
        return None, None


def zeta_stats(signal, eigenvalues, eigenvectors,
               s_values=(1.0, 2.0), eps=1e-8):
    nz   = eigenvalues > eps
    lam  = eigenvalues[nz]
    vecs = eigenvectors[:, nz]
    if len(lam) == 0:
        return {f"Z_s{s}": np.nan for s in s_values} | {
            "spectral_gini": np.nan, "frac_energy_low_half": np.nan}

    alpha = np.array([(signal @ vecs[:, k])**2 for k in range(len(lam))])
    total = alpha.sum()
    if total < 1e-30:
        return {f"Z_s{s}": np.nan for s in s_values} | {
            "spectral_gini": np.nan, "frac_energy_low_half": np.nan}

    a_norm = alpha / total
    result = {}
    for s in s_values:
        w = lam**(-s);  w = w / w.sum()
        result[f"Z_s{s}"] = float(np.dot(a_norm, w * len(lam)))

    sorted_a = np.sort(a_norm)
    n = len(sorted_a)
    result["spectral_gini"] = float(
        1.0 - 2.0 * np.cumsum(sorted_a)[:-1].sum() / n - sorted_a[-1] / n)

    half = max(1, len(lam) // 2)
    result["frac_energy_low_half"] = float(alpha[:half].sum() / total)
    return result


def spatial_perm_null(signal, eigenvalues, eigenvectors,
                      n_perm=1000, seed=42, s_values=(1.0, 2.0)):
    rng  = np.random.default_rng(seed)
    null = {f"Z_s{s}": [] for s in s_values}
    null["spectral_gini"] = []; null["frac_energy_low_half"] = []
    for _ in range(n_perm):
        st = zeta_stats(rng.permutation(signal), eigenvalues, eigenvectors,
                        s_values=s_values)
        for k, v in st.items():
            if k in null:
                null[k].append(v)
    return {k: np.array(v, dtype=float) for k, v in null.items()}


def emp_p(obs, null):
    nc = null[np.isfinite(null)]
    if not len(nc) or not np.isfinite(obs):
        return np.nan
    return float((np.sum(nc >= obs) + 1) / (len(nc) + 1))


# =============================================================================
# SAMPLE MODE
# =============================================================================

def run_sample(sample_id, flux_tag, stats_dir, out_dir, max_modes, n_perm, seed):
    print(f"\nLoading files for {sample_id}...")

    node_file = require(stats_dir / f"{sample_id}_step6_nodes_hodge_{flux_tag}.csv")

    # Edge file: prefer step3_edges, fall back to step6 edges
    edge_candidates = [
        stats_dir / f"{sample_id}_step3_edges.csv",
        stats_dir / f"{sample_id}_step6_edges_hodge_{flux_tag}.csv",
    ]
    edge_file = next((f for f in edge_candidates if f.exists()), None)
    if edge_file is None:
        print(f"  [skip] No edge file found for {sample_id}.")
        return

    nodes = pd.read_csv(node_file)
    edges = pd.read_csv(edge_file)

    for col in ["node_id", "node_energy_coexact"]:
        if col not in nodes.columns:
            raise ValueError(f"Missing column '{col}' in node file.")
    for col in ["tail", "head"]:
        if col not in edges.columns:
            raise ValueError(f"Missing column '{col}' in edge file.")

    n_nodes = len(nodes); n_edges = len(edges)
    print(f"  n_nodes={n_nodes}, n_edges={n_edges}")

    # Build L0 and eigenmodes
    print("  Building L0 and eigenmodes...")
    L0 = build_L0(edges, n_nodes)
    eigenvalues, eigenvectors = compute_eigenmodes(L0, k=min(max_modes, n_nodes-2))
    if eigenvalues is None:
        return

    nz = np.sum(eigenvalues > 1e-8)
    print(f"  Nonzero modes: {nz}")
    if nz < 5:
        print("  [skip] Too few nonzero modes.")
        return

    # Signal: log(1 + E_coexact) per node, sorted by node_id
    nodes_s = nodes.sort_values("node_id").reset_index(drop=True)
    signal  = np.log1p(nodes_s["node_energy_coexact"].fillna(0.0).to_numpy(dtype=float))

    print("  Computing observed statistics...")
    obs = zeta_stats(signal, eigenvalues, eigenvectors)
    print(f"  Z(s=1)={obs.get('Z_s1.0', np.nan):.4f}  "
          f"Z(s=2)={obs.get('Z_s2.0', np.nan):.4f}  "
          f"Gini={obs.get('spectral_gini', np.nan):.4f}  "
          f"f_low={obs.get('frac_energy_low_half', np.nan):.4f}")

    print(f"  Generating {n_perm} spatial permutations...")
    null = spatial_perm_null(signal, eigenvalues, eigenvectors,
                             n_perm=n_perm, seed=seed)

    rows = []
    for metric, obs_val in obs.items():
        na = null.get(metric, np.array([]))
        rows.append({
            "sample_id": sample_id, "metric": metric,
            "observed":  obs_val,
            "null_mean": float(np.nanmean(na)) if len(na) else np.nan,
            "null_std":  float(np.nanstd(na))  if len(na) else np.nan,
            "empirical_p_greater": emp_p(obs_val, na),
            "n_modes": int(nz), "n_nodes": n_nodes,
        })

    df = pd.DataFrame(rows)
    out = out_dir / f"{sample_id}_step21_zeta_stats_{flux_tag}.csv"
    df.to_csv(out, index=False)
    print(f"  Saved → {out}")

    print(f"\n  === ZETA SUMMARY — {sample_id} ===")
    for _, r in df.iterrows():
        flag = "✓" if pd.notna(r["empirical_p_greater"]) and r["empirical_p_greater"] < 0.05 else "✗"
        print(f"  {r['metric']:<30}  obs={r['observed']:.4f}  "
              f"null={r['null_mean']:.4f}  p={r['empirical_p_greater']:.4f}  {flag}")


# =============================================================================
# COHORT MODE
# =============================================================================

def run_cohort(flux_tag, stats_dir, out_dir):
    files = sorted(stats_dir.glob(f"*_step21_zeta_stats_{flux_tag}.csv"))
    if not files:
        print(f"[cohort] No files found.")
        return

    print(f"[cohort] Found {len(files)} per-sample stats files.")
    all_dfs = []
    for f in files:
        sid = f.name.replace(f"_step21_zeta_stats_{flux_tag}.csv", "")
        df = pd.read_csv(f); df["sample_id"] = sid
        all_dfs.append(df)

    long_df = pd.concat(all_dfs, ignore_index=True)
    print(f"[cohort] Metric names: {sorted(long_df['metric'].unique().tolist())}")

    long_df.to_csv(out_dir / f"cohort_step21_zeta_per_sample_{flux_tag}.csv", index=False)

    label_map = {
        "Z_s1.0": "Z(s=1)", "Z_s2.0": "Z(s=2)",
        "spectral_gini": "Spectral Gini",
        "frac_energy_low_half": "Frac low half",
    }
    rows = []
    for raw, label in label_map.items():
        sub = long_df[(long_df["metric"] == raw) &
                      long_df["observed"].notna() &
                      long_df["empirical_p_greater"].notna()]
        if sub.empty:
            print(f"  [skip] No valid data for '{raw}'")
            continue
        ov = sub["observed"].to_numpy()
        nm = sub["null_mean"].to_numpy()
        pv = sub["empirical_p_greater"].to_numpy()
        n  = len(ov)
        n_ab = int(np.sum(ov > nm))
        sp_p = binomtest(n_ab, n, p=0.5, alternative="greater").pvalue
        rows.append({
            "metric": label, "n_sections": n,
            "median_observed": round(float(np.median(ov)), 4),
            "n_above_null": f"{n_ab}/{n}",
            "sign_test_p":  round(float(sp_p), 5),
            "n_sig_p05":    int(np.sum(pv < 0.05)),
            "interpretation": "low-freq concentrated" if sp_p < 0.05 else "not concentrated",
        })

    if not rows:
        print("[cohort] ERROR: No valid data found in per-sample CSVs.")
        return

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / f"cohort_step21_zeta_summary_{flux_tag}.csv", index=False)

    print("\n=== STEP 21 ZETA COHORT SUMMARY ===")
    print(f"{'Metric':<20} {'Median':>8} {'N>null':>8} {'Sign p':>10} {'N sig':>6}  Interpretation")
    print("-"*70)
    for _, r in summary.iterrows():
        print(f"{r['metric']:<20} {r['median_observed']:>8.4f} "
              f"{r['n_above_null']:>8} {r['sign_test_p']:>10.5f} "
              f"{r['n_sig_p05']:>6}  {r['interpretation']}")

    print(f"\nSections: {long_df['sample_id'].nunique()}")
    print("\n=== INTERPRETATION ===")
    for _, r in summary.iterrows():
        flag = "✓" if r["sign_test_p"] < 0.05 else "✗"
        print(f"{flag} {r['metric']}: p={r['sign_test_p']:.5f}  → {r['interpretation']}")
    print("\n[cohort] Done.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if _ARGS.mode == "sample":
    if _ARGS.sample_id is None:
        import sys; print("Error: --sample-id required."); sys.exit(1)
    run_sample(_ARGS.sample_id, FLUX_TAG, STATS_DIR, OUT_DIR,
               _ARGS.max_modes, _ARGS.n_perm, _ARGS.seed)
elif _ARGS.mode == "cohort":
    run_cohort(FLUX_TAG, STATS_DIR, OUT_DIR)
