"""
Step 22 — Independent Biological Marker Validation of NC Norm

Purpose
-------
Test whether the per-node non-commutativity norm NC_i predicts independent
biological signals that were NOT used to construct the wedge flux.

Circularity boundary
--------------------
The wedge flux uses:
  Tumor score A  : EPCAM, KRT8, KRT18, KRT19, ERBB2, MUC1, TACSTD2
  Immune score B : PTPRC, CD3D, CD3E, NKG7, CD68, C1QA, CXCL9, CXCL10
  Stroma score   : COL1A1, COL1A2, DCN, LUM, POSTN, FAP, TAGLN

All validation markers below are OUTSIDE these gene sets.

Independent validation markers (safe)
--------------------------------------
  Cytotoxic activity : GZMB, PRF1, GNLY, IFNG
  CD8 T cells        : CD8A, CD8B
  Chemokine recruit  : CXCL12, CCL5, CCL2
  Antigen present    : HLA-DRA, HLA-DRB1, B2M
  Tumor hypoxia      : HIF1A, VEGFA, LDHA
  M2 macrophage      : CD163, MRC1

Three tests per marker
----------------------
  Test 1 — Node-level Spearman ρ(NC_i, marker_i)  [global and within-interface]
  Test 2 — Interface specificity Δ = ρ_interface − ρ_other_mean
  Test 3 — Top-k enrichment ratio (permutation-tested)
           top_k nodes by NC_norm → mean(marker) vs mean(marker) all nodes

Modes
-----
  --mode sample   Per-sample analysis → stats CSV
  --mode cohort   Aggregate → manuscript summary table

Usage
-----
  python step22_ncg_bio_validation.py \\
    --mode sample --sample-id GSM_6433619 \\
    --flux-tag flux_tumor_immune_region_interface_weighted \\
    --h5-file path/to/GSM_6433619_filtered_feature_bc_matrix.h5

  python step22_ncg_bio_validation.py \\
    --mode cohort \\
    --flux-tag flux_tumor_immune_region_interface_weighted
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr, binomtest
from statsmodels.stats.multitest import multipletests


# =============================================================================
# INDEPENDENT MARKER GENE SETS (outside the wedge construction)
# =============================================================================

MARKER_SETS: dict[str, list[str]] = {
    "cytotoxic":     ["GZMB", "PRF1", "GNLY", "IFNG"],
    "cd8_tcell":     ["CD8A", "CD8B"],
    "chemokine":     ["CXCL12", "CCL5", "CCL2"],
    "antigen_pres":  ["HLA-DRA", "HLA-DRB1", "B2M"],
    "hypoxia":       ["HIF1A", "VEGFA", "LDHA"],
    "m2_macrophage": ["CD163", "MRC1"],
}

TOP_K_FRAC = 0.10   # top 10% of nodes by NC_norm for enrichment test
N_PERM     = 1000


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", choices=["sample", "cohort"], default="sample")
    p.add_argument("--sample-id",  default=None)
    p.add_argument("--flux-tag",
                   default="flux_tumor_immune_region_interface_weighted")
    p.add_argument("--statsdir",   default="stats/CSV_GSM")
    p.add_argument("--h5-file",    default=None,
                   help="Path to filtered_feature_bc_matrix.h5 for this sample.")
    p.add_argument("--h5-dir",     default=None,
                   help="Directory containing per-sample H5 files "
                        "(named {sample_id}_filtered_feature_bc_matrix.h5).")
    p.add_argument("--outdir",     default="stats/CSV_GSM")
    p.add_argument("--top-k-frac", type=float, default=TOP_K_FRAC)
    p.add_argument("--n-perm",     type=int,   default=N_PERM)
    p.add_argument("--seed",       type=int,   default=42)
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


def spearman_safe(x, y) -> tuple[float, float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 5:
        return np.nan, np.nan, n
    r, p = spearmanr(x[mask], y[mask])
    return float(r), float(p), n


# =============================================================================
# LOAD MARKER EXPRESSION FROM H5
# =============================================================================

def load_marker_scores(
    h5_path: Path,
    node_df: pd.DataFrame,
    marker_sets: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Load Visium H5, compute mean log-normalised expression per marker set,
    align to node_df by barcode.

    Returns node_df with additional columns: one per marker set name.
    """
    print(f"  Loading H5: {h5_path.name}")
    adata = sc.read_10x_h5(str(h5_path))
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Map barcode → marker scores
    bc_to_scores: dict[str, dict[str, float]] = {}
    available_genes = set(adata.var_names)

    for set_name, genes in marker_sets.items():
        present = [g for g in genes if g in available_genes]
        absent  = [g for g in genes if g not in available_genes]
        if absent:
            print(f"    {set_name}: {len(present)}/{len(genes)} genes found "
                  f"(missing: {absent})")
        if not present:
            print(f"    [skip] {set_name}: no genes found in dataset")
            continue

        gene_idx = [adata.var_names.get_loc(g) for g in present]
        expr     = adata.X[:, gene_idx]
        if hasattr(expr, "toarray"):
            expr = expr.toarray()
        scores   = expr.mean(axis=1)  # mean across genes per spot

        for bc, sc_val in zip(adata.obs_names, scores):
            bc_to_scores.setdefault(bc, {})[set_name] = float(sc_val)

    if not bc_to_scores:
        print("  [warning] No marker scores computed — check gene names.")
        return node_df

    score_df = pd.DataFrame.from_dict(bc_to_scores, orient="index")
    score_df.index.name = "barcode"
    score_df = score_df.reset_index()

    merged = node_df.merge(score_df, on="barcode", how="left")
    return merged


# =============================================================================
# TEST 1 & 2 — NODE-LEVEL SPEARMAN + INTERFACE SPECIFICITY
# =============================================================================

def run_correlation_tests(
    df: pd.DataFrame,
    marker_cols: list[str],
    nc_col: str = "log1p_nc_norm",
    region_col: str = "region_step2",
) -> list[dict]:
    results = []

    for marker in marker_cols:
        if marker not in df.columns:
            continue
        x = df[nc_col].to_numpy(dtype=float)
        y = df[marker].to_numpy(dtype=float)

        # Global
        rho_g, p_g, n_g = spearman_safe(x, y)

        # Per region
        rho_int = np.nan; p_int = np.nan; n_int = 0
        rho_other_vals = []

        for region, sub in df.groupby(region_col):
            xi = sub[nc_col].to_numpy(dtype=float)
            yi = sub[marker].to_numpy(dtype=float)
            r, p, n = spearman_safe(xi, yi)
            if region == "interface_like":
                rho_int, p_int, n_int = r, p, n
            elif np.isfinite(r):
                rho_other_vals.append(r)

        rho_other_mean = float(np.mean(rho_other_vals)) if rho_other_vals else np.nan
        delta = float(rho_int - rho_other_mean) if np.isfinite(rho_int) else np.nan

        results.append({
            "marker":          marker,
            "n_global":        n_g,
            "rho_global":      rho_g,
            "p_global":        p_g,
            "n_interface":     n_int,
            "rho_interface":   rho_int,
            "p_interface":     p_int,
            "rho_other_mean":  rho_other_mean,
            "delta":           delta,
        })

    return results


# =============================================================================
# TEST 3 — TOP-K ENRICHMENT
# =============================================================================

def top_k_enrichment(
    df: pd.DataFrame,
    marker_cols: list[str],
    nc_col: str = "log1p_nc_norm",
    k_frac: float = 0.10,
    n_perm: int = 1000,
    seed: int = 42,
) -> list[dict]:
    """
    Top-k enrichment: take nodes in top k_frac by NC_norm.
    Compute mean(marker) in top-k vs mean(marker) in all nodes.
    Permutation null: shuffle NC_norm labels and recompute ratio.
    """
    rng = np.random.default_rng(seed)
    k   = max(1, int(len(df) * k_frac))
    nc  = df[nc_col].to_numpy(dtype=float)
    top_mask = nc >= np.nanpercentile(nc, 100 * (1 - k_frac))

    results = []
    for marker in marker_cols:
        if marker not in df.columns:
            continue
        y = df[marker].to_numpy(dtype=float)
        valid = np.isfinite(y) & np.isfinite(nc)
        if valid.sum() < 10:
            continue

        mean_top = float(np.nanmean(y[top_mask & valid]))
        mean_all = float(np.nanmean(y[valid]))
        if mean_all < 1e-10:
            ratio = np.nan
        else:
            ratio = mean_top / mean_all

        # Permutation null
        null_ratios = []
        for _ in range(n_perm):
            nc_perm  = rng.permutation(nc[valid])
            top_p    = nc_perm >= np.nanpercentile(nc_perm, 100 * (1 - k_frac))
            m_top_p  = float(np.nanmean(y[valid][top_p]))
            null_ratios.append(m_top_p / mean_all if mean_all > 1e-10 else np.nan)

        null_arr  = np.array(null_ratios, dtype=float)
        null_arr  = null_arr[np.isfinite(null_arr)]
        emp_p     = float((np.sum(null_arr >= ratio) + 1) / (len(null_arr) + 1)) \
                    if len(null_arr) and np.isfinite(ratio) else np.nan

        results.append({
            "marker":          marker,
            "k":               k,
            "mean_top_k":      mean_top,
            "mean_all":        mean_all,
            "enrichment_ratio":ratio,
            "null_mean_ratio": float(np.nanmean(null_arr)) if len(null_arr) else np.nan,
            "perm_p_greater":  emp_p,
        })

    return results


# =============================================================================
# SAMPLE MODE
# =============================================================================

def run_sample(sample_id, flux_tag, stats_dir, out_dir,
               h5_path, top_k_frac, n_perm, seed):

    print(f"\n=== {sample_id} ===")

    # ── Load Step 20 NC norm node file ────────────────────────────────────────
    node_file = require(stats_dir / f"{sample_id}_step20_ncg_stats_{flux_tag}.csv")
    # We need the per-node data — use the Step 6 node file merged with NC norm
    step6_node = require(stats_dir / f"{sample_id}_step6_nodes_hodge_{flux_tag}.csv")

    ncg_df  = pd.read_csv(node_file)
    node_df = pd.read_csv(step6_node)

    # The NCG stats CSV is per-region; we need per-node NC norm.
    # Recompute NC norm from Step 6 edge file.
    edge_file = require(stats_dir / f"{sample_id}_step6_edges_hodge_{flux_tag}.csv")
    edges = pd.read_csv(edge_file)

    if flux_tag not in edges.columns:
        raise ValueError(f"Column '{flux_tag}' not in edge file.")

    # Per-node NC norm = mean of squared raw flux over incident edges
    nc_rows = []
    for nid, grp in pd.concat([
        edges.rename(columns={"tail": "node"}).assign(node=edges["tail"]),
        edges.rename(columns={"head": "node"}).assign(node=edges["head"]),
    ]).groupby("node"):
        nc_rows.append({
            "node_id": int(nid),
            "nc_norm": float(np.mean(grp[flux_tag].to_numpy() ** 2)),
        })
    nc_df = pd.DataFrame(nc_rows)

    merged = node_df.merge(nc_df, on="node_id", how="left")
    merged["log1p_nc_norm"] = np.log1p(merged["nc_norm"])

    # ── Load independent marker expression ───────────────────────────────────
    if h5_path is None or not h5_path.exists():
        print(f"  [skip] H5 file not found: {h5_path}")
        return

    merged = load_marker_scores(h5_path, merged, MARKER_SETS)

    # Identify which marker set columns were loaded
    marker_cols = [col for col in MARKER_SETS.keys() if col in merged.columns]
    if not marker_cols:
        print("  [skip] No marker columns loaded.")
        return
    print(f"  Marker sets loaded: {marker_cols}")
    print(f"  N nodes: {len(merged)}")

    # ── Test 1 & 2 — Spearman correlations ───────────────────────────────────
    corr_results = run_correlation_tests(merged, marker_cols)

    # ── Test 3 — Top-k enrichment ─────────────────────────────────────────────
    topk_results = top_k_enrichment(merged, marker_cols,
                                    k_frac=top_k_frac, n_perm=n_perm, seed=seed)

    # ── FDR correction on interface p-values ─────────────────────────────────
    corr_df = pd.DataFrame(corr_results)
    corr_df["sample_id"] = sample_id
    mask = corr_df["p_interface"].notna()
    if mask.sum() > 1:
        _, pfc, _, _ = multipletests(corr_df.loc[mask, "p_interface"], method="fdr_bh")
        corr_df.loc[mask, "p_interface_fdr"] = pfc

    topk_df = pd.DataFrame(topk_results)
    topk_df["sample_id"] = sample_id

    # ── Save ──────────────────────────────────────────────────────────────────
    corr_out = out_dir / f"{sample_id}_step22_ncg_biomarker_corr_{flux_tag}.csv"
    topk_out = out_dir / f"{sample_id}_step22_ncg_biomarker_topk_{flux_tag}.csv"
    corr_df.to_csv(corr_out, index=False)
    topk_df.to_csv(topk_out, index=False)
    print(f"  Saved → {corr_out.name}")
    print(f"  Saved → {topk_out.name}")

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n  TEST 1+2 — Spearman ρ(NC, marker):")
    print(f"  {'Marker':<18} {'ρ_global':>10} {'ρ_interface':>12} {'Δ':>8} {'p_int_fdr':>12}")
    print(f"  {'-'*60}")
    for _, r in corr_df.iterrows():
        flag = "✓" if (pd.notna(r.get("p_interface_fdr"))
                       and r["p_interface_fdr"] < 0.05) else "  "
        print(f"  {r['marker']:<18} {r['rho_global']:>10.3f} "
              f"{r['rho_interface']:>12.3f} {r['delta']:>8.3f} "
              f"{r.get('p_interface_fdr', np.nan):>12.3e}  {flag}")

    print(f"\n  TEST 3 — Top-{int(top_k_frac*100)}% NC enrichment:")
    print(f"  {'Marker':<18} {'Ratio':>8} {'Null':>8} {'p_perm':>10}")
    print(f"  {'-'*48}")
    for _, r in topk_df.iterrows():
        flag = "✓" if (pd.notna(r["perm_p_greater"])
                       and r["perm_p_greater"] < 0.05) else "  "
        print(f"  {r['marker']:<18} {r['enrichment_ratio']:>8.3f} "
              f"{r['null_mean_ratio']:>8.3f} {r['perm_p_greater']:>10.4f}  {flag}")


# =============================================================================
# COHORT MODE
# =============================================================================

def run_cohort(flux_tag, stats_dir, out_dir):

    corr_files = sorted(stats_dir.glob(f"*_step22_ncg_biomarker_corr_{flux_tag}.csv"))
    topk_files = sorted(stats_dir.glob(f"*_step22_ncg_biomarker_topk_{flux_tag}.csv"))

    if not corr_files:
        print(f"[cohort] No corr files found. Run --mode sample first.")
        return

    print(f"[cohort] Found {len(corr_files)} correlation files, "
          f"{len(topk_files)} top-k files.")

    # ── Correlation summary ───────────────────────────────────────────────────
    all_corr = pd.concat([pd.read_csv(f) for f in corr_files], ignore_index=True)
    all_topk = pd.concat([pd.read_csv(f) for f in topk_files], ignore_index=True) \
               if topk_files else pd.DataFrame()

    # Per marker: sign test on ρ_interface > 0 and Δ > 0
    markers = [m for m in all_corr["marker"].unique() if pd.notna(m)]
    corr_summary = []

    for marker in sorted(markers):
        sub = all_corr[all_corr["marker"] == marker].dropna(subset=["rho_interface"])
        if sub.empty:
            continue
        ri = sub["rho_interface"].to_numpy()
        dl = sub["delta"].dropna().to_numpy()
        n  = len(ri)

        n_pos_ri = int(np.sum(ri > 0))
        n_pos_dl = int(np.sum(dl > 0)) if len(dl) else 0
        sp_ri = binomtest(n_pos_ri, n, p=0.5, alternative="greater").pvalue
        sp_dl = binomtest(n_pos_dl, len(dl), p=0.5, alternative="greater").pvalue \
                if len(dl) else np.nan

        corr_summary.append({
            "marker":              marker,
            "n_sections":          n,
            "median_rho_int":      round(float(np.median(ri)), 3),
            "n_rho_int_pos":       f"{n_pos_ri}/{n}",
            "sign_p_rho_int":      round(float(sp_ri), 5),
            "median_delta":        round(float(np.median(dl)), 3) if len(dl) else np.nan,
            "n_delta_pos":         f"{n_pos_dl}/{len(dl)}" if len(dl) else "—",
            "sign_p_delta":        round(float(sp_dl), 5) if not np.isnan(sp_dl) else np.nan,
        })

    corr_sum_df = pd.DataFrame(corr_summary)

    # Top-k enrichment ratio summary
    topk_summary = []
    if not all_topk.empty:
        for marker in sorted(all_topk["marker"].unique()):
            sub = all_topk[all_topk["marker"] == marker].dropna(subset=["enrichment_ratio"])
            if sub.empty:
                continue
            ratios = sub["enrichment_ratio"].to_numpy()
            ns = len(ratios)
            n_above_1 = int(np.sum(ratios > 1.0))
            sp = binomtest(n_above_1, ns, p=0.5, alternative="greater").pvalue
            topk_summary.append({
                "marker":            marker,
                "n_sections":        ns,
                "median_ratio":      round(float(np.median(ratios)), 3),
                "n_ratio_gt1":       f"{n_above_1}/{ns}",
                "sign_p_ratio_gt1":  round(float(sp), 5),
            })
    topk_sum_df = pd.DataFrame(topk_summary)

    # ── Save ──────────────────────────────────────────────────────────────────
    corr_sum_df.to_csv(out_dir / f"cohort_step22_corr_summary_{flux_tag}.csv", index=False)
    if not topk_sum_df.empty:
        topk_sum_df.to_csv(out_dir / f"cohort_step22_topk_summary_{flux_tag}.csv", index=False)

    # ── Console tables ────────────────────────────────────────────────────────
    print("\n=== STEP 22 — CORRELATION SUMMARY (Tests 1+2) ===")
    print(f"{'Marker':<18} {'Med ρ_int':>10} {'N ρ>0':>8} {'Sign p':>10} "
          f"{'Med Δ':>8} {'N Δ>0':>8} {'Sign p Δ':>10}")
    print("-"*78)
    for _, r in corr_sum_df.iterrows():
        print(f"{r['marker']:<18} {r['median_rho_int']:>10.3f} "
              f"{r['n_rho_int_pos']:>8} {r['sign_p_rho_int']:>10.5f} "
              f"{r['median_delta']:>8.3f} {r['n_delta_pos']:>8} "
              f"{r['sign_p_delta']:>10.5f}")

    if not topk_sum_df.empty:
        print("\n=== STEP 22 — TOP-K ENRICHMENT SUMMARY (Test 3) ===")
        print(f"{'Marker':<18} {'Med ratio':>10} {'N>1.0':>8} {'Sign p':>10}")
        print("-"*50)
        for _, r in topk_sum_df.iterrows():
            flag = "✓" if r["sign_p_ratio_gt1"] < 0.05 else "✗"
            print(f"{r['marker']:<18} {r['median_ratio']:>10.3f} "
                  f"{r['n_ratio_gt1']:>8} {r['sign_p_ratio_gt1']:>10.5f}  {flag}")

    print(f"\nSections: {all_corr['sample_id'].nunique()}")
    print("[cohort] Done.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if _ARGS.mode == "sample":
    if _ARGS.sample_id is None:
        import sys; print("Error: --sample-id required."); sys.exit(1)

    # Resolve H5 path
    h5 = None
    if _ARGS.h5_file:
        h5 = Path(_ARGS.h5_file)
    elif _ARGS.h5_dir:
        h5_dir = Path(_ARGS.h5_dir)
        # Try common naming conventions
        candidates = [
            h5_dir / _ARGS.sample_id / f"{_ARGS.sample_id}_filtered_feature_bc_matrix.h5",
            h5_dir / f"{_ARGS.sample_id}_filtered_feature_bc_matrix.h5",
            h5_dir / _ARGS.sample_id / "filtered_feature_bc_matrix.h5",
        ]
        h5 = next((f for f in candidates if f.exists()), None)
        if h5 is None:
            print(f"[warning] H5 not found in {h5_dir} for {_ARGS.sample_id}")
            print(f"  Tried: {[str(c) for c in candidates]}")

    run_sample(
        sample_id=_ARGS.sample_id,
        flux_tag=FLUX_TAG,
        stats_dir=STATS_DIR,
        out_dir=OUT_DIR,
        h5_path=h5,
        top_k_frac=_ARGS.top_k_frac,
        n_perm=_ARGS.n_perm,
        seed=_ARGS.seed,
    )

elif _ARGS.mode == "cohort":
    run_cohort(FLUX_TAG, STATS_DIR, OUT_DIR)
