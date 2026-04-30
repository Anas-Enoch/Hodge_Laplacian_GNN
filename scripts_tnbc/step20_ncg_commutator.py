"""
Step 20 — Non-Commutative Geometry (NCG) Commutator Grounding

Theoretical basis
-----------------
The wedge flux f_ij = ã_i b̃_j - ã_j b̃_i is the discrete analogue of the
commutator [M_A, M_B] of two biological program operators on the spatial graph.
Its coexact component therefore measures not arbitrary rotational structure, but
the geometric image of operator non-commutativity between programs A and B.

If the coexact enrichment at interfaces is a genuine geometric invariant — not
an artifact of the antisymmetric construction — then:
  (1) the per-node non-commutativity norm NC_i should correlate strongly with
      node-level coexact energy E_coexact_i, and
  (2) this correlation should be specifically elevated within interface-like
      nodes relative to other regions.

A construction artifact would produce NC_i ~ E_coexact_i uniformly everywhere.
Interface-specific elevation of the NC–coexact correlation distinguishes
geometric invariant from noise.

Primary test
------------
  Spearman ρ(NC_i, E_coexact_i) within interface_like nodes vs other regions.
  Sign test across sections: is ρ_interface consistently > ρ_other?

Modes
-----
  --mode sample   Per-sample analysis → stats CSV
  --mode cohort   Aggregate all per-sample CSVs → manuscript summary table

Usage
-----
  # Single sample
  python step20_ncg_commutator.py \\
    --mode sample --sample-id GSM_6433619 \\
    --flux-tag flux_tumor_immune_region_interface_weighted

  # Cohort aggregate (after running sample mode for all sections)
  python step20_ncg_commutator.py \\
    --mode cohort \\
    --flux-tag flux_tumor_immune_region_interface_weighted
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, binomtest
from statsmodels.stats.multitest import multipletests


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", choices=["sample", "cohort"], default="sample")
    p.add_argument("--sample-id", default=None,
                   help="GSM sample ID (required for --mode sample).")
    p.add_argument("--flux-tag",
                   default="flux_tumor_immune_region_interface_weighted",
                   help="Flux column tag matching Step 6 output.")
    p.add_argument("--statsdir", default="stats/CSV_GSM",
                   help="Directory of Step 6 CSVs.")
    p.add_argument("--outdir", default="stats/CSV_GSM",
                   help="Output directory.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


_ARGS = _parse_args()
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


def spearman_safe(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    """Returns (rho, p, n) with nan-safety."""
    mask = x.notna() & y.notna()
    n = int(mask.sum())
    if n < 5:
        return np.nan, np.nan, n
    rho, p = spearmanr(x[mask], y[mask])
    return float(rho), float(p), n


# =============================================================================
# CORE: PER-NODE NON-COMMUTATIVITY NORM
# =============================================================================

def compute_nc_norm(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    flux_col: str,
) -> pd.DataFrame:
    """
    Compute the per-node non-commutativity norm:

        NC_i = mean_{j ∈ N(i)} C_ij²

    where C_ij = f_ij (the wedge flux = commutator [M_A, M_B] on the edge).

    This is identical to node_energy_total in Step 6, but we recompute it
    here explicitly from the raw commutator field to make the NCG connection
    transparent and to verify internal consistency.
    """
    rows = []
    node_ids = nodes["node_id"].tolist()

    for nid in node_ids:
        incident = edges[(edges["tail"] == nid) | (edges["head"] == nid)]
        if len(incident) == 0:
            nc_norm = 0.0
            n_edges = 0
        else:
            vals = incident[flux_col].to_numpy()
            nc_norm = float(np.mean(vals ** 2))
            n_edges = len(incident)
        rows.append({"node_id": nid, "nc_norm": nc_norm, "n_incident_edges": n_edges})

    return pd.DataFrame(rows)


# =============================================================================
# CORE: REGION-STRATIFIED NC–COEXACT CORRELATION
# =============================================================================

def ncg_correlation_analysis(merged: pd.DataFrame) -> list[dict]:
    """
    For each region (and globally), compute Spearman ρ between NC norm and
    node-level coexact energy.

    Primary test: within interface_like nodes — high ρ here means the
    coexact component tracks non-commutativity structure specifically at
    biologically active boundaries.
    """
    results = []

    # log-transform both (heavy right tails)
    merged = merged.copy()
    merged["log1p_nc_norm"]      = np.log1p(merged["nc_norm"])
    merged["log1p_coexact_energy"] = np.log1p(merged["node_energy_coexact"])

    # ── Global ────────────────────────────────────────────────────────────
    rho, p, n = spearman_safe(
        merged["log1p_nc_norm"],
        merged["log1p_coexact_energy"],
    )
    results.append({
        "region": "global",
        "n": n,
        "rho_nc_coexact": rho,
        "p_nc_coexact": p,
    })

    # ── Per region ────────────────────────────────────────────────────────
    for region, sub in merged.groupby("region_step2"):
        rho, p, n = spearman_safe(
            sub["log1p_nc_norm"],
            sub["log1p_coexact_energy"],
        )
        results.append({
            "region": region,
            "n": n,
            "rho_nc_coexact": rho,
            "p_nc_coexact": p,
        })

    return results


def interface_vs_other_delta(region_results: list[dict]) -> dict:
    """
    Key test: ρ_interface - ρ_other.

    A positive Δ means the NC–coexact relationship is specifically elevated
    at tumor–immune interfaces, distinguishing geometric invariant from
    construction-wide artifact.
    """
    rho_interface = np.nan
    rho_other_vals = []

    for r in region_results:
        if r["region"] == "interface_like":
            rho_interface = r["rho_nc_coexact"]
        elif r["region"] not in ("global",):
            if not np.isnan(r["rho_nc_coexact"]):
                rho_other_vals.append(r["rho_nc_coexact"])

    rho_other_mean = float(np.mean(rho_other_vals)) if rho_other_vals else np.nan
    delta = float(rho_interface - rho_other_mean) if not np.isnan(rho_interface) else np.nan

    return {
        "rho_interface":  rho_interface,
        "rho_other_mean": rho_other_mean,
        "delta_interface_vs_other": delta,
    }


# =============================================================================
# SAMPLE MODE
# =============================================================================

def run_sample(sample_id: str, flux_tag: str,
               stats_dir: Path, out_dir: Path) -> None:

    print(f"\nLoading files for {sample_id}...")

    edge_file  = require(stats_dir / f"{sample_id}_step6_edges_hodge_{flux_tag}.csv")
    nodes_file = require(stats_dir / f"{sample_id}_step6_nodes_hodge_{flux_tag}.csv")

    edges = pd.read_csv(edge_file)
    nodes = pd.read_csv(nodes_file)

    # Validate columns
    if flux_tag not in edges.columns:
        raise ValueError(
            f"Flux column '{flux_tag}' not found in {edge_file.name}.\n"
            f"Available: {edges.columns.tolist()}"
        )
    required_node_cols = ["node_id", "node_energy_coexact", "region_step2"]
    missing = [c for c in required_node_cols if c not in nodes.columns]
    if missing:
        raise ValueError(f"Missing node columns: {missing}")

    # ── Compute NC norm ──────────────────────────────────────────────────────
    print("  Computing per-node non-commutativity norm...")
    nc_df = compute_nc_norm(edges, nodes, flux_col=flux_tag)

    # ── Merge with node metadata ─────────────────────────────────────────────
    merged = nodes[["node_id", "node_energy_coexact",
                     "frac_coexact", "region_step2"]].merge(
        nc_df, on="node_id", how="inner"
    )

    # Consistency check: NC norm should equal node_energy_total
    # (both = mean of squared raw flux over incident edges)
    corr_check, _ = spearmanr(
        np.log1p(merged["nc_norm"]),
        np.log1p(merged["node_energy_coexact"].fillna(0))
    )
    print(f"  NC norm vs coexact energy global Spearman ρ = {corr_check:.3f}")

    # ── Region-stratified correlation ─────────────────────────────────────────
    region_results = ncg_correlation_analysis(merged)
    delta_stats    = interface_vs_other_delta(region_results)

    # ── Assemble stats table ──────────────────────────────────────────────────
    rows = []
    for r in region_results:
        rows.append({
            "sample_id":      sample_id,
            "region":         r["region"],
            "n":              r["n"],
            "rho_nc_coexact": r["rho_nc_coexact"],
            "p_nc_coexact":   r["p_nc_coexact"],
            "rho_interface":  delta_stats["rho_interface"],
            "rho_other_mean": delta_stats["rho_other_mean"],
            "delta_interface_vs_other": delta_stats["delta_interface_vs_other"],
        })

    stats_df = pd.DataFrame(rows)

    # FDR correction across regions (excluding global)
    mask = (stats_df["region"] != "global") & stats_df["p_nc_coexact"].notna()
    if mask.sum() > 1:
        _, pvals_fdr, _, _ = multipletests(
            stats_df.loc[mask, "p_nc_coexact"].values, method="fdr_bh"
        )
        stats_df.loc[mask, "p_nc_coexact_fdr"] = pvals_fdr
    else:
        stats_df["p_nc_coexact_fdr"] = stats_df["p_nc_coexact"]

    # ── Save ──────────────────────────────────────────────────────────────────
    out_file = out_dir / f"{sample_id}_step20_ncg_stats_{flux_tag}.csv"
    stats_df.to_csv(out_file, index=False)
    print(f"  Saved → {out_file}")

    # Console summary
    interface_row = stats_df[stats_df["region"] == "interface_like"]
    global_row    = stats_df[stats_df["region"] == "global"]

    print(f"\n  === NCG SUMMARY — {sample_id} ===")
    if not global_row.empty:
        g = global_row.iloc[0]
        print(f"  Global ρ(NC, coexact)     = {g['rho_nc_coexact']:.3f}  (n={g['n']})")
    if not interface_row.empty:
        irow = interface_row.iloc[0]
        print(f"  Interface ρ(NC, coexact)  = {irow['rho_nc_coexact']:.3f}  "
              f"(n={irow['n']}, FDR p={irow.get('p_nc_coexact_fdr', np.nan):.3e})")
    print(f"  Δ(interface − other mean) = {delta_stats['delta_interface_vs_other']:.3f}")
    print(f"  Interpretation: {'interface-specific elevation ✓' if delta_stats['delta_interface_vs_other'] > 0 else 'no interface specificity ✗'}")


# =============================================================================
# COHORT MODE
# =============================================================================

def run_cohort(flux_tag: str, stats_dir: Path, out_dir: Path) -> None:

    pattern = f"*_step20_ncg_stats_{flux_tag}.csv"
    files   = sorted(stats_dir.glob(pattern))

    if not files:
        print(f"[cohort] No files found matching {stats_dir / pattern}")
        print("         Run --mode sample for each section first.")
        return

    print(f"[cohort] Found {len(files)} per-sample stats files.")

    # ── Collect per-sample interface ρ and Δ ──────────────────────────────────
    records = []
    for f in files:
        sample_id = f.name.replace(f"_step20_ncg_stats_{flux_tag}.csv", "")
        df = pd.read_csv(f)

        irow = df[df["region"] == "interface_like"]
        grow = df[df["region"] == "global"]

        if irow.empty:
            print(f"  [skip] {sample_id}: no interface_like row")
            continue

        irow = irow.iloc[0]
        grow = grow.iloc[0] if not grow.empty else None

        records.append({
            "sample_id":               sample_id,
            "n_interface":             int(irow["n"]) if pd.notna(irow["n"]) else np.nan,
            "rho_interface":           irow["rho_nc_coexact"],
            "rho_global":              grow["rho_nc_coexact"] if grow is not None else np.nan,
            "delta_interface_vs_other":irow["delta_interface_vs_other"],
        })

    if not records:
        print("[cohort] No valid records collected.")
        return

    long_df = pd.DataFrame(records)

    # Exclude n_interface < 10 (same criterion as Steps 7 and 19)
    excluded = long_df[long_df["n_interface"] < 10]["sample_id"].tolist()
    valid    = long_df[long_df["n_interface"] >= 10].copy()
    print(f"[cohort] Valid sections (n_interface ≥ 10): {len(valid)}")
    if excluded:
        print(f"[cohort] Excluded (n_interface < 10): {excluded}")

    # ── Sign test: ρ_interface > 0 ────────────────────────────────────────────
    rho_vals  = valid["rho_interface"].dropna().to_numpy()
    n_total   = len(rho_vals)
    n_pos     = int(np.sum(rho_vals > 0))
    median_rho = float(np.median(rho_vals))
    binom_p    = binomtest(n_pos, n_total, p=0.5, alternative="greater").pvalue

    # Sign test: Δ > 0 (interface specifically elevated)
    delta_vals = valid["delta_interface_vs_other"].dropna().to_numpy()
    n_delta_pos = int(np.sum(delta_vals > 0))
    delta_p     = binomtest(n_delta_pos, len(delta_vals),
                            p=0.5, alternative="greater").pvalue
    median_delta = float(np.median(delta_vals))

    # ── Save per-sample long form ─────────────────────────────────────────────
    long_out = out_dir / f"cohort_step20_ncg_per_sample_{flux_tag}.csv"
    long_df.to_csv(long_out, index=False)
    print(f"[cohort] Saved per-sample table → {long_out}")

    # ── Build manuscript summary table ────────────────────────────────────────
    summary_rows = [
        {
            "metric":         "ρ(NC, coexact) within interface_like nodes",
            "n_sections":     n_total,
            "median":         round(median_rho, 3),
            "n_positive":     f"{n_pos}/{n_total}",
            "sign_test_p":    round(binom_p, 5),
            "interpretation": "positive consistent" if binom_p < 0.05 and median_rho > 0 else "not consistent",
        },
        {
            "metric":         "Δ ρ_interface − ρ_other_mean",
            "n_sections":     len(delta_vals),
            "median":         round(median_delta, 3),
            "n_positive":     f"{n_delta_pos}/{len(delta_vals)}",
            "sign_test_p":    round(delta_p, 5),
            "interpretation": "interface-specific elevation" if delta_p < 0.05 and median_delta > 0 else "no interface specificity",
        },
    ]
    summary_df = pd.DataFrame(summary_rows)

    summary_out = out_dir / f"cohort_step20_ncg_summary_{flux_tag}.csv"
    summary_df.to_csv(summary_out, index=False)
    print(f"[cohort] Saved summary table → {summary_out}")

    # ── Console table (paste into manuscript) ────────────────────────────────
    print("\n=== STEP 20 NCG COHORT SUMMARY (paste into manuscript) ===")
    print(f"{'Metric':<45} {'Median':>8} {'N pos':>8} {'Sign p':>10}  Interpretation")
    print("-" * 85)
    for _, r in summary_df.iterrows():
        print(f"{r['metric']:<45} {r['median']:>8.3f} {r['n_positive']:>8} "
              f"{r['sign_test_p']:>10.5f}  {r['interpretation']}")

    print(f"\nSections contributing: {len(valid)}")
    print("\n[cohort] Done.")

    # ── Biological interpretation guide ───────────────────────────────────────
    print("\n=== INTERPRETATION GUIDE ===")
    if binom_p < 0.05 and median_rho > 0:
        print("✓ NC norm predicts coexact energy within interface nodes across sections.")
        print("  → Coexact component tracks operator non-commutativity, not free noise.")
    else:
        print("✗ NC–coexact correlation within interface nodes is not consistent.")
        print("  → Construction artifact hypothesis cannot be ruled out by this test alone.")

    if delta_p < 0.05 and median_delta > 0:
        print("✓ NC–coexact correlation is specifically elevated at interfaces vs other regions.")
        print("  → Interface localization of coexact energy reflects where non-commutativity")
        print("    is geometrically maximal, consistent with NCG grounding.")
    else:
        print("✗ No interface-specific elevation of NC–coexact correlation.")
        print("  → The relationship is spatially uniform; localization claim is weakened.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if _ARGS.mode == "sample":
    if _ARGS.sample_id is None:
        import sys
        print("Error: --sample-id is required for --mode sample.")
        sys.exit(1)
    run_sample(
        sample_id=_ARGS.sample_id,
        flux_tag=FLUX_TAG,
        stats_dir=STATS_DIR,
        out_dir=OUT_DIR,
    )

elif _ARGS.mode == "cohort":
    run_cohort(
        flux_tag=FLUX_TAG,
        stats_dir=STATS_DIR,
        out_dir=OUT_DIR,
    )
