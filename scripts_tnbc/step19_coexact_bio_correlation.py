from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, binomtest
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf


# =========================
# CLI
# =========================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Step 19 — coexact energy biological anchoring.\n\n"
            "Two modes:\n"
            "  sample   Run analysis for a single sample and save per-sample stats CSV.\n"
            "  cohort   Aggregate all per-sample stats CSVs and output cohort summary table.\n\n"
            "Example — single sample:\n"
            "  python step19_coexact_bio_correlation.py --mode sample "
            "--sample-id GSM_6433619 --flux-tag flux_tumor_immune\n\n"
            "Example — cohort aggregate:\n"
            "  python step19_coexact_bio_correlation.py --mode cohort "
            "--flux-tag flux_tumor_immune\n\n"
            "Bash loop for all samples:\n"
            "  for sid in $(cat sample_ids.txt); do\n"
            "      python step19_coexact_bio_correlation.py --mode sample "
            "--sample-id $sid --flux-tag flux_tumor_immune\n"
            "  done\n"
            "  python step19_coexact_bio_correlation.py --mode cohort "
            "--flux-tag flux_tumor_immune"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=["sample", "cohort"],
        default="sample",
        help="'sample': run per-sample analysis.  'cohort': aggregate all per-sample stats. (default: sample)",
    )
    p.add_argument(
        "--sample-id",
        default=None,
        help="GSM sample ID (required for --mode sample).",
    )
    p.add_argument(
        "--flux-tag",
        default="flux_tumor_immune",
        help="Flux tag used in file naming (default: flux_tumor_immune).",
    )
    p.add_argument(
        "--statsdir",
        default="stats/CSV_GSM",
        help="Directory containing per-sample CSV outputs (default: stats/CSV_GSM).",
    )
    p.add_argument(
        "--gnndir",
        default="stats/gnn_data",
        help="Directory containing nodes_for_gnn CSVs (default: stats/gnn_data).",
    )
    p.add_argument(
        "--outdir",
        default="stats",
        help="Output directory for cohort-level files (default: stats).",
    )
    p.add_argument(
        "--n-perm",
        type=int,
        default=1000,
        help="Number of permutations for regionwise permutation test (default: 1000).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed (default: 42).",
    )
    return p.parse_args()


_ARGS = _parse_args()

# Per-sample resolved config (only used in sample mode)
SAMPLE_ID = _ARGS.sample_id or "GSM_6433619"
FLUX_TAG  = _ARGS.flux_tag
N_PERM    = _ARGS.n_perm
RNG_SEED  = _ARGS.seed

STATS_DIR = Path(_ARGS.statsdir)
GNN_DIR   = Path(_ARGS.gnndir)
OUT_DIR   = Path(_ARGS.outdir)
OUT_DIR.mkdir(exist_ok=True, parents=True)

PROXY_NODE_FILE = STATS_DIR / f"{SAMPLE_ID}_step6_nodes_hodge_{FLUX_TAG}.csv"
BIO_NODE_FILE   = GNN_DIR   / f"{SAMPLE_ID}_{FLUX_TAG}_nodes_for_gnn.csv"

MERGED_OUT = STATS_DIR / f"{SAMPLE_ID}_step19_coexact_bio_merged_{FLUX_TAG}.csv"
STATS_OUT  = STATS_DIR / f"{SAMPLE_ID}_step19_coexact_bio_stats_{FLUX_TAG}.csv"
PLOT_OUT   = STATS_DIR / f"{SAMPLE_ID}_step19_coexact_bio_plot_{FLUX_TAG}.png"


# =========================
# HELPERS
# =========================

def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def raw_spearman(x: pd.Series, y: pd.Series) -> dict:
    mask = x.notna() & y.notna()
    x_ = x[mask]
    y_ = y[mask]

    if len(x_) < 5:
        return {"n": len(x_), "rho": np.nan, "p_value": np.nan}

    rho, p = spearmanr(x_, y_)
    return {"n": len(x_), "rho": float(rho), "p_value": float(p)}


def region_demeaned_spearman(
    df: pd.DataFrame,
    response_col: str,
    predictor_col: str,
    region_col: str = "region_step2",
) -> dict:
    """
    Spearman correlation after subtracting region means from both variables.
    This is a region-demeaned nonparametric robustness check, not a partial correlation.
    """
    work = df[[response_col, predictor_col, region_col]].dropna().copy()

    if len(work) < 5:
        return {
            "n": len(work),
            "rho": np.nan,
            "p_value": np.nan,
            "note": "insufficient_data",
        }

    n_regions = work[region_col].nunique()
    if n_regions == 1:
        rho, p = spearmanr(work[response_col], work[predictor_col])
        return {
            "n": len(work),
            "rho": float(rho),
            "p_value": float(p),
            "note": "single_region_raw_spearman",
        }

    for col in [response_col, predictor_col]:
        region_means = work.groupby(region_col)[col].transform("mean")
        work[f"{col}_resid_region"] = work[col] - region_means

    rho, p = spearmanr(
        work[f"{response_col}_resid_region"],
        work[f"{predictor_col}_resid_region"],
    )
    return {
        "n": len(work),
        "rho": float(rho),
        "p_value": float(p),
        "note": "region_demeaned",
    }


def regionwise_permutation_test(
    df: pd.DataFrame,
    response_col: str,
    predictor_col: str,
    region_col: str = "region_step2",
    n_perm: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Shuffle predictor within region. In a single-region subset this reduces
    to a global permutation; that case is flagged explicitly.
    """
    rng = np.random.default_rng(seed)

    work = df[[response_col, predictor_col, region_col]].dropna().copy()
    if len(work) < 5:
        return {
            "observed_rho": np.nan,
            "empirical_p_two_sided": np.nan,
            "null_mean": np.nan,
            "null_std": np.nan,
            "note": "insufficient_data",
        }

    observed_rho, _ = spearmanr(work[response_col], work[predictor_col])
    n_regions = work[region_col].nunique()

    null_rhos = []
    for _ in range(n_perm):
        permuted = work.copy()

        for _, idx in permuted.groupby(region_col).groups.items():
            vals = permuted.loc[idx, predictor_col].to_numpy(copy=True)
            rng.shuffle(vals)
            permuted.loc[idx, predictor_col] = vals

        rho_perm, _ = spearmanr(permuted[response_col], permuted[predictor_col])
        null_rhos.append(rho_perm)

    null_rhos = np.asarray(null_rhos, dtype=float)
    empirical_p = (np.sum(np.abs(null_rhos) >= abs(observed_rho)) + 1) / (len(null_rhos) + 1)

    return {
        "observed_rho": float(observed_rho),
        "empirical_p_two_sided": float(empirical_p),
        "null_mean": float(np.nanmean(null_rhos)),
        "null_std": float(np.nanstd(null_rhos)),
        "note": "single_region_global_shuffle" if n_regions == 1 else "within_region_permutation",
    }


def ols_with_region(
    df: pd.DataFrame,
    response_col: str,
    predictor_col: str,
    region_col: str = "region_step2",
) -> dict:
    """
    OLS with region as categorical covariate.
    This is the primary region-adjusted effect-size estimate for global analysis.
    """
    model_df = df[[response_col, predictor_col, region_col]].dropna().copy()

    if len(model_df) < 5:
        return {
            "n": len(model_df),
            "coef": np.nan,
            "p_value": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }

    model_df[region_col] = model_df[region_col].astype("category")
    formula = f"{response_col} ~ {predictor_col} + C({region_col})"
    fit = smf.ols(formula, data=model_df).fit()

    coef = float(fit.params[predictor_col])
    pval = float(fit.pvalues[predictor_col])
    ci_low, ci_high = fit.conf_int().loc[predictor_col].tolist()

    return {
        "n": len(model_df),
        "coef": coef,
        "p_value": pval,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def interface_only_subset(df: pd.DataFrame, region_col: str = "region_step2") -> tuple[pd.DataFrame, str]:
    regions = set(df[region_col].dropna().unique().tolist())
    if "interface_like" not in regions:
        raise ValueError(
            "The analysis requires region_step2 == 'interface_like'. "
            "No fallback subset is used."
        )
    subset = df[df[region_col] == "interface_like"].copy()
    return subset, "interface_like"


def annotate_panel(ax, results_df: pd.DataFrame, predictor: str, analysis: str = "global") -> None:
    row = results_df[
        (results_df["predictor"] == predictor) &
        (results_df["analysis"] == analysis)
    ]
    if row.empty:
        return

    row = row.iloc[0]

    rho = row["spearman_region_demeaned_rho"]
    p_fdr = row["spearman_region_demeaned_p_fdr"]
    n = int(row["n"])
    note = row["spearman_region_demeaned_note"]

    label = (
        f"Region-demeaned ρ = {rho:.3f}\n"
        f"FDR-adj p = {p_fdr:.3g}\n"
        f"n = {n}\n"
        f"{note}"
    )
    ax.text(
        0.05, 0.95, label,
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )


# =========================
# SAMPLE MODE EXECUTION
# =========================

if _ARGS.mode == "sample":

    if _ARGS.sample_id is None:
        import sys
        print("Error: --sample-id is required for --mode sample.")
        sys.exit(1)

    # ── LOAD ─────────────────────────────────────────────────────────────────

    print(f"Loading files for {SAMPLE_ID}...")

    proxy_df = pd.read_csv(require_file(PROXY_NODE_FILE))
    bio_df   = pd.read_csv(require_file(BIO_NODE_FILE))

    print("Proxy columns:", proxy_df.columns.tolist())
    print("Bio columns:", bio_df.columns.tolist())

    required_proxy_cols = [
        "node_id",
        "node_energy_coexact",
        "frac_coexact",
        "region_step2",
    ]
    required_bio_cols = [
        "node_id",
        "immune_residual",
        "tumor_residual",
        "stroma_residual",
        "region_step2",
    ]

    missing_proxy = [c for c in required_proxy_cols if c not in proxy_df.columns]
    missing_bio = [c for c in required_bio_cols if c not in bio_df.columns]

    if missing_proxy:
        raise ValueError(f"Missing required proxy columns: {missing_proxy}")
    if missing_bio:
        raise ValueError(f"Missing required bio columns: {missing_bio}")


    # =========================
    # MERGE
    # =========================

    bio_keep = [
        "node_id",
        "immune_residual",
        "tumor_residual",
        "stroma_residual",
        "tumor_score",
        "stroma_score",
        "immune_score",
    ]
    bio_keep = [c for c in bio_keep if c in bio_df.columns]

    merged = proxy_df.merge(
        bio_df[bio_keep],
        on="node_id",
        how="inner",
        validate="one_to_one",
    )

    merged["log1p_node_energy_coexact"] = np.log1p(merged["node_energy_coexact"])

    merged.to_csv(MERGED_OUT, index=False)
    print(f"Saved merged table → {MERGED_OUT}")


    # =========================
    # GLOBAL ANALYSIS
    # =========================

    results = []
    predictors = ["immune_residual", "tumor_residual", "stroma_residual"]

    for predictor in predictors:
        raw = raw_spearman(merged["log1p_node_energy_coexact"], merged[predictor])

        rd = region_demeaned_spearman(
            merged,
            response_col="log1p_node_energy_coexact",
            predictor_col=predictor,
            region_col="region_step2",
        )

        ols = ols_with_region(
            merged,
            response_col="log1p_node_energy_coexact",
            predictor_col=predictor,
            region_col="region_step2",
        )

        perm = regionwise_permutation_test(
            merged,
            response_col="log1p_node_energy_coexact",
            predictor_col=predictor,
            region_col="region_step2",
            n_perm=N_PERM,
            seed=RNG_SEED,
        )

        results.append(
            {
                "analysis": "global",
                "subset": "all_nodes",
                "predictor": predictor,
                "n": raw["n"],
                "spearman_raw_rho": raw["rho"],
                "spearman_raw_p": raw["p_value"],
                "spearman_region_demeaned_rho": rd["rho"],
                "spearman_region_demeaned_p": rd["p_value"],
                "spearman_region_demeaned_note": rd["note"],
                "ols_coef": ols["coef"],
                "ols_p": ols["p_value"],
                "ols_ci_low": ols["ci_low"],
                "ols_ci_high": ols["ci_high"],
                "perm_observed_rho": perm["observed_rho"],
                "perm_p_two_sided": perm["empirical_p_two_sided"],
                "perm_null_mean": perm["null_mean"],
                "perm_null_std": perm["null_std"],
                "perm_note": perm["note"],
            }
        )


    # =========================
    # INTERFACE-ONLY ANALYSIS
    # =========================

    subset_df, subset_name = interface_only_subset(merged)
    print(f"Interface-focused subset: {subset_name} (n={len(subset_df)})")

    for predictor in predictors:
        raw = raw_spearman(subset_df["log1p_node_energy_coexact"], subset_df[predictor])

        rd = region_demeaned_spearman(
            subset_df,
            response_col="log1p_node_energy_coexact",
            predictor_col=predictor,
            region_col="region_step2",
        )

        perm = regionwise_permutation_test(
            subset_df,
            response_col="log1p_node_energy_coexact",
            predictor_col=predictor,
            region_col="region_step2",
            n_perm=N_PERM,
            seed=RNG_SEED,
        )

        results.append(
            {
                "analysis": "subset",
                "subset": subset_name,
                "predictor": predictor,
                "n": raw["n"],
                "spearman_raw_rho": raw["rho"],
                "spearman_raw_p": raw["p_value"],
                "spearman_region_demeaned_rho": rd["rho"],
                "spearman_region_demeaned_p": rd["p_value"],
                "spearman_region_demeaned_note": rd["note"],
                "ols_coef": np.nan,
                "ols_p": np.nan,
                "ols_ci_low": np.nan,
                "ols_ci_high": np.nan,
                "perm_observed_rho": perm["observed_rho"],
                "perm_p_two_sided": perm["empirical_p_two_sided"],
                "perm_null_mean": perm["null_mean"],
                "perm_null_std": perm["null_std"],
                "perm_note": perm["note"],
            }
        )

    results_df = pd.DataFrame(results)


    # =========================
    # MULTIPLE TEST CORRECTION
    # =========================

    results_df["spearman_region_demeaned_p_fdr"] = np.nan
    results_df["ols_p_fdr"] = np.nan

    # FDR for global region-demeaned Spearman only
    mask_global_rd = (
        (results_df["analysis"] == "global") &
        results_df["spearman_region_demeaned_p"].notna()
    )
    if mask_global_rd.sum() > 0:
        _, pvals_corr, _, _ = multipletests(
            results_df.loc[mask_global_rd, "spearman_region_demeaned_p"].values,
            method="fdr_bh",
        )
        results_df.loc[mask_global_rd, "spearman_region_demeaned_p_fdr"] = pvals_corr

    # FDR for subset region-demeaned Spearman only
    mask_subset_rd = (
        (results_df["analysis"] == "subset") &
        results_df["spearman_region_demeaned_p"].notna()
    )
    if mask_subset_rd.sum() > 0:
        _, pvals_corr, _, _ = multipletests(
            results_df.loc[mask_subset_rd, "spearman_region_demeaned_p"].values,
            method="fdr_bh",
        )
        results_df.loc[mask_subset_rd, "spearman_region_demeaned_p_fdr"] = pvals_corr

    # FDR for OLS only on global rows
    mask_global_ols = (
        (results_df["analysis"] == "global") &
        results_df["ols_p"].notna()
    )
    if mask_global_ols.sum() > 0:
        _, pvals_corr, _, _ = multipletests(
            results_df.loc[mask_global_ols, "ols_p"].values,
            method="fdr_bh",
        )
        results_df.loc[mask_global_ols, "ols_p_fdr"] = pvals_corr

    results_df.to_csv(STATS_OUT, index=False)
    print(f"Saved stats table → {STATS_OUT}")


    # =========================
    # PLOT
    # =========================

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    # Panel A
    axes[0].scatter(
        merged["immune_residual"],
        merged["log1p_node_energy_coexact"],
        s=10,
        alpha=0.6,
    )
    axes[0].set_xlabel("Immune residual")
    axes[0].set_ylabel("log(1 + node coexact energy)")
    axes[0].set_title("Coexact energy vs immune residual\n(OLS line for reference)")

    x = merged["immune_residual"].to_numpy()
    y = merged["log1p_node_energy_coexact"].to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() > 2:
        coef = np.polyfit(x[mask], y[mask], deg=1)
        xx = np.linspace(x[mask].min(), x[mask].max(), 100)
        yy = coef[0] * xx + coef[1]
        axes[0].plot(xx, yy)

    annotate_panel(axes[0], results_df, "immune_residual", analysis="global")

    # Panel B
    axes[1].scatter(
        merged["tumor_residual"],
        merged["log1p_node_energy_coexact"],
        s=10,
        alpha=0.6,
    )
    axes[1].set_xlabel("Tumor residual")
    axes[1].set_ylabel("log(1 + node coexact energy)")
    axes[1].set_title("Coexact energy vs tumor residual\n(OLS line for reference)")

    x = merged["tumor_residual"].to_numpy()
    y = merged["log1p_node_energy_coexact"].to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() > 2:
        coef = np.polyfit(x[mask], y[mask], deg=1)
        xx = np.linspace(x[mask].min(), x[mask].max(), 100)
        yy = coef[0] * xx + coef[1]
        axes[1].plot(xx, yy)

    annotate_panel(axes[1], results_df, "tumor_residual", analysis="global")

    fig.savefig(PLOT_OUT, dpi=300)
    print(f"Saved plot → {PLOT_OUT}")


    # =========================
    # CONSOLE SUMMARY
    # =========================

    print("\n=== STEP 19 SUMMARY ===")
    print(results_df.to_string(index=False))
    print("\nStep 19 complete.")


# =========================
# COHORT AGGREGATION
# =========================

def run_cohort(flux_tag: str, statsdir: Path, outdir: Path) -> None:
    """
    Aggregate all per-sample Step 19 stats CSVs and produce:
      1. cohort_step19_within_interface_rho.csv  — one row per sample × predictor
      2. cohort_step19_summary.csv               — median ρ, fraction > 0, sign test
         (ready to paste into tab:step19_cohort in the manuscript)

    The sign test uses a one-sided binomial H₀: ρ_int ≤ 0 (i.e. tests whether
    within-interface coexact energy is positively associated with each predictor
    across sections, not just in individual sections).
    """
    pattern = f"*_step19_coexact_bio_stats_{flux_tag}.csv"
    csv_files = sorted(statsdir.glob(pattern))

    if not csv_files:
        print(
            f"[cohort] No stats files found matching {statsdir / pattern}\n"
            "         Run --mode sample for each sample first."
        )
        return

    print(f"[cohort] Found {len(csv_files)} per-sample stats files.")

    # ── collect within-interface ρ per sample × predictor ────────────────────
    records = []
    for f in csv_files:
        sample_id = f.name.replace(f"_step19_coexact_bio_stats_{flux_tag}.csv", "")
        df = pd.read_csv(f)

        # Keep only within-interface rows
        subset = df[df["analysis"] == "subset"].copy()
        if subset.empty:
            print(f"  [skip] {sample_id}: no subset rows in stats CSV")
            continue

        for _, row in subset.iterrows():
            records.append({
                "sample_id":  sample_id,
                "predictor":  row["predictor"],
                "n_interface": int(row["n"]) if pd.notna(row["n"]) else np.nan,
                "rho_int":     row.get("spearman_raw_rho", np.nan),
                # primary metric: raw Spearman within interface-like nodes
                # (region-demeaning is not meaningful when region = interface_like only)
                "perm_p":      row.get("perm_p_two_sided", np.nan),
            })

    if not records:
        print("[cohort] No valid within-interface rows collected.")
        return

    long_df = pd.DataFrame(records)

    # Save long form
    long_out = outdir / f"cohort_step19_within_interface_rho_{flux_tag}.csv"
    long_df.to_csv(long_out, index=False)
    print(f"[cohort] Saved per-sample within-interface ρ → {long_out}")

    # ── cohort-level summary per predictor ───────────────────────────────────
    predictors = ["tumor_residual", "immune_residual", "stroma_residual"]
    summary_rows = []

    for pred in predictors:
        sub = long_df[long_df["predictor"] == pred].dropna(subset=["rho_int"])
        if sub.empty:
            continue

        rho_vals  = sub["rho_int"].to_numpy()
        n_total   = len(rho_vals)
        n_pos     = int(np.sum(rho_vals > 0))
        median_rho = float(np.median(rho_vals))
        mean_rho   = float(np.mean(rho_vals))

        # One-sided binomial sign test: H₀ prob(ρ > 0) = 0.5
        binom_result = binomtest(k=n_pos, n=n_total, p=0.5, alternative="greater")
        sign_p = binom_result.pvalue

        # Interpretation heuristic
        if sign_p < 0.05 and median_rho > 0.05:
            interp = "positive, consistent"
        elif sign_p < 0.05 and median_rho <= 0.05:
            interp = "weak positive trend"
        else:
            interp = "negligible / inconsistent"

        summary_rows.append({
            "predictor":    pred,
            "n_sections":   n_total,
            "median_rho":   round(median_rho, 3),
            "mean_rho":     round(mean_rho, 3),
            "n_pos":        n_pos,
            "frac_pos":     f"{n_pos}/{n_total}",
            "sign_test_p":  round(sign_p, 4),
            "interpretation": interp,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_out = outdir / f"cohort_step19_summary_{flux_tag}.csv"
    summary_df.to_csv(summary_out, index=False)
    print(f"[cohort] Saved cohort summary → {summary_out}")

    # ── console table (copy into manuscript) ─────────────────────────────────
    print("\n=== COHORT STEP 19 SUMMARY (paste into tab:step19_cohort) ===")
    print(f"{'Predictor':<20} {'Median ρ':>10} {'Sections ρ>0':>14} {'Sign test p':>12} {'Interpretation'}")
    print("-" * 75)
    for _, r in summary_df.iterrows():
        print(
            f"{r['predictor']:<20} {r['median_rho']:>10.3f} "
            f"{r['frac_pos']:>14} {r['sign_test_p']:>12.4f}  {r['interpretation']}"
        )

    print(f"\nTotal sections contributing: {long_df['sample_id'].nunique()}")
    print("[cohort] Done.")


# =========================
# ENTRY POINT
# =========================

if _ARGS.mode == "cohort":
    run_cohort(flux_tag=FLUX_TAG, statsdir=STATS_DIR, outdir=OUT_DIR)
else:
    # sample mode: nothing extra — the per-sample code above already ran
    pass
