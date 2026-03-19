from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf


# =========================
# CONFIG
# =========================

SAMPLE_ID = "GSM_6433619"
FLUX_TAG = "flux_tumor_immune"

PROXY_NODE_FILE = Path(
    f"stats/CSV_GSM/{SAMPLE_ID}_step6_nodes_hodge_{FLUX_TAG}.csv"
)

BIO_NODE_FILE = Path(
    f"stats/gnn_data/{SAMPLE_ID}_{FLUX_TAG}_nodes_for_gnn.csv"
)

OUT_DIR = Path("stats")
OUT_DIR.mkdir(exist_ok=True, parents=True)

MERGED_OUT = OUT_DIR / f"{SAMPLE_ID}_step19_coexact_bio_merged_{FLUX_TAG}.csv"
STATS_OUT = OUT_DIR / f"{SAMPLE_ID}_step19_coexact_bio_stats_{FLUX_TAG}.csv"
PLOT_OUT = OUT_DIR / f"{SAMPLE_ID}_step19_coexact_bio_plot_{FLUX_TAG}.png"

N_PERM = 1000
RNG_SEED = 42


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
# LOAD
# =========================

print("Loading files...")

proxy_df = pd.read_csv(require_file(PROXY_NODE_FILE))
bio_df = pd.read_csv(require_file(BIO_NODE_FILE))

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
