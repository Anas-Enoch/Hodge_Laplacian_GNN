from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


REGION_ORDER = [
    "tumor_enriched",
    "stroma_enriched",
    "immune_enriched",
    "interface_like",
    "other",
]


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def safe_mean(x: np.ndarray) -> float:
    if len(x) == 0:
        return np.nan
    return float(np.mean(x))


def safe_median(x: np.ndarray) -> float:
    if len(x) == 0:
        return np.nan
    return float(np.median(x))


def safe_std(x: np.ndarray) -> float:
    if len(x) == 0:
        return np.nan
    return float(np.std(x))


def compute_enrichment(a: np.ndarray, b: np.ndarray, eps: float = 1e-18) -> float:
    return safe_median(a) / max(safe_median(b), eps)


def permutation_pvalue(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_perm: int = 1000,
    alternative: str = "greater",
    seed: int = 0,
) -> tuple[float, float]:
    """
    Permutation p-value for difference in medians.
    Returns:
      observed_stat, p_value
    where observed_stat = median(x) - median(y)
    """
    rng = np.random.default_rng(seed)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    pooled = np.concatenate([x, y])
    n_x = len(x)

    obs = np.median(x) - np.median(y)

    perm_stats = np.empty(n_perm, dtype=float)
    for k in range(n_perm):
        perm = rng.permutation(len(pooled))
        px = pooled[perm[:n_x]]
        py = pooled[perm[n_x:]]
        perm_stats[k] = np.median(px) - np.median(py)

    if alternative == "greater":
        p = (1.0 + np.sum(perm_stats >= obs)) / (n_perm + 1.0)
    elif alternative == "less":
        p = (1.0 + np.sum(perm_stats <= obs)) / (n_perm + 1.0)
    else:
        p = (1.0 + np.sum(np.abs(perm_stats) >= abs(obs))) / (n_perm + 1.0)

    return float(obs), float(p)


def mwu_pvalue(x: np.ndarray, y: np.ndarray, alternative: str = "greater") -> tuple[float, float]:
    if len(x) == 0 or len(y) == 0:
        return np.nan, np.nan
    stat, p = mannwhitneyu(x, y, alternative=alternative)
    return float(stat), float(p)


def region_pair_tests(
    df: pd.DataFrame,
    metric_col: str,
    region_a: str,
    region_b: str,
    *,
    n_perm: int = 1000,
    seed: int = 0,
    alternative: str = "greater",
) -> dict:
    xa = df.loc[df["region_step2"] == region_a, metric_col].to_numpy(dtype=float)
    xb = df.loc[df["region_step2"] == region_b, metric_col].to_numpy(dtype=float)

    mwu_stat, mwu_p = mwu_pvalue(xa, xb, alternative=alternative)
    obs_diff, perm_p = permutation_pvalue(
        xa, xb, n_perm=n_perm, alternative=alternative, seed=seed
    )

    return {
        "metric": metric_col,
        "region_a": region_a,
        "region_b": region_b,
        "n_a": len(xa),
        "n_b": len(xb),
        "mean_a": safe_mean(xa),
        "mean_b": safe_mean(xb),
        "median_a": safe_median(xa),
        "median_b": safe_median(xb),
        "std_a": safe_std(xa),
        "std_b": safe_std(xb),
        "median_diff_a_minus_b": obs_diff,
        "median_ratio_a_over_b": compute_enrichment(xa, xb),
        "mwu_stat": mwu_stat,
        "mwu_p": mwu_p,
        "perm_p": perm_p,
    }


def save_metric_maps(
    df: pd.DataFrame,
    sample_id: str,
    flux_name: str,
    outpath: Path,
) -> None:
    x = df["x_fullres"].to_numpy(dtype=float)
    y = df["y_fullres"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    panels = [
        ("node_energy_coexact", "Absolute coexact energy"),
        ("frac_coexact", "Coexact fraction"),
    ]

    for ax, (col, title) in zip(axes, panels):
        sca = ax.scatter(
            x,
            y,
            c=df[col].to_numpy(dtype=float),
            s=15,
            alpha=0.90,
        )
        ax.set_title(title)
        ax.invert_yaxis()
        ax.axis("off")
        plt.colorbar(sca, ax=ax, fraction=0.04, pad=0.02)

    plt.suptitle(f"{sample_id}: Step 7 coexact enrichment maps — {flux_name}", y=0.98)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close(fig)


def save_region_boxplots(
    df: pd.DataFrame,
    sample_id: str,
    flux_name: str,
    outpath: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    specs = [
        ("node_energy_coexact", "Absolute coexact energy by region"),
        ("frac_coexact", "Coexact fraction by region"),
    ]

    for ax, (col, title) in zip(axes, specs):
        data = [
            df.loc[df["region_step2"] == reg, col].to_numpy(dtype=float)
            for reg in REGION_ORDER
        ]
        ax.boxplot(data, tick_labels=REGION_ORDER, showfliers=False)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)

    plt.suptitle(f"{sample_id}: Step 7 region tests — {flux_name}", y=0.98)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 7 TNBC: statistical enrichment testing for coexact energy."
    )
    parser.add_argument("--sample_id", required=True)
    parser.add_argument(
        "--flux_name",
        required=True,
        choices=["flux_tumor_immune", "flux_tumor_stroma", "flux_immune_stroma"],
    )
    parser.add_argument("--statsdir", default="stats")
    parser.add_argument("--figdir", default="visium_figures")
    parser.add_argument("--n_perm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sample_id = args.sample_id
    flux_name = args.flux_name
    statsdir = Path(args.statsdir)
    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)

    node_file = require_file(statsdir / f"{sample_id}_step6_nodes_hodge_{flux_name}.csv")
    df = pd.read_csv(node_file)

    print("=" * 72)
    print(f"STEP 7: TNBC region enrichment tests for {sample_id}")
    print("=" * 72)
    print(f"Input node file : {node_file}")
    print(f"Flux            : {flux_name}")
    print(f"n_perm          : {args.n_perm}")
    print(f"seed            : {args.seed}")

    # Primary comparisons: immune/interface against tumor/stroma/other
    tests = []

    primary_pairs = [
        ("immune_enriched", "tumor_enriched"),
        ("immune_enriched", "stroma_enriched"),
        ("immune_enriched", "other"),
        ("interface_like", "tumor_enriched"),
        ("interface_like", "stroma_enriched"),
        ("interface_like", "other"),
    ]

    for metric in ["node_energy_coexact", "frac_coexact"]:
        for ra, rb in primary_pairs:
            tests.append(
                region_pair_tests(
                    df,
                    metric,
                    ra,
                    rb,
                    n_perm=args.n_perm,
                    seed=args.seed,
                    alternative="greater",
                )
            )

    tests_df = pd.DataFrame(tests)
    tests_out = statsdir / f"{sample_id}_step7_region_tests_{flux_name}.csv"
    tests_df.to_csv(tests_out, index=False)

    print("\nPrimary test summary")
    print("-" * 72)
    print(tests_df[["metric", "region_a", "region_b", "median_ratio_a_over_b", "mwu_p", "perm_p"]])

    # Region summary table
    region_summary = (
        df.groupby("region_step2")[
            ["node_energy_total", "node_energy_exact", "node_energy_coexact", "node_energy_harmonic", "frac_exact", "frac_coexact", "frac_harmonic"]
        ]
        .agg(["mean", "median", "std", "count"])
    )
    region_summary_out = statsdir / f"{sample_id}_step7_region_summary_{flux_name}.csv"
    region_summary.to_csv(region_summary_out)

    # Compact energy fractions across all nodes
    global_summary = pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "flux_name": flux_name,
                "mean_total_energy": float(df["node_energy_total"].mean()),
                "mean_exact_energy": float(df["node_energy_exact"].mean()),
                "mean_coexact_energy": float(df["node_energy_coexact"].mean()),
                "mean_harmonic_energy": float(df["node_energy_harmonic"].mean()),
                "mean_frac_exact": float(df["frac_exact"].mean()),
                "mean_frac_coexact": float(df["frac_coexact"].mean()),
                "mean_frac_harmonic": float(df["frac_harmonic"].mean()),
            }
        ]
    )
    global_out = statsdir / f"{sample_id}_step7_global_summary_{flux_name}.csv"
    global_summary.to_csv(global_out, index=False)

    # Figures
    maps_png = figdir / f"{sample_id}_step7_coexact_maps_{flux_name}.png"
    save_metric_maps(df, sample_id, flux_name, maps_png)

    boxplots_png = figdir / f"{sample_id}_step7_coexact_boxplots_{flux_name}.png"
    save_region_boxplots(df, sample_id, flux_name, boxplots_png)

    print(f"\nSaved: {tests_out}")
    print(f"Saved: {region_summary_out}")
    print(f"Saved: {global_out}")
    print(f"Saved: {maps_png}")
    print(f"Saved: {boxplots_png}")
    print("\nStep 7 completed successfully.")


if __name__ == "__main__":
    main()
