from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ============================================================
# BUG / CHANGE INVENTORY
# ============================================================
#
# Bug S7-1 (agg_frac > 1.0):
#   compute_region_summary used sum_total as denominator for agg_frac_*.
#   Because node_energy_total (omega-derived) can be less than sum_exact
#   in small regions, agg_frac_exact > 1.0 was possible.
#   Fix: use sum_component_total = sum_exact + sum_coexact + sum_harmonic.
#
# Bug S7-2 (NameError: n_focus, n_ref):
#   Both referenced in return dicts before being defined.
#   Fix: define immediately after the zero-fraction guard.
#
# Bug S7-3 (TypeError: unexpected keyword argument):
#   min_nodes_per_region passed by main() but absent from function signature.
#   Fix: add to signature; implement quality-filter check.
#
# Bug S7-4 (KeyError in main):
#   Early-exit returns missing n_focus and n_ref keys.
#   Fix: add n_focus=0, n_ref=0 to all early-exit returns.
#
# Bug S7-5 (stale comment):
#   "# ... rest of permutation loop" implied omitted code.
#   Fix: removed.
#
# Change S7-6 (harmonic suppression):
#   When global E_harmonic / E_coexact < suppress_harmonic_threshold,
#   the harmonic component is operating at floating-point noise level.
#   Any nominal significance for node_energy_harmonic in that regime
#   is a numerical artifact, not a biological signal.
#   Fix: after building enrich_df, suppress harmonic p-value and annotate
#   the note field.  Global energies are read from the Step 6 energy
#   summary CSV when available; otherwise computed from nodes_df sums.
#   The suppression threshold is exposed as --suppress-harmonic-threshold
#   (default 1e-6) so it can be adjusted without editing the script.
# ============================================================


# ============================================================
# Helpers
# ============================================================

def require_cols(df: pd.DataFrame, cols: Iterable[str], df_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {df_name}: {missing}")


def sanitize_flux_name(flux_name: str) -> str:
    return flux_name.replace("/", "_").replace(" ", "_")


def safe_mean(x: pd.Series) -> float:
    arr = x.to_numpy(dtype=float)
    if len(arr) == 0:
        return np.nan
    return float(np.nanmean(arr))


def safe_median(x: pd.Series) -> float:
    arr = x.to_numpy(dtype=float)
    if len(arr) == 0:
        return np.nan
    return float(np.nanmedian(arr))


def choose_default_reference_region(regions: list[str]) -> str | None:
    preferences = ["tumor_enriched", "tumor_core", "other"]
    for r in preferences:
        if r in regions:
            return r
    return regions[0] if regions else None


# ============================================================
# Region summary
# ============================================================

def compute_region_summary(
    nodes_df: pd.DataFrame,
    region_col: str = "region_step2",
) -> pd.DataFrame:
    """
    Region-level summaries of node energies and fractions.

    agg_frac_* use the component sum (exact + coexact + harmonic) as
    denominator, guaranteeing values in [0, 1] regardless of whether
    node_energy_total was computed from omega or from components upstream.

    sum_total is reported for reference only and NOT used as a denominator.
    """
    required = [
        region_col,
        "node_energy_total",
        "node_energy_exact",
        "node_energy_coexact",
        "node_energy_harmonic",
    ]
    require_cols(nodes_df, required, "nodes_df")

    rows = []
    for region, sub in nodes_df.groupby(region_col, dropna=False):
        E_total    = sub["node_energy_total"].to_numpy(dtype=float)
        E_exact    = sub["node_energy_exact"].to_numpy(dtype=float)
        E_coexact  = sub["node_energy_coexact"].to_numpy(dtype=float)
        E_harmonic = sub["node_energy_harmonic"].to_numpy(dtype=float)

        sum_total    = float(np.nansum(E_total))
        sum_exact    = float(np.nansum(E_exact))
        sum_coexact  = float(np.nansum(E_coexact))
        sum_harmonic = float(np.nansum(E_harmonic))

        # FIX S7-1: component sum — always >= each numerator
        sum_component_total = sum_exact + sum_coexact + sum_harmonic

        if sum_component_total > 1e-12:
            agg_frac_exact    = sum_exact    / sum_component_total
            agg_frac_coexact  = sum_coexact  / sum_component_total
            agg_frac_harmonic = sum_harmonic / sum_component_total
        else:
            agg_frac_exact = agg_frac_coexact = agg_frac_harmonic = np.nan

        # Validate bounds — should always pass after the fix above
        for label, val in [
            ("agg_frac_exact",    agg_frac_exact),
            ("agg_frac_coexact",  agg_frac_coexact),
            ("agg_frac_harmonic", agg_frac_harmonic),
        ]:
            if np.isfinite(val) and (val < -1e-6 or val > 1.0 + 1e-6):
                warnings.warn(
                    f"Region '{region}': {label} = {val:.6f} outside [0, 1] "
                    f"even after component-sum fix. "
                    f"sum_exact={sum_exact:.3e}, "
                    f"sum_component_total={sum_component_total:.3e}. "
                    "Investigate Step 6 projection quality."
                )

        # Per-node fractions using component-sum denominator
        node_component_total = E_exact + E_coexact + E_harmonic
        valid = node_component_total > 1e-12
        if int(valid.sum()) > 0:
            denom_v = node_component_total[valid]
            mean_node_frac_exact    = float(np.nanmean(E_exact[valid]    / denom_v))
            mean_node_frac_coexact  = float(np.nanmean(E_coexact[valid]  / denom_v))
            mean_node_frac_harmonic = float(np.nanmean(E_harmonic[valid] / denom_v))
        else:
            mean_node_frac_exact = mean_node_frac_coexact = mean_node_frac_harmonic = np.nan

        rows.append({
            "region":               region,
            "n_nodes":              int(len(sub)),
            "n_nodes_valid":        int(valid.sum()),
            "sum_total":            sum_total,
            "sum_component_total":  sum_component_total,
            "sum_exact":            sum_exact,
            "sum_coexact":          sum_coexact,
            "sum_harmonic":         sum_harmonic,
            "agg_frac_exact":       agg_frac_exact,
            "agg_frac_coexact":     agg_frac_coexact,
            "agg_frac_harmonic":    agg_frac_harmonic,
            "mean_node_frac_exact":    mean_node_frac_exact,
            "mean_node_frac_coexact":  mean_node_frac_coexact,
            "mean_node_frac_harmonic": mean_node_frac_harmonic,
            "mean_total":    safe_mean(sub["node_energy_total"]),
            "mean_exact":    safe_mean(sub["node_energy_exact"]),
            "mean_coexact":  safe_mean(sub["node_energy_coexact"]),
            "mean_harmonic": safe_mean(sub["node_energy_harmonic"]),
            "median_total":    safe_median(sub["node_energy_total"]),
            "median_exact":    safe_median(sub["node_energy_exact"]),
            "median_coexact":  safe_median(sub["node_energy_coexact"]),
            "median_harmonic": safe_median(sub["node_energy_harmonic"]),
        })

    return pd.DataFrame(rows).sort_values("region").reset_index(drop=True)


# ============================================================
# Enrichment ratio
# ============================================================

def enrichment_ratio(
    nodes_df: pd.DataFrame,
    numerator_region: str,
    denominator_region: str,
    value_col: str,
    region_col: str = "region_step2",
) -> float:
    """
    Mean(value_col | numerator) / Mean(value_col | denominator).
    Returns NaN if either mean is non-finite or denominator mean is zero.
    """
    require_cols(nodes_df, [region_col, value_col], "nodes_df")

    num = nodes_df.loc[nodes_df[region_col] == numerator_region,   value_col]
    den = nodes_df.loc[nodes_df[region_col] == denominator_region, value_col]

    num_mean = safe_mean(num)
    den_mean = safe_mean(den)

    if not np.isfinite(num_mean) or not np.isfinite(den_mean) or den_mean == 0:
        return np.nan
    return float(num_mean / den_mean)


# ============================================================
# Permutation enrichment test
# ============================================================

def permutation_enrichment_test(
    nodes_df: pd.DataFrame,
    numerator_region: str,
    denominator_region: str,
    value_col: str,
    region_col: str = "region_step2",
    n_perm: int = 1000,
    seed: int = 42,
    zero_fraction_threshold: float = 0.95,
    min_nodes_per_region: int = 10,
) -> dict:
    """
    Permutation enrichment test for a scalar node metric between two regions.

    Parameters
    ----------
    zero_fraction_threshold : float
        Fraction of exactly-zero values above which the metric is treated as
        structurally degenerate.  Uses exact zeros (not np.allclose) so that
        small-but-nonzero energy scales are handled correctly.
    min_nodes_per_region : int
        Minimum nodes required in both focus and reference regions.
        Samples below this threshold are excluded (note = low_sample_size)
        but retained for global Hodge analysis.
    """
    require_cols(nodes_df, [region_col, value_col], "nodes_df")

    work = nodes_df[[region_col, value_col]].dropna().copy()

    # Guard: empty dataframe
    if len(work) == 0:
        return {
            "observed_ratio":   np.nan,
            "perm_p_two_sided": np.nan,
            "null_mean":        np.nan,
            "null_std":         np.nan,
            "n_focus":          0,
            "n_ref":            0,
            "note":             "insufficient_data",
        }

    vals = work[value_col].to_numpy(dtype=float)

    # Guard: degenerate metric (fraction-based, not np.allclose)
    n_zero = int(np.sum(vals == 0.0))
    zero_fraction = n_zero / len(vals)
    if zero_fraction >= zero_fraction_threshold:
        return {
            "observed_ratio":   np.nan,
            "perm_p_two_sided": np.nan,
            "null_mean":        np.nan,
            "null_std":         np.nan,
            "n_focus":          0,
            "n_ref":            0,
            "note":             f"all_zero_metric (zero_fraction={zero_fraction:.3f})",
        }

    # FIX S7-2: define before any return that references them
    n_focus = int((work[region_col] == numerator_region).sum())
    n_ref   = int((work[region_col] == denominator_region).sum())

    # FIX S7-3: quality filter
    if n_focus < min_nodes_per_region or n_ref < min_nodes_per_region:
        return {
            "observed_ratio":   np.nan,
            "perm_p_two_sided": np.nan,
            "null_mean":        np.nan,
            "null_std":         np.nan,
            "n_focus":          n_focus,
            "n_ref":            n_ref,
            "note": (
                f"low_sample_size "
                f"(focus={n_focus}, ref={n_ref}, min={min_nodes_per_region})"
            ),
        }

    obs = enrichment_ratio(
        work,
        numerator_region=numerator_region,
        denominator_region=denominator_region,
        value_col=value_col,
        region_col=region_col,
    )

    labels = work[region_col].to_numpy(copy=True)
    values = vals.copy()
    rng = np.random.default_rng(seed)
    null_vals: list[float] = []

    for _ in range(n_perm):
        shuffled = labels.copy()
        rng.shuffle(shuffled)
        perm_df = pd.DataFrame({region_col: shuffled, value_col: values})
        r = enrichment_ratio(
            perm_df,
            numerator_region=numerator_region,
            denominator_region=denominator_region,
            value_col=value_col,
            region_col=region_col,
        )
        if np.isfinite(r):
            null_vals.append(r)

    if len(null_vals) == 0 or not np.isfinite(obs):
        return {
            "observed_ratio":   float(obs) if np.isfinite(obs) else np.nan,
            "perm_p_two_sided": np.nan,
            "null_mean":        np.nan,
            "null_std":         np.nan,
            "n_focus":          n_focus,
            "n_ref":            n_ref,
            "note":             "null_failed",
        }

    null_arr = np.asarray(null_vals, dtype=float)
    p_two = (
        np.sum(np.abs(null_arr - 1.0) >= abs(obs - 1.0)) + 1
    ) / (len(null_arr) + 1)

    return {
        "observed_ratio":   float(obs),
        "perm_p_two_sided": float(p_two),
        "null_mean":        float(np.nanmean(null_arr)),
        "null_std":         float(np.nanstd(null_arr)),
        "n_focus":          n_focus,
        "n_ref":            n_ref,
        "note":             "ok",
    }


# ============================================================
# Change S7-6: Harmonic suppression
# ============================================================

def get_global_energies(
    nodes_df: pd.DataFrame,
    step6_summary_file: Path | None,
) -> tuple[float, float]:
    """
    Return (E_harmonic_global, E_coexact_global).

    Preference order:
    1. Step 6 energy summary CSV — uses the exact global L2 norms from the
       Hodge decomposition (E_coexact = ||omega_coexact||^2).
    2. Sum of node_energy_* columns — approximate but consistent direction.
       Used as fallback when the Step 6 summary is unavailable.
    """
    if step6_summary_file is not None and step6_summary_file.exists():
        try:
            summary = pd.read_csv(step6_summary_file)
            if "E_harmonic" in summary.columns and "E_coexact" in summary.columns:
                E_harm = float(summary["E_harmonic"].iloc[0])
                E_coex = float(summary["E_coexact"].iloc[0])
                print(
                    f"Harmonic suppression: loaded global energies from "
                    f"{step6_summary_file.name} "
                    f"(E_harmonic={E_harm:.3e}, E_coexact={E_coex:.3e})"
                )
                return E_harm, E_coex
            else:
                print(
                    f"Harmonic suppression: Step 6 summary found but missing "
                    f"E_harmonic / E_coexact columns — falling back to nodes_df sums."
                )
        except Exception as exc:
            print(
                f"Harmonic suppression: could not read {step6_summary_file}: "
                f"{exc} — falling back to nodes_df sums."
            )

    # Fallback: sum of per-node mean squared energies
    E_harm = float(nodes_df["node_energy_harmonic"].sum())
    E_coex = float(nodes_df["node_energy_coexact"].sum())
    print(
        f"Harmonic suppression: using nodes_df sums "
        f"(E_harmonic≈{E_harm:.3e}, E_coexact≈{E_coex:.3e})"
    )
    return E_harm, E_coex


def suppress_harmonic_if_negligible(
    enrich_df: pd.DataFrame,
    E_harmonic_global: float,
    E_coexact_global: float,
    threshold: float = 1e-6,
) -> pd.DataFrame:
    """
    Suppress harmonic enrichment significance when the global harmonic energy
    is negligible relative to global coexact energy.

    When E_harmonic / E_coexact < threshold, the harmonic component is at
    floating-point noise level.  Any nominal p-value for node_energy_harmonic
    in that regime reflects numerical artifacts of the lsqr projection, not
    biology.  The p-value is set to NaN and the note field is annotated.

    A new boolean column 'harmonic_suppressed' is added to make downstream
    filtering explicit.

    Parameters
    ----------
    threshold : float
        Ratio E_harmonic / E_coexact below which harmonic significance is
        suppressed.  Default 1e-6 — ten orders of magnitude separation.
    """
    enrich_df = enrich_df.copy()

    # Add column — default False
    enrich_df["harmonic_suppressed"] = False

    if E_coexact_global <= 0:
        return enrich_df

    harmonic_fraction = E_harmonic_global / E_coexact_global

    if harmonic_fraction < threshold:
        mask = enrich_df["metric"] == "node_energy_harmonic"

        enrich_df.loc[mask, "perm_p_two_sided"] = np.nan
        enrich_df.loc[mask, "harmonic_suppressed"] = True
        enrich_df.loc[mask, "note"] = enrich_df.loc[mask, "note"].apply(
            lambda n: (
                n + f" [harmonic suppressed: "
                f"E_harm/E_coexact={harmonic_fraction:.2e} < {threshold:.0e}]"
            )
        )
        print(
            f"Harmonic suppression applied: "
            f"E_harmonic/E_coexact = {harmonic_fraction:.2e} < {threshold:.0e}. "
            f"node_energy_harmonic p-value set to NaN."
        )
    else:
        print(
            f"Harmonic suppression NOT applied: "
            f"E_harmonic/E_coexact = {harmonic_fraction:.2e} >= {threshold:.0e}."
        )

    return enrich_df


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TNBC Step 7: region enrichment from node-level Hodge summaries."
    )
    parser.add_argument("--sample-id", default="GSM_6433618")
    parser.add_argument(
        "--flux-col", default="flux_tumor_immune",
        help="Flux tag from Step 6, e.g. flux_tumor_immune_region_interface_weighted",
    )
    parser.add_argument("--stats-dir", default="stats/CSV_GSM")
    parser.add_argument("--region-col", default="region_step2")
    parser.add_argument(
        "--focus-region", default="interface_like",
        help="Numerator region for enrichment ratio",
    )
    parser.add_argument(
        "--reference-region", default=None,
        help="Denominator region. Auto-selected from [tumor_enriched, tumor_core, other].",
    )
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument(
        "--min-nodes-per-region", type=int, default=10,
        help=(
            "Minimum nodes in both focus and reference regions. "
            "Samples below threshold are excluded from enrichment analysis "
            "(note = low_sample_size) but retained for global Hodge statistics."
        ),
    )
    # Change S7-6
    parser.add_argument(
        "--suppress-harmonic-threshold", type=float, default=1e-6,
        help=(
            "Suppress harmonic enrichment p-value when "
            "E_harmonic_global / E_coexact_global < this value. "
            "Values below this ratio indicate the harmonic component is at "
            "floating-point noise level and any significance is artifactual. "
            "Default: 1e-6. Set to 0.0 to disable suppression entirely."
        ),
    )
    args = parser.parse_args()

    sample_id  = args.sample_id
    flux_col   = args.flux_col
    flux_tag   = sanitize_flux_name(flux_col)
    stats_dir  = Path(args.stats_dir)

    nodes_file = stats_dir / f"{sample_id}_step6_nodes_hodge_{flux_tag}.csv"
    nodes_df   = pd.read_csv(nodes_file)

    require_cols(
        nodes_df,
        [
            "node_id",
            args.region_col,
            "node_energy_total",
            "node_energy_exact",
            "node_energy_coexact",
            "node_energy_harmonic",
        ],
        "nodes_df",
    )

    # ---- Region summary (always computed, no size filter) ----
    region_summary = compute_region_summary(nodes_df, region_col=args.region_col)

    # ---- Reference region ----
    regions = nodes_df[args.region_col].dropna().astype(str).unique().tolist()
    reference_region = args.reference_region or choose_default_reference_region(regions)
    if reference_region is None:
        raise ValueError("Could not determine a reference region.")

    # ---- Enrichment tests ----
    metrics = [
        "node_energy_exact",
        "node_energy_coexact",
        "node_energy_harmonic",
        "node_energy_total",
    ]

    rows = []
    for metric in metrics:
        test_out = permutation_enrichment_test(
            nodes_df,
            numerator_region=args.focus_region,
            denominator_region=reference_region,
            value_col=metric,
            region_col=args.region_col,
            n_perm=args.n_perm,
            min_nodes_per_region=args.min_nodes_per_region,
        )

        rows.append({
            "sample_id":        sample_id,
            "target_flux":      flux_col,
            "focus_region":     args.focus_region,
            "reference_region": reference_region,
            "metric":           metric,
            "observed_ratio":   test_out["observed_ratio"],
            "perm_p_two_sided": test_out["perm_p_two_sided"],
            "null_mean":        test_out["null_mean"],
            "null_std":         test_out["null_std"],
            "n_focus":          test_out["n_focus"],
            "n_ref":            test_out["n_ref"],
            "note":             test_out["note"],
        })

    enrich_df = pd.DataFrame(rows)

    # ---- Change S7-6: harmonic suppression ----
    if args.suppress_harmonic_threshold > 0.0:
        # Try to load Step 6 energy summary for precise global energies
        step6_summary_file = (
            stats_dir / f"{sample_id}_step6_energy_summary_{flux_tag}.csv"
        )
        E_harm_global, E_coex_global = get_global_energies(
            nodes_df, step6_summary_file
        )
        enrich_df = suppress_harmonic_if_negligible(
            enrich_df,
            E_harmonic_global=E_harm_global,
            E_coexact_global=E_coex_global,
            threshold=args.suppress_harmonic_threshold,
        )
    else:
        enrich_df["harmonic_suppressed"] = False
        print("Harmonic suppression disabled (--suppress-harmonic-threshold=0.0).")

    # ---- Save ----
    out_region = stats_dir / f"{sample_id}_step7_region_summary_{flux_tag}.csv"
    out_enrich = stats_dir / f"{sample_id}_step7_region_enrichment_{flux_tag}.csv"

    region_summary.to_csv(out_region, index=False)
    enrich_df.to_csv(out_enrich,      index=False)

    print(f"\nSaved region summary   -> {out_region}")
    print(f"Saved enrichment tests -> {out_enrich}")

    print("\nRegion summary:")
    print(region_summary.to_string(index=False))

    print("\nEnrichment tests:")
    print(enrich_df.to_string(index=False))


if __name__ == "__main__":
    main()
