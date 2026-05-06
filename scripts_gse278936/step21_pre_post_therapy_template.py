#!/usr/bin/env python3
"""
step21_pre_post_therapy_template.py — Pre/Post Therapy Response Template
=========================================================================
PROSPECTIVE-USE TEMPLATE ONLY.

This script was not used as evidence in the current study. It provides a
reusable template for future paired pre-treatment / post-treatment spatial
transcriptomics analyses. Outputs are research metrics, not clinical endpoints.

To use this template, paired pre- and post-treatment samples processed
through Steps 15–19 must be available. The script computes signed Delta
metrics (post − pre) at the interface region for each operator measure.

SAFETY STATEMENT:
-----------------
None of these metrics are validated clinical endpoints. This framework
does not predict treatment outcome and must not be used for clinical
decision-making. All outputs are pre-specified research-level hypotheses
requiring prospective validation.

Usage
-----
  python step21_pre_post_therapy_template.py \\
      --pre-id   GSM_pre_treatment \\
      --post-id  GSM_post_treatment \\
      --statsdir results_paired \\
      --outdir   results_paired \\
      --label    "Patient_01_anti_VEGF"
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp


def build_laplacian(edges, n):
    i = edges["i"].values.astype(int)
    j = edges["j"].values.astype(int)
    w = np.abs(edges["flux_coexact"].values.astype(float))
    A = sp.coo_matrix((w, (i, j)), shape=(n, n)).tocsr()
    A = A + A.T
    D = sp.diags(np.array(A.sum(axis=1)).ravel())
    return (D - A).tocsr()


def regime_summary(sid: str, statsdir: Path) -> dict | None:
    """Compute per-regime metric medians for one sample."""
    spots_path  = statsdir / f"{sid}_spots_coexact_energy.csv"
    edges_path  = statsdir / f"{sid}_edges_hodge.csv"
    regime_path = statsdir / f"{sid}_regime_assignment.csv"
    cdis_path   = statsdir / f"{sid}_constraint_score.csv"

    if not all(p.exists() for p in [spots_path, edges_path, regime_path]):
        print(f"  [{sid}] SKIP — run steps 15–19 first")
        return None

    spots  = pd.read_csv(spots_path)
    edges  = pd.read_csv(edges_path)
    regime = pd.read_csv(regime_path)

    if "node_id" not in spots.columns:
        spots = spots.reset_index().rename(columns={"index": "node_id"})
    spots = spots.merge(regime[["node_id", "regime"]], on="node_id", how="left")

    if cdis_path.exists():
        cdis_df = pd.read_csv(cdis_path)[["node_id", "CDIS"]]
        spots = spots.merge(cdis_df, on="node_id", how="left")
    else:
        spots["CDIS"] = np.nan

    n = len(spots)
    u = spots["coexact_energy"].values.astype(float)
    L = build_laplacian(edges, n)
    Lu  = L @ u
    L2u = L @ Lu
    i_idx = edges["i"].values.astype(int)
    j_idx = edges["j"].values.astype(int)
    deg   = np.maximum(
        np.bincount(i_idx, minlength=n) + np.bincount(j_idx, minlength=n), 1
    ).astype(float)
    diff2  = (u[i_idx] - u[j_idx]) ** 2
    nonlin = np.bincount(i_idx, weights=diff2, minlength=n) / deg
    ks     = np.abs(-Lu - L2u - nonlin)

    if "flux_exact" in edges.columns:
        exact_e = np.bincount(
            i_idx, weights=edges["flux_exact"].values**2, minlength=n
        ) / deg
        ratio = u / (exact_e + 1e-12)
    else:
        ratio = np.zeros(n)

    spots["ks_like"]             = ks
    spots["coexact_exact_ratio"] = ratio

    int_mask = spots["regime"].isin({"interface_like"})
    if int_mask.sum() < 1:
        print(f"  [{sid}] no interface nodes found")
        return None

    def med(col):
        return float(np.median(spots.loc[int_mask, col].dropna()))

    # KTS bias — load if available
    kts_path = statsdir / f"{sid}_kts_transition_bias.csv"
    kts_bias = np.nan
    if kts_path.exists():
        kts = pd.read_csv(kts_path)
        ie_rows = kts[kts["target"] == "IMMUNE_EXHAUSTED"]["bias_ratio"]
        if len(ie_rows):
            kts_bias = float(ie_rows.median())

    return {
        "sample_id":                 sid,
        "n_interface_nodes":         int(int_mask.sum()),
        "coexact_energy_median":     med("coexact_energy"),
        "coexact_exact_ratio_median":med("coexact_exact_ratio"),
        "ks_like_median":            med("ks_like"),
        "CDIS_median":               med("CDIS") if "CDIS" in spots.columns else np.nan,
        "kts_exhaustion_bias_median":kts_bias,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Step 21: Pre/post therapy response template"
    )
    ap.add_argument("--pre-id",   required=True, help="Pre-treatment sample ID")
    ap.add_argument("--post-id",  required=True, help="Post-treatment sample ID")
    ap.add_argument("--statsdir", type=Path, default=Path("results_paired"))
    ap.add_argument("--outdir",   type=Path, default=None)
    ap.add_argument("--label",    type=str,  default="paired_sample",
                    help="Label for the pair (e.g., Patient_01_intervention)")
    args = ap.parse_args()

    outdir = args.outdir or args.statsdir
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Pre:  {args.pre_id}")
    print(f"Post: {args.post_id}")
    print(f"Label: {args.label}")

    pre  = regime_summary(args.pre_id,  args.statsdir)
    post = regime_summary(args.post_id, args.statsdir)

    if pre is None or post is None:
        print("Cannot compute deltas — one or both samples failed.")
        return

    metrics = ["coexact_energy_median", "coexact_exact_ratio_median",
               "ks_like_median", "CDIS_median", "kts_exhaustion_bias_median"]

    rows = []
    for m in metrics:
        pre_val  = pre.get(m, np.nan)
        post_val = post.get(m, np.nan)
        delta    = post_val - pre_val if not (np.isnan(pre_val) or np.isnan(post_val)) else np.nan
        rows.append({
            "label":    args.label,
            "metric":   m,
            "pre":      pre_val,
            "post":     post_val,
            "delta":    delta,
            "direction": ("decrease" if delta < 0 else "increase" if delta > 0 else "unchanged")
                         if not np.isnan(delta) else "NA",
        })

    df = pd.DataFrame(rows)
    out = outdir / "paired_response_metrics.csv"
    df.to_csv(out, index=False)

    print("\n=== PAIRED RESPONSE METRICS ===")
    print(df[["metric", "pre", "post", "delta", "direction"]].to_string(index=False))
    print(f"\n[done] {out}")
    print("\nSAFETY: These metrics are research outputs, not clinical endpoints.")


if __name__ == "__main__":
    main()
