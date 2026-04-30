#!/usr/bin/env python3

"""
Step 07 — Cohort-Level Summary (Visium)

Input:
    *_enrichment.csv

Output:
    cohort_summary.csv

Computes:
    - median R
    - fraction R > 1
    - sign test
    - fraction significant (p < 0.05)
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import binomtest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    args = parser.parse_args()

    files = list(args.input_dir.glob("*_enrichment.csv"))

    if not files:
        raise ValueError("No enrichment files found")

    rows = []

    for f in files:
        df = pd.read_csv(f)

        R = df["R_interface_over_tumor"].iloc[0]
        p = df["p_value"].iloc[0]

        rows.append({
            "sample": f.stem.replace("_enrichment", ""),
            "R": R,
            "p_value": p
        })

    df_all = pd.DataFrame(rows)

    # ── Core stats ─────────────────────────────────

    median_R = df_all["R"].median()
    mean_R = df_all["R"].mean()

    n_total = len(df_all)
    n_gt1 = (df_all["R"] > 1).sum()

    frac_gt1 = n_gt1 / n_total

    # sign test: H0 = 0.5
    sign_test = binomtest(n_gt1, n_total, p=0.5, alternative="greater")

    # significance
    n_sig = (df_all["p_value"] < 0.05).sum()
    frac_sig = n_sig / n_total

    # ── Print summary ───────────────────────────────

    print("\n=== COHORT SUMMARY ===")
    print(f"n_samples            = {n_total}")
    print(f"median_R             = {median_R:.3f}")
    print(f"mean_R               = {mean_R:.3f}")
    print(f"R > 1                = {n_gt1}/{n_total} ({frac_gt1:.2%})")
    print(f"sign_test_p          = {sign_test.pvalue:.3e}")
    print(f"p < 0.05             = {n_sig}/{n_total} ({frac_sig:.2%})")

    # ── Save ───────────────────────────────────────

    df_all.to_csv(args.output_csv, index=False)

    print(f"\n[done] → {args.output_csv}")


if __name__ == "__main__":
    main()
