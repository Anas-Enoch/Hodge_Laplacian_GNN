#!/usr/bin/env python3
"""
step15_regime_assignment.py — Regime Classification
====================================================
Classifies each spot into one of five operator regimes based on
region labels and coexact energy quantile within the section.

SAFETY STATEMENT:
-----------------
These regime labels are operator-level classifications derived from
spatial transcriptomics decomposition. They are not clinical diagnoses
and do not imply specific biological mechanisms. The classification
reflects the energy distribution of the Hodge coexact component, not
a direct measurement of transport dynamics.

Regimes
-------
  bulk_like       : tumor-core region, coexact_energy < Q50 (gradient-compatible)
  interface_like  : interface_like or tumor-adjacent region
  stromal_like    : stroma region
  immune_like     : immune region
  other           : all remaining nodes

Input
-----
  {sid}_spots_coexact_energy.csv   — node table with [region, coexact_energy, ...]
  {sid}_edges_hodge.csv            — edge table (used for degree normalization)

Output
------
  {sid}_regime_assignment.csv
  Columns: node_id, x, y, region, coexact_energy, regime,
           coexact_quantile_within_section

Usage
-----
  python step15_regime_assignment.py \\
      --statsdir results_gse278936 \\
      --sample-id GSM_ID \\
      [--coexact-bulk-q 0.50]   # quantile threshold below which tumor-core = bulk_like

  Cohort mode (all *_spots_coexact_energy.csv in statsdir):
      python step15_regime_assignment.py --statsdir results_gse278936
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


BULK_LIKE_REGIONS   = {"tumor_core", "tumor_enriched", "tumor"}
INTERFACE_REGIONS   = {"interface", "interface_like"}
STROMAL_REGIONS     = {"stroma", "stroma_enriched", "stromal"}
IMMUNE_REGIONS      = {"immune", "immune_core", "immune_enriched"}


def classify_regime(row: pd.Series, q50: float) -> str:
    r = str(row["region"]).strip().lower()
    e = float(row["coexact_energy"])
    if r in BULK_LIKE_REGIONS:
        return "bulk_like" if e < q50 else "interface_adjacent"
    if r in INTERFACE_REGIONS:
        return "interface_like"
    if r in STROMAL_REGIONS:
        return "stromal_like"
    if r in IMMUNE_REGIONS:
        return "immune_like"
    return "other"


def process_sample(sid: str, statsdir: Path, outdir: Path,
                   bulk_q: float = 0.50) -> pd.DataFrame:
    spots_path = statsdir / f"{sid}_spots_coexact_energy.csv"
    if not spots_path.exists():
        raise FileNotFoundError(spots_path)

    spots = pd.read_csv(spots_path)
    required = {"region", "coexact_energy"}
    if not required <= set(spots.columns):
        raise ValueError(f"{sid}: missing {required - set(spots.columns)}")

    # Add node_id if absent
    if "node_id" not in spots.columns:
        spots = spots.reset_index().rename(columns={"index": "node_id"})

    # Section-wide quantile
    q50_val = spots["coexact_energy"].quantile(bulk_q)
    spots["coexact_quantile_within_section"] = spots["coexact_energy"].rank(pct=True)

    spots["regime"] = spots.apply(
        lambda r: classify_regime(r, q50_val), axis=1
    )

    out_cols = ["node_id", "region", "coexact_energy",
                "coexact_quantile_within_section", "regime"]
    for col in ["x", "y", "x_fullres", "y_fullres"]:
        if col in spots.columns:
            out_cols.insert(1, col)

    out = spots[[c for c in out_cols if c in spots.columns]].copy()
    out.insert(0, "sample_id", sid)

    out_path = outdir / f"{sid}_regime_assignment.csv"
    out.to_csv(out_path, index=False)

    regime_summary = (
        out.groupby("regime")["coexact_energy"]
        .agg(n="count", median="median", mean="mean")
        .reset_index()
    )
    print(f"  [{sid}] {len(out)} nodes classified")
    print(regime_summary.to_string(index=False))
    return out


def main():
    ap = argparse.ArgumentParser(description="Step 15: Regime assignment")
    ap.add_argument("--statsdir",   type=Path, default=Path("results_gse278936"))
    ap.add_argument("--outdir",     type=Path, default=None)
    ap.add_argument("--sample-id",  type=str,  default=None)
    ap.add_argument("--coexact-bulk-q", type=float, default=0.50,
                    help="Energy quantile below which tumor-core nodes = bulk_like")
    args = ap.parse_args()

    outdir = args.outdir or args.statsdir
    outdir.mkdir(parents=True, exist_ok=True)

    if args.sample_id:
        sample_ids = [args.sample_id]
    else:
        sample_ids = sorted([
            p.name.replace("_spots_coexact_energy.csv", "")
            for p in args.statsdir.glob("*_spots_coexact_energy.csv")
        ])

    print(f"Processing {len(sample_ids)} samples …")
    for sid in sample_ids:
        print(f"\n── {sid} ──")
        try:
            process_sample(sid, args.statsdir, outdir, args.coexact_bulk_q)
        except Exception as exc:
            print(f"  SKIP: {exc}")


if __name__ == "__main__":
    main()
