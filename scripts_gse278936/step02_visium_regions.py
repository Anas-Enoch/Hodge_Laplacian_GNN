#!/usr/bin/env python3

"""
Step 02 — Visium Interface Detection

Input:
    results_gse278936/{sample}_spots.csv

Output:
    results_gse278936/{sample}_spots_regions.csv

Adds:
    region ∈ {tumor_core, immune_core, interface, other}
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# ── Parameters ───────────────────────────────────────────────

K = 6
TUMOR_Q = 0.75
IMMUNE_Q = 0.75


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    print(f"[load] {args.input_csv.name} ({len(df)} spots)")

    # ── Define tumor / immune cores ──────────────────────────

    tumor_thr = df["tumor_score"].quantile(TUMOR_Q)
    immune_thr = df["immune_score"].quantile(IMMUNE_Q)

    df["is_tumor"] = df["tumor_score"] >= tumor_thr
    df["is_immune"] = df["immune_score"] >= immune_thr

    # ── Build spatial kNN graph ─────────────────────────────

    coords = df[["x", "y"]].values

    nbrs = NearestNeighbors(n_neighbors=K + 1).fit(coords)
    _, indices = nbrs.kneighbors(coords)

    # remove self neighbor
    neighbors = indices[:, 1:]

    # ── Interface detection ─────────────────────────────────

    region = []

    for i in range(len(df)):
        neigh = neighbors[i]

        neigh_tumor = df.iloc[neigh]["is_tumor"].any()
        neigh_immune = df.iloc[neigh]["is_immune"].any()

        if df.iloc[i]["is_tumor"] and not neigh_immune:
            region.append("tumor_core")

        elif df.iloc[i]["is_immune"] and not neigh_tumor:
            region.append("immune_core")

        elif neigh_tumor and neigh_immune:
            region.append("interface")

        else:
            region.append("other")

    df["region"] = region

    # ── Summary ─────────────────────────────────────────────

    print("[region counts]")
    print(df["region"].value_counts())

    # ── Save ────────────────────────────────────────────────

    df.to_csv(args.output_csv, index=False)

    print(f"[done] → {args.output_csv}")


if __name__ == "__main__":
    main()
