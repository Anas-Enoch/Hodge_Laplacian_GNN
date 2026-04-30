#!/usr/bin/env python3

"""
Step 03 — Visium Wedge Flux Construction

Input:
    *_spots_regions.csv

Output:
    *_edges_wedge.csv

Computes:
    F_ij = tumor_i * immune_j - tumor_j * immune_i
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# ── Parameters ─────────────────────────────────────────────

K = 6


# ── Main ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    print(f"[load] {args.input_csv.name} ({len(df)} spots)")

    coords = df[["x", "y"]].values

    # ── Build kNN graph ───────────────────────────────────

    nbrs = NearestNeighbors(n_neighbors=K + 1).fit(coords)
    _, indices = nbrs.kneighbors(coords)

    neighbors = indices[:, 1:]  # remove self

    edges = []

    # ── Compute wedge flux ────────────────────────────────

    for i in range(len(df)):
        for j in neighbors[i]:
            if i < j:  # avoid duplicates

                A_i = df.iloc[i]["tumor_score"]
                B_i = df.iloc[i]["immune_score"]

                A_j = df.iloc[j]["tumor_score"]
                B_j = df.iloc[j]["immune_score"]

                # wedge (antisymmetric)
                flux = A_i * B_j - A_j * B_i

                edges.append({
                    "i": i,
                    "j": j,
                    "flux_wedge": flux,
                    "abs_flux": abs(flux),

                    # optional (useful later)
                    "region_i": df.iloc[i]["region"],
                    "region_j": df.iloc[j]["region"]
                })

    edges = pd.DataFrame(edges)

    print(f"[edges] {len(edges)}")

    # ── Basic sanity check ────────────────────────────────

    print("[flux stats]")
    print(edges["flux_wedge"].describe())

    # ── Save ─────────────────────────────────────────────

    edges.to_csv(args.output_csv, index=False)

    print(f"[done] → {args.output_csv}")


if __name__ == "__main__":
    main()
