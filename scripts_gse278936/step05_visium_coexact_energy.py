#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges_csv", type=Path, required=True)
    parser.add_argument("--spots_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    args = parser.parse_args()

    edges = pd.read_csv(args.edges_csv)
    spots = pd.read_csv(args.spots_csv)

    # 🔴 HARD CHECK (prevents this bug forever)
    if "region" not in spots.columns:
        raise ValueError(f"[FATAL] region column missing in {args.spots_csv}")

    n_nodes = len(spots)

    print(f"[nodes] {n_nodes}  [edges] {len(edges)}")

    # ── energy accumulation ─────────────────────

    energy = np.zeros(n_nodes)
    degree = np.zeros(n_nodes)

    for _, e in edges.iterrows():
        i = int(e["i"])
        j = int(e["j"])
        f = e["flux_coexact"]

        val = f * f

        energy[i] += val
        energy[j] += val

        degree[i] += 1
        degree[j] += 1

    degree = np.maximum(degree, 1)
    energy = energy / degree

    # ── attach WITHOUT dropping anything ────────

    spots["coexact_energy"] = energy

    # 🔴 DEBUG CHECK
    print("[columns]", spots.columns.tolist())

    spots.to_csv(args.output_csv, index=False)

    print(f"[done] → {args.output_csv}")


if __name__ == "__main__":
    main()
