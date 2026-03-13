from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse


def compute_residual(B1, flux):

    """
    r = B1 f
    """

    return B1 @ flux


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_id", required=True)
    parser.add_argument("--statsdir", default="stats")

    args = parser.parse_args()

    sample = args.sample_id
    statsdir = Path(args.statsdir)

    nodes = pd.read_csv(statsdir / f"{sample}_step3_nodes.csv")
    edges = pd.read_csv(statsdir / f"{sample}_step4_edge_fluxes.csv")

    B1 = sparse.load_npz(statsdir / f"{sample}_step3_B1.npz")

    print("======================================")
    print("STEP 5: Conservation residuals")
    print("Sample:", sample)
    print("Nodes:", len(nodes))
    print("Edges:", len(edges))
    print("======================================")

    flux_ti = edges["flux_tumor_immune"].values
    flux_ts = edges["flux_tumor_stroma"].values
    flux_is = edges["flux_immune_stroma"].values

    r_ti = compute_residual(B1, flux_ti)
    r_ts = compute_residual(B1, flux_ts)
    r_is = compute_residual(B1, flux_is)

    nodes["residual_tumor_immune"] = r_ti
    nodes["residual_tumor_stroma"] = r_ts
    nodes["residual_immune_stroma"] = r_is

    out = statsdir / f"{sample}_step5_node_residuals.csv"
    nodes.to_csv(out, index=False)

    print("Saved:", out)

    print("\nResidual summary")
    print("--------------------------------")
    print("tumor-immune:", np.mean(np.abs(r_ti)), np.std(r_ti))
    print("tumor-stroma:", np.mean(np.abs(r_ts)), np.std(r_ts))
    print("immune-stroma:", np.mean(np.abs(r_is)), np.std(r_is))


if __name__ == "__main__":
    main()
