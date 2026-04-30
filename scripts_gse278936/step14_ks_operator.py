import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, diags
from pathlib import Path
import argparse


def build_graph(n, edges):
    i = edges["i"].values
    j = edges["j"].values
    w = np.abs(edges["flux_coexact"].values)

    A = coo_matrix((w, (i, j)), shape=(n, n))
    A = A + A.T

    d = np.array(A.sum(axis=1)).flatten()
    D = diags(d)

    L = D - A
    return L


def gradient_energy(n, edges, u):
    i = edges["i"].values
    j = edges["j"].values

    diff = u[i] - u[j]
    return np.bincount(i, weights=diff**2, minlength=n)


def process_sample(sid, statsdir, outdir):
    nodes = pd.read_csv(statsdir / f"{sid}_spots_coexact_energy.csv")
    edges = pd.read_csv(statsdir / f"{sid}_edges_hodge.csv")

    u = nodes["coexact_energy"].values
    n = len(u)

    L = build_graph(n, edges)

    Lu = L @ u
    L2u = L @ Lu
    nonlin = gradient_energy(n, edges, u)

    ks = -Lu - L2u - nonlin

    df = pd.DataFrame({
        "node": np.arange(n),
        "ks_value": ks,
        "coexact_energy": u,
        "region": nodes["region"]
    })

    df.to_csv(outdir / f"{sid}_ks_operator.csv", index=False)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--statsdir", required=True)
    parser.add_argument("--outdir", required=True)

    args = parser.parse_args()

    statsdir = Path(args.statsdir)
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    sample_ids = sorted([
        p.name.replace("_spots_coexact_energy.csv", "")
        for p in statsdir.glob("*_spots_coexact_energy.csv")
    ])

    for sid in sample_ids:
        process_sample(sid, statsdir, outdir)
        print(f"[done] {sid}")


if __name__ == "__main__":
    main()
