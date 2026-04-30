import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def process_sample(sid, statsdir):

    nodes = pd.read_csv(statsdir / f"{sid}_kts_states.csv")
    edges = pd.read_csv(statsdir / f"{sid}_edges_hodge.csv")

    states = nodes["state"].values
    n = len(nodes)

    transition_counts = {}

    for _, e in edges.iterrows():
        i = int(e["i"])
        j = int(e["j"])
        w = abs(e["flux_coexact"])

        s_i = states[i]
        s_j = states[j]

        key = (s_i, s_j)

        transition_counts[key] = transition_counts.get(key, 0) + w

    # convert to matrix
    states_unique = sorted(set(states))
    idx = {s: k for k, s in enumerate(states_unique)}

    M = np.zeros((len(states_unique), len(states_unique)))

    for (a, b), val in transition_counts.items():
        M[idx[a], idx[b]] += val

    # normalize row-wise
    row_sums = M.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P = M / row_sums

    df = pd.DataFrame(P, index=states_unique, columns=states_unique)
    df.to_csv(statsdir / f"{sid}_kts_transition_matrix.csv")

    return P


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--statsdir", required=True)

    args = parser.parse_args()

    statsdir = Path(args.statsdir)

    sample_ids = sorted([
        p.name.replace("_kts_states.csv", "")
        for p in statsdir.glob("*_kts_states.csv")
    ])

    for sid in sample_ids:
        process_sample(sid, statsdir)
        print(f"[done] {sid}")


if __name__ == "__main__":
    main()
