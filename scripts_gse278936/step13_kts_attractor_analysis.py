import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def compute_entropy(P):
    P_safe = np.where(P > 0, P, 1)
    return -np.sum(P * np.log(P_safe))


def process_sample(sid, statsdir):

    P = pd.read_csv(
        statsdir / f"{sid}_kts_transition_matrix.csv",
        index_col=0
    ).values

    entropy = compute_entropy(P)

    eigvals = np.linalg.eigvals(P.T)

    # stationary distribution
    stat = np.real(eigvals[np.argmax(np.real(eigvals))])

    return {
        "sample": sid,
        "entropy": entropy,
        "dominant_eigenvalue": float(stat)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--statsdir", required=True)

    args = parser.parse_args()

    statsdir = Path(args.statsdir)

    sample_ids = sorted([
        p.name.replace("_kts_transition_matrix.csv", "")
        for p in statsdir.glob("*_kts_transition_matrix.csv")
    ])

    rows = []

    for sid in sample_ids:
        rows.append(process_sample(sid, statsdir))

    df = pd.DataFrame(rows)
    df.to_csv(statsdir / "kts_summary.csv", index=False)

    print(df.describe())


if __name__ == "__main__":
    main()
