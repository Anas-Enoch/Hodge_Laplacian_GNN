import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def assign_state(row):
    # Example thresholds (adjust later)
    if row["tumor_score"] > 0.6 and row["immune_score"] < 0.3:
        return "TUMOR"
    elif row["immune_score"] > 0.6 and row["exhaustion_score"] < 0.4:
        return "IMMUNE_ACTIVE"
    elif row["immune_score"] > 0.6 and row["exhaustion_score"] > 0.4:
        return "IMMUNE_EXHAUSTED"
    elif row["stroma_score"] > 0.6:
        return "STROMA"
    else:
        return "MIXED"


def process_sample(sid, statsdir, outdir):
    df = pd.read_csv(statsdir / f"{sid}_spots_coexact_energy.csv")

    # must exist from Step01
    required = ["tumor_score", "immune_score", "stroma_score"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"{sid}: missing {c}")

    # fake exhaustion proxy (can refine later)
    df["exhaustion_score"] = df["immune_score"] * 0.5

    df["state"] = df.apply(assign_state, axis=1)

    df.to_csv(outdir / f"{sid}_kts_states.csv", index=False)

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
