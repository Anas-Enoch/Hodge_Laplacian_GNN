#!/usr/bin/env python3

"""
Step 06 — Interface Enrichment of Coexact Energy

Input:
    *_spots_coexact_energy.csv

Output:
    *_enrichment.csv

Computes:
    R = interface / tumor_core
    permutation p-value
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


# ── Parameters ─────────────────────────────────────────

N_PERM = 300  # keep fast, increase later if needed


# ── Main ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    print(f"[load] {args.input_csv.name} ({len(df)} spots)")

    # ── Extract groups ───────────────────────────────

    interface = df[df["region"] == "interface"]["coexact_energy"]
    tumor = df[df["region"] == "tumor_core"]["coexact_energy"]

    if len(interface) == 0 or len(tumor) == 0:
        print("[SKIP] missing interface or tumor_core")
        return

    mu_i = interface.mean()
    mu_t = tumor.mean()

    R = mu_i / (mu_t + 1e-12)

    print(f"[R] interface/tumor = {R:.3f}")

    # ── Permutation test ─────────────────────────────

    combined = df["coexact_energy"].values
    labels = df["region"].values

    R_perm = []

    for _ in range(N_PERM):
        perm_labels = np.random.permutation(labels)

        interface_perm = combined[perm_labels == "interface"]
        tumor_perm = combined[perm_labels == "tumor_core"]

        if len(interface_perm) == 0 or len(tumor_perm) == 0:
            continue

        mu_i_p = interface_perm.mean()
        mu_t_p = tumor_perm.mean()

        R_perm.append(mu_i_p / (mu_t_p + 1e-12))

    R_perm = np.array(R_perm)

    p_value = (R_perm >= R).mean()

    print(f"[p-value] {p_value:.4f}")

    # ── Save ─────────────────────────────────────────

    out = pd.DataFrame({
        "R_interface_over_tumor": [R],
        "p_value": [p_value],
        "n_interface": [len(interface)],
        "n_tumor": [len(tumor)]
    })

    out.to_csv(args.output_csv, index=False)

    print(f"[done] → {args.output_csv}")


if __name__ == "__main__":
    main()
