#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh
from scipy.stats import spearmanr, binomtest


# ─────────────────────────────────────────────
# Laplacian
# ─────────────────────────────────────────────
def build_laplacian(edges, n):
    rows, cols, data = [], [], []

    for _, e in edges.iterrows():
        i, j = int(e["i"]), int(e["j"])
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([1.0, 1.0])

    A = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()

    return diags(deg) - A


# ─────────────────────────────────────────────
# Zeta
# ─────────────────────────────────────────────
def compute_zeta(signal, L, k=50):
    """Normalized Zeta — see step08 for full docstring."""
    n = L.shape[0]
    k_eff = min(k, n - 2)
    if k_eff < 2:
        return np.nan

    vals, vecs = eigsh(L, k=k_eff, which="SM")

    numerator = 0.0
    alpha_sum = 0.0
    for lam, u in zip(vals, vecs.T):
        if lam < 1e-8:
            continue
        proj      = np.dot(signal, u)
        alpha_k   = proj * proj
        numerator += alpha_k / lam
        alpha_sum += alpha_k

    if alpha_sum < 1e-12:
        return 0.0
    return float(numerator / alpha_sum)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-perm", type=int, default=300)
    parser.add_argument("--k-eigs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    zeta_files = list(args.dir.glob("*_zeta_interface.csv"))

    rows = []

    for zf in zeta_files:
        sid = zf.name.replace("_zeta_interface.csv", "")

        spots_file = args.dir / f"{sid}_spots_coexact_energy.csv"
        edges_file = args.dir / f"{sid}_edges_hodge.csv"
        enrich_file = args.dir / f"{sid}_enrichment.csv"

        if not (spots_file.exists() and edges_file.exists() and enrich_file.exists()):
            print(f"[skip] {sid}")
            continue

        spots = pd.read_csv(spots_file)
        edges = pd.read_csv(edges_file)
        enrich = pd.read_csv(enrich_file).iloc[0]
        zeta = pd.read_csv(zf).iloc[0]

        if "region" not in spots.columns:
            print(f"[skip] {sid} (no region)")
            continue

        n = len(spots)

        signal = spots["coexact_energy"].values
        mask = (spots["region"] == "interface").values

        n_interface = mask.sum()
        frac_interface = n_interface / n

        L = build_laplacian(edges, n)

        Z_global = zeta["Z_global"]
        Z_interface = zeta["Z_interface"]
        Z_ratio = zeta["Z_interface_over_global"]

        # ── Size normalization
        Z_density = Z_ratio / max(frac_interface, 1e-12)

        # ── Null model
        null_vals = []

        idx = np.arange(n)

        for _ in range(args.n_perm):
            perm_idx = rng.choice(idx, size=n_interface, replace=False)

            perm_mask = np.zeros(n, dtype=bool)
            perm_mask[perm_idx] = True

            perm_signal = signal * perm_mask

            Z_perm = compute_zeta(perm_signal, L, k=args.k_eigs)

            null_vals.append(Z_perm / (Z_global + 1e-12))

        null_vals = np.array(null_vals)

        p_null = (np.sum(null_vals >= Z_ratio) + 1) / (len(null_vals) + 1)

        rows.append({
            "sample": sid,
            "R": enrich["R_interface_over_tumor"],
            "p_enrichment": enrich["p_value"],
            "Z_ratio": Z_ratio,
            "frac_interface": frac_interface,
            "Z_density": Z_density,
            "Z_null_mean": np.mean(null_vals),
            "Z_null_p": p_null
        })

        print(f"[{sid}] R={enrich['R_interface_over_tumor']:.2f} "
              f"Zratio={Z_ratio:.3f} "
              f"Zdens={Z_density:.3f} "
              f"p_null={p_null:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print("\n=== COHORT SUMMARY ===")
    print(f"n = {len(df)}")
    print(f"median Z_ratio   = {df['Z_ratio'].median():.3f}")
    print(f"median Z_density = {df['Z_density'].median():.3f}")
    print(f"Z_density > 1    = {(df['Z_density'] > 1).sum()}/{len(df)}")
    print(f"p_null < 0.05    = {(df['Z_null_p'] < 0.05).sum()}/{len(df)}")

    # ── Spearman: local enrichment vs spectral organization ────────────────
    r, p = spearmanr(df["R"], df["Z_density"])
    print("\n=== LOCAL–SPECTRAL COUPLING ===")
    print(f"Spearman(R, Z_density) = {r:.3f}, p = {p:.4f}")
    print("Interpretation: R measures local coexact enrichment at the interface;")
    print("Z_density measures whether that signal is carried by large-scale")
    print("spectral modes. Independence (rho ≈ 0) means these are orthogonal")
    print("axes — not redundant descriptors of the same phenomenon.")

    # ── Named dissociation cases ───────────────────────────────────────────
    # Dissociation type A: high local enrichment (R > median), low spectral (p_null >= 0.05)
    # Dissociation type B: low local enrichment (R <= median), significant spectral (p_null < 0.05)
    R_median = df["R"].median()
    sig_local    = df["p_enrichment"] < 0.05
    sig_spectral = df["Z_null_p"] < 0.05

    type_A = df[sig_local & ~sig_spectral].copy()   # high R, null spectral
    type_B = df[~sig_local & sig_spectral].copy()   # low R, significant spectral

    print("\n=== DISSOCIATION CASES ===")
    print("These sections demonstrate independence of local and spectral axes.")
    print()

    if len(type_A):
        print("Type A — locally enriched, spectrally null")
        print("(high R: coexact energy concentrated at interface,")
        print(" but not carried by large-scale eigenmodes)")
        print(type_A[["sample", "R", "p_enrichment", "Z_density", "Z_null_p"]]
              .sort_values("R", ascending=False).to_string(index=False))
    else:
        print("Type A — none detected")

    print()

    if len(type_B):
        print("Type B — locally non-enriched, spectrally significant")
        print("(low R: coexact energy not strongly concentrated at interface,")
        print(" but the distributed signal is spectrally organized)")
        print(type_B[["sample", "R", "p_enrichment", "Z_density", "Z_null_p"]]
              .sort_values("Z_density", ascending=False).to_string(index=False))
    else:
        print("Type B — none detected")

    # Append dissociation classification to output
    df["dissociation_type"] = "concordant"
    df.loc[type_A.index, "dissociation_type"] = "type_A_local_only"
    df.loc[type_B.index, "dissociation_type"] = "type_B_spectral_only"
    df.to_csv(args.out, index=False)   # overwrite with classification column

    print(f"\n[done] → {args.out}")


if __name__ == "__main__":
    main()
