#!/usr/bin/env python3
"""
Step 10 — Energy-Matched Zeta Null

Purpose
-------
Test whether the tumor–immune interface has spectral geometry beyond its
elevated coexact-energy magnitude.

This fixes the previous error:
    "missing spectral columns"

because this version DOES NOT require lambda_k / alpha_k / node_id columns
inside *_zeta_interface.csv. It recomputes normalized Zeta directly from:

    *_spots_coexact_energy.csv
    *_edges_hodge.csv

Required input files per sample:
    results_gse278936/GSMxxxx_spots_coexact_energy.csv
    results_gse278936/GSMxxxx_edges_hodge.csv

Output:
    results_gse278936/cohort_zeta_energy_matched_null.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh
from scipy.stats import binomtest


def build_laplacian(edges: pd.DataFrame, n_nodes: int):
    required = {"i", "j"}
    if not required <= set(edges.columns):
        raise ValueError(f"edges missing columns: {required - set(edges.columns)}")

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for row in edges.itertuples(index=False):
        i = int(getattr(row, "i"))
        j = int(getattr(row, "j"))

        if i < 0 or j < 0 or i >= n_nodes or j >= n_nodes:
            continue

        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([1.0, 1.0])

    A = coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    degree = np.asarray(A.sum(axis=1)).ravel()

    return diags(degree) - A


def compute_zeta_normalized(signal: np.ndarray, laplacian, k_eigs: int = 50) -> float:
    """
    Normalized Zeta:

        Z = [sum_k alpha_k / lambda_k] / [sum_k alpha_k]

    where:
        alpha_k = <signal, u_k>^2

    This removes scale dependence on total signal magnitude.
    """

    signal = np.asarray(signal, dtype=float)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale normalization for cross-cohort numerical stability.
    # This does not change normalized Zeta because the statistic is scale-invariant.
    signal = signal - np.mean(signal)
    norm = np.linalg.norm(signal)

    if norm <= 1e-300:
        return np.nan

    signal = signal / norm

    n = laplacian.shape[0]

    k_eff = min(int(k_eigs), n - 2)
    if k_eff < 2:
        return np.nan

    vals, vecs = eigsh(laplacian, k=k_eff, which="SM")

    numerator = 0.0
    denominator = 0.0

    for lam, u in zip(vals, vecs.T):
        if lam < 1e-8:
            continue

        alpha = float(np.dot(signal, u) ** 2)
        numerator += alpha / float(lam)
        denominator += alpha

    if denominator <= 1e-300:
        return np.nan

    return float(numerator / denominator)


def energy_matched_candidate_pool(
    signal: np.ndarray,
    interface_mask: np.ndarray,
    low_q: float,
    high_q: float,
):
    interface_values = signal[interface_mask]

    if len(interface_values) == 0:
        return np.array([], dtype=int), np.nan, np.nan

    q_low, q_high = np.percentile(interface_values, [low_q, high_q])

    candidates = np.where(
        (signal >= q_low) &
        (signal <= q_high) &
        (signal > 0)
    )[0]

    return candidates.astype(int), float(q_low), float(q_high)


def process_sample(
    sid: str,
    statsdir: Path,
    n_perm: int,
    k_eigs: int,
    seed: int,
    low_q: float,
    high_q: float,
) -> dict:
    spots_path = statsdir / f"{sid}_spots_coexact_energy.csv"
    edges_path = statsdir / f"{sid}_edges_hodge.csv"
    enrichment_path = statsdir / f"{sid}_enrichment.csv"

    if not spots_path.exists():
        return {"sample": sid, "status": "missing_spots_file"}
    if not edges_path.exists():
        return {"sample": sid, "status": "missing_edges_file"}

    spots = pd.read_csv(spots_path)
    edges = pd.read_csv(edges_path)

    required_spots = {"coexact_energy", "region"}
    if not required_spots <= set(spots.columns):
        return {
            "sample": sid,
            "status": "missing_spots_columns",
            "missing": ",".join(sorted(required_spots - set(spots.columns))),
        }

    n_nodes = len(spots)
    signal = spots["coexact_energy"].to_numpy(dtype=float)
    
    interface_mask = (
        (spots["region"] == "interface") &
        (spots["coexact_energy"].astype(float) > 0)
    ).to_numpy(dtype=bool)

    n_interface = int(interface_mask.sum())
    frac_interface = n_interface / max(n_nodes, 1)

    if n_interface < 5:
        return {
            "sample": sid,
            "status": "not_testable_low_interface",
            "n_nodes": n_nodes,
            "n_interface": n_interface,
        }

    rng = np.random.default_rng(seed)

    L = build_laplacian(edges, n_nodes)

    Z_global = compute_zeta_normalized(signal, L, k_eigs=k_eigs)
    Z_interface = compute_zeta_normalized(signal * interface_mask, L, k_eigs=k_eigs)
    Z_ratio = Z_interface / (Z_global + 1e-12)

    # Candidate pool matched to the interquartile energy range of interface nodes.
    candidates, q_low, q_high = energy_matched_candidate_pool(
        signal=signal,
        interface_mask=interface_mask,
        low_q=low_q,
        high_q=high_q,
    )

    energy_window = f"{low_q:g}-{high_q:g}"
    widened = False

    # If the IQR pool is too small, widen to 10–90.
    if len(candidates) < n_interface:
        candidates, q_low, q_high = energy_matched_candidate_pool(
            signal=signal,
            interface_mask=interface_mask,
            low_q=10,
            high_q=90,
        )
        energy_window = "10-90_widened"
        widened = True

    if len(candidates) < n_interface:
        return {
            "sample": sid,
            "status": "not_testable_insufficient_energy_matched_candidates",
            "n_nodes": n_nodes,
            "n_interface": n_interface,
            "n_candidates": len(candidates),
            "energy_q_low": q_low,
            "energy_q_high": q_high,
            "energy_window": energy_window,
        }

    null_ratios = []

    for _ in range(n_perm):
        perm_idx = rng.choice(candidates, size=n_interface, replace=False)

        perm_mask = np.zeros(n_nodes, dtype=bool)
        perm_mask[perm_idx] = True

        Z_perm = compute_zeta_normalized(signal * perm_mask, L, k_eigs=k_eigs)
        null_ratios.append(Z_perm / (Z_global + 1e-12))

    null_ratios = np.asarray(null_ratios, dtype=float)
    null_ratios = null_ratios[np.isfinite(null_ratios)]

    if len(null_ratios) == 0:
        return {
            "sample": sid,
            "status": "not_testable_all_null_nan",
            "n_nodes": n_nodes,
            "n_interface": n_interface,
        }

    p_energy = (np.sum(null_ratios >= Z_ratio) + 1) / (len(null_ratios) + 1)

    R = np.nan
    enrichment_p = np.nan

    if enrichment_path.exists():
        try:
            enrich = pd.read_csv(enrichment_path).iloc[0]
            R = float(enrich.get("R_interface_over_tumor", np.nan))
            enrichment_p = float(enrich.get("p_value", np.nan))
        except Exception:
            pass

    return {
        "sample": sid,
        "status": "ok",
        "n_nodes": n_nodes,
        "n_interface": n_interface,
        "frac_interface": frac_interface,
        "n_candidates": len(candidates),
        "energy_q_low": q_low,
        "energy_q_high": q_high,
        "energy_window": energy_window,
        "energy_window_widened": widened,
        "R_interface_over_tumor": R,
        "enrichment_p_value": enrichment_p,
        "Z_global_normalized": Z_global,
        "Z_interface_normalized": Z_interface,
        "Z_interface_over_global": Z_ratio,
        "Z_energy_matched_null_mean": float(np.mean(null_ratios)),
        "Z_energy_matched_null_median": float(np.median(null_ratios)),
        "Z_energy_matched_null_sd": float(np.std(null_ratios, ddof=1)),
        "p_energy_matched": float(p_energy),
        "interpretation": (
            "interface_spectrally_special_beyond_energy"
            if p_energy < 0.05
            else "not_spectrally_special_beyond_energy"
        ),
        "zeta_definition": "normalized: sum(alpha/lambda)/sum(alpha)",
        "null_model": "energy-matched random interface mask using interface coexact-energy quantiles",
        "n_perm": int(n_perm),
        "k_eigs": int(k_eigs),
        "seed": int(seed),
        "signal_preprocessing": "nan_to_zero; mean_centered; l2_normalized inside compute_zeta_normalized",
        "energy_positive_filter": "signal > 0",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--statsdir", type=Path, default=Path("results_gse278936"))
    parser.add_argument("--out", type=Path, default=Path("results_gse278936/cohort_zeta_energy_matched_null.csv"))
    parser.add_argument("--n-perm", type=int, default=300)
    parser.add_argument("--k-eigs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--low-q", type=float, default=25)
    parser.add_argument("--high-q", type=float, default=75)
    parser.add_argument("--sample-ids", type=str, default=None)
    args = parser.parse_args()

    if args.sample_ids:
        sample_ids = [s.strip() for s in args.sample_ids.split(",") if s.strip()]
    else:
        sample_ids = sorted(
            p.name.replace("_spots_coexact_energy.csv", "")
            for p in args.statsdir.glob("*_spots_coexact_energy.csv")
        )

    if not sample_ids:
        raise ValueError("No *_spots_coexact_energy.csv files found.")

    rows = []

    for sid in sample_ids:
        # Stable per-sample seed independent of Python hash randomization.
        sid_seed = args.seed + sum(ord(c) for c in sid)

        row = process_sample(
            sid=sid,
            statsdir=args.statsdir,
            n_perm=args.n_perm,
            k_eigs=args.k_eigs,
            seed=sid_seed,
            low_q=args.low_q,
            high_q=args.high_q,
        )

        rows.append(row)

        if row.get("status") == "ok":
            print(
                f"[{sid}] "
                f"Zratio={row['Z_interface_over_global']:.3f} "
                f"null_med={row['Z_energy_matched_null_median']:.3f} "
                f"p_energy={row['p_energy_matched']:.4f} "
                f"{row['interpretation']}"
            )
        else:
            print(f"[{sid}] {row.get('status')}")

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    ok = out[out["status"] == "ok"].copy()

    print("\n=== STEP10 ENERGY-MATCHED ZETA NULL SUMMARY ===")
    print(f"n_total = {len(out)}")
    print(f"n_ok = {len(ok)}")

    if len(ok) > 0:
        n_sig = int((ok["p_energy_matched"] < 0.05).sum())
        n_gt = int((ok["Z_interface_over_global"] > ok["Z_energy_matched_null_median"]).sum())

        print(f"interface > energy-matched null median = {n_gt}/{len(ok)}")
        print(f"p_energy_matched < 0.05 = {n_sig}/{len(ok)}")

        sign_p = binomtest(n_gt, len(ok), p=0.5, alternative="greater").pvalue
        print(f"sign_test_p_interface_gt_null = {sign_p:.3e}")

    print(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
