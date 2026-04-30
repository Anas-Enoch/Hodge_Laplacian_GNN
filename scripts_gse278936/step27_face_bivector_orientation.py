#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def process_sample(sid: str, statsdir: Path, outdir: Path):
    spots_path = statsdir / f"{sid}_spots_coexact_energy.csv"
    edges_path = statsdir / f"{sid}_edges_hodge.csv"

    spots = pd.read_csv(spots_path)
    edges = pd.read_csv(edges_path)

    required_edges = {"i", "j", "flux_coexact"}
    required_spots = {"region"}

    if not required_edges <= set(edges.columns):
        raise ValueError(f"{sid}: edges missing {required_edges - set(edges.columns)}")

    if not required_spots <= set(spots.columns):
        raise ValueError(f"{sid}: spots missing {required_spots - set(spots.columns)}")

    n = len(spots)

    curl = np.zeros(n, dtype=float)
    degree = np.zeros(n, dtype=float)

    for _, row in edges.iterrows():
        i = int(row["i"])
        j = int(row["j"])
        f = float(row["flux_coexact"])

        # signed local flux imbalance
        curl[i] += f
        curl[j] -= f

        degree[i] += 1
        degree[j] += 1

    degree[degree == 0] = 1.0

    spots["graph_curl_proxy"] = curl / degree
    spots["abs_graph_curl_proxy"] = np.abs(spots["graph_curl_proxy"])

    out_path = outdir / f"{sid}_graph_curl_proxy.csv"
    spots.to_csv(out_path, index=False)

    med = spots.groupby("region")["abs_graph_curl_proxy"].median()

    interface = float(med.get("interface", np.nan))
    tumor = float(med.get("tumor_core", np.nan))

    fold = interface / max(tumor, 1e-12) if not np.isnan(interface) and not np.isnan(tumor) else np.nan

    signed_interface = spots.loc[
        spots["region"] == "interface",
        "graph_curl_proxy"
    ].dropna()

    if len(signed_interface) > 1:
        signed_mean = signed_interface.mean()
        signed_std = signed_interface.std(ddof=1)
        signed_se = signed_std / np.sqrt(len(signed_interface))
        signed_z = signed_mean / signed_se if signed_se > 0 else np.nan
        frac_positive = (signed_interface > 0).mean()
    else:
        signed_mean = signed_z = frac_positive = np.nan

    print(f"\n=== {sid} ===")
    print(f"[done] {out_path}")
    print(f"interface median abs curl = {interface:.6e}")
    print(f"tumor_core median abs curl = {tumor:.6e}")
    print(f"fold interface/tumor_core = {fold:.3f}")
    print(f"signed_z = {signed_z:.3f}")
    print(f"frac_positive = {frac_positive:.3f}")

    return {
        "sample": sid,
        "interface_median_abs_curl": interface,
        "tumor_core_median_abs_curl": tumor,
        "fold_interface_vs_tumor_core": fold,
        "interface_signed_mean": signed_mean,
        "interface_signed_z": signed_z,
        "interface_frac_positive": frac_positive,
        "n_interface": int((spots["region"] == "interface").sum()),
        "n_tumor_core": int((spots["region"] == "tumor_core").sum()),
        "note": "Graph-based curl proxy from signed coexact edge-flux imbalance; not exact DEC face circulation."
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", type=str, default=None)
    parser.add_argument("--sample-ids", type=str, default=None)
    parser.add_argument("--statsdir", type=Path, default=Path("results_gse278936"))
    parser.add_argument("--outdir", type=Path, default=Path("results_gse278936"))
    args = parser.parse_args()

    if args.sample_ids:
        sample_ids = [s.strip() for s in args.sample_ids.split(",") if s.strip()]
    elif args.sample_id:
        sample_ids = [args.sample_id]
    else:
        sample_ids = sorted([
            p.name.replace("_edges_hodge.csv", "")
            for p in args.statsdir.glob("*_edges_hodge.csv")
        ])

    summaries = []
    for sid in sample_ids:
        summaries.append(process_sample(sid, args.statsdir, args.outdir))

    summary = pd.DataFrame(summaries)
    summary_path = args.outdir / "step27_graph_curl_proxy_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n=== STEP27 GRAPH CURL PROXY SUMMARY ===")
    print(f"n_samples = {len(summary)}")
    print(f"median fold interface/tumor_core = {summary['fold_interface_vs_tumor_core'].median():.3f}")
    print(f"[done] {summary_path}")


if __name__ == "__main__":
    main()
