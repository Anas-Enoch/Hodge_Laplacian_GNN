#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from sklearn.neighbors import kneighbors_graph


def assign_state(row):
    tumor = row.get("tumor_score", 0)
    immune = row.get("immune_score", 0)
    stroma = row.get("stroma_score", 0)
    exhaustion = row.get("exhaustion_score", 0)
    cytotoxic = row.get("cytotoxic_score", 0)

    if immune > 0.5 and exhaustion > 0.4:
        return "IMMUNE_EXHAUSTED"
    if immune > 0.5 and cytotoxic > 0.4:
        return "IMMUNE_ACTIVE"
    if stroma > 0.5:
        return "STROMA"
    if tumor > 0.5:
        return "TUMOR"
    return "MIXED"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--k", type=int, default=6)
    args = parser.parse_args()

    adata = sc.read_h5ad(args.adata)
    obs = adata.obs.copy()

    if "sample_id" not in obs.columns:
        raise ValueError("Missing sample_id column in adata.obs")

    if "barcode" not in obs.columns:
        obs["barcode"] = obs.index.astype(str)

    # coordinates
    if {"x", "y"}.issubset(obs.columns):
        xcol, ycol = "x", "y"
    elif {"x_fullres", "y_fullres"}.issubset(obs.columns):
        xcol, ycol = "x_fullres", "y_fullres"
    elif "spatial" in adata.obsm:
        obs["x"] = adata.obsm["spatial"][:, 0]
        obs["y"] = adata.obsm["spatial"][:, 1]
        xcol, ycol = "x", "y"
    else:
        raise ValueError("No spatial coordinates found")

    obs["kts_state"] = obs.apply(assign_state, axis=1)

    rows = []

    for sid, sub in obs.groupby("sample_id"):
        sub = sub.copy()
        coords = sub[[xcol, ycol]].to_numpy(float)

        if len(sub) <= args.k:
            continue

        A = kneighbors_graph(coords, n_neighbors=args.k, mode="connectivity", include_self=False)
        A = A.maximum(A.T).tocoo()

        local_index = sub.index.to_numpy()

        for i, j in zip(A.row, A.col):
            if i >= j:
                continue

            src = sub.iloc[i]
            tgt = sub.iloc[j]

            rows.append({
                "sample_id": sid,
                "source_barcode": src["barcode"],
                "target_barcode": tgt["barcode"],
                "source_state": src["kts_state"],
                "target_state": tgt["kts_state"],
                "source_x": src[xcol],
                "source_y": src[ycol],
                "target_x": tgt[xcol],
                "target_y": tgt[ycol],
            })

            rows.append({
                "sample_id": sid,
                "source_barcode": tgt["barcode"],
                "target_barcode": src["barcode"],
                "source_state": tgt["kts_state"],
                "target_state": src["kts_state"],
                "source_x": tgt[xcol],
                "source_y": tgt[ycol],
                "target_x": src[xcol],
                "target_y": src[ycol],
            })

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"[done] wrote {args.out}")
    print(out["source_state"].value_counts())
    print(out["target_state"].value_counts())


if __name__ == "__main__":
    main()
