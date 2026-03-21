from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.mean(x)
    sd = np.std(x)
    if sd < 1e-12:
        return np.zeros_like(x)
    return (x - mu) / sd


def minmax01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    xmin = np.min(x)
    xmax = np.max(x)
    if np.isclose(xmax, xmin):
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


def build_oriented_edge_index(edges_df: pd.DataFrame) -> np.ndarray:
    """
    Return edge_index with shape (2, n_edges), using the stored tail/head orientation.
    """
    tail = edges_df["tail"].to_numpy(dtype=np.int64)
    head = edges_df["head"].to_numpy(dtype=np.int64)
    edge_index = np.vstack([tail, head])
    return edge_index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 13 TNBC: prepare tensors/files for PDE-constrained GNN training."
    )
    parser.add_argument("--sample_id", required=True)
    parser.add_argument(
        "--target_flux",
        required=True,
        choices=["flux_tumor_immune", "flux_tumor_stroma", "flux_immune_stroma"],
        help="Which residualized proxy flux column to use as soft target.",
    )
    parser.add_argument("--statsdir", default="stats")
    parser.add_argument("--outdir", default="stats/gnn_data")
    args = parser.parse_args()

    sample_id = args.sample_id
    target_flux = args.target_flux
    statsdir = Path(args.statsdir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes_csv = require_file(statsdir / f"{sample_id}_step4_node_residualized_fields.csv")
    edges_csv = require_file(statsdir / f"{sample_id}_step4_edge_fluxes.csv")
    B1_path = require_file(statsdir / f"{sample_id}_step3_B1.npz")
    B2_path = require_file(statsdir / f"{sample_id}_step3_B2.npz")
    L1_path = require_file(statsdir / f"{sample_id}_step3_L1_edge_hodge.npz")

    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)

    B1 = sparse.load_npz(B1_path).tocsr()
    B2 = sparse.load_npz(B2_path).tocsr()
    L1 = sparse.load_npz(L1_path).tocsr()

    required_node_cols = [
        "node_id",
        "barcode",
        "x_fullres",
        "y_fullres",
        "tumor_score",
        "stroma_score",
        "immune_score",
        "tumor_residual",
        "stroma_residual",
        "immune_residual",
        "region_step2",
    ]
    missing_node = [c for c in required_node_cols if c not in nodes_df.columns]
    if missing_node:
        raise KeyError(f"Missing required node columns: {missing_node}")

    required_edge_cols = [
        "edge_id",
        "tail",
        "head",
        "length",
        target_flux,
        "tail_region",
        "head_region",
    ]
    missing_edge = [c for c in required_edge_cols if c not in edges_df.columns]
    if missing_edge:
        raise KeyError(f"Missing required edge columns: {missing_edge}")

    # -----------------------------
    # Node feature matrix
    # -----------------------------
    x_coord = zscore(nodes_df["x_fullres"].to_numpy(dtype=float))
    y_coord = zscore(nodes_df["y_fullres"].to_numpy(dtype=float))

    tumor_score = zscore(nodes_df["tumor_score"].to_numpy(dtype=float))
    stroma_score = zscore(nodes_df["stroma_score"].to_numpy(dtype=float))
    immune_score = zscore(nodes_df["immune_score"].to_numpy(dtype=float))

    tumor_res = zscore(nodes_df["tumor_residual"].to_numpy(dtype=float))
    stroma_res = zscore(nodes_df["stroma_residual"].to_numpy(dtype=float))
    immune_res = zscore(nodes_df["immune_residual"].to_numpy(dtype=float))

    X = np.column_stack(
        [
            tumor_score,
            stroma_score,
            immune_score,
            tumor_res,
            stroma_res,
            immune_res,
            x_coord,
            y_coord,
        ]
    ).astype(np.float32)

    feature_names = np.array(
        [
            "tumor_score_z",
            "stroma_score_z",
            "immune_score_z",
            "tumor_residual_z",
            "stroma_residual_z",
            "immune_residual_z",
            "x_coord_z",
            "y_coord_z",
        ],
        dtype=object,
    )

    # -----------------------------
    # Edge-level tensors
    # -----------------------------
    edge_index = build_oriented_edge_index(edges_df).astype(np.int64)
    edge_length = edges_df["length"].to_numpy(dtype=np.float32)
    edge_length_scaled = minmax01(edge_length).astype(np.float32)

    y_edge = edges_df[target_flux].to_numpy(dtype=np.float32)

    # Helpful edge features for a later edge decoder
    tail = edge_index[0]
    head = edge_index[1]

    dx = (nodes_df.loc[head, "x_fullres"].to_numpy(dtype=float) - nodes_df.loc[tail, "x_fullres"].to_numpy(dtype=float)).astype(np.float32)
    dy = (nodes_df.loc[head, "y_fullres"].to_numpy(dtype=float) - nodes_df.loc[tail, "y_fullres"].to_numpy(dtype=float)).astype(np.float32)
    dx_z = zscore(dx).astype(np.float32)
    dy_z = zscore(dy).astype(np.float32)

    edge_attr = np.column_stack(
        [
            edge_length,
            edge_length_scaled,
            dx_z,
            dy_z,
        ]
    ).astype(np.float32)

    edge_attr_names = np.array(
        [
            "length_raw",
            "length_minmax",
            "dx_z",
            "dy_z",
        ],
        dtype=object,
    )

    # -----------------------------
    # Region metadata
    # -----------------------------
    node_region = nodes_df["region_step2"].astype(str).to_numpy()
    edge_region = np.where(
        edges_df["tail_region"].astype(str).to_numpy() == edges_df["head_region"].astype(str).to_numpy(),
        edges_df["tail_region"].astype(str).to_numpy(),
        "mixed_edge",
    )

    # -----------------------------
    # Save dense arrays
    # -----------------------------
    prefix = f"{sample_id}_{target_flux}"

    np.save(outdir / f"{prefix}_X.npy", X)
    np.save(outdir / f"{prefix}_feature_names.npy", feature_names)
    np.save(outdir / f"{prefix}_edge_index.npy", edge_index)
    np.save(outdir / f"{prefix}_edge_attr.npy", edge_attr)
    np.save(outdir / f"{prefix}_edge_attr_names.npy", edge_attr_names)
    np.save(outdir / f"{prefix}_y_edge.npy", y_edge)
    np.save(outdir / f"{prefix}_node_region.npy", node_region)
    np.save(outdir / f"{prefix}_edge_region.npy", edge_region)

    # Save sparse operators in one place for training convenience
    sparse.save_npz(outdir / f"{prefix}_B1.npz", B1)
    sparse.save_npz(outdir / f"{prefix}_B2.npz", B2)
    sparse.save_npz(outdir / f"{prefix}_L1.npz", L1)

    # -----------------------------
    # Save metadata table
    # -----------------------------
    meta = pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "target_flux": target_flux,
                "n_nodes": X.shape[0],
                "n_edges": edge_index.shape[1],
                "n_features": X.shape[1],
                "B1_shape_0": B1.shape[0],
                "B1_shape_1": B1.shape[1],
                "B2_shape_0": B2.shape[0],
                "B2_shape_1": B2.shape[1],
                "L1_shape_0": L1.shape[0],
                "L1_shape_1": L1.shape[1],
                "target_mean": float(np.mean(y_edge)),
                "target_std": float(np.std(y_edge)),
                "target_min": float(np.min(y_edge)),
                "target_max": float(np.max(y_edge)),
            }
        ]
    )
    meta_csv = outdir / f"{prefix}_meta.csv"
    meta.to_csv(meta_csv, index=False)

    # Save compact inspection tables
    node_export = nodes_df[
        [
            "node_id",
            "barcode",
            "x_fullres",
            "y_fullres",
            "tumor_score",
            "stroma_score",
            "immune_score",
            "tumor_residual",
            "stroma_residual",
            "immune_residual",
            "region_step2",
        ]
    ].copy()
    node_export.to_csv(outdir / f"{prefix}_nodes_for_gnn.csv", index=False)

    edge_export = edges_df[
        [
            "edge_id",
            "tail",
            "head",
            "length",
            target_flux,
            "tail_region",
            "head_region",
        ]
    ].copy()
    edge_export.to_csv(outdir / f"{prefix}_edges_for_gnn.csv", index=False)

    print("=" * 72)
    print(f"STEP 13: prepare GNN data for {sample_id}")
    print("=" * 72)
    print(f"target_flux   : {target_flux}")
    print(f"n_nodes       : {X.shape[0]}")
    print(f"n_edges       : {edge_index.shape[1]}")
    print(f"n_features    : {X.shape[1]}")
    print(f"X shape       : {X.shape}")
    print(f"edge_index    : {edge_index.shape}")
    print(f"edge_attr     : {edge_attr.shape}")
    print(f"y_edge        : {y_edge.shape}")
    print(f"target mean   : {np.mean(y_edge):.6e}")
    print(f"target std    : {np.std(y_edge):.6e}")
    print("-" * 72)
    print(f"Saved: {outdir / f'{prefix}_X.npy'}")
    print(f"Saved: {outdir / f'{prefix}_edge_index.npy'}")
    print(f"Saved: {outdir / f'{prefix}_edge_attr.npy'}")
    print(f"Saved: {outdir / f'{prefix}_y_edge.npy'}")
    print(f"Saved: {outdir / f'{prefix}_B1.npz'}")
    print(f"Saved: {outdir / f'{prefix}_B2.npz'}")
    print(f"Saved: {outdir / f'{prefix}_L1.npz'}")
    print(f"Saved: {meta_csv}")


if __name__ == "__main__":
    main()
