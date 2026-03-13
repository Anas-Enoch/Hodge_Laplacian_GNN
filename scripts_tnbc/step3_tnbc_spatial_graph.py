from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import sparse
from scipy.spatial import Delaunay


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def load_sample_image_and_scale(sample_dir: Path) -> tuple[np.ndarray, float]:
    img_files = sorted(sample_dir.glob("*tissue_hires_image.png"))
    scale_files = sorted(sample_dir.glob("*scalefactors_json.json"))

    if len(img_files) != 1:
        raise RuntimeError(
            f"Expected exactly one hires image in {sample_dir}, found {[p.name for p in img_files]}"
        )
    if len(scale_files) != 1:
        raise RuntimeError(
            f"Expected exactly one scalefactors json in {sample_dir}, found {[p.name for p in scale_files]}"
        )

    image = np.array(Image.open(img_files[0]))
    with open(scale_files[0], "r", encoding="utf-8") as f:
        scalefactors = json.load(f)

    hires_scale = float(scalefactors.get("tissue_hires_scalef", 1.0))
    return image, hires_scale


def build_edges_and_faces(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build Delaunay triangles and a unique undirected edge list.

    Returns
    -------
    edges : (m, 2) int array
        Unique edges with convention i < j.
    faces : (t, 3) int array
        Triangle vertex indices.
    """
    tri = Delaunay(coords)
    faces = tri.simplices.copy()

    edge_set: set[tuple[int, int]] = set()
    for a, b, c in faces:
        for i, j in [(a, b), (b, c), (a, c)]:
            u, v = (i, j) if i < j else (j, i)
            edge_set.add((u, v))

    edges = np.array(sorted(edge_set), dtype=int)
    return edges, faces


def build_B1(n_nodes: int, edges: np.ndarray) -> sparse.csr_matrix:
    """
    Node-edge incidence matrix B1 with orientation tail -> head, using i<j.
    For edge e=(i,j):
      B1[i,e] = -1
      B1[j,e] = +1
    """
    rows = []
    cols = []
    data = []

    for e_idx, (i, j) in enumerate(edges):
        rows.extend([i, j])
        cols.extend([e_idx, e_idx])
        data.extend([-1.0, +1.0])

    B1 = sparse.coo_matrix((data, (rows, cols)), shape=(n_nodes, len(edges))).tocsr()
    return B1


def build_B2(edges: np.ndarray, faces: np.ndarray) -> sparse.csr_matrix:
    """
    Edge-face incidence matrix B2.

    Triangle orientation convention:
      for sorted triangle (i, j, k) with i < j < k,
      boundary = (j,k) - (i,k) + (i,j)

    Since edge list is always stored with smaller -> larger index:
      +1 on e_ij
      -1 on e_ik
      +1 on e_jk
    """
    edge_lookup = {tuple(edge): idx for idx, edge in enumerate(edges)}

    rows = []
    cols = []
    data = []

    for f_idx, tri_nodes in enumerate(faces):
        i, j, k = sorted(map(int, tri_nodes))
        e_ij = edge_lookup[(i, j)]
        e_ik = edge_lookup[(i, k)]
        e_jk = edge_lookup[(j, k)]

        rows.extend([e_ij, e_ik, e_jk])
        cols.extend([f_idx, f_idx, f_idx])
        data.extend([+1.0, -1.0, +1.0])

    B2 = sparse.coo_matrix((data, (rows, cols)), shape=(len(edges), len(faces))).tocsr()
    return B2


def save_graph_plot(
    df: pd.DataFrame,
    edges: np.ndarray,
    image: np.ndarray,
    hires_scale: float,
    outpath: Path,
    sample_id: str,
    max_edges_to_draw: int = 8000,
) -> None:
    """
    Draw spot graph on top of tissue image.
    """
    x = df["x_fullres"].to_numpy(dtype=float) * hires_scale
    y = df["y_fullres"].to_numpy(dtype=float) * hires_scale

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(image)

    # Draw a subset of edges if needed to avoid clutter
    draw_edges = edges
    if len(edges) > max_edges_to_draw:
        rng = np.random.default_rng(0)
        keep_idx = rng.choice(len(edges), size=max_edges_to_draw, replace=False)
        draw_edges = edges[np.sort(keep_idx)]

    for i, j in draw_edges:
        ax.plot([x[i], x[j]], [y[i], y[j]], linewidth=0.4, alpha=0.15)

    # Node colors by region
    color_map = {
        "tumor_enriched": "red",
        "stroma_enriched": "mediumpurple",
        "immune_enriched": "limegreen",
        "interface_like": "deepskyblue",
        "other": "lightgray",
    }
    region_order = [
        "tumor_enriched",
        "stroma_enriched",
        "immune_enriched",
        "interface_like",
        "other",
    ]

    for reg in region_order:
        sub = df.loc[df["region_step2"] == reg]
        if sub.empty:
            continue
        xs = sub["x_fullres"].to_numpy(dtype=float) * hires_scale
        ys = sub["y_fullres"].to_numpy(dtype=float) * hires_scale
        ax.scatter(xs, ys, s=16, alpha=0.90, c=color_map[reg], label=f"{reg} (n={len(sub)})")

    ax.set_title(f"{sample_id}: Step 3 spatial graph")
    ax.invert_yaxis()
    ax.axis("off")
    ax.legend(loc="upper right", fontsize=8, frameon=True)

    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3 for TNBC cohort: build Delaunay spatial graph, incidence matrices, and graph outputs."
    )
    parser.add_argument(
        "--sample_id",
        type=str,
        required=True,
        help="Sample ID, e.g. GSM_6433618",
    )
    parser.add_argument(
        "--sample_dir",
        type=str,
        required=True,
        help="Path to sample folder, e.g. data/TNBC_GSE210616/GSM_6433618",
    )
    parser.add_argument(
        "--statsdir",
        type=str,
        default="stats",
        help="Directory for CSV / NPZ outputs.",
    )
    parser.add_argument(
        "--figdir",
        type=str,
        default="visium_figures",
        help="Directory for figure outputs.",
    )
    args = parser.parse_args()

    sample_id = args.sample_id
    sample_dir = Path(args.sample_dir).resolve()
    statsdir = Path(args.statsdir)
    figdir = Path(args.figdir)
    statsdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    region_csv = require_file(statsdir / f"{sample_id}_step2_region_assignments.csv")
    image, hires_scale = load_sample_image_and_scale(sample_dir)

    print("=" * 70)
    print(f"STEP 3: TNBC spatial graph for {sample_id}")
    print("=" * 70)
    print(f"Reading regions from: {region_csv}")

    df = pd.read_csv(region_csv)

    required_cols = [
        "barcode",
        "x_fullres",
        "y_fullres",
        "tumor_score",
        "stroma_score",
        "immune_score",
        "region_step2",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {region_csv}: {missing}")

    # Node table
    node_df = df.copy().reset_index(drop=True)
    node_df["node_id"] = np.arange(len(node_df), dtype=int)

    coords = node_df[["x_fullres", "y_fullres"]].to_numpy(dtype=float)

    edges, faces = build_edges_and_faces(coords)
    B1 = build_B1(len(node_df), edges)
    B2 = build_B2(edges, faces)
    L1 = (B1.T @ B1) + (B2 @ B2.T)

    # Edge table
    tail = edges[:, 0]
    head = edges[:, 1]
    x_i = coords[tail, 0]
    y_i = coords[tail, 1]
    x_j = coords[head, 0]
    y_j = coords[head, 1]
    lengths = np.sqrt((x_j - x_i) ** 2 + (y_j - y_i) ** 2)

    edge_df = pd.DataFrame({
        "edge_id": np.arange(len(edges), dtype=int),
        "tail": tail,
        "head": head,
        "tail_barcode": node_df.loc[tail, "barcode"].to_numpy(),
        "head_barcode": node_df.loc[head, "barcode"].to_numpy(),
        "x_tail": x_i,
        "y_tail": y_i,
        "x_head": x_j,
        "y_head": y_j,
        "length": lengths,
        "tail_region": node_df.loc[tail, "region_step2"].to_numpy(),
        "head_region": node_df.loc[head, "region_step2"].to_numpy(),
    })

    # Face table
    face_df = pd.DataFrame({
        "face_id": np.arange(len(faces), dtype=int),
        "i": faces[:, 0],
        "j": faces[:, 1],
        "k": faces[:, 2],
    })

    # Save outputs
    nodes_csv = statsdir / f"{sample_id}_step3_nodes.csv"
    edges_csv = statsdir / f"{sample_id}_step3_edges.csv"
    faces_csv = statsdir / f"{sample_id}_step3_faces.csv"

    node_df[
        [
            "node_id",
            "barcode",
            "x_fullres",
            "y_fullres",
            "tumor_score",
            "stroma_score",
            "immune_score",
            "interface_score",
            "region_step2",
        ]
    ].to_csv(nodes_csv, index=False)

    edge_df.to_csv(edges_csv, index=False)
    face_df.to_csv(faces_csv, index=False)

    B1_path = statsdir / f"{sample_id}_step3_B1.npz"
    B2_path = statsdir / f"{sample_id}_step3_B2.npz"
    L1_path = statsdir / f"{sample_id}_step3_L1_edge_hodge.npz"

    sparse.save_npz(B1_path, B1)
    sparse.save_npz(B2_path, B2)
    sparse.save_npz(L1_path, L1)

    print("\nGraph summary")
    print("-" * 70)
    print(f"n_nodes       : {len(node_df)}")
    print(f"n_edges       : {len(edge_df)}")
    print(f"n_faces       : {len(face_df)}")
    print(f"B1 shape      : {B1.shape}")
    print(f"B2 shape      : {B2.shape}")
    print(f"L1 shape      : {L1.shape}")
    print(f"edge length mean/std : {edge_df['length'].mean():.3f} / {edge_df['length'].std():.3f}")

    print(f"\nSaved: {nodes_csv}")
    print(f"Saved: {edges_csv}")
    print(f"Saved: {faces_csv}")
    print(f"Saved: {B1_path}")
    print(f"Saved: {B2_path}")
    print(f"Saved: {L1_path}")

    graph_png = figdir / f"{sample_id}_step3_spatial_graph.png"
    save_graph_plot(node_df, edges, image, hires_scale, graph_png, sample_id)
    print(f"Saved: {graph_png}")

    print("\nStep 3 completed successfully.")


if __name__ == "__main__":
    main()
