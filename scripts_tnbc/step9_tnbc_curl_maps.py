from __future__ import annotations

import argparse
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import sparse


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


def triangle_centroids(node_df: pd.DataFrame, face_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    coords = node_df[["x_fullres", "y_fullres"]].to_numpy(dtype=float)

    tri = face_df[["i", "j", "k"]].to_numpy(dtype=int)
    tri_pts = coords[tri]  # (n_faces, 3, 2)

    centroids = tri_pts.mean(axis=1)
    x = centroids[:, 0]
    y = centroids[:, 1]
    return x, y


def plot_curl_maps(
    image: np.ndarray,
    hires_scale: float,
    x_face: np.ndarray,
    y_face: np.ndarray,
    curl_total_abs: np.ndarray,
    curl_coexact_abs: np.ndarray,
    sample_id: str,
    flux_name: str,
    outpath: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    panels = [
        (curl_total_abs, r"Total curl magnitude $|B_2^\top f|$"),
        (curl_coexact_abs, r"Coexact curl magnitude $|B_2^\top f_{\mathrm{coexact}}|$"),
    ]

    for ax, (vals, title) in zip(axes, panels):
        ax.imshow(image)

        vmax = np.quantile(vals, 0.99) if np.any(np.isfinite(vals)) else None

        sca = ax.scatter(
            x_face * hires_scale,
            y_face * hires_scale,
            c=vals,
            s=18,
            alpha=0.95,
            vmin=0,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.invert_yaxis()
        ax.axis("off")
        plt.colorbar(sca, ax=ax, fraction=0.04, pad=0.02)

    plt.suptitle(f"{sample_id}: Step 9 curl maps — {flux_name}", y=0.98)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 9 TNBC: make discrete curl density maps from Hodge outputs."
    )
    parser.add_argument("--sample_id", required=True)
    parser.add_argument("--sample_dir", required=True)
    parser.add_argument(
        "--flux_name",
        required=True,
        choices=["flux_tumor_immune", "flux_tumor_stroma", "flux_immune_stroma"],
    )
    parser.add_argument("--statsdir", default="stats")
    parser.add_argument("--figdir", default="visium_figures")
    args = parser.parse_args()

    sample_id = args.sample_id
    flux_name = args.flux_name
    statsdir = Path(args.statsdir)
    figdir = Path(args.figdir)
    sample_dir = Path(args.sample_dir).resolve()

    nodes_csv = require_file(statsdir / f"{sample_id}_step3_nodes.csv")
    faces_csv = require_file(statsdir / f"{sample_id}_step3_faces.csv")
    edges_hodge_csv = require_file(statsdir / f"{sample_id}_step6_edges_hodge_{flux_name}.csv")
    B2_path = require_file(statsdir / f"{sample_id}_step3_B2.npz")

    node_df = pd.read_csv(nodes_csv)
    face_df = pd.read_csv(faces_csv)
    edge_df = pd.read_csv(edges_hodge_csv)
    B2 = sparse.load_npz(B2_path).tocsr()

    image, hires_scale = load_sample_image_and_scale(sample_dir)
    x_face, y_face = triangle_centroids(node_df, face_df)

    f_total = edge_df["flux_total"].to_numpy(dtype=float)
    f_coexact = edge_df["flux_coexact"].to_numpy(dtype=float)

    curl_total_abs = np.abs(np.asarray(B2.T @ f_total).ravel())
    curl_coexact_abs = np.abs(np.asarray(B2.T @ f_coexact).ravel())

    out_csv = statsdir / f"{sample_id}_step9_face_curl_{flux_name}.csv"
    face_out = face_df.copy()
    face_out["x_centroid"] = x_face
    face_out["y_centroid"] = y_face
    face_out["curl_total_abs"] = curl_total_abs
    face_out["curl_coexact_abs"] = curl_coexact_abs
    face_out.to_csv(out_csv, index=False)

    out_png = figdir / f"{sample_id}_step9_curl_maps_{flux_name}.png"
    plot_curl_maps(
        image=image,
        hires_scale=hires_scale,
        x_face=x_face,
        y_face=y_face,
        curl_total_abs=curl_total_abs,
        curl_coexact_abs=curl_coexact_abs,
        sample_id=sample_id,
        flux_name=flux_name,
        outpath=out_png,
    )

    print("=" * 72)
    print(f"STEP 9: curl maps for {sample_id}")
    print("=" * 72)
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_png}")
    print(f"mean |B2^T f|              : {np.mean(curl_total_abs):.6e}")
    print(f"mean |B2^T f_coexact|      : {np.mean(curl_coexact_abs):.6e}")


if __name__ == "__main__":
    main()
