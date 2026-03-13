from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc
from PIL import Image


def find_one(sample_dir: Path, pattern: str) -> Path:
    matches = sorted(sample_dir.glob(pattern))
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No file matching pattern '{pattern}' found in: {sample_dir}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple files matching pattern '{pattern}' found in {sample_dir}: "
            f"{[m.name for m in matches]}"
        )
    return matches[0]


def load_visium_geo_sample(sample_dir: str | Path) -> dict[str, Any]:
    sample_dir = Path(sample_dir).resolve()
    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample directory does not exist: {sample_dir}")
    if not sample_dir.is_dir():
        raise NotADirectoryError(f"Sample path is not a directory: {sample_dir}")

    h5_file = find_one(sample_dir, "*filtered_feature_bc_matrix.h5")
    pos_file = find_one(sample_dir, "*tissue_positions_list.csv")
    img_file = find_one(sample_dir, "*tissue_hires_image.png")
    scale_file = find_one(sample_dir, "*scalefactors_json.json")

    print("=" * 70)
    print("Loading TNBC Visium sample")
    print(f"sample_dir : {sample_dir}")
    print(f"matrix     : {h5_file.name}")
    print(f"positions  : {pos_file.name}")
    print(f"image      : {img_file.name}")
    print(f"scalefactor: {scale_file.name}")
    print("=" * 70)

    adata = sc.read_10x_h5(h5_file)
    adata.var_names_make_unique()

    pos = pd.read_csv(
        pos_file,
        header=None,
        names=[
            "barcode",
            "in_tissue",
            "array_row",
            "array_col",
            "pxl_row_in_fullres",
            "pxl_col_in_fullres",
        ],
    )

    pos = pos[pos["barcode"].isin(adata.obs_names)].copy()
    if pos.empty:
        raise ValueError(
            "No overlapping barcodes between expression matrix and tissue positions."
        )

    pos = pos.set_index("barcode").loc[adata.obs_names].copy()

    with open(scale_file, "r", encoding="utf-8") as f:
        scalefactors = json.load(f)

    image = np.array(Image.open(img_file))

    adata.obs["in_tissue"] = pos["in_tissue"].astype(int).values
    adata.obs["array_row"] = pos["array_row"].astype(int).values
    adata.obs["array_col"] = pos["array_col"].astype(int).values
    adata.obs["y_fullres"] = pos["pxl_row_in_fullres"].astype(float).values
    adata.obs["x_fullres"] = pos["pxl_col_in_fullres"].astype(float).values

    coords_fullres = adata.obs[["x_fullres", "y_fullres"]].to_numpy(dtype=float)
    barcodes = adata.obs_names.to_numpy()

    print("\nLoaded sample successfully.")
    print(f"adata shape          : {adata.shape}")
    print(f"matched barcodes     : {len(barcodes)}")
    print(f"in-tissue spots      : {int((adata.obs['in_tissue'] == 1).sum())}")
    print(f"hires image shape    : {image.shape}")
    print(f"scalefactors keys    : {list(scalefactors.keys())}")

    return {
        "sample_dir": sample_dir,
        "sample_id": sample_dir.name,
        "adata": adata,
        "positions": pos.reset_index(),
        "coords_fullres": coords_fullres,
        "barcodes": barcodes,
        "image_hires": image,
        "scalefactors": scalefactors,
        "matrix_file": h5_file,
        "positions_file": pos_file,
        "image_file": img_file,
        "scalefactors_file": scale_file,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load one GEO Visium TNBC sample and validate required files."
    )
    parser.add_argument(
        "--sample_dir",
        type=str,
        required=True,
        help="Path to one sample directory, e.g. data/TNBC_GSE210616/GSM_6433626",
    )
    args = parser.parse_args()

    result = load_visium_geo_sample(args.sample_dir)

    adata = result["adata"]
    sample_id = result["sample_id"]

    print("\nSummary")
    print("-" * 70)
    print(f"sample_id            : {sample_id}")
    print(f"n_genes              : {adata.n_vars}")
    print(f"n_spots              : {adata.n_obs}")
    print(f"n_in_tissue          : {int((adata.obs['in_tissue'] == 1).sum())}")
    print(f"x range              : "
          f"{adata.obs['x_fullres'].min():.1f} -> {adata.obs['x_fullres'].max():.1f}")
    print(f"y range              : "
          f"{adata.obs['y_fullres'].min():.1f} -> {adata.obs['y_fullres'].max():.1f}")


if __name__ == "__main__":
    main()
