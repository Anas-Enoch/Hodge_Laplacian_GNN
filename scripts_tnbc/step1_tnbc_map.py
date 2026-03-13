from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from .load_tnbc_visium_sample import load_visium_geo_sample

def keep_present(genes: list[str], var_names: pd.Index) -> list[str]:
    return [g for g in genes if g in var_names]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 1 for TNBC cohort: load one sample, score marker programs, and make spatial maps."
    )
    parser.add_argument(
        "--sample_dir",
        type=str,
        required=True,
        help="Path to one TNBC GEO sample folder, e.g. data/TNBC_GSE210616/GSM_6433626",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="visium_figures",
        help="Directory where figures and summary CSV will be written.",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    result = load_visium_geo_sample(args.sample_dir)
    sample_id = result["sample_id"]
    adata = result["adata"].copy()
    image = result["image_hires"]
    scalefactors = result["scalefactors"]

    hires_scale = float(scalefactors.get("tissue_hires_scalef", 1.0))

    print("\n" + "=" * 70)
    print(f"STEP 1: TNBC marker scoring for {sample_id}")
    print("=" * 70)

    # ------------------------------------------------------------
    # Marker programs
    # ------------------------------------------------------------
    tumor_genes = ["EPCAM", "KRT8", "KRT18", "KRT19", "ERBB2", "MUC1", "TACSTD2"]
    stroma_genes = ["COL1A1", "COL1A2", "DCN", "LUM", "POSTN", "FAP", "TAGLN"]
    immune_genes = ["PTPRC", "CD3D", "CD3E", "NKG7", "CD68", "C1QA", "CXCL9", "CXCL10"]

    adata.var_names_make_unique()

    tumor_genes_present = keep_present(tumor_genes, adata.var_names)
    stroma_genes_present = keep_present(stroma_genes, adata.var_names)
    immune_genes_present = keep_present(immune_genes, adata.var_names)

    print("\nGenes present in this sample")
    print("-" * 70)
    print("tumor  :", tumor_genes_present)
    print("stroma :", stroma_genes_present)
    print("immune :", immune_genes_present)

    # ------------------------------------------------------------
    # Normalize and score
    # ------------------------------------------------------------
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    if tumor_genes_present:
        sc.tl.score_genes(
            adata,
            gene_list=tumor_genes_present,
            score_name="tumor_score",
            use_raw=False,
        )
    else:
        adata.obs["tumor_score"] = 0.0

    if stroma_genes_present:
        sc.tl.score_genes(
            adata,
            gene_list=stroma_genes_present,
            score_name="stroma_score",
            use_raw=False,
        )
    else:
        adata.obs["stroma_score"] = 0.0

    if immune_genes_present:
        sc.tl.score_genes(
            adata,
            gene_list=immune_genes_present,
            score_name="immune_score",
            use_raw=False,
        )
    else:
        adata.obs["immune_score"] = 0.0

    # Total counts after log normalization are less interpretable,
    # so keep a simple raw-like proxy from the normalized object.
    adata.obs["total_counts_proxy"] = np.asarray(adata.X.sum(axis=1)).ravel()

    # ------------------------------------------------------------
    # Restrict to in-tissue spots
    # ------------------------------------------------------------
    obs_plot = adata.obs.loc[adata.obs["in_tissue"] == 1].copy()

    print("\nSample summary")
    print("-" * 70)
    print(f"in-tissue spots : {obs_plot.shape[0]}")
    print(f"tumor_score  mean/std : {obs_plot['tumor_score'].mean():.4f} / {obs_plot['tumor_score'].std():.4f}")
    print(f"stroma_score mean/std : {obs_plot['stroma_score'].mean():.4f} / {obs_plot['stroma_score'].std():.4f}")
    print(f"immune_score mean/std : {obs_plot['immune_score'].mean():.4f} / {obs_plot['immune_score'].std():.4f}")

    # ------------------------------------------------------------
    # Save spot-level scores
    # ------------------------------------------------------------
    score_df = obs_plot[
        [
            "array_row",
            "array_col",
            "x_fullres",
            "y_fullres",
            "tumor_score",
            "stroma_score",
            "immune_score",
            "total_counts_proxy",
        ]
    ].copy()
    score_df["barcode"] = obs_plot.index

    score_csv = outdir / f"{sample_id}_step1_marker_scores.csv"
    score_df.to_csv(score_csv, index=False)
    print(f"\nSaved: {score_csv}")

    # ------------------------------------------------------------
    # Spatial plotting coordinates
    # ------------------------------------------------------------
    x = obs_plot["x_fullres"].to_numpy(dtype=float) * hires_scale
    y = obs_plot["y_fullres"].to_numpy(dtype=float) * hires_scale

    # ------------------------------------------------------------
    # Figure: 2x2 spatial maps
    # ------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    panels = [
        ("total_counts_proxy", "Total counts proxy"),
        ("tumor_score", "Tumor score"),
        ("stroma_score", "Stroma score"),
        ("immune_score", "Immune score"),
    ]

    for ax, (col, title) in zip(axes.ravel(), panels):
        ax.imshow(image)
        sca = ax.scatter(
            x,
            y,
            c=obs_plot[col].to_numpy(dtype=float),
            s=18,
            alpha=0.95,
        )
        ax.set_title(title)
        ax.invert_yaxis()
        ax.axis("off")
        plt.colorbar(sca, ax=ax, fraction=0.03, pad=0.02)

    plt.suptitle(f"{sample_id}: TNBC Step 1 marker maps", y=0.98)
    plt.tight_layout()

    fig_path = outdir / f"{sample_id}_step1_marker_maps.png"
    plt.savefig(fig_path, dpi=220)
    plt.close(fig)
    print(f"Saved: {fig_path}")

    # ------------------------------------------------------------
    # Simple boxplot figure for score distributions
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    data_to_plot = [
        obs_plot["tumor_score"].to_numpy(dtype=float),
        obs_plot["stroma_score"].to_numpy(dtype=float),
        obs_plot["immune_score"].to_numpy(dtype=float),
    ]
    ax.boxplot(data_to_plot, tick_labels=["tumor", "stroma", "immune"], showfliers=False)
    ax.set_title(f"{sample_id}: marker score distributions")
    ax.set_ylabel("score")
    plt.tight_layout()

    boxplot_path = outdir / f"{sample_id}_step1_marker_boxplots.png"
    plt.savefig(boxplot_path, dpi=220)
    plt.close(fig)
    print(f"Saved: {boxplot_path}")

    print("\nStep 1 completed successfully.")


if __name__ == "__main__":
    main()
