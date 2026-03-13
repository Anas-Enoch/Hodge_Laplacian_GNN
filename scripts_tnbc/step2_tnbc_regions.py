from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def load_sample_image(sample_dir: Path) -> tuple[np.ndarray, float]:
    img_files = sorted(sample_dir.glob("*tissue_hires_image.png"))
    scale_files = sorted(sample_dir.glob("*scalefactors_json.json"))

    if len(img_files) != 1:
        raise RuntimeError(
            f"Expected exactly one hires image in {sample_dir}, found: {[p.name for p in img_files]}"
        )
    if len(scale_files) != 1:
        raise RuntimeError(
            f"Expected exactly one scalefactors json in {sample_dir}, found: {[p.name for p in scale_files]}"
        )

    image = np.array(Image.open(img_files[0]))

    import json
    with open(scale_files[0], "r", encoding="utf-8") as f:
        scalefactors = json.load(f)

    hires_scale = float(scalefactors.get("tissue_hires_scalef", 1.0))
    return image, hires_scale


def normalize_01(x: pd.Series) -> pd.Series:
    xmin = float(x.min())
    xmax = float(x.max())
    if np.isclose(xmax, xmin):
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - xmin) / (xmax - xmin)


def assign_regions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Standardized helper scores
    out["tumor_q"] = normalize_01(out["tumor_score"])
    out["stroma_q"] = normalize_01(out["stroma_score"])
    out["immune_q"] = normalize_01(out["immune_score"])

    # Sample-adaptive thresholds
    tumor_hi = out["tumor_score"].quantile(0.80)
    stroma_hi = out["stroma_score"].quantile(0.70)
    immune_hi = out["immune_score"].quantile(0.75)

    tumor_mid = out["tumor_score"].quantile(0.60)
    stroma_mid = out["stroma_score"].quantile(0.55)
    immune_mid = out["immune_score"].quantile(0.55)

    # Composite interface score:
    # high if two programs are simultaneously elevated, especially tumor with stroma/immune
    out["interface_score"] = (
        0.45 * np.minimum(out["tumor_q"], out["stroma_q"])
        + 0.45 * np.minimum(out["tumor_q"], out["immune_q"])
        + 0.10 * np.minimum(out["stroma_q"], out["immune_q"])
    )
    interface_hi = out["interface_score"].quantile(0.75)

    # Default assignment
    region = np.array(["other"] * len(out), dtype=object)

    # Order matters.
    # 1) Tumor-enriched: tumor high and stronger than both others
    tumor_mask = (
        (out["tumor_score"] >= tumor_hi)
        & (out["tumor_score"] >= out["stroma_score"])
        & (out["tumor_score"] >= out["immune_score"])
    )
    region[tumor_mask.to_numpy()] = "tumor_enriched"

    # 2) Immune-enriched: immune high, not strongly tumor-dominant
    immune_mask = (
        (out["immune_score"] >= immune_hi)
        & (out["tumor_score"] < tumor_hi)
        & (out["immune_score"] >= out["stroma_score"] * 0.75)
    )
    region[immune_mask.to_numpy()] = "immune_enriched"

    # 3) Stroma-enriched: stroma high and not tumor-dominant
    stroma_mask = (
        (out["stroma_score"] >= stroma_hi)
        & (out["tumor_score"] < tumor_hi)
        & (out["stroma_score"] >= out["immune_score"])
    )
    region[stroma_mask.to_numpy()] = "stroma_enriched"

    # 4) Interface-like: moderate/high mixed programs
    interface_mask = (
        (out["interface_score"] >= interface_hi)
        & (
            ((out["tumor_score"] >= tumor_mid) & (out["stroma_score"] >= stroma_mid))
            | ((out["tumor_score"] >= tumor_mid) & (out["immune_score"] >= immune_mid))
            | ((out["stroma_score"] >= stroma_mid) & (out["immune_score"] >= immune_mid))
        )
    )
    region[(region == "other") & interface_mask.to_numpy()] = "interface_like"

    out["region_step2"] = region
    return out


def save_boxplots(df: pd.DataFrame, outpath: Path) -> None:
    region_order = [
        "tumor_enriched",
        "stroma_enriched",
        "immune_enriched",
        "interface_like",
        "other",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=False)

    specs = [
        ("tumor_score", "Tumor score by region"),
        ("stroma_score", "Stroma score by region"),
        ("immune_score", "Immune score by region"),
    ]

    for ax, (col, title) in zip(axes, specs):
        data = [
            df.loc[df["region_step2"] == reg, col].to_numpy(dtype=float)
            for reg in region_order
        ]
        ax.boxplot(data, tick_labels=region_order, showfliers=False)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close(fig)


def save_region_map(
    df: pd.DataFrame,
    image: np.ndarray,
    hires_scale: float,
    outpath: Path,
    sample_id: str,
) -> None:
    region_order = [
        "tumor_enriched",
        "stroma_enriched",
        "immune_enriched",
        "interface_like",
        "other",
    ]
    color_map = {
        "tumor_enriched": "red",
        "stroma_enriched": "mediumpurple",
        "immune_enriched": "limegreen",
        "interface_like": "deepskyblue",
        "other": "lightgray",
    }

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(image)

    x = df["x_fullres"].to_numpy(dtype=float) * hires_scale
    y = df["y_fullres"].to_numpy(dtype=float) * hires_scale

    for reg in region_order:
        sub = df.loc[df["region_step2"] == reg]
        if sub.empty:
            continue
        xs = sub["x_fullres"].to_numpy(dtype=float) * hires_scale
        ys = sub["y_fullres"].to_numpy(dtype=float) * hires_scale
        ax.scatter(
            xs,
            ys,
            s=18,
            alpha=0.90,
            c=color_map[reg],
            label=f"{reg} (n={len(sub)})",
        )

    ax.set_title(f"{sample_id}: Step 2 region definitions")
    ax.invert_yaxis()
    ax.axis("off")
    ax.legend(loc="upper right", fontsize=8, frameon=True)

    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 2 for TNBC cohort: define region labels from Step 1 marker scores."
    )
    parser.add_argument(
        "--sample_id",
        type=str,
        required=True,
        help="Sample ID, e.g. GSM_6433626",
    )
    parser.add_argument(
        "--sample_dir",
        type=str,
        required=True,
        help="Path to one TNBC sample folder, e.g. data/TNBC_GSE210616/GSM_6433626",
    )
    parser.add_argument(
        "--figdir",
        type=str,
        default="visium_figures",
        help="Directory containing Step 1 outputs and where Step 2 figures will be written.",
    )
    parser.add_argument(
        "--statsdir",
        type=str,
        default="stats",
        help="Directory where Step 2 CSV outputs will be written.",
    )
    args = parser.parse_args()

    sample_id = args.sample_id
    sample_dir = Path(args.sample_dir).resolve()
    figdir = Path(args.figdir)
    statsdir = Path(args.statsdir)
    figdir.mkdir(parents=True, exist_ok=True)
    statsdir.mkdir(parents=True, exist_ok=True)

    score_csv = require_file(figdir / f"{sample_id}_step1_marker_scores.csv")
    image, hires_scale = load_sample_image(sample_dir)

    print("=" * 70)
    print(f"STEP 2: TNBC region definition for {sample_id}")
    print("=" * 70)
    print(f"Reading marker scores from: {score_csv}")

    df = pd.read_csv(score_csv)

    required_cols = [
        "barcode",
        "x_fullres",
        "y_fullres",
        "tumor_score",
        "stroma_score",
        "immune_score",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {score_csv}: {missing}")

    region_df = assign_regions(df)

    print("\nRegion counts")
    print("-" * 70)
    print(region_df["region_step2"].value_counts(dropna=False))

    out_csv = statsdir / f"{sample_id}_step2_region_assignments.csv"
    region_df[
        [
            "barcode",
            "x_fullres",
            "y_fullres",
            "tumor_score",
            "stroma_score",
            "immune_score",
            "interface_score",
            "region_step2",
        ]
    ].to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    map_path = figdir / f"{sample_id}_step2_regions_map.png"
    save_region_map(region_df, image, hires_scale, map_path, sample_id)
    print(f"Saved: {map_path}")

    boxplot_path = figdir / f"{sample_id}_step2_region_boxplots.png"
    save_boxplots(region_df, boxplot_path)
    print(f"Saved: {boxplot_path}")

    print("\nStep 2 completed successfully.")


if __name__ == "__main__":
    main()
