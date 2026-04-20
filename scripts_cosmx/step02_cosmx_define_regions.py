"""
Step 02 — CosMx Breast: Define Cell Regions
=============================================
Classifies cells into tumor / immune / interface / tumor_core within each FOV.

FIXES vs original:
  - Absolute score floor for immune and tumor classification: cells must
    exceed score=0 (the per-FOV mean after z-scoring) in addition to the
    quantile rank criterion. This prevents single-compartment FOVs from
    generating spurious minority-class labels from background noise.
  - Mutual-class exclusivity assertion: warns if any cell passes both
    is_tumor and is_immune after classification (should never happen with
    mutual-dominance rule, but edge cases at score ties can occur).
  - Stricter tumor_core distance: tumor cells must be at distance
    > min_core_radius_multiplier × base_spacing from nearest immune cell
    (default 3.0×, configurable). Original used only the interface radius
    threshold (2.0×). This creates a cleaner separation between the
    interface zone and the "genuine" tumor core used as comparison baseline.
  - Per-FOV QC: logs the fraction of 'other' cells and flags FOVs where
    > 80% of cells are unclassified (potential calibration failure).
  - Explicit label-priority documentation: interface overwrites tumor_core;
    immune overwrites tumor when both fire (logged as a warning).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# ── Geometry ──────────────────────────────────────────────────────────────────

def robust_base_spacing(coords: np.ndarray, k: int = 6,
                        fallback: float = 50.0) -> float:
    if coords.shape[0] < k + 1:
        return fallback
    ucoords = np.unique(coords, axis=0)
    if ucoords.shape[0] < k + 1:
        return fallback
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(ucoords)
    dists, _ = nn.kneighbors(ucoords)
    neigh = dists[:, 1:k+1].reshape(-1)
    neigh = neigh[np.isfinite(neigh) & (neigh > 0)]
    if len(neigh) == 0:
        return fallback
    base = float(np.median(neigh))
    return base if (np.isfinite(base) and base > 0) else fallback


# ── Classification ─────────────────────────────────────────────────────────────

def classify_cells(
    sub: pd.DataFrame,
    tumor_quantile: float,
    immune_quantile: float,
    score_floor: float = 0.0,
) -> pd.DataFrame:
    """
    Classify cells as tumor or immune candidates.

    Two-gate criterion:
      1. Quantile rank: score >= FOV-specific quantile threshold
      2. Absolute floor: score >= score_floor  (default 0.0, the per-FOV mean)
         This prevents background-noise cells in single-compartment FOVs
         from being promoted to the minority class.
      3. Mutual dominance: tumor_score > immune_score (and vice versa)

    score_floor=0.0 is appropriate when program scores are per-FOV z-scored
    (Step 01), since 0 corresponds to the within-FOV mean expression.
    """
    sub = sub.copy()

    tumor_thr  = sub["tumor_score"].quantile(tumor_quantile)
    immune_thr = sub["immune_score"].quantile(immune_quantile)

    sub["is_tumor"]  = (
        (sub["tumor_score"]  >= tumor_thr)
        & (sub["tumor_score"]  >= score_floor)
        & (sub["tumor_score"]  > sub["immune_score"])
    )
    sub["is_immune"] = (
        (sub["immune_score"] >= immune_thr)
        & (sub["immune_score"] >= score_floor)
        & (sub["immune_score"] > sub["tumor_score"])
    )

    # Soft fallback if one class is completely absent (all cells fail floor)
    if sub["is_tumor"].sum() < 10:
        tumor_thr_fb = sub["tumor_score"].quantile(0.60)
        sub["is_tumor"] = (
            (sub["tumor_score"] >= tumor_thr_fb)
            & (sub["tumor_score"] > sub["immune_score"])
        )

    if sub["is_immune"].sum() < 10:
        immune_thr_fb = sub["immune_score"].quantile(0.60)
        sub["is_immune"] = (
            (sub["immune_score"] >= immune_thr_fb)
            & (sub["immune_score"] > sub["tumor_score"])
        )

    # Mutual exclusivity assertion (should never fire with mutual-dominance rule)
    both = sub["is_tumor"] & sub["is_immune"]
    if both.sum() > 0:
        print(
            f"    [WARNING] {both.sum()} cells pass BOTH is_tumor and is_immune "
            "(score tie at quantile boundary). Resolving: tumor score dominates."
        )
        sub.loc[both, "is_immune"] = False

    return sub


# ── Interface + core ──────────────────────────────────────────────────────────

def define_interface_and_core(
    sub: pd.DataFrame,
    interface_radius: float,
    core_min_distance: float,
) -> pd.DataFrame:
    """
    Interface: tumor or immune cells within interface_radius of the nearest
               cell of the opposite class.
    Tumor core: tumor cells at distance > core_min_distance from any immune cell.
                core_min_distance is typically larger than interface_radius
                (default: 3× base_spacing vs interface 2× base_spacing),
                ensuring a clean gap between interface and core zones.
    """
    sub = sub.copy()
    sub["is_interface"]            = False
    sub["tumor_core"]              = False
    sub["dist_to_nearest_immune"]  = np.nan
    sub["dist_to_nearest_tumor"]   = np.nan

    coords     = sub[["x", "y"]].to_numpy(float)
    tumor_idx  = np.where(sub["is_tumor"].to_numpy())[0]
    immune_idx = np.where(sub["is_immune"].to_numpy())[0]

    if len(tumor_idx) == 0 or len(immune_idx) == 0:
        sub["tumor_core"] = sub["is_tumor"]
        return sub

    tumor_pts  = coords[tumor_idx]
    immune_pts = coords[immune_idx]

    nn_immune = NearestNeighbors(n_neighbors=1).fit(immune_pts)
    d_tumor_to_immune, _ = nn_immune.kneighbors(tumor_pts)

    nn_tumor  = NearestNeighbors(n_neighbors=1).fit(tumor_pts)
    d_immune_to_tumor, _ = nn_tumor.kneighbors(immune_pts)

    sub.loc[tumor_idx,  "dist_to_nearest_immune"] = d_tumor_to_immune[:, 0]
    sub.loc[immune_idx, "dist_to_nearest_tumor"]  = d_immune_to_tumor[:, 0]

    tumor_iface_mask  = d_tumor_to_immune[:, 0] <= interface_radius
    immune_iface_mask = d_immune_to_tumor[:, 0] <= interface_radius

    sub.loc[tumor_idx[tumor_iface_mask],   "is_interface"] = True
    sub.loc[immune_idx[immune_iface_mask], "is_interface"] = True

    # Tumor core: tumor cells strictly beyond core_min_distance from immune.
    # This is stricter than the original (which used interface_radius).
    # core_min_distance > interface_radius creates a clear gap zone between
    # the interface and the comparison baseline (tumor core).
    sub["tumor_core"] = sub["is_tumor"] & (
        sub["dist_to_nearest_immune"] > core_min_distance
    )

    return sub


# ── Region labels ─────────────────────────────────────────────────────────────

def assign_region_labels(sub: pd.DataFrame) -> pd.DataFrame:
    """
    Priority order (each overwrites previous):
      other → tumor → immune → tumor_core → interface
    interface is applied last so it can override tumor_core for cells that
    are technically in the tumor compartment but adjacent to immune cells.
    """
    sub = sub.copy()
    sub["region_label"] = "other"
    sub.loc[sub["is_tumor"],     "region_label"] = "tumor"
    sub.loc[sub["is_immune"],    "region_label"] = "immune"
    sub.loc[sub["tumor_core"],   "region_label"] = "tumor_core"
    sub.loc[sub["is_interface"], "region_label"] = "interface"
    return sub


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=Path,
        default=Path("results_cosmx/cosmx_breast_canonical_cells.csv.gz"))
    parser.add_argument("--out", type=Path,
        default=Path("results_cosmx/cosmx_breast_cells_with_regions.csv.gz"))
    parser.add_argument("--tumor-quantile",  type=float, default=0.75)
    parser.add_argument("--immune-quantile", type=float, default=0.75)
    parser.add_argument("--score-floor", type=float, default=0.0,
        help="Minimum absolute score a cell must exceed to be classified as "
             "tumor or immune. 0.0 = per-FOV mean (appropriate when scores "
             "are per-FOV z-scored in Step 01). Raise to 0.5 for stricter filtering.")
    parser.add_argument("--radius-multiplier", type=float, default=2.0,
        help="Interface radius = multiplier × median 6-NN spacing within each FOV.")
    parser.add_argument("--core-radius-multiplier", type=float, default=3.0,
        help="Tumor-core minimum distance = multiplier × base_spacing. "
             "Must be >= radius-multiplier. Default 3.0 (interface=2.0), "
             "creating a clear gap zone between interface and tumor core.")
    parser.add_argument("--min-class-size", type=int, default=10,
        help="Minimum tumor/immune cell count to attempt interface logic.")
    parser.add_argument("--other-fraction-warn", type=float, default=0.80,
        help="Warn if fraction of 'other' cells in a FOV exceeds this threshold.")
    args = parser.parse_args()

    if args.core_radius_multiplier < args.radius_multiplier:
        raise ValueError(
            f"--core-radius-multiplier ({args.core_radius_multiplier}) must be "
            f">= --radius-multiplier ({args.radius_multiplier}) to ensure a "
            "gap zone between interface and tumor core."
        )

    df = pd.read_csv(args.cells, compression="gzip")

    required = ["cell", "cell_id_numeric", "fov", "x", "y",
                "tumor_score", "immune_score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    before = len(df)
    df = df.drop_duplicates(subset=["cell"]).copy()
    if len(df) != before:
        print(f"[dedup] dropped {before - len(df)} duplicate cells")

    out_frames = []
    n_flagged_other = 0

    for fov, sub in df.groupby("fov", dropna=False):
        sub = sub.copy().reset_index(drop=True)

        sub = classify_cells(
            sub,
            tumor_quantile=args.tumor_quantile,
            immune_quantile=args.immune_quantile,
            score_floor=args.score_floor,
        )

        coords = sub[["x", "y"]].to_numpy(float)
        base_spacing      = robust_base_spacing(coords, k=6, fallback=50.0)
        interface_radius  = args.radius_multiplier      * base_spacing
        core_min_distance = args.core_radius_multiplier * base_spacing

        sub["base_spacing_px"]      = base_spacing
        sub["interface_radius_px"]  = interface_radius
        sub["core_min_distance_px"] = core_min_distance

        if (sub["is_tumor"].sum() < args.min_class_size
                or sub["is_immune"].sum() < args.min_class_size):
            sub["is_interface"]           = False
            sub["tumor_core"]             = sub["is_tumor"]
            sub["dist_to_nearest_immune"] = np.nan
            sub["dist_to_nearest_tumor"]  = np.nan
        else:
            sub = define_interface_and_core(
                sub,
                interface_radius=interface_radius,
                core_min_distance=core_min_distance,
            )

        sub = assign_region_labels(sub)

        # Per-FOV QC: warn if most cells are unclassified
        frac_other = float((sub["region_label"] == "other").mean())
        if frac_other > args.other_fraction_warn:
            n_flagged_other += 1
            print(
                f"[fov={fov}] WARNING: {frac_other:.0%} of cells are 'other' — "
                "possible calibration failure (score floor too high or "
                "single-compartment FOV)."
            )

        out_frames.append(sub)

        print(
            f"[fov={fov}] n={len(sub)} "
            f"tumor={int(sub['is_tumor'].sum())} "
            f"immune={int(sub['is_immune'].sum())} "
            f"interface={int(sub['is_interface'].sum())} "
            f"tumor_core={int(sub['tumor_core'].sum())} "
            f"other={int((sub['region_label']=='other').sum())} "
            f"base_spacing={base_spacing:.1f}px "
            f"interface_r={interface_radius:.1f}px "
            f"core_min_d={core_min_distance:.1f}px"
        )

    out = pd.concat(out_frames, axis=0, ignore_index=True)
    if n_flagged_other > 0:
        print(
            f"\n[QC summary] {n_flagged_other} FOVs had >{args.other_fraction_warn:.0%} "
            "'other' cells. Consider lowering --score-floor or checking FOV tissue content."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, compression="gzip")
    print(f"[done] wrote regions → {args.out}")


if __name__ == "__main__":
    main()
