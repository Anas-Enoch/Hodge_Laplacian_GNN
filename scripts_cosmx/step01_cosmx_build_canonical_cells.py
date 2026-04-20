"""
Step 01 — CosMx Breast: Build Canonical Cell Table
====================================================
Reads CosMx RNA expression matrix + metadata, computes per-FOV z-scored
program scores (tumor, immune, stroma), and writes one row per cell.

FIXES vs original:
  - z-scoring is now PER-FOV (not global). Per-FOV z-scoring is internally
    consistent with the per-FOV quantile thresholds used in Step 02, and
    prevents aberrant FOVs from distorting the score distribution globally.
  - detect_expr_columns uses nrows=0 (column names only) — no risk of
    excluding genes whose first 3 rows happen to be NaN.
  - Deduplication asserts uniqueness on the composite ('fov', 'cell_id_numeric')
    key, not only on the cell string (which depends on CosMx naming convention).
  - Row-alignment fallback logs the specific cell_IDs that disagree.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TUMOR_GENES  = ["EPCAM", "KRT8", "KRT18", "KRT19", "ERBB2", "MUC1", "TACSTD2"]
IMMUNE_GENES = ["PTPRC", "CD3D", "CD3E", "NKG7", "CD68", "C1QA", "CXCL9", "CXCL10"]
STROMA_GENES = ["COL1A1", "COL1A2", "DCN", "LUM", "TAGLN", "ACTA2", "VIM"]

ALL_SCORE_GENES = sorted(set(TUMOR_GENES + IMMUNE_GENES + STROMA_GENES))


# ── Scoring ─────────────────────────────────────────────────────────────────

def zscore_series(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mu, sd = s.mean(), s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd


def score_from_genes_fov(df_fov: pd.DataFrame, genes: list[str]) -> pd.Series:
    """
    Compute program score within a single FOV.
    log1p → per-gene z-score within FOV → mean across genes.
    Per-FOV normalization is consistent with the per-FOV quantile
    thresholds used in Step 02 classification.
    """
    genes = [g for g in genes if g in df_fov.columns]
    if not genes:
        return pd.Series(np.zeros(len(df_fov)), index=df_fov.index)
    x = np.log1p(df_fov[genes].astype(float))
    x = x.apply(zscore_series, axis=0)
    return x.mean(axis=1)


def add_scores_per_fov(df: pd.DataFrame) -> pd.DataFrame:
    """Apply per-FOV z-scored program scoring to the full cell table."""
    tumor_scores  = pd.Series(np.nan, index=df.index)
    immune_scores = pd.Series(np.nan, index=df.index)
    stroma_scores = pd.Series(np.nan, index=df.index)

    for fov, idx in df.groupby("fov").groups.items():
        sub = df.loc[idx]
        tumor_scores.loc[idx]  = score_from_genes_fov(sub, TUMOR_GENES).values
        immune_scores.loc[idx] = score_from_genes_fov(sub, IMMUNE_GENES).values
        stroma_scores.loc[idx] = score_from_genes_fov(sub, STROMA_GENES).values

    df["tumor_score"]  = tumor_scores
    df["immune_score"] = immune_scores
    df["stroma_score"] = stroma_scores
    return df


# ── Column detection ─────────────────────────────────────────────────────────

def detect_expr_columns(expr_path: Path, id_candidates: list[str]) -> list[str]:
    """
    Read column names only (nrows=0) to avoid any risk of NaN-value
    misidentification in the first few rows.
    """
    header = pd.read_csv(expr_path, compression="gzip", nrows=0)
    cols = list(header.columns)

    id_cols = [c for c in cols if c in set(id_candidates)]
    if not id_cols:
        raise ValueError(
            f"No cell id column found in expr matrix. "
            f"Expected one of {id_candidates}. Got: {cols[:20]}"
        )

    keep = id_cols[:1]                             # first id col
    if "fov" in cols:
        keep.append("fov")
    keep += [g for g in ALL_SCORE_GENES if g in cols]

    found = [g for g in ALL_SCORE_GENES if g in cols]
    missing = [g for g in ALL_SCORE_GENES if g not in cols]
    print(f"[expr genes found]   {found}")
    if missing:
        print(f"[expr genes missing] {missing}")
    return sorted(set(keep), key=keep.index)


# ── Helpers ──────────────────────────────────────────────────────────────────

def require_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"No usable {label} column found. Available: {list(df.columns)}"
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expr", type=Path,
        default=Path("data/Breast_Multiomic/Flatfiles_RNA/flatFiles/BreastCancer/"
                     "BreastCancer_exprMat_file.csv.gz"))
    parser.add_argument("--meta", type=Path,
        default=Path("data/Breast_Multiomic/Flatfiles_RNA/flatFiles/BreastCancer/"
                     "BreastCancer_metadata_file.csv.gz"))
    parser.add_argument("--out", type=Path,
        default=Path("results_cosmx/cosmx_breast_canonical_cells.csv.gz"))
    args = parser.parse_args()

    # ── Metadata ────────────────────────────────────────────────────────────
    print("[1/5] reading metadata...")
    meta = pd.read_csv(args.meta, compression="gzip")

    meta_keep = [c for c in [
        "cell", "cell_id", "cell_ID", "fov",
        "CenterX_global_px", "CenterY_global_px",
        "CenterX_local_px",  "CenterY_local_px",
        "Area", "Area.um2",
        "nCount_RNA", "nFeature_RNA",
        "Mean.PanCK", "Mean.CD45", "Mean.CD68",
    ] if c in meta.columns]
    meta = meta[meta_keep].copy()

    meta_cell_col   = require_column(meta, ["cell", "cell_id"],   "cell string")
    meta_cellid_col = require_column(meta, ["cell_ID", "cell_id"], "numeric cell_ID")
    meta_fov_col    = require_column(meta, ["fov"],                "fov")
    meta_x_col      = require_column(meta,
        ["CenterX_global_px", "CenterX_local_px"], "x coordinate")
    meta_y_col      = require_column(meta,
        ["CenterY_global_px", "CenterY_local_px"], "y coordinate")

    meta = meta.rename(columns={
        meta_cell_col:   "cell",
        meta_cellid_col: "cell_ID",
        meta_fov_col:    "fov",
        meta_x_col:      "x",
        meta_y_col:      "y",
    })

    meta["cell_ID"] = pd.to_numeric(meta["cell_ID"], errors="coerce")
    meta["fov"]     = pd.to_numeric(meta["fov"],     errors="coerce")
    meta = meta.dropna(subset=["cell_ID", "fov"]).copy()
    meta["cell_ID"] = meta["cell_ID"].astype(int)
    meta["fov"]     = meta["fov"].astype(int)
    meta = meta.reset_index(drop=True)
    meta["_row_order"] = np.arange(len(meta))

    # ── Expression ──────────────────────────────────────────────────────────
    print("[2/5] detecting expression columns (header-only scan)...")
    id_candidates = ["cell_ID", "cell_id", "cell"]
    expr_usecols = detect_expr_columns(args.expr, id_candidates)
    print("[expr columns to load]", expr_usecols)

    print("[3/5] reading restricted expression matrix...")
    expr = pd.read_csv(args.expr, compression="gzip", usecols=expr_usecols).copy()

    expr_cellid_col = require_column(expr, id_candidates, "expression cell_ID")
    expr = expr.rename(columns={expr_cellid_col: "cell_ID"})
    expr["cell_ID"] = pd.to_numeric(expr["cell_ID"], errors="coerce")
    expr = expr.dropna(subset=["cell_ID"]).copy()
    expr["cell_ID"] = expr["cell_ID"].astype(int)

    # ── Merge ────────────────────────────────────────────────────────────────
    print("[4/5] aligning metadata and expression...")

    if "fov" in expr.columns:
        expr["fov"] = pd.to_numeric(expr["fov"], errors="coerce")
        expr = expr.dropna(subset=["fov"]).copy()
        expr["fov"] = expr["fov"].astype(int)
        before = len(meta)
        df = meta.merge(expr, on=["fov", "cell_ID"], how="inner", suffixes=("", "_expr"))
        print(f"[merge] keyed on ['fov','cell_ID']:  {before} meta → {len(df)} merged")
    else:
        if len(expr) != len(meta):
            raise ValueError(
                f"Expression lacks fov column and row counts differ: "
                f"len(expr)={len(expr)} vs len(meta)={len(meta)}."
            )
        meta_ids = meta["cell_ID"].to_numpy()
        expr_ids = expr["cell_ID"].to_numpy()
        same = np.mean(meta_ids == expr_ids)
        print(f"[row alignment] exact cell_ID agreement = {same:.6f}")

        if same < 0.999:
            # Log specific mismatches so user can audit
            bad_idx = np.where(meta_ids != expr_ids)[0]
            print(f"[row alignment] {len(bad_idx)} mismatched rows:")
            for i in bad_idx[:20]:
                print(f"   row {i}: meta cell_ID={meta_ids[i]}  expr cell_ID={expr_ids[i]}")
            raise ValueError(
                f"cell_ID agreement {same:.6f} < 0.999. Refusing unsafe join."
            )
        expr = expr.reset_index(drop=True)
        df = pd.concat([
            meta.reset_index(drop=True),
            expr.drop(columns=["cell_ID"]).reset_index(drop=True),
        ], axis=1)
        print("[merge] row-wise alignment (expr has no fov column)")

    # ── Deduplication with composite-key assertion ────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["cell"]).copy()
    after_str_dedup = len(df)
    if before != after_str_dedup:
        print(f"[dedup by cell string] dropped {before - after_str_dedup} rows")

    # Assert composite key uniqueness — catches cases where cell string is not unique
    dup_composite = df.duplicated(subset=["fov", "cell_ID"]).sum()
    if dup_composite > 0:
        print(
            f"[WARNING] {dup_composite} rows share (fov, cell_ID) after string dedup. "
            "This indicates the cell string may not uniquely encode FOV+cellID. "
            "Dropping composite duplicates (keeping first occurrence)."
        )
        df = df.drop_duplicates(subset=["fov", "cell_ID"]).copy()

    # ── Scores (per-FOV z-scoring) ───────────────────────────────────────────
    print("[5/5] computing per-FOV program scores and writing output...")
    df = add_scores_per_fov(df)

    # ── Build output table ───────────────────────────────────────────────────
    out = pd.DataFrame({
        "cell":                df["cell"],
        "cell_id_numeric":     df["cell_ID"],
        "fov":                 df["fov"],
        "x":                   df["x"],
        "y":                   df["y"],
        "cell_area_px":        df["Area"]       if "Area"       in df.columns else np.nan,
        "cell_area_um2":       df["Area.um2"]   if "Area.um2"   in df.columns else np.nan,
        "total_counts":        df["nCount_RNA"] if "nCount_RNA" in df.columns else np.nan,
        "n_features":          df["nFeature_RNA"] if "nFeature_RNA" in df.columns else np.nan,
        "protein_tumor_proxy":   df["Mean.PanCK"] if "Mean.PanCK" in df.columns else np.nan,
        "protein_immune_proxy":  df["Mean.CD45"]  if "Mean.CD45"  in df.columns else np.nan,
        "protein_myeloid_proxy": df["Mean.CD68"]  if "Mean.CD68"  in df.columns else np.nan,
        "tumor_score":   df["tumor_score"],
        "immune_score":  df["immune_score"],
        "stroma_score":  df["stroma_score"],
        "log_total_counts": np.log1p(
            pd.to_numeric(df.get("nCount_RNA", pd.Series(np.nan, index=df.index)),
                          errors="coerce").fillna(0).astype(float)
        ),
    })

    # Per-FOV UMI z-score (for Step 25 density control)
    umi_global = pd.to_numeric(
        df.get("nCount_RNA", pd.Series(np.nan, index=df.index)),
        errors="coerce"
    ).fillna(0).astype(float)
    out["local_umi_zscore_global"] = zscore_series(umi_global)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, compression="gzip")

    print(f"[done] wrote canonical cells → {args.out}")
    print(f"[n_cells]           {len(out)}")
    print(f"[unique FOVs]       {out['fov'].nunique()}")
    print(f"[score range tumor] {out['tumor_score'].min():.3f} – {out['tumor_score'].max():.3f}")
    print(f"[columns]           {list(out.columns)}")


if __name__ == "__main__":
    main()
