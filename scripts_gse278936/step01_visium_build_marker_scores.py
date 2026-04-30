#!/usr/bin/env python3
"""
Step 01 — GSE278936 Visium: Build Canonical Spot Table (ROBUST)

- Auto-detects GEO filenames (*barcodes.tsv.gz etc.)
- Handles .gz transparently
- Skips broken samples cleanly
- Logs exactly what is used

Output:
    results_gse278936/{sample}_spots.csv
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.io import mmread
from gzip import open as gzopen


# ── Gene sets ────────────────────────────────────────────────

TUMOR_GENES  = ["EPCAM", "KRT8", "KRT18", "KRT19", "ERBB2", "MUC1", "TACSTD2"]
IMMUNE_GENES = ["PTPRC", "CD3D", "CD3E", "NKG7", "CD68", "C1QA", "CXCL9", "CXCL10"]
# CIRCULARITY BOUNDARY — these gene sets must not appear in any downstream
# independent validation step (Steps 06, 09, 27).
# Aligned with main TNBC manuscript (GSE210616) gene panel exactly.
# Removed ACTA2, VIM (present in earlier draft); added POSTN, FAP.
STROMA_GENES = ["COL1A1", "COL1A2", "DCN", "LUM", "POSTN", "FAP", "TAGLN"]


# ── Utilities ───────────────────────────────────────────────

def score_program(df, genes):
    genes = [g for g in genes if g in df.columns]
    if not genes:
        return np.zeros(len(df))
    x = np.log1p(df[genes].values)
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)
    return x.mean(axis=1)


def load_matrix_safe(path):
    if str(path).endswith(".gz"):
        with gzopen(path, "rb") as f:
            return mmread(f).tocsr()
    else:
        return mmread(path).tocsr()


def find_file(sdir, pattern, label):
    files = list(sdir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"{label} not found in {sdir}")
    return files[0]


# ── Main ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", type=Path, required=True)
    parser.add_argument("--sample_id", type=str, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("results_gse278936"))
    args = parser.parse_args()

    sdir = args.sample_dir

    print(f"[load] {args.sample_id}")

    try:
        # ── Detect files ─────────────────────────────────────

        matrix_file   = find_file(sdir, "*matrix.mtx*", "matrix")
        features_file = find_file(sdir, "*features.tsv*", "features")
        barcodes_file = find_file(sdir, "*barcodes.tsv*", "barcodes")
        pos_file      = find_file(sdir, "*tissue_positions_list.csv*", "positions")

        print(f"[files] "
              f"{matrix_file.name}, "
              f"{features_file.name}, "
              f"{barcodes_file.name}, "
              f"{pos_file.name}")

        # ── Load matrix ─────────────────────────────────────

        mtx = load_matrix_safe(matrix_file)

        features = pd.read_csv(
            features_file,
            sep="\t",
            header=None,
            compression="infer"
        )
        features.columns = ["gene_id", "gene_name", "type"]

        barcodes = pd.read_csv(
            barcodes_file,
            header=None,
            compression="infer"
        )
        barcodes.columns = ["barcode"]

        # spots × genes
        X = pd.DataFrame(
            mtx.T.toarray(),
            columns=features["gene_name"],
            index=barcodes["barcode"]
        )

        # ── Load positions ─────────────────────────────────

        pos = pd.read_csv(
            pos_file,
            header=None,
            compression="infer"
        )

        pos.columns = [
            "barcode", "in_tissue",
            "array_row", "array_col",
            "pxl_row", "pxl_col"
        ]

        pos = pos.set_index("barcode")

        # ── Merge ─────────────────────────────────────────

        df = X.join(pos, how="inner")

        df = df[df["in_tissue"] == 1]

        print(f"[spots] {len(df)}")

        if len(df) == 0:
            raise ValueError("No tissue spots found")

        # ── Scores ────────────────────────────────────────

        df["tumor_score"]  = score_program(df, TUMOR_GENES)
        df["immune_score"] = score_program(df, IMMUNE_GENES)
        df["stroma_score"] = score_program(df, STROMA_GENES)

        # ── Output ───────────────────────────────────────

        out = pd.DataFrame({
            "spot": df.index,
            "x": df["pxl_col"],
            "y": df["pxl_row"],
            "tumor_score": df["tumor_score"],
            "immune_score": df["immune_score"],
            "stroma_score": df["stroma_score"],
        })

        args.out_dir.mkdir(exist_ok=True)
        out_file = args.out_dir / f"{args.sample_id}_spots.csv"
        out.to_csv(out_file, index=False)

        print(f"[done] → {out_file}")

    except Exception as e:
        print(f"[SKIP] {args.sample_id} → {e}")


if __name__ == "__main__":
    main()
