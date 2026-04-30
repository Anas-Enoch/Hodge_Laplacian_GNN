"""
Step 25 Utility — Extract UMI Counts from Visium h5 Files
===========================================================
Self-contained. Run this ONCE to add total_umi to your step6 node CSVs,
then rerun step25_density_nuisance_control.py.

Reads filtered_feature_bc_matrix.h5 for each sample, computes total UMI
per barcode, matches to node_id via barcode index from step3 node CSV, and
appends a total_umi column to the step6 node CSV.

Usage:
  python scripts_tnbc/step25_extract_umi.py \
    --sample-ids-file valid_sample_ids.txt \
    --data-dir data/TNBC_GSE210616 \
    --statsdir stats/CSV_GSM

  # Or single section:
  python scripts_tnbc/step25_extract_umi.py \
    --sample-id GSM_6433618 \
    --data-dir data/TNBC_GSE210616 \
    --statsdir stats/CSV_GSM
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample-id",       default=None)
    p.add_argument("--sample-ids-file", default=None,
                   help="Text file with one sample ID per line.")
    p.add_argument("--data-dir",   default="data/TNBC_GSE210616")
    p.add_argument("--statsdir",   default="stats/CSV_GSM")
    p.add_argument("--flux-tag",   default="flux_tumor_immune_region_interface_weighted")
    p.add_argument("--dry-run",    action="store_true",
                   help="Report what would be done without writing.")
    return p.parse_args()

def load_umi_from_h5(h5_path: Path) -> pd.Series | None:
    """
    Load total UMI per barcode from Visium filtered_feature_bc_matrix.h5.
    Returns a pd.Series indexed by barcode string.
    """
    try:
        import scanpy as sc
        adata = sc.read_10x_h5(str(h5_path))
        umi = np.asarray(adata.X.sum(axis=1)).ravel()
        return pd.Series(umi, index=adata.obs_names, name="total_umi")
    except ImportError:
        pass

    # Fallback: manual h5 read without scanpy
    try:
        import h5py
        with h5py.File(h5_path, "r") as f:
            # 10x HDF5 format: /matrix/barcodes and /matrix/data (CSC)
            barcodes = f["matrix/barcodes"][:].astype(str)
            # Sum all features per barcode from CSC sparse matrix
            data     = f["matrix/data"][:]
            indices  = f["matrix/indices"][:]   # row indices (features)
            indptr   = f["matrix/indptr"][:]    # column pointers (barcodes)
            n_barcodes = len(indptr) - 1
            umi = np.array([data[indptr[i]:indptr[i+1]].sum()
                            for i in range(n_barcodes)], dtype=float)
            return pd.Series(umi, index=barcodes, name="total_umi")
    except Exception as ex:
        print(f"    [warn] h5 read failed: {ex}")
        return None

def process_sample(sid: str, data_dir: Path, statsdir: Path,
                   flux_tag: str, dry_run: bool) -> bool:
    print(f"\n=== {sid} ===")
    sample_dir = data_dir / sid
    if not sample_dir.exists():
        # try without underscore in folder name
        alt = data_dir / sid.replace("_","")
        sample_dir = alt if alt.exists() else sample_dir

    # Find h5 file
    h5_candidates = list(sample_dir.glob("*filtered_feature_bc_matrix.h5"))
    if not h5_candidates:
        h5_candidates = list(sample_dir.glob("*.h5"))
    if not h5_candidates:
        print(f"  [skip] no h5 file found in {sample_dir}"); return False
    h5_path = h5_candidates[0]
    print(f"  h5: {h5_path.name}")

    # Find step6 node CSV
    nf = statsdir / f"{sid}_step6_nodes_hodge_{flux_tag}.csv"
    if not nf.exists():
        print(f"  [skip] step6 node CSV not found: {nf.name}"); return False

    # Find step3 node CSV for barcode mapping
    nf3 = statsdir / f"{sid}_step3_nodes.csv"
    nf3_alt = statsdir / f"{sid}_step3_spatial_nodes.csv"
    nf3 = nf3 if nf3.exists() else (nf3_alt if nf3_alt.exists() else None)

    # Load UMI
    umi_series = load_umi_from_h5(h5_path)
    if umi_series is None:
        print("  [fail] could not load UMI from h5"); return False
    print(f"  UMI loaded: n_barcodes={len(umi_series)}  "
          f"mean={umi_series.mean():.0f}  std={umi_series.std():.0f}")

    # Load step6 node CSV
    nodes = pd.read_csv(nf)
    n_nodes = len(nodes)

    # Match barcodes to node_ids
    umi_out = None

    # Option 1: step6 node CSV has a 'barcode' column
    if "barcode" in nodes.columns:
        merged = nodes[["node_id","barcode"]].merge(
            umi_series.reset_index().rename(columns={"index":"barcode"}),
            on="barcode", how="left")
        umi_out = merged.sort_values("node_id")["total_umi"].to_numpy()
        print(f"  matched via barcode column: {(~np.isnan(umi_out)).sum()}/{n_nodes}")

    # Option 2: step3 node CSV has barcode + node_id
    elif nf3 is not None:
        nodes3 = pd.read_csv(nf3)
        if "barcode" in nodes3.columns and "node_id" in nodes3.columns:
            merged = nodes3[["node_id","barcode"]].merge(
                umi_series.reset_index().rename(columns={"index":"barcode"}),
                on="barcode", how="left")
            umi_map = merged.set_index("node_id")["total_umi"]
            umi_out = nodes.sort_values("node_id")["node_id"].map(umi_map).to_numpy()
            print(f"  matched via step3 barcode: {(~np.isnan(umi_out)).sum()}/{n_nodes}")

    # Option 3: assume barcodes are in sorted order (same as node_id order)
    else:
        if len(umi_series) == n_nodes:
            umi_out = umi_series.to_numpy()
            print(f"  matched by position (n_barcodes == n_nodes = {n_nodes})")
        else:
            print(f"  [warn] cannot match barcodes: h5 has {len(umi_series)} barcodes, "
                  f"node CSV has {n_nodes} nodes. Need barcode column in node CSV.")
            return False

    if umi_out is None or np.all(np.isnan(umi_out)):
        print("  [fail] UMI match produced all NaN"); return False

    if dry_run:
        print(f"  [dry-run] would write total_umi to {nf.name}")
        return True

    # Write back
    nodes_sorted = nodes.sort_values("node_id").reset_index(drop=True)
    nodes_sorted["total_umi"] = umi_out
    # Reorder columns to put total_umi near front
    cols = ["node_id","total_umi"] + [c for c in nodes_sorted.columns
                                       if c not in ("node_id","total_umi")]
    nodes_sorted[cols].to_csv(nf, index=False)
    print(f"  ✓ wrote total_umi to {nf.name}  "
          f"(mean={np.nanmean(umi_out):.0f}, std={np.nanstd(umi_out):.0f})")
    return True

if __name__ == "__main__":
    a = _args()
    sd = Path(a.statsdir); dd = Path(a.data_dir)
    sids = []
    if a.sample_id:
        sids = [a.sample_id]
    elif a.sample_ids_file:
        sids = [l.strip() for l in open(a.sample_ids_file) if l.strip()]
    else:
        print("Provide --sample-id or --sample-ids-file"); sys.exit(1)

    ok = 0
    for sid in sids:
        if process_sample(sid, dd, sd, a.flux_tag, a.dry_run):
            ok += 1
    print(f"\nDone: {ok}/{len(sids)} sections processed successfully.")
    if ok == len(sids):
        print("Now rerun: step25_density_nuisance_control.py --mode cohort")
    else:
        print(f"  {len(sids)-ok} sections failed — check h5 file locations.")
