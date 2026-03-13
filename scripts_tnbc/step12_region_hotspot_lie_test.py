from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def majority_region(regions: list[str]) -> str:
    vals, counts = np.unique(regions, return_counts=True)
    idx = np.argmax(counts)
    if counts[idx] >= 2:
        return str(vals[idx])
    return "mixed_face"


def assign_face_regions(nodes_df: pd.DataFrame, faces_df: pd.DataFrame) -> pd.DataFrame:
    node_region = dict(zip(nodes_df["node_id"], nodes_df["region_step2"]))
    face_regions = []

    for _, row in faces_df.iterrows():
        regs = [
            node_region[int(row["i"])],
            node_region[int(row["j"])],
            node_region[int(row["k"])],
        ]
        face_regions.append(majority_region(regs))

    out = faces_df.copy()
    out["face_region"] = face_regions
    return out


def hotspot_enrichment(curl_vals: np.ndarray, region_mask: np.ndarray, q: float = 0.95) -> dict:
    thr = np.quantile(curl_vals, q)
    hotspot = curl_vals >= thr

    n_hot = int(hotspot.sum())
    n_reg = int(region_mask.sum())
    n_all = len(curl_vals)

    if n_hot == 0 or n_reg == 0:
        return {
            "threshold": float(thr),
            "n_hotspots": n_hot,
            "n_region_faces": n_reg,
            "hotspots_in_region": 0,
            "frac_hot_in_region": np.nan,
            "frac_region": np.nan,
            "enrichment_ratio": np.nan,
        }

    hot_in_region = int(np.sum(hotspot & region_mask))
    frac_hot_in_region = hot_in_region / n_hot
    frac_region = n_reg / n_all
    enrich = frac_hot_in_region / frac_region if frac_region > 0 else np.nan

    return {
        "threshold": float(thr),
        "n_hotspots": n_hot,
        "n_region_faces": n_reg,
        "hotspots_in_region": hot_in_region,
        "frac_hot_in_region": float(frac_hot_in_region),
        "frac_region": float(frac_region),
        "enrichment_ratio": float(enrich),
    }


def empirical_pvalue(null_vals: np.ndarray, real_val: float) -> float:
    return float((1.0 + np.sum(null_vals >= real_val)) / (len(null_vals) + 1.0))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 12 TNBC: region-localized hotspot enrichment versus Lie null."
    )
    parser.add_argument("--sample_id", required=True)
    parser.add_argument(
        "--flux_name",
        required=True,
        choices=["flux_tumor_immune", "flux_tumor_stroma", "flux_immune_stroma"],
    )
    parser.add_argument("--statsdir", default="stats")
    parser.add_argument("--q", type=float, default=0.95)
    args = parser.parse_args()

    sample_id = args.sample_id
    flux_name = args.flux_name
    statsdir = Path(args.statsdir)

    nodes_csv = require_file(statsdir / f"{sample_id}_step3_nodes.csv")
    faces_csv = require_file(statsdir / f"{sample_id}_step3_faces.csv")
    real_curl_csv = require_file(statsdir / f"{sample_id}_step9_face_curl_{flux_name}.csv")
    lie_null_csv = require_file(statsdir / f"{sample_id}_step11_lie_null_distribution_{flux_name}.csv")

    nodes_df = pd.read_csv(nodes_csv)
    faces_df = pd.read_csv(faces_csv)
    curl_df = pd.read_csv(real_curl_csv)
    null_df = pd.read_csv(lie_null_csv)

    faces_with_regions = assign_face_regions(nodes_df, faces_df)

    # merge region labels into curl table by face_id
    merged = curl_df.merge(
        faces_with_regions[["face_id", "face_region"]],
        on="face_id",
        how="left",
    )

    # Real curl values
    curl_real = merged["curl_coexact_abs"].to_numpy(dtype=float)

    summaries = []
    for region_name in ["immune_enriched", "interface_like"]:
        mask = (merged["face_region"] == region_name).to_numpy()

        real_stats = hotspot_enrichment(curl_real, mask, q=args.q)

        # For null, we only have scalar summary stats from step11 currently.
        # So we compare real enrichment to the null hotspot statistic already saved there,
        # which is the region-focused top-5% edge hotspot proxy.
        #
        # This is not perfect face-level symmetry, but it is the best compatible comparison
        # with your existing step11 output.
        null_col = f"edge_hotspot_top95_{region_name}"
        if null_col not in null_df.columns:
            # if absent, fall back to top95_mean_curl
            null_vals = null_df["top95_mean_curl"].to_numpy(dtype=float)
        else:
            null_vals = null_df[null_col].to_numpy(dtype=float)

        pval = empirical_pvalue(null_vals, real_stats["enrichment_ratio"])

        summaries.append({
            "sample_id": sample_id,
            "flux_name": flux_name,
            "region_name": region_name,
            "q_hotspot": args.q,
            **real_stats,
            "null_mean_stat": float(np.mean(null_vals)),
            "null_std_stat": float(np.std(null_vals)),
            "empirical_p": pval,
        })

    out_df = pd.DataFrame(summaries)
    out_csv = statsdir / f"{sample_id}_step12_region_hotspot_lie_test_{flux_name}.csv"
    out_df.to_csv(out_csv, index=False)

    print("=" * 72)
    print(f"STEP 12: region-localized hotspot enrichment vs Lie null for {sample_id}")
    print("=" * 72)
    print(out_df[[
        "region_name",
        "enrichment_ratio",
        "null_mean_stat",
        "null_std_stat",
        "empirical_p",
    ]])
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
