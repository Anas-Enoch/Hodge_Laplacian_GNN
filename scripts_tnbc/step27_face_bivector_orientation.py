from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def triangle_area(p0, p1, p2):
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    return 0.5 * abs(cross)


def process_sample(sid: str, statsdir: Path, outdir: Path) -> dict:
    nodes_path = statsdir / f"{sid}_step6_nodes_hodge_flux_tumor_immune_region_interface_weighted.csv"
    edges_path = statsdir / f"{sid}_step6_edges_hodge_flux_tumor_immune_region_interface_weighted.csv"
    faces_path = statsdir / f"{sid}_step3_faces.csv"
    out_path = outdir / f"{sid}_step27_face_bivector_orientation.csv"

    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)
    faces = pd.read_csv(faces_path)

    required_nodes = {"node_id", "x_fullres", "y_fullres", "region_step2"}
    required_edges = {"tail", "head", "flux_coexact"}
    required_faces = {"face_id", "i", "j", "k"}

    if not required_nodes <= set(nodes.columns):
        raise ValueError(f"{sid}: nodes missing {required_nodes - set(nodes.columns)}")
    if not required_edges <= set(edges.columns):
        raise ValueError(f"{sid}: edges missing {required_edges - set(edges.columns)}")
    if not required_faces <= set(faces.columns):
        raise ValueError(f"{sid}: faces missing {required_faces - set(faces.columns)}")

    coord = {
        int(r.node_id): np.array([float(r.x_fullres), float(r.y_fullres)])
        for r in nodes.itertuples()
    }

    region = {
        int(r.node_id): r.region_step2
        for r in nodes.itertuples()
    }

    edge_flux = {}
    for r in edges.itertuples():
        u, v = int(r.tail), int(r.head)
        val = float(r.flux_coexact)
        edge_flux[(u, v)] = val
        edge_flux[(v, u)] = -val

    rows = []

    for r in faces.itertuples():
        a, b, c = int(r.i), int(r.j), int(r.k)

        if a not in coord or b not in coord or c not in coord:
            continue

        p0, p1, p2 = coord[a], coord[b], coord[c]
        area = triangle_area(p0, p1, p2)

        if area <= 0:
            continue

        # B2^T f_coexact on oriented triangle boundary
        circ = (
            edge_flux.get((a, b), 0.0)
            + edge_flux.get((b, c), 0.0)
            + edge_flux.get((c, a), 0.0)
        )

        bivector_density = circ / area
        centroid = (p0 + p1 + p2) / 3.0

        regs = [
            region.get(a, "unknown"),
            region.get(b, "unknown"),
            region.get(c, "unknown"),
        ]
        face_region = max(set(regs), key=regs.count)

        rows.append({
            "sample_id": sid,
            "face_id": r.face_id,
            "a": a,
            "b": b,
            "c": c,
            "centroid_x": centroid[0],
            "centroid_y": centroid[1],
            "area_px2": area,
            "circulation": circ,
            "abs_circulation": abs(circ),
            "bivector_density": bivector_density,
            "abs_bivector_density": abs(bivector_density),
            "sign": np.sign(bivector_density),
            "face_region": face_region,
        })

    out = pd.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"\n=== {sid} ===")
    print(f"[done] wrote {out_path}")
    print(f"[n_faces] {len(out)}")

    region_medians = (
        out.groupby("face_region")["abs_bivector_density"]
        .median()
        .sort_values(ascending=False)
    )
    print("\n[regional median abs_bivector_density]")
    print(region_medians)

    interface = out[out["face_region"] == "interface_like"].copy()
    if len(interface) == 0:
        return {
            "sample_id": sid,
            "n_faces": len(out),
            "n_interface_faces": 0,
            "interface_median_abs": np.nan,
            "next_highest_median_abs": np.nan,
            "fold_vs_next": np.nan,
            "signed_mean": np.nan,
            "signed_std": np.nan,
            "signed_se": np.nan,
            "z_mean_over_se": np.nan,
            "frac_positive": np.nan,
            "frac_negative": np.nan,
            "orientation_bias": "no_interface_faces",
        }

    x = interface["bivector_density"].dropna().to_numpy()
    signed_mean = float(np.mean(x))
    signed_std = float(np.std(x, ddof=1)) if len(x) > 1 else np.nan
    signed_se = signed_std / np.sqrt(len(x)) if len(x) > 1 and signed_std > 0 else np.nan
    z = signed_mean / signed_se if signed_se and not np.isnan(signed_se) else np.nan
    frac_positive = float(np.mean(x > 0))
    frac_negative = float(np.mean(x < 0))

    interface_median = float(region_medians.get("interface_like", np.nan))
    non_interface = region_medians.drop(labels=["interface_like"], errors="ignore")
    next_highest = float(non_interface.max()) if len(non_interface) else np.nan
    fold = interface_median / next_highest if next_highest and next_highest > 0 else np.nan

    # Conservative heuristic, not a formal p-value.
    if np.isnan(z):
        orientation_bias = "not_testable"
    elif abs(z) >= 2 and (frac_positive >= 0.60 or frac_positive <= 0.40):
        orientation_bias = "directional_bias"
    else:
        orientation_bias = "balanced_orientation"

    print("\n[interface signed orientation test]")
    print(f"n_interface_faces = {len(x)}")
    print(f"signed_mean       = {signed_mean:.6e}")
    print(f"signed_std        = {signed_std:.6e}")
    print(f"signed_se         = {signed_se:.6e}")
    print(f"z_mean_over_se    = {z:.3f}")
    print(f"frac_positive     = {frac_positive:.3f}")
    print(f"frac_negative     = {frac_negative:.3f}")
    print(f"orientation_bias  = {orientation_bias}")
    print(f"fold_vs_next      = {fold:.3f}")

    return {
        "sample_id": sid,
        "n_faces": len(out),
        "n_interface_faces": len(x),
        "interface_median_abs": interface_median,
        "next_highest_median_abs": next_highest,
        "fold_vs_next": fold,
        "signed_mean": signed_mean,
        "signed_std": signed_std,
        "signed_se": signed_se,
        "z_mean_over_se": z,
        "frac_positive": frac_positive,
        "frac_negative": frac_negative,
        "orientation_bias": orientation_bias,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", type=str, default=None)
    parser.add_argument("--sample-ids", type=str, default=None)
    parser.add_argument("--statsdir", type=Path, default=Path("stats/CSV_GSM"))
    parser.add_argument("--outdir", type=Path, default=Path("stats/CSV_GSM"))
    args = parser.parse_args()

    if args.sample_ids:
        sample_ids = [s.strip() for s in args.sample_ids.split(",") if s.strip()]
    elif args.sample_id:
        sample_ids = [args.sample_id]
    else:
        raise ValueError("Provide --sample-id or --sample-ids")

    summaries = []
    for sid in sample_ids:
        summaries.append(process_sample(sid, args.statsdir, args.outdir))

    summary = pd.DataFrame(summaries)
    summary_path = args.outdir / "step27_face_bivector_orientation_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n=== STEP 27 SUMMARY ===")
    print(summary)
    print(f"\n[done] wrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
