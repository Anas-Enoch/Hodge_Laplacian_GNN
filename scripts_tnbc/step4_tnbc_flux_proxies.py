from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from scripts_tnbc.wedge_variants import (
    WedgeConfig,
    attach_interface_to_edges,
    attach_node_programs_to_edges,
    attach_binary_interface_weight_to_edges,
    build_region_interface_wedge,
    build_wedge_variants,
    compute_edge_length,
    compute_interface_score_on_nodes,
    compute_interface_weight_from_region,
    summarize_wedge_variants,
)


# ============================================================
# Helpers
# ============================================================

def require_cols(df: pd.DataFrame, cols: Iterable[str], df_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {df_name}: {missing}")


def pick_first_existing(df: pd.DataFrame, candidates: list[str], df_name: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of the candidate columns exist in {df_name}: {candidates}")


def zscore(x: pd.Series) -> pd.Series:
    arr = x.to_numpy(dtype=float)
    mu = np.nanmean(arr)
    sd = np.nanstd(arr)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros_like(arr), index=x.index)
    return pd.Series((arr - mu) / sd, index=x.index)


def region_demean(df: pd.DataFrame, value_col: str, region_col: str) -> pd.Series:
    means = df.groupby(region_col)[value_col].transform("mean")
    return df[value_col] - means


def attach_coordinates_to_edges(
    edges_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    tail_col: str = "tail",
    head_col: str = "head",
    node_id_col: str = "node_id",
) -> pd.DataFrame:
    """
    Ensures x_tail, y_tail, x_head, y_head exist on edges_df.
    Uses x_fullres / y_fullres if present, else x / y.
    """
    x_col = pick_first_existing(nodes_df, ["x_fullres", "x", "x_coord"], "nodes_df")
    y_col = pick_first_existing(nodes_df, ["y_fullres", "y", "y_coord"], "nodes_df")

    require_cols(edges_df, [tail_col, head_col], "edges_df")
    require_cols(nodes_df, [node_id_col, x_col, y_col], "nodes_df")

    node_small = nodes_df[[node_id_col, x_col, y_col]].copy()

    tail_map = node_small.rename(
        columns={
            node_id_col: tail_col,
            x_col: "x_tail",
            y_col: "y_tail",
        }
    )
    head_map = node_small.rename(
        columns={
            node_id_col: head_col,
            x_col: "x_head",
            y_col: "y_head",
        }
    )

    out = edges_df.copy()
    if not {"x_tail", "y_tail"}.issubset(out.columns):
        out = out.merge(tail_map, on=tail_col, how="left", validate="many_to_one")
    if not {"x_head", "y_head"}.issubset(out.columns):
        out = out.merge(head_map, on=head_col, how="left", validate="many_to_one")

    require_cols(out, ["x_tail", "y_tail", "x_head", "y_head"], "edge-coordinate merged dataframe")
    return out


def compute_scalar_gradient_flux(
    edges_df: pd.DataFrame,
    value_tail_col: str,
    value_head_col: str,
    edge_length_col: str = "edge_length",
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Gradient-like scalar proxy:
      (value_head - value_tail) / (edge_length + eps)
    """
    require_cols(edges_df, [value_tail_col, value_head_col, edge_length_col], "edges_df")
    v_tail = edges_df[value_tail_col].to_numpy(dtype=float)
    v_head = edges_df[value_head_col].to_numpy(dtype=float)
    ell = edges_df[edge_length_col].to_numpy(dtype=float)
    return (v_head - v_tail) / (ell + eps)


def attach_single_program_to_edges(
    edges_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    program_col: str,
    tail_col: str = "tail",
    head_col: str = "head",
    node_id_col: str = "node_id",
    prefix: str = "tumor",
) -> pd.DataFrame:
    """
    Adds:
      {prefix}_tail, {prefix}_head
    """
    require_cols(edges_df, [tail_col, head_col], "edges_df")
    require_cols(nodes_df, [node_id_col, program_col], "nodes_df")

    node_small = nodes_df[[node_id_col, program_col]].copy()

    tail_map = node_small.rename(
        columns={
            node_id_col: tail_col,
            program_col: f"{prefix}_tail",
        }
    )
    head_map = node_small.rename(
        columns={
            node_id_col: head_col,
            program_col: f"{prefix}_head",
        }
    )

    out = edges_df.copy()
    out = out.merge(tail_map, on=tail_col, how="left", validate="many_to_one")
    out = out.merge(head_map, on=head_col, how="left", validate="many_to_one")
    return out


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Build TNBC step4 proxy fluxes.")
    parser.add_argument("--sample-id", default="GSM_6433618", help="Sample ID, e.g. GSM_6433618")
    parser.add_argument("--stats-dir", default="stats/CSV_GSM", help="Directory containing step3/step4 CSV files")
    parser.add_argument("--eps", type=float, default=1e-8, help="Small epsilon for length normalization")
    parser.add_argument("--interface-power", type=float, default=1.0, help="Interface weight exponent")
    parser.add_argument("--bio-power", type=float, default=0.5, help="Biological reliability weight exponent")
    args = parser.parse_args()

    sample_id = args.sample_id
    stats_dir = Path(args.stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)

    nodes_file = stats_dir / f"{sample_id}_step3_nodes.csv"
    edges_file = stats_dir / f"{sample_id}_step3_edges.csv"

    out_nodes_resid = stats_dir / f"{sample_id}_step4_node_residualized_fields.csv"
    out_edges_flux = stats_dir / f"{sample_id}_step4_edge_fluxes.csv"
    out_wedge_summary = stats_dir / f"{sample_id}_step4_wedge_variant_summary.csv"

    nodes_df = pd.read_csv(nodes_file)
    edges_df = pd.read_csv(edges_file)

    # --------------------------------------------------------
    # Column detection
    # --------------------------------------------------------
    node_id_col = pick_first_existing(nodes_df, ["node_id"], "nodes_df")
    region_col = pick_first_existing(nodes_df, ["region_step2", "region", "region_label"], "nodes_df")

    tumor_col = pick_first_existing(nodes_df, ["tumor_score"], "nodes_df")
    stroma_col = pick_first_existing(nodes_df, ["stroma_score"], "nodes_df")
    immune_col = pick_first_existing(nodes_df, ["immune_score"], "nodes_df")

    tail_col = pick_first_existing(edges_df, ["tail"], "edges_df")
    head_col = pick_first_existing(edges_df, ["head"], "edges_df")

    # --------------------------------------------------------
    # Step 4A: node-level residualized fields
    # --------------------------------------------------------
    require_cols(nodes_df, [node_id_col, region_col, tumor_col, stroma_col, immune_col], "nodes_df")

    nodes_out = nodes_df.copy()

    nodes_out["tumor_residual"] = region_demean(nodes_out, tumor_col, region_col)
    nodes_out["stroma_residual"] = region_demean(nodes_out, stroma_col, region_col)
    nodes_out["immune_residual"] = region_demean(nodes_out, immune_col, region_col)

    nodes_out["tumor_score_z"] = zscore(nodes_out[tumor_col])
    nodes_out["stroma_score_z"] = zscore(nodes_out[stroma_col])
    nodes_out["immune_score_z"] = zscore(nodes_out[immune_col])

    nodes_out["tumor_residual_z"] = zscore(nodes_out["tumor_residual"])
    nodes_out["stroma_residual_z"] = zscore(nodes_out["stroma_residual"])
    nodes_out["immune_residual_z"] = zscore(nodes_out["immune_residual"])

    nodes_out.to_csv(out_nodes_resid, index=False)
    print(f"Saved node residualized fields -> {out_nodes_resid}")

    # --------------------------------------------------------
    # Step 4B: prepare edges with geometry and node values
    # --------------------------------------------------------
    edges_out = edges_df.copy()
    edges_out = attach_coordinates_to_edges(
        edges_out,
        nodes_out,
        tail_col=tail_col,
        head_col=head_col,
        node_id_col=node_id_col,
    )
    edges_out["edge_length"] = compute_edge_length(edges_out)

    # Single-program attachments for scalar gradient proxies
    edges_out = attach_single_program_to_edges(
        edges_out, nodes_out,
        program_col=tumor_col,
        tail_col=tail_col, head_col=head_col,
        node_id_col=node_id_col,
        prefix="tumor",
    )
    edges_out = attach_single_program_to_edges(
        edges_out, nodes_out,
        program_col=stroma_col,
        tail_col=tail_col, head_col=head_col,
        node_id_col=node_id_col,
        prefix="stroma",
    )
    edges_out = attach_single_program_to_edges(
        edges_out, nodes_out,
        program_col=immune_col,
        tail_col=tail_col, head_col=head_col,
        node_id_col=node_id_col,
        prefix="immune",
    )

    # Scalar gradient proxy fluxes
    edges_out["flux_tumor"] = compute_scalar_gradient_flux(
        edges_out, "tumor_tail", "tumor_head", edge_length_col="edge_length", eps=args.eps
    )
    edges_out["flux_stroma"] = compute_scalar_gradient_flux(
        edges_out, "stroma_tail", "stroma_head", edge_length_col="edge_length", eps=args.eps
    )
    edges_out["flux_immune"] = compute_scalar_gradient_flux(
        edges_out, "immune_tail", "immune_head", edge_length_col="edge_length", eps=args.eps
    )

    # --------------------------------------------------------
    # Step 4C: specialized wedge variants for each pair
    # --------------------------------------------------------
    cfg = WedgeConfig(
        eps=args.eps,
        interface_power=args.interface_power,
        bio_power=args.bio_power,
    )

    wedge_pairs = [
        ("tumor_score", "immune_score", "flux_tumor_immune"),
        ("tumor_score", "stroma_score", "flux_tumor_stroma"),
        ("immune_score", "stroma_score", "flux_immune_stroma"),
    ]

    wedge_summary_blocks = []

    for prog_a, prog_b, prefix in wedge_pairs:
        # Interface score specific to this pair
        nodes_pair = compute_interface_score_on_nodes(
            nodes_out,
            program_a=prog_a,
            program_b=prog_b,
            out_col="interface_score_simple",
        )

        # Attach pair-specific programs
        pair_edges = attach_node_programs_to_edges(
            edges_out,
            nodes_pair,
            tail_col=tail_col,
            head_col=head_col,
            node_id_col=node_id_col,
            program_a=prog_a,
            program_b=prog_b,
        )

        # Attach interface score
        pair_edges = attach_interface_to_edges(
            pair_edges,
            nodes_pair,
            tail_col=tail_col,
            head_col=head_col,
            node_id_col=node_id_col,
            interface_col="interface_score_simple",
        )

        # Build wedge variants
        pair_edges = build_wedge_variants(
            pair_edges,
            cfg=cfg,
            a_tail_col="A_tail",
            a_head_col="A_head",
            b_tail_col="B_tail",
            b_head_col="B_head",
            interface_tail_col="I_tail",
            interface_head_col="I_head",
            edge_length_col="edge_length",
            prefix=prefix,
        )

                # Region-based binary interface weighting
        nodes_region_weighted = compute_interface_weight_from_region(
            nodes_pair,
            region_col=region_col,
            interface_region="interface_like",
            out_col="interface_weight_binary",
        )

        pair_edges_region = attach_binary_interface_weight_to_edges(
            pair_edges,
            nodes_region_weighted,
            tail_col=tail_col,
            head_col=head_col,
            node_id_col=node_id_col,
            interface_weight_col="interface_weight_binary",
        )

        pair_edges_region = build_region_interface_wedge(
            pair_edges_region,
            cfg=cfg,
            a_tail_col="A_tail",
            a_head_col="A_head",
            b_tail_col="B_tail",
            b_head_col="B_head",
            edge_weight_col="W_edge",
            edge_length_col="edge_length",
            prefix=prefix,
        )

        region_variant_col = f"{prefix}_region_interface_weighted"
        edges_out[region_variant_col] = pair_edges_region[region_variant_col].to_numpy()

        

        # Copy resulting columns back into main edge table
        copy_cols = [
            f"{prefix}_baseline",
            f"{prefix}_length_norm",
            f"{prefix}_interface_weighted",
            f"{prefix}_bio_weight",
            f"{prefix}_interface_weight",
        ]
        for c in copy_cols:
            edges_out[c] = pair_edges[c].to_numpy()

        # Legacy alias for compatibility:
        # keep old name = baseline wedge
        if prefix == "flux_tumor_immune":
            edges_out["flux_tumor_immune"] = edges_out[f"{prefix}_baseline"]
        elif prefix == "flux_tumor_stroma":
            edges_out["flux_tumor_stroma"] = edges_out[f"{prefix}_baseline"]
        elif prefix == "flux_immune_stroma":
            edges_out["flux_immune_stroma"] = edges_out[f"{prefix}_baseline"]

        

        summary_df = summarize_wedge_variants(pair_edges, prefix=prefix)

        region_variant_col = f"{prefix}_region_interface_weighted"
        x = pair_edges_region[region_variant_col].to_numpy(dtype=float)
        summary_region = pd.DataFrame(
            [{
                "flux_name": region_variant_col,
                "mean": float(np.mean(x)),
                "std": float(np.std(x)),
                "mean_abs": float(np.mean(np.abs(x))),
                "max_abs": float(np.max(np.abs(x))),
            }]
        )

        summary_df = pd.concat([summary_df, summary_region], ignore_index=True)
        summary_df["sample_id"] = sample_id
        summary_df["pair"] = prefix
        wedge_summary_blocks.append(summary_df)
        
        

        
    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------
    edges_out.to_csv(out_edges_flux, index=False)
    print(f"Saved edge fluxes -> {out_edges_flux}")

    wedge_summary = pd.concat(wedge_summary_blocks, ignore_index=True)
    wedge_summary.to_csv(out_wedge_summary, index=False)
    print(f"Saved wedge summary -> {out_wedge_summary}")

    # Console preview
    print("\nNode residualized field preview:")
    print(
        nodes_out[
            [
                node_id_col,
                region_col,
                tumor_col,
                stroma_col,
                immune_col,
                "tumor_residual",
                "stroma_residual",
                "immune_residual",
            ]
        ].head().to_string(index=False)
    )

    print("\nEdge flux preview:")
    preview_cols = [
        tail_col,
        head_col,
        "edge_length",
        "flux_tumor",
        "flux_stroma",
        "flux_immune",
        "flux_tumor_immune_baseline",
        "flux_tumor_immune_length_norm",
        "flux_tumor_immune_interface_weighted",
    ]
    print(edges_out[preview_cols].head().to_string(index=False))


if __name__ == "__main__":
    main()
