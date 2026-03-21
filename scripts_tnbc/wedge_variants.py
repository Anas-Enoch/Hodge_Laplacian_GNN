from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class WedgeConfig:
    eps: float = 1e-8
    interface_power: float = 1.0
    bio_power: float = 0.5


def _require_cols(df: pd.DataFrame, cols: Iterable[str], df_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {df_name}: {missing}")


def compute_edge_length(
    edges_df: pd.DataFrame,
    x_tail_col: str = "x_tail",
    y_tail_col: str = "y_tail",
    x_head_col: str = "x_head",
    y_head_col: str = "y_head",
) -> np.ndarray:
    """
    Compute Euclidean length for each edge.
    """
    _require_cols(edges_df, [x_tail_col, y_tail_col, x_head_col, y_head_col], "edges_df")

    dx = edges_df[x_head_col].to_numpy(dtype=float) - edges_df[x_tail_col].to_numpy(dtype=float)
    dy = edges_df[y_head_col].to_numpy(dtype=float) - edges_df[y_tail_col].to_numpy(dtype=float)
    return np.sqrt(dx * dx + dy * dy)


def attach_node_programs_to_edges(
    edges_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    tail_col: str = "tail",
    head_col: str = "head",
    node_id_col: str = "node_id",
    program_a: str = "tumor_score",
    program_b: str = "immune_score",
) -> pd.DataFrame:
    """
    Add node-level program values to each edge as:
      A_tail, A_head, B_tail, B_head
    """
    _require_cols(edges_df, [tail_col, head_col], "edges_df")
    _require_cols(nodes_df, [node_id_col, program_a, program_b], "nodes_df")

    node_small = nodes_df[[node_id_col, program_a, program_b]].copy()
    node_small = node_small.rename(
        columns={
            node_id_col: "node_id",
            program_a: "A",
            program_b: "B",
        }
    )

    out = edges_df.copy()

    tail_map = node_small.rename(
        columns={"node_id": tail_col, "A": "A_tail", "B": "B_tail"}
    )
    head_map = node_small.rename(
        columns={"node_id": head_col, "A": "A_head", "B": "B_head"}
    )

    out = out.merge(tail_map, on=tail_col, how="left", validate="many_to_one")
    out = out.merge(head_map, on=head_col, how="left", validate="many_to_one")

    _require_cols(out, ["A_tail", "A_head", "B_tail", "B_head"], "merged edge dataframe")
    return out


def compute_interface_score_on_nodes(
    nodes_df: pd.DataFrame,
    program_a: str = "tumor_score",
    program_b: str = "immune_score",
    out_col: str = "interface_score_simple",
) -> pd.DataFrame:
    """
    Simple node-level interface contrast:
      |A - B|
    """
    _require_cols(nodes_df, [program_a, program_b], "nodes_df")

    out = nodes_df.copy()
    out[out_col] = np.abs(
        out[program_a].to_numpy(dtype=float) - out[program_b].to_numpy(dtype=float)
    )
    return out


def attach_interface_to_edges(
    edges_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    tail_col: str = "tail",
    head_col: str = "head",
    node_id_col: str = "node_id",
    interface_col: str = "interface_score_simple",
) -> pd.DataFrame:
    """
    Add node-level interface score to each edge as:
      I_tail, I_head
    """
    _require_cols(edges_df, [tail_col, head_col], "edges_df")
    _require_cols(nodes_df, [node_id_col, interface_col], "nodes_df")

    node_small = nodes_df[[node_id_col, interface_col]].copy()

    out = edges_df.copy()

    tail_map = node_small.rename(
        columns={node_id_col: tail_col, interface_col: "I_tail"}
    )
    head_map = node_small.rename(
        columns={node_id_col: head_col, interface_col: "I_head"}
    )

    out = out.merge(tail_map, on=tail_col, how="left", validate="many_to_one")
    out = out.merge(head_map, on=head_col, how="left", validate="many_to_one")

    _require_cols(out, ["I_tail", "I_head"], "merged edge dataframe")
    return out


def build_wedge_variants(
    edges_df: pd.DataFrame,
    cfg: WedgeConfig | None = None,
    a_tail_col: str = "A_tail",
    a_head_col: str = "A_head",
    b_tail_col: str = "B_tail",
    b_head_col: str = "B_head",
    interface_tail_col: str = "I_tail",
    interface_head_col: str = "I_head",
    edge_length_col: str = "edge_length",
    prefix: str = "flux_tumor_immune",
) -> pd.DataFrame:
    """
    Build three wedge variants:

      {prefix}_baseline
      {prefix}_length_norm
      {prefix}_interface_weighted

    Also saves:
      {prefix}_bio_weight
      {prefix}_interface_weight

    Required columns:
      A_tail, A_head, B_tail, B_head, I_tail, I_head, edge_length
    """
    if cfg is None:
        cfg = WedgeConfig()

    _require_cols(
        edges_df,
        [
            a_tail_col,
            a_head_col,
            b_tail_col,
            b_head_col,
            interface_tail_col,
            interface_head_col,
            edge_length_col,
        ],
        "edges_df",
    )

    out = edges_df.copy()

    A_i = out[a_tail_col].to_numpy(dtype=float)
    A_j = out[a_head_col].to_numpy(dtype=float)
    B_i = out[b_tail_col].to_numpy(dtype=float)
    B_j = out[b_head_col].to_numpy(dtype=float)

    I_i = out[interface_tail_col].to_numpy(dtype=float)
    I_j = out[interface_head_col].to_numpy(dtype=float)

    ell = out[edge_length_col].to_numpy(dtype=float)

    # Core antisymmetric term
    wedge_raw = A_i * B_j - A_j * B_i

    # Variant 1: baseline
    flux_baseline = wedge_raw

    # Variant 2: length-normalized
    flux_length_norm = wedge_raw / (ell + cfg.eps)

    # Variant 3: interface-weighted
    bio_weight = (
        ((np.abs(A_i) + np.abs(A_j)) / 2.0) *
        ((np.abs(B_i) + np.abs(B_j)) / 2.0)
    ) ** cfg.bio_power

    interface_weight = ((I_i + I_j) / 2.0) ** cfg.interface_power

    flux_interface_weighted = (
        bio_weight * interface_weight * wedge_raw / (ell + cfg.eps)
    )

    out[f"{prefix}_baseline"] = flux_baseline
    out[f"{prefix}_length_norm"] = flux_length_norm
    out[f"{prefix}_interface_weighted"] = flux_interface_weighted
    out[f"{prefix}_bio_weight"] = bio_weight
    out[f"{prefix}_interface_weight"] = interface_weight

    return out


def summarize_wedge_variants(
    edges_df: pd.DataFrame,
    prefix: str = "flux_tumor_immune",
) -> pd.DataFrame:
    """
    Small diagnostic summary for the wedge variants.
    """
    cols = [
        f"{prefix}_baseline",
        f"{prefix}_length_norm",
        f"{prefix}_interface_weighted",
    ]
    _require_cols(edges_df, cols, "edges_df")

    rows = []
    for c in cols:
        x = edges_df[c].to_numpy(dtype=float)
        rows.append(
            {
                "flux_name": c,
                "mean": float(np.mean(x)),
                "std": float(np.std(x)),
                "mean_abs": float(np.mean(np.abs(x))),
                "max_abs": float(np.max(np.abs(x))),
            }
        )

    return pd.DataFrame(rows)

def compute_interface_weight_from_region(
    nodes_df: pd.DataFrame,
    region_col: str = "region_step2",
    interface_region: str = "interface_like",
    out_col: str = "interface_weight_binary",
) -> pd.DataFrame:
    """
    Binary node-level interface weight:
      1.0 for interface_like
      0.1 otherwise
    """
    _require_cols(nodes_df, [region_col], "nodes_df")

    out = nodes_df.copy()
    out[out_col] = np.where(
        out[region_col] == interface_region,
        1.0,
        0.1,
    )
    return out

def attach_binary_interface_weight_to_edges(
    edges_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    tail_col: str = "tail",
    head_col: str = "head",
    node_id_col: str = "node_id",
    interface_weight_col: str = "interface_weight_binary",
) -> pd.DataFrame:
    """
    Adds W_tail, W_head and edge-level geometric mean weight W_edge.
    """
    _require_cols(edges_df, [tail_col, head_col], "edges_df")
    _require_cols(nodes_df, [node_id_col, interface_weight_col], "nodes_df")

    node_small = nodes_df[[node_id_col, interface_weight_col]].copy()

    tail_map = node_small.rename(
        columns={node_id_col: tail_col, interface_weight_col: "W_tail"}
    )
    head_map = node_small.rename(
        columns={node_id_col: head_col, interface_weight_col: "W_head"}
    )

    out = edges_df.copy()
    out = out.merge(tail_map, on=tail_col, how="left", validate="many_to_one")
    out = out.merge(head_map, on=head_col, how="left", validate="many_to_one")

    _require_cols(out, ["W_tail", "W_head"], "merged edge dataframe")

    out["W_edge"] = np.sqrt(
        out["W_tail"].to_numpy(dtype=float) * out["W_head"].to_numpy(dtype=float)
    )
    return out

def build_region_interface_wedge(
    edges_df: pd.DataFrame,
    cfg: WedgeConfig | None = None,
    a_tail_col: str = "A_tail",
    a_head_col: str = "A_head",
    b_tail_col: str = "B_tail",
    b_head_col: str = "B_head",
    edge_weight_col: str = "W_edge",
    edge_length_col: str = "edge_length",
    prefix: str = "flux_tumor_immune",
) -> pd.DataFrame:
    """
    Region-based interface-weighted wedge:
      W_edge * (A_i B_j - A_j B_i) / (ell + eps)
    """
    if cfg is None:
        cfg = WedgeConfig()

    _require_cols(
        edges_df,
        [a_tail_col, a_head_col, b_tail_col, b_head_col, edge_weight_col, edge_length_col],
        "edges_df",
    )

    out = edges_df.copy()

    A_i = out[a_tail_col].to_numpy(dtype=float)
    A_j = out[a_head_col].to_numpy(dtype=float)
    B_i = out[b_tail_col].to_numpy(dtype=float)
    B_j = out[b_head_col].to_numpy(dtype=float)
    W = out[edge_weight_col].to_numpy(dtype=float)
    ell = out[edge_length_col].to_numpy(dtype=float)

    wedge_raw = A_i * B_j - A_j * B_i
    flux_region_interface = W * wedge_raw / (ell + cfg.eps)

    out[f"{prefix}_region_interface_weighted"] = flux_region_interface
    return out
