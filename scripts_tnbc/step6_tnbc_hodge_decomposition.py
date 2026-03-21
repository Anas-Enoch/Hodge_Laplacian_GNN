from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
from scipy.spatial import Delaunay


# ============================================================
# BUG INVENTORY — Step 6
# ============================================================
#
# Bug S6-1 (ROOT CAUSE of agg_frac > 1.0):
#   node_energy_total was computed independently from omega using
#   node_energy_from_edge_component(edges_df, node_ids, omega).
#   node_energy_exact was computed from omega_exact the same way.
#   These are INDEPENDENT means of squared values:
#     node_energy_total_i  = mean_j( omega_j^2 )
#     node_energy_exact_i  = mean_j( omega_exact_j^2 )
#   There is NO guarantee that mean(omega_exact^2) <= mean(omega^2)
#   at a single node, because the Hodge projection is orthogonal in
#   the GLOBAL L2 sense, not locally.  lsqr can give individual
#   edge values of omega_exact that exceed omega, while satisfying
#   ||omega_exact|| <= ||omega|| globally.
#   Result: node_energy_exact > node_energy_total at some nodes,
#           causing agg_frac_exact > 1.0 in small regions.
#
#   Fix: recompute node_energy_total as the sum of the three component
#        node energies (exact + coexact + harmonic).  This is internally
#        consistent by construction and bounded by the triangle inequality.
#        The original omega-derived total is retained as
#        node_energy_total_raw for diagnostic comparison only.
#
# Bug S6-2 (wrong fraction denominator):
#   frac_exact / frac_coexact / frac_harmonic at node level used
#   node_energy_total (omega-derived) as denominator.
#   With Bug S6-1 present, this produces frac > 1.
#
#   Fix: use node_energy_total_consistent (component sum) as denominator.
#
# Bug S6-3 (no global leakage diagnostic):
#   The global energy fracs used E_total = ||omega||^2.  Due to lsqr
#   numerical error, E_exact + E_coexact + E_harmonic can differ from
#   E_total at the ~1e-6 level.  No warning was emitted.
#
#   Fix: compute global fracs from component sum; warn if the discrepancy
#        with ||omega||^2 exceeds a tolerance.
# ============================================================


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


def sanitize_flux_name(flux_name: str) -> str:
    return flux_name.replace("/", "_").replace(" ", "_")


def load_sparse_npz_if_exists(path: Path) -> sp.csr_matrix | None:
    if path.exists():
        return sp.load_npz(path).tocsr()
    return None


# ============================================================
# Incidence matrix constructors
# ============================================================

def build_B1_from_edges(edges_df: pd.DataFrame, node_ids: np.ndarray) -> sp.csr_matrix:
    """
    Build node-edge incidence matrix B1: rows = edges, cols = nodes.
    Convention: tail -> -1, head -> +1.
    """
    require_cols(edges_df, ["tail", "head"], "edges_df")
    node_to_idx = {int(n): i for i, n in enumerate(node_ids)}

    m = len(edges_df)
    n = len(node_ids)
    rows, cols, vals = [], [], []

    for e_idx, row in edges_df.reset_index(drop=True).iterrows():
        tail = int(row["tail"])
        head = int(row["head"])
        if tail not in node_to_idx or head not in node_to_idx:
            raise ValueError(f"Edge references missing node_id: tail={tail}, head={head}")
        rows.extend([e_idx, e_idx])
        cols.extend([node_to_idx[tail], node_to_idx[head]])
        vals.extend([-1.0, 1.0])

    return sp.csr_matrix((vals, (rows, cols)), shape=(m, n))


def build_B2_from_faces(edges_df: pd.DataFrame, faces_df: pd.DataFrame) -> sp.csr_matrix | None:
    """
    Build edge-face incidence matrix B2 from a triangular face table.
    Supported column schemas: (edge_1/2/3), (e1/2/3), (edge_a/b/c).
    Negative edge ids indicate reversed orientation.
    """
    if faces_df is None or len(faces_df) == 0:
        return None

    candidate_sets = [
        ["edge_1", "edge_2", "edge_3"],
        ["e1", "e2", "e3"],
        ["edge_a", "edge_b", "edge_c"],
    ]
    edge_cols = None
    for cand in candidate_sets:
        if all(c in faces_df.columns for c in cand):
            edge_cols = cand
            break
    if edge_cols is None:
        return None

    if "edge_id" in edges_df.columns:
        edge_id_to_idx = {int(eid): i for i, eid in enumerate(edges_df["edge_id"].tolist())}
    else:
        edge_id_to_idx = {int(i): i for i in range(len(edges_df))}

    p = len(faces_df)
    m = len(edges_df)
    rows, cols, vals = [], [], []

    for f_idx, row in faces_df.reset_index(drop=True).iterrows():
        for c in edge_cols:
            raw = row[c]
            if pd.isna(raw):
                continue
            edge_id_signed = int(raw)
            sign = 1.0
            edge_id = edge_id_signed
            if edge_id_signed < 0:
                sign = -1.0
                edge_id = abs(edge_id_signed)
            if edge_id not in edge_id_to_idx:
                if edge_id_signed in edge_id_to_idx:
                    edge_id = edge_id_signed
                    sign = 1.0
                else:
                    continue
            e_idx = edge_id_to_idx[edge_id]
            rows.append(f_idx)
            cols.append(e_idx)
            vals.append(sign)

    if len(rows) == 0:
        return None
    return sp.csr_matrix((vals, (rows, cols)), shape=(p, m))


def build_B2_from_coordinates(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    node_id_col: str = "node_id",
) -> sp.csr_matrix | None:
    """
    Construct B2 from Delaunay triangulation of node coordinates.
    Face boundary cycle: (v0->v1), (v1->v2), (v2->v0).
    Sign is +1 if face direction matches stored edge orientation, else -1.
    """
    x_col = pick_first_existing(nodes_df, ["x_fullres", "x", "x_coord"], "nodes_df")
    y_col = pick_first_existing(nodes_df, ["y_fullres", "y", "y_coord"], "nodes_df")

    require_cols(nodes_df, [node_id_col, x_col, y_col], "nodes_df")
    require_cols(edges_df, ["tail", "head"], "edges_df")

    node_ids = nodes_df[node_id_col].to_numpy(dtype=int)
    coords   = nodes_df[[x_col, y_col]].to_numpy(dtype=float)

    tri = Delaunay(coords)
    simplices = tri.simplices

    edge_lookup: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for e_idx, row in edges_df.reset_index(drop=True).iterrows():
        t = int(row["tail"])
        h = int(row["head"])
        key = (min(t, h), max(t, h))
        edge_lookup[key] = (e_idx, (t, h))

    m = len(edges_df)
    rows, cols, vals = [], [], []
    face_cycle = [(0, 1), (1, 2), (2, 0)]

    for f_idx, simplex in enumerate(simplices):
        for a_loc, b_loc in face_cycle:
            nid_a = int(node_ids[simplex[a_loc]])
            nid_b = int(node_ids[simplex[b_loc]])
            key   = (min(nid_a, nid_b), max(nid_a, nid_b))
            if key not in edge_lookup:
                continue
            e_idx, stored_dir = edge_lookup[key]
            face_sign = 1.0 if stored_dir == (nid_a, nid_b) else -1.0
            rows.append(f_idx)
            cols.append(e_idx)
            vals.append(face_sign)

    if len(rows) == 0:
        print("WARNING: B2 construction from coordinates produced no entries.")
        return None

    n_faces = len(simplices)
    B2 = sp.csr_matrix((vals, (rows, cols)), shape=(n_faces, m))
    print(f"B2 built from Delaunay: {n_faces} faces, shape={B2.shape}, nnz={B2.nnz}")
    return B2


# ============================================================
# Hodge projection functions
# ============================================================

def exact_projection(B1: sp.csr_matrix, omega: np.ndarray) -> np.ndarray:
    """
    Project omega onto im(B1) = exact subspace.
    Uses least-squares: phi = argmin ||B1*phi - omega||.
    Orthogonality holds globally; individual edge values may exceed omega.
    """
    phi = lsqr(B1, omega)[0]
    return np.asarray(B1 @ phi).reshape(-1)


def coexact_projection(B2: sp.csr_matrix | None, omega_resid: np.ndarray) -> np.ndarray:
    """
    Project omega_resid onto im(B2^T) = coexact subspace.
    Returns zeros if B2 is unavailable.
    """
    if B2 is None or B2.shape[0] == 0:
        return np.zeros_like(omega_resid)
    psi = lsqr(B2.T, omega_resid)[0]
    return np.asarray(B2.T @ psi).reshape(-1)


# ============================================================
# Node-level energy aggregation
# ============================================================

def node_energy_from_edge_component(
    edges_df: pd.DataFrame,
    node_ids: np.ndarray,
    component: np.ndarray,
) -> pd.DataFrame:
    """
    For each node, compute the mean squared value of incident edge components.

    NOTE: This function computes mean(component²) over incident edges, NOT
    the squared L2 norm of the component vector.  The result is therefore
    NOT additive across Hodge components (i.e. node_energy_exact +
    node_energy_coexact + node_energy_harmonic ≠ node_energy_total in
    general, because the cross terms from squaring a sum are non-zero).

    This is handled upstream: node_energy_total_consistent is recomputed
    as the sum of the three component energies in main() to ensure that
    fractions computed from it are always in [0, 1].
    """
    require_cols(edges_df, ["tail", "head"], "edges_df")

    edge_work = edges_df.copy().reset_index(drop=True)
    edge_work["component_sq"] = component ** 2

    rows = []
    for node_id in node_ids:
        sub = edge_work[(edge_work["tail"] == node_id) | (edge_work["head"] == node_id)]
        energy = float(sub["component_sq"].mean()) if len(sub) > 0 else 0.0
        rows.append({"node_id": int(node_id), "node_energy": energy})

    return pd.DataFrame(rows)


# ============================================================
# Face-level curl
# ============================================================

def face_curl_from_B2(B2: sp.csr_matrix | None, omega: np.ndarray) -> np.ndarray:
    if B2 is None or B2.shape[0] == 0:
        return np.zeros(0, dtype=float)
    curl = B2 @ omega
    return np.asarray(curl).reshape(-1)


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TNBC Step 6: Hodge decomposition for a chosen flux column."
    )
    parser.add_argument("--sample-id", default="GSM_6433618")
    parser.add_argument(
        "--flux-col", default="flux_tumor_immune",
        help=(
            "Flux column in step4_edge_fluxes.csv "
            "(e.g. flux_tumor_immune_region_interface_weighted)"
        ),
    )
    parser.add_argument("--stats-dir", default="stats/CSV_GSM")
    parser.add_argument("--root-stats-dir", default="stats",
                        help="Directory containing optional precomputed B1/B2 npz files")
    args = parser.parse_args()

    sample_id = args.sample_id
    flux_col  = args.flux_col
    flux_tag  = sanitize_flux_name(flux_col)

    stats_dir      = Path(args.stats_dir)
    root_stats_dir = Path(args.root_stats_dir)

    # ---- Input files ----
    nodes_file = stats_dir / f"{sample_id}_step3_nodes.csv"
    edges_file = stats_dir / f"{sample_id}_step4_edge_fluxes.csv"
    faces_file = stats_dir / f"{sample_id}_step3_faces.csv"

    nodes_df = pd.read_csv(nodes_file)
    edges_df = pd.read_csv(edges_file)
    faces_df = pd.read_csv(faces_file) if faces_file.exists() else None

    require_cols(nodes_df, ["node_id"], "nodes_df")
    require_cols(edges_df, ["tail", "head", flux_col], "edges_df")

    node_ids = nodes_df["node_id"].to_numpy(dtype=int)

    # ---- Build B1 ----
    B1 = load_sparse_npz_if_exists(root_stats_dir / f"{sample_id}_step3_B1.npz")
    if B1 is None:
        B1 = build_B1_from_edges(edges_df, node_ids)

    # ---- Build B2 ----
    B2 = load_sparse_npz_if_exists(root_stats_dir / f"{sample_id}_step3_B2.npz")
    if B2 is None:
        B2 = build_B2_from_faces(edges_df, faces_df)

    if B2 is None:
        if faces_df is None:
            print(f"WARNING: B2 is None — faces file not found: {faces_file}")
        elif len(faces_df) == 0:
            print("WARNING: B2 is None — faces_df is empty")
        else:
            print(
                f"WARNING: B2 is None — faces_df has {len(faces_df)} rows "
                f"but no recognized edge column schema.\n"
                f"  Available columns: {faces_df.columns.tolist()}"
            )
    else:
        print(f"B2 loaded/built: shape={B2.shape}, nnz={B2.nnz}")

    if B2 is None:
        print("Falling back to coordinate-based B2 construction...")
        B2 = build_B2_from_coordinates(nodes_df, edges_df, node_id_col="node_id")

    if B2 is None:
        print(
            "CRITICAL: B2 unavailable. Coexact and harmonic cannot be separated; "
            "all non-exact residual energy will appear in harmonic."
        )
    else:
        # Orientation consistency check: B2 @ B1 should be zero
        product = B2 @ B1
        max_entry = float(np.abs(product.data).max()) if product.nnz > 0 else 0.0
        if max_entry > 1e-10:
            print(
                f"WARNING: B2 @ B1 not zero (max entry = {max_entry:.3e}). "
                "Orientation inconsistency may cause Hodge leakage."
            )
        else:
            print(f"Orientation check passed: B2 @ B1 max entry = {max_entry:.2e}")

    # ---- Hodge decomposition ----
    omega = edges_df[flux_col].to_numpy(dtype=float)

    omega_exact   = exact_projection(B1, omega)
    resid         = omega - omega_exact
    omega_coexact = coexact_projection(B2, resid)
    omega_harmonic = omega - omega_exact - omega_coexact

    # ---- Global energy summary ----
    # FIX S6-3: compute global fracs from component sum, not ||omega||^2,
    # to avoid lsqr numerical leakage in reported fractions.
    E_total_raw  = float(np.sum(omega ** 2))        # from original signal
    E_exact      = float(np.sum(omega_exact ** 2))
    E_coexact    = float(np.sum(omega_coexact ** 2))
    E_harmonic   = float(np.sum(omega_harmonic ** 2))
    E_component_sum = E_exact + E_coexact + E_harmonic  # consistent denominator

    # Warn if global leakage is non-trivial
    if E_total_raw > 0:
        leakage = abs(E_component_sum - E_total_raw) / E_total_raw
        if leakage > 1e-4:
            print(
                f"WARNING: global energy leakage = {leakage:.2e}. "
                f"E_total_raw={E_total_raw:.4e}, "
                f"E_component_sum={E_component_sum:.4e}. "
                "Using component sum for reported fractions."
            )

    if E_component_sum > 0:
        frac_exact    = E_exact    / E_component_sum
        frac_coexact  = E_coexact  / E_component_sum
        frac_harmonic = E_harmonic / E_component_sum
    else:
        frac_exact = frac_coexact = frac_harmonic = np.nan

    summary_df = pd.DataFrame([{
        "sample_id":         sample_id,
        "target_flux":       flux_col,
        "E_total_raw":       E_total_raw,        # ||omega||^2
        "E_component_sum":   E_component_sum,    # sum of component L2^2
        "E_exact":           E_exact,
        "E_coexact":         E_coexact,
        "E_harmonic":        E_harmonic,
        "frac_exact":        frac_exact,         # from component sum
        "frac_coexact":      frac_coexact,
        "frac_harmonic":     frac_harmonic,
    }])

    # ---- Edge-level output ----
    edges_out = edges_df.copy().reset_index(drop=True)
    edges_out["flux_exact"]    = omega_exact
    edges_out["flux_coexact"]  = omega_coexact
    edges_out["flux_harmonic"] = omega_harmonic

    # ---- Node-level output ----
    # Each node_energy_* = mean squared incident edge values for that component.
    node_total    = node_energy_from_edge_component(
        edges_df, node_ids, omega).rename(columns={"node_energy": "node_energy_total_raw"})
    node_exact    = node_energy_from_edge_component(
        edges_df, node_ids, omega_exact).rename(columns={"node_energy": "node_energy_exact"})
    node_coexact  = node_energy_from_edge_component(
        edges_df, node_ids, omega_coexact).rename(columns={"node_energy": "node_energy_coexact"})
    node_harmonic = node_energy_from_edge_component(
        edges_df, node_ids, omega_harmonic).rename(columns={"node_energy": "node_energy_harmonic"})

    nodes_out = nodes_df.copy()
    nodes_out = nodes_out.merge(node_total,    on="node_id", how="left", validate="one_to_one")
    nodes_out = nodes_out.merge(node_exact,    on="node_id", how="left", validate="one_to_one")
    nodes_out = nodes_out.merge(node_coexact,  on="node_id", how="left", validate="one_to_one")
    nodes_out = nodes_out.merge(node_harmonic, on="node_id", how="left", validate="one_to_one")

    # FIX S6-1 + S6-2: recompute node_energy_total as component sum.
    # This guarantees that fractions are in [0, 1] at every node and
    # that agg_frac_* computed downstream in Step 7 are always valid.
    # The original omega-derived energy is preserved as node_energy_total_raw.
    nodes_out["node_energy_total"] = (
        nodes_out["node_energy_exact"].fillna(0.0)
        + nodes_out["node_energy_coexact"].fillna(0.0)
        + nodes_out["node_energy_harmonic"].fillna(0.0)
    )

    # Per-node fractions — use component-sum total as denominator (FIX S6-2)
    denom = nodes_out["node_energy_total"].replace(0.0, np.nan)
    nodes_out["frac_exact"]    = nodes_out["node_energy_exact"]    / denom
    nodes_out["frac_coexact"]  = nodes_out["node_energy_coexact"]  / denom
    nodes_out["frac_harmonic"] = nodes_out["node_energy_harmonic"] / denom

    # Validate: warn if any frac column is outside [0, 1]
    for col in ["frac_exact", "frac_coexact", "frac_harmonic"]:
        bad = nodes_out[col].dropna()
        n_bad = int(((bad < -1e-6) | (bad > 1.0 + 1e-6)).sum())
        if n_bad > 0:
            print(
                f"WARNING: {n_bad} nodes have {col} outside [0,1] after fix. "
                "This indicates a deeper numerical issue in the projection."
            )

    # ---- Face-level curl output ----
    curl = face_curl_from_B2(B2, omega)
    if faces_df is not None and len(curl) == len(faces_df):
        faces_out = faces_df.copy().reset_index(drop=True)
        faces_out["curl"]     = curl
        faces_out["abs_curl"] = np.abs(curl)
    else:
        faces_out = pd.DataFrame({
            "face_id":  np.arange(len(curl)),
            "curl":     curl,
            "abs_curl": np.abs(curl),
        })

    # ---- Save ----
    out_edges   = stats_dir / f"{sample_id}_step6_edges_hodge_{flux_tag}.csv"
    out_nodes   = stats_dir / f"{sample_id}_step6_nodes_hodge_{flux_tag}.csv"
    out_faces   = stats_dir / f"{sample_id}_step6_face_curl_{flux_tag}.csv"
    out_summary = stats_dir / f"{sample_id}_step6_energy_summary_{flux_tag}.csv"

    edges_out.to_csv(out_edges,   index=False)
    nodes_out.to_csv(out_nodes,   index=False)
    faces_out.to_csv(out_faces,   index=False)
    summary_df.to_csv(out_summary, index=False)

    print(f"Saved edge components   -> {out_edges}")
    print(f"Saved node summaries    -> {out_nodes}")
    print(f"Saved face curl         -> {out_faces}")
    print(f"Saved energy summary    -> {out_summary}")

    print("\nEnergy summary:")
    print(summary_df.to_string(index=False))

    preview_cols = [
        c for c in ["tail", "head", flux_col, "flux_exact", "flux_coexact", "flux_harmonic"]
        if c in edges_out.columns
    ]
    print("\nEdge preview:")
    print(edges_out[preview_cols].head().to_string(index=False))


if __name__ == "__main__":
    main()
