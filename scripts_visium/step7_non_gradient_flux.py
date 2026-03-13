#!/usr/bin/env python3
"""
step7_non_gradient_flux.py

Robust pipeline to:
  - construct non-gradient antisymmetric edge fluxes on a Visium spatial graph
  - build triangles (2-simplices) -> B2
  - run ridge-stabilized Hodge decomposition (exact / coexact / harmonic)
  - compute region-level summaries + enrichment ratios
  - run Mann–Whitney U tests + label-shuffle permutation controls (default 1000)
  - write manuscript-ready CSVs and PNG maps/boxplots
  - save B2 and L1 matrices + metadata for reproducibility

Inputs expected in working directory:
  step3_nodes.csv
  step3_edges.csv
  step3_B1_incidence.npz
  step2_region_assignments.csv
  Visium_Human_Breast_Cancer_filtered_feature_bc_matrix.h5
  spatial/scalefactors_json.json
  spatial/tissue_hires_image.png

Outputs (required by spec):
  step7_edges_fluxes.csv
  step7_edges_hodge_{fluxname}.csv
  step7_nodes_hodge_{fluxname}.csv
  step7_region_summary_{fluxname}.csv
  step7_permutation_results_{fluxname}.csv
  step7_maps_{fluxname}.png
  step7_boxplots_{fluxname}.png

Additional helpful outputs:
  step7_region_comparisons_{fluxname}.csv
  step7_energy_fractions.csv
  step7_run_metadata.json
  step7_B2_faces.npz
  step7_L1_edge_hodge.npz
  step7_triangles.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd

import scanpy as sc
import matplotlib.pyplot as plt
from PIL import Image

from scipy import sparse
from scipy.sparse import linalg as spla
from scipy.stats import mannwhitneyu


# ============================================================
# Marker programs (edit if desired)
# ============================================================
TUMOR_GENES = ["EPCAM", "KRT8", "KRT18", "KRT19", "ERBB2", "MUC1", "TACSTD2"]
STROMA_GENES = ["COL1A1", "COL1A2", "DCN", "LUM", "POSTN", "FAP", "TAGLN"]
IMMUNE_GENES = ["PTPRC", "CD3D", "CD3E", "NKG7", "CD68", "C1QA", "CXCL9", "CXCL10"]

REGION_ORDER = ["tumor_core", "invasive_margin", "stroma", "immune_rich", "mixed_unassigned"]

DEFAULT_COMPARISONS = [
    ("tumor_core", "invasive_margin"),
    ("tumor_core", "immune_rich"),
    ("tumor_core", "stroma"),
    ("invasive_margin", "immune_rich"),
]

METRICS_FOR_TESTS = [
    "node_energy_total",
    "exact_fraction",
    "coexact_fraction",
    "harmonic_fraction",
]


# ============================================================
# Utilities
# ============================================================
def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def require_file(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p.resolve()}")


def safe_version(pkg_name: str) -> str:
    try:
        import importlib.metadata as md
        return md.version(pkg_name)
    except Exception:
        return "unknown"


def robust_zscore(x: np.ndarray, clip: float = 6.0) -> np.ndarray:
    """
    Robust z-score using median and MAD (scaled by 1.4826).
    Clipping prevents extreme products in interaction fluxes.
    """
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad < 1e-12:
        return np.zeros_like(x)
    z = (x - med) / (1.4826 * mad)
    z = np.clip(z, -clip, clip)
    z[~np.isfinite(z)] = 0.0
    return z


def flatten_agg_columns(df: pd.DataFrame, sep: str = "__") -> pd.DataFrame:
    """
    Flatten groupby-agg MultiIndex columns into stable strings.
    ('metric','median') -> 'metric__median'
    Avoids Pandas MultiIndex tuple-indexing pitfalls.
    """
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [f"{a}{sep}{b}" for (a, b) in out.columns.to_list()]
    else:
        out.columns = [str(c) for c in out.columns]
    return out


# ============================================================
# Loading inputs
# ============================================================
def load_graph(nodes_csv: Path, edges_csv: Path, b1_npz: Path) -> tuple[pd.DataFrame, pd.DataFrame, sparse.csr_matrix]:
    nodes = pd.read_csv(nodes_csv)
    edges = pd.read_csv(edges_csv)
    B1 = sparse.load_npz(b1_npz).tocsr()

    need_nodes = {"node_id", "barcode", "x_fullres", "y_fullres"}
    need_edges = {"edge_id", "tail", "head", "length"}

    if not need_nodes.issubset(nodes.columns):
        raise ValueError(f"{nodes_csv} missing columns: {sorted(need_nodes - set(nodes.columns))}")
    if not need_edges.issubset(edges.columns):
        raise ValueError(f"{edges_csv} missing columns: {sorted(need_edges - set(edges.columns))}")

    nodes = nodes.sort_values("node_id").reset_index(drop=True)
    edges = edges.sort_values("edge_id").reset_index(drop=True)

    n_nodes = nodes.shape[0]
    n_edges = edges.shape[0]

    if not np.array_equal(nodes["node_id"].values, np.arange(n_nodes)):
        raise ValueError("step3_nodes.csv node_id must be 0..n-1 after sorting.")
    if not np.array_equal(edges["edge_id"].values, np.arange(n_edges)):
        raise ValueError("step3_edges.csv edge_id must be 0..m-1 after sorting.")
    if B1.shape != (n_nodes, n_edges):
        raise ValueError(f"B1 shape {B1.shape} != ({n_nodes},{n_edges})")

    return nodes, edges, B1


def load_regions(region_csv: Path, barcodes: np.ndarray) -> np.ndarray:
    df = pd.read_csv(region_csv)
    if "barcode" not in df.columns or "region_step2" not in df.columns:
        raise ValueError("step2_region_assignments.csv must contain columns: barcode, region_step2")
    df = df.set_index("barcode")

    missing = [b for b in barcodes if b not in df.index]
    if missing:
        raise ValueError(f"{len(missing)} barcodes from graph not found in region CSV (first 5): {missing[:5]}")

    regions = df.loc[barcodes, "region_step2"].astype(str).values
    regions = np.where(regions == "other", "mixed_unassigned", regions)
    return regions


def load_spatial_image(spatial_dir: Path) -> tuple[np.ndarray, float]:
    sf = spatial_dir / "scalefactors_json.json"
    imgp = spatial_dir / "tissue_hires_image.png"
    require_file(sf)
    require_file(imgp)
    with open(sf, "r") as f:
        scalef = json.load(f)
    scale = float(scalef.get("tissue_hires_scalef", 1.0))
    img = np.asarray(Image.open(imgp))
    return img, scale


def compute_scores(h5_file: Path, barcodes: np.ndarray, seed: int) -> pd.DataFrame:
    """
    Compute tumor/stroma/immune scores in barcode order.
    Uses Scanpy normalize_total + log1p + tl.score_genes(random_state=seed).
    """
    require_file(h5_file)

    adata = sc.read_10x_h5(str(h5_file))
    adata.var_names_make_unique()  # reduce duplicate-name issues downstream

    missing = [b for b in barcodes if b not in adata.obs_names]
    if missing:
        raise ValueError(f"{len(missing)} graph barcodes not found in H5 (first 5): {missing[:5]}")

    adata = adata[barcodes].copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    def present(gl):
        return [g for g in gl if g in adata.var_names]

    tumor = present(TUMOR_GENES)
    stroma = present(STROMA_GENES)
    immune = present(IMMUNE_GENES)

    if tumor:
        sc.tl.score_genes(adata, tumor, score_name="tumor_score", random_state=seed, use_raw=False)
    else:
        adata.obs["tumor_score"] = 0.0

    if stroma:
        sc.tl.score_genes(adata, stroma, score_name="stroma_score", random_state=seed, use_raw=False)
    else:
        adata.obs["stroma_score"] = 0.0

    if immune:
        sc.tl.score_genes(adata, immune, score_name="immune_score", random_state=seed, use_raw=False)
    else:
        adata.obs["immune_score"] = 0.0

    out = adata.obs[["tumor_score", "stroma_score", "immune_score"]].copy()
    out.index.name = "barcode"
    return out


# ============================================================
# Build triangles and B2
# ============================================================
def build_edge_lookup_and_neighbors(n_nodes: int, edges: pd.DataFrame) -> tuple[dict[tuple[int, int], int], list[set[int]]]:
    edge_lookup: dict[tuple[int, int], int] = {}
    neighbors: list[set[int]] = [set() for _ in range(n_nodes)]

    tail = edges["tail"].to_numpy(dtype=int)
    head = edges["head"].to_numpy(dtype=int)

    for eid, (i, j) in enumerate(zip(tail, head)):
        a, b = (i, j) if i < j else (j, i)
        edge_lookup[(a, b)] = int(eid)
        neighbors[a].add(b)
        neighbors[b].add(a)

    return edge_lookup, neighbors


def enumerate_triangles(edge_lookup: dict[tuple[int, int], int], neighbors: list[set[int]]) -> list[tuple[int, int, int]]:
    """
    Enumerate triangles (i<j<k) via intersection-based enumeration.
    Efficient for low-degree Visium graphs.
    """
    tris: list[tuple[int, int, int]] = []
    n = len(neighbors)
    for i in range(n):
        nbrs_i = sorted([j for j in neighbors[i] if j > i])
        nbrs_i_set = set(nbrs_i)
        for j in nbrs_i:
            common = nbrs_i_set.intersection(neighbors[j])
            for k in common:
                if k <= j:
                    continue
                if (i, j) in edge_lookup and (i, k) in edge_lookup and (j, k) in edge_lookup:
                    tris.append((i, j, k))
    tris.sort()
    return tris


def build_B2(n_edges: int, triangles: list[tuple[int, int, int]], edge_lookup: dict[tuple[int, int], int]) -> sparse.csr_matrix:
    """
    Orientation convention for triangle (i<j<k):
      boundary = +e_ij - e_ik + e_jk
    where edges are oriented from smaller->larger node id.
    """
    n_faces = len(triangles)
    rows, cols, data = [], [], []

    for fid, (i, j, k) in enumerate(triangles):
        e_ij = edge_lookup[(i, j)]
        e_ik = edge_lookup[(i, k)]
        e_jk = edge_lookup[(j, k)]
        rows.extend([e_ij, e_ik, e_jk])
        cols.extend([fid, fid, fid])
        data.extend([+1.0, -1.0, +1.0])

    return sparse.coo_matrix((data, (rows, cols)), shape=(n_edges, n_faces)).tocsr()


# ============================================================
# Flux constructions
# ============================================================
def wedge_flux(a: np.ndarray, b: np.ndarray, edges: pd.DataFrame) -> np.ndarray:
    """
    f_ij = (a_i b_j - a_j b_i) / length_ij, for oriented edge tail=i -> head=j.
    """
    tail = edges["tail"].to_numpy(dtype=int)
    head = edges["head"].to_numpy(dtype=int)
    length = edges["length"].to_numpy(dtype=float)

    ai, aj = a[tail], a[head]
    bi, bj = b[tail], b[head]
    return (ai * bj - aj * bi) / np.maximum(length, 1e-12)


def local_gradients(values: np.ndarray, coords: np.ndarray, neighbors: list[set[int]], ridge: float = 1e-3) -> np.ndarray:
    """
    Estimate gradient at node i by least squares:
      dv ≈ gx*dx + gy*dy over neighbors
    Returns (n_nodes, 2).
    """
    n = coords.shape[0]
    grads = np.zeros((n, 2), dtype=float)

    for i in range(n):
        nbrs = list(neighbors[i])
        if len(nbrs) < 2:
            continue

        xi, yi = coords[i]
        vi = values[i]
        dx, dy, dv = [], [], []

        for j in nbrs:
            xj, yj = coords[j]
            dx.append(xj - xi)
            dy.append(yj - yi)
            dv.append(values[j] - vi)

        dx = np.asarray(dx, float)
        dy = np.asarray(dy, float)
        dv = np.asarray(dv, float)

        Sxx = float(dx @ dx) + ridge
        Sxy = float(dx @ dy)
        Syy = float(dy @ dy) + ridge
        bx = float(dx @ dv)
        by = float(dy @ dv)

        det = Sxx * Syy - Sxy * Sxy
        if abs(det) < 1e-18:
            continue

        gx = (Syy * bx - Sxy * by) / det
        gy = (-Sxy * bx + Sxx * by) / det
        grads[i] = (gx, gy)

    return grads


def vector_interaction_flux(a: np.ndarray, b: np.ndarray, coords: np.ndarray, neighbors: list[set[int]], edges: pd.DataFrame, grad_ridge: float = 1e-3) -> np.ndarray:
    """
    Optional alternative:
      v_i = a_i R90(∇b_i) - b_i R90(∇a_i),
      f_ij = ((v_i + v_j)/2) · (x_j - x_i)/||x_j-x_i||
    """
    ga = local_gradients(a, coords, neighbors, ridge=grad_ridge)
    gb = local_gradients(b, coords, neighbors, ridge=grad_ridge)

    rot_gb = np.column_stack([-gb[:, 1], gb[:, 0]])  # R90
    rot_ga = np.column_stack([-ga[:, 1], ga[:, 0]])

    v = a[:, None] * rot_gb - b[:, None] * rot_ga

    tail = edges["tail"].to_numpy(dtype=int)
    head = edges["head"].to_numpy(dtype=int)
    length = edges["length"].to_numpy(dtype=float)

    d = coords[head] - coords[tail]
    unit = d / np.maximum(length[:, None], 1e-12)

    v_edge = 0.5 * (v[tail] + v[head])
    return np.sum(v_edge * unit, axis=1)


# ============================================================
# Solvers and decomposition
# ============================================================
@dataclass
class SolveInfo:
    method: str
    ok: bool
    info: int | None
    iters: int | None


def cg_compat(A, b, M=None, rtol=1e-10, atol=0.0, maxiter=5000, callback=None):
    """
    Compatibility wrapper for SciPy cg():
    newer SciPy uses (rtol, atol); older uses (tol).
    """
    try:
        return spla.cg(A, b, M=M, rtol=rtol, atol=atol, maxiter=maxiter, callback=callback)
    except TypeError:
        return spla.cg(A, b, M=M, tol=rtol, maxiter=maxiter, callback=callback)


def jacobi_preconditioner(A: sparse.spmatrix) -> spla.LinearOperator:
    d = A.diagonal()
    d = np.where(np.abs(d) < 1e-12, 1.0, d)
    inv = 1.0 / d
    return spla.LinearOperator(A.shape, matvec=lambda x: inv * x, dtype=float)


def prepare_solvers(B1: sparse.csr_matrix, B2: sparse.csr_matrix, ridge: float, solver: str, cg_rtol: float, cg_maxiter: int):
    """
    Prepare reusable solvers for:
      A_exact = B1 B1^T + ridge I      (n_nodes x n_nodes)
      A_co    = B2^T B2 + ridge I      (n_faces x n_faces)
    """
    n_nodes = B1.shape[0]
    n_faces = B2.shape[1]

    t0 = time.perf_counter()
    A_exact = (B1 @ B1.T).tocsr() + ridge * sparse.eye(n_nodes, format="csr")
    A_co = (B2.T @ B2).tocsr() + ridge * sparse.eye(n_faces, format="csr")
    t1 = time.perf_counter()

    prep = {"time_build_systems_sec": float(t1 - t0), "solver_requested": solver}

    if solver == "factorized":
        try:
            t2 = time.perf_counter()
            se = spla.factorized(A_exact.tocsc())
            sc_ = spla.factorized(A_co.tocsc())
            t3 = time.perf_counter()
            prep["time_factorized_sec"] = float(t3 - t2)
            prep["solver_used"] = "factorized"

            def solve_exact(rhs):
                return se(np.asarray(rhs, float)), SolveInfo("factorized", True, None, None)

            def solve_co(rhs):
                return sc_(np.asarray(rhs, float)), SolveInfo("factorized", True, None, None)

            return solve_exact, solve_co, A_exact, A_co, prep
        except Exception as e:
            print("[WARN] factorized() failed; falling back to CG. Error:", repr(e))
            solver = "cg"

    if solver == "spsolve":
        prep["solver_used"] = "spsolve"

        def solve_exact(rhs):
            x = spla.spsolve(A_exact.tocsc(), np.asarray(rhs, float))
            return x, SolveInfo("spsolve", True, None, None)

        def solve_co(rhs):
            x = spla.spsolve(A_co.tocsc(), np.asarray(rhs, float))
            return x, SolveInfo("spsolve", True, None, None)

        return solve_exact, solve_co, A_exact, A_co, prep

    if solver == "cg":
        prep["solver_used"] = "cg"
        M_exact = jacobi_preconditioner(A_exact)
        M_co = jacobi_preconditioner(A_co)

        def solve_cg(A, M, rhs):
            it = 0

            def cb(_):
                nonlocal it
                it += 1

            x, info = cg_compat(A, np.asarray(rhs, float), M=M, rtol=cg_rtol, atol=0.0, maxiter=cg_maxiter, callback=cb)
            ok = (info == 0)
            return x, SolveInfo("cg", ok, int(info), int(it))

        def solve_exact(rhs):
            return solve_cg(A_exact, M_exact, rhs)

        def solve_co(rhs):
            return solve_cg(A_co, M_co, rhs)

        return solve_exact, solve_co, A_exact, A_co, prep

    raise ValueError(f"Unknown solver: {solver}")


def hodge_decompose(f: np.ndarray, B1: sparse.csr_matrix, B2: sparse.csr_matrix, solve_exact, solve_co):
    """
    Decompose f into exact/coexact/harmonic components using stabilized solves.
    Returns:
      f_exact, f_coexact, f_harmonic,
      energies dict, solve_info_exact, solve_info_coexact,
      diagnostics dict (mean |div| and mean |curl| per component)
    """
    rhs_e = B1 @ f
    alpha, info_e = solve_exact(rhs_e)
    f_exact = B1.T @ alpha

    rhs_c = B2.T @ (f - f_exact)
    beta, info_c = solve_co(rhs_c)
    f_co = B2 @ beta

    f_harm = f - f_exact - f_co

    def nsq(x): return float(np.dot(x, x))
    E_total = nsq(f)
    E_exact = nsq(f_exact)
    E_co = nsq(f_co)
    E_harm = nsq(f_harm)

    energies = {
        "E_total": E_total,
        "E_exact": E_exact,
        "E_coexact": E_co,
        "E_harmonic": E_harm,
        "frac_exact": (E_exact / E_total) if E_total > 0 else np.nan,
        "frac_coexact": (E_co / E_total) if E_total > 0 else np.nan,
        "frac_harmonic": (E_harm / E_total) if E_total > 0 else np.nan,
    }

    div_total = B1 @ f
    div_exact = B1 @ f_exact
    div_co = B1 @ f_co
    div_harm = B1 @ f_harm

    curl_total = B2.T @ f
    curl_exact = B2.T @ f_exact
    curl_co = B2.T @ f_co
    curl_harm = B2.T @ f_harm

    diag = {
        "mean_abs_div_total": float(np.mean(np.abs(div_total))),
        "mean_abs_div_exact": float(np.mean(np.abs(div_exact))),
        "mean_abs_div_coexact": float(np.mean(np.abs(div_co))),
        "mean_abs_div_harmonic": float(np.mean(np.abs(div_harm))),
        "mean_abs_curl_total": float(np.mean(np.abs(curl_total))),
        "mean_abs_curl_exact": float(np.mean(np.abs(curl_exact))),
        "mean_abs_curl_coexact": float(np.mean(np.abs(curl_co))),
        "mean_abs_curl_harmonic": float(np.mean(np.abs(curl_harm))),
    }

    return f_exact, f_co, f_harm, energies, info_e, info_c, diag


# ============================================================
# Region stats + permutation controls
# ============================================================
def mwu_table(values: np.ndarray, regions: np.ndarray, comparisons: list[tuple[str, str]], flux: str, metric: str) -> pd.DataFrame:
    rows = []
    for ra, rb in comparisons:
        xa = values[regions == ra]
        xb = values[regions == rb]
        if xa.size < 10 or xb.size < 10:
            continue

        med_a = float(np.median(xa))
        med_b = float(np.median(xb))
        diff = med_a - med_b
        ratio = (med_a / med_b) if med_b != 0 else np.nan

        mwu = mannwhitneyu(xa, xb, alternative="two-sided", method="auto")

        rows.append({
            "flux": flux,
            "metric": metric,
            "region_a": ra,
            "region_b": rb,
            "n_a": int(xa.size),
            "n_b": int(xb.size),
            "median_a": med_a,
            "median_b": med_b,
            "median_diff_a_minus_b": diff,
            "median_ratio_a_over_b": ratio,
            "mwu_u_stat": float(mwu.statistic),
            "mwu_p_value": float(mwu.pvalue),
        })
    return pd.DataFrame(rows)


def permutation_median_diff(values: np.ndarray, regions: np.ndarray, comparisons: list[tuple[str, str]], n_perm: int, seed: int, flux: str, metric: str) -> pd.DataFrame:
    """
    Label-shuffle null via value shuffling across fixed region indices.
    Statistic: absolute median difference.
    """
    rng = np.random.default_rng(seed)
    n = values.size

    idx = {r: np.where(regions == r)[0] for r in np.unique(regions)}
    regs_needed = sorted({r for ab in comparisons for r in ab if r in idx})

    obs_med = {r: float(np.median(values[idx[r]])) for r in regs_needed}
    obs_abs = {(ra, rb): abs(obs_med[ra] - obs_med[rb]) for (ra, rb) in comparisons if ra in idx and rb in idx}
    ge = {(ra, rb): 0 for (ra, rb) in obs_abs.keys()}

    t0 = time.perf_counter()
    for _ in range(n_perm):
        vp = values[rng.permutation(n)]
        medp = {r: float(np.median(vp[idx[r]])) for r in regs_needed}
        for (ra, rb), thr in obs_abs.items():
            if abs(medp[ra] - medp[rb]) >= thr:
                ge[(ra, rb)] += 1
    t1 = time.perf_counter()

    rows = []
    for (ra, rb), thr in obs_abs.items():
        p = (ge[(ra, rb)] + 1) / (n_perm + 1)  # add-one smoothing
        rows.append({
            "flux": flux,
            "metric": metric,
            "region_a": ra,
            "region_b": rb,
            "observed_abs_median_diff": float(thr),
            "n_perm": int(n_perm),
            "n_ge": int(ge[(ra, rb)]),
            "perm_p_value_two_sided": float(p),
            "seed": int(seed),
            "perm_time_sec": float(t1 - t0),
        })
    return pd.DataFrame(rows)


# ============================================================
# Plotting
# ============================================================
def plot_maps(node_df: pd.DataFrame, tissue_img: np.ndarray, scale: float, fluxname: str, out_png: Path):
    x = node_df["x_fullres"].to_numpy(float) * scale
    y = node_df["y_fullres"].to_numpy(float) * scale

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    items = [
        ("node_energy_total", "Total node energy"),
        ("exact_fraction", "Exact fraction"),
        ("coexact_fraction", "Coexact fraction"),
        ("harmonic_fraction", "Harmonic fraction"),
    ]
    for ax, (col, title) in zip(axes.ravel(), items):
        ax.imshow(tissue_img)
        sca = ax.scatter(x, y, s=18, c=node_df[col].to_numpy(), alpha=0.95)
        ax.set_title(title)
        ax.invert_yaxis()
        ax.axis("off")
        plt.colorbar(sca, ax=ax, fraction=0.03, pad=0.02)

    plt.suptitle(f"Hodge maps — {fluxname}", y=0.98)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_boxplots(node_df: pd.DataFrame, fluxname: str, out_png: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    items = [
        ("node_energy_total", "Total node energy"),
        ("exact_fraction", "Exact fraction"),
        ("coexact_fraction", "Coexact fraction"),
        ("harmonic_fraction", "Harmonic fraction"),
    ]
    for ax, (col, title) in zip(axes.ravel(), items):
        data = [node_df.loc[node_df["region"] == r, col].values for r in REGION_ORDER]
        ax.boxplot(data, tick_labels=REGION_ORDER, showfliers=False)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)

    plt.suptitle(f"Region boxplots — {fluxname}", y=0.98)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


# ============================================================
# Main
# ============================================================
def main():
    warnings.filterwarnings("ignore", message="Variable names are not unique.*", category=UserWarning)

    ap = argparse.ArgumentParser(description="Non-gradient fluxes + Hodge decomposition + region stats + permutation controls")
    ap.add_argument("--perm", type=int, default=1000, help="Permutation count (default: 1000)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    ap.add_argument("--ridge", type=float, default=1e-6, help="Ridge stabilizer λ (default: 1e-6)")
    ap.add_argument("--solver", type=str, default="factorized", choices=["factorized", "cg", "spsolve"], help="Linear solver (default: factorized)")
    ap.add_argument("--cg-rtol", type=float, default=1e-10, help="CG rtol (if solver=cg)")
    ap.add_argument("--cg-maxiter", type=int, default=5000, help="CG maxiter (if solver=cg)")
    ap.add_argument("--include-vector-flux", action="store_true", help="Also compute vector-projection flux variants")
    ap.add_argument("--grad-ridge", type=float, default=1e-3, help="Ridge for local gradient estimation (vector flux)")
    ap.add_argument("--outdir", type=str, default=".", help="Output directory (default: current)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Inputs
    nodes_csv = Path("step3_nodes.csv")
    edges_csv = Path("step3_edges.csv")
    b1_npz = Path("step3_B1_incidence.npz")
    region_csv = Path("step2_region_assignments.csv")
    h5_file = Path("Visium_Human_Breast_Cancer_filtered_feature_bc_matrix.h5")
    spatial_dir = Path("spatial")

    for p in [nodes_csv, edges_csv, b1_npz, region_csv, h5_file]:
        require_file(p)
    require_file(spatial_dir / "scalefactors_json.json")
    require_file(spatial_dir / "tissue_hires_image.png")

    # Metadata
    meta = {
        "run_started": now_iso(),
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            "numpy": safe_version("numpy"),
            "pandas": safe_version("pandas"),
            "scipy": safe_version("scipy"),
            "scanpy": safe_version("scanpy"),
            "matplotlib": safe_version("matplotlib"),
            "pillow": safe_version("Pillow"),
        },
        "args": vars(args),
    }

    print("\n[Run] Loading graph inputs...")
    nodes, edges, B1 = load_graph(nodes_csv, edges_csv, b1_npz)
    n_nodes, n_edges = nodes.shape[0], edges.shape[0]
    print(f"Nodes={n_nodes}, Edges={n_edges}, B1={B1.shape}")

    print("\n[Run] Loading tissue image + scalefactors...")
    tissue_img, scale = load_spatial_image(spatial_dir)
    print(f"Hires image shape={tissue_img.shape}, hires_scale={scale}")

    print("\n[Run] Building triangles and B2...")
    edge_lookup, neighbors = build_edge_lookup_and_neighbors(n_nodes, edges)
    triangles = enumerate_triangles(edge_lookup, neighbors)
    if len(triangles) == 0:
        raise RuntimeError("No triangles detected. Cannot build B2 or coexact component.")
    B2 = build_B2(n_edges, triangles, edge_lookup)
    print(f"Triangles={len(triangles)}, B2 shape={B2.shape}, nnz(B2)={B2.nnz}")

    # Save B2/triangles + L1
    sparse.save_npz(outdir / "step7_B2_faces.npz", B2)
    tri_df = pd.DataFrame(triangles, columns=["i", "j", "k"])
    tri_df["triangle_id"] = np.arange(len(triangles), dtype=int)
    tri_df.to_csv(outdir / "step7_triangles.csv", index=False)

    print("[Run] Computing and saving L1 (edge Hodge Laplacian)...")
    tL0 = time.perf_counter()
    L1 = (B1.T @ B1) + (B2 @ B2.T)
    sparse.save_npz(outdir / "step7_L1_edge_hodge.npz", L1)
    tL1 = time.perf_counter()
    print(f"L1 saved. Time={tL1 - tL0:.2f} sec")

    meta["graph"] = {
        "n_nodes": int(n_nodes),
        "n_edges": int(n_edges),
        "n_faces": int(len(triangles)),
        "B2_nnz": int(B2.nnz),
        "L1_nnz": int(L1.nnz),
        "time_L1_sec": float(tL1 - tL0),
    }

    print("\n[Run] Loading region assignments...")
    barcodes = nodes["barcode"].astype(str).values
    regions = load_regions(region_csv, barcodes)
    print(pd.Series(regions).value_counts())

    print("\n[Run] Computing program scores (tumor/stroma/immune)...")
    tS0 = time.perf_counter()
    scores = compute_scores(h5_file, barcodes, seed=args.seed)
    tS1 = time.perf_counter()
    print(f"Scoring done in {tS1 - tS0:.2f} sec")
    meta["scoring_time_sec"] = float(tS1 - tS0)

    # Robust-scaled scores for interaction fluxes
    I = robust_zscore(scores["immune_score"].to_numpy())
    T = robust_zscore(scores["tumor_score"].to_numpy())
    S = robust_zscore(scores["stroma_score"].to_numpy())
    coords = nodes[["x_fullres", "y_fullres"]].to_numpy(float)

    print("\n[Run] Constructing non-gradient antisymmetric fluxes...")
    fluxes: dict[str, np.ndarray] = {
        "immune_tumor_wedge": wedge_flux(I, T, edges),
        "stroma_tumor_wedge": wedge_flux(S, T, edges),
        "immune_stroma_wedge": wedge_flux(I, S, edges),
    }
    if args.include_vector_flux:
        fluxes.update({
            "immune_tumor_vec": vector_interaction_flux(I, T, coords, neighbors, edges, grad_ridge=args.grad_ridge),
            "stroma_tumor_vec": vector_interaction_flux(S, T, coords, neighbors, edges, grad_ridge=args.grad_ridge),
            "immune_stroma_vec": vector_interaction_flux(I, S, coords, neighbors, edges, grad_ridge=args.grad_ridge),
        })

    # Save step7_edges_fluxes.csv
    flux_table = edges[["edge_id", "tail", "head", "length"]].copy()
    for name, f in fluxes.items():
        flux_table[name] = f
    flux_table.to_csv(outdir / "step7_edges_fluxes.csv", index=False)
    print(f"Saved: {outdir / 'step7_edges_fluxes.csv'}")

    print("\n[Run] Preparing reusable solvers...")
    solve_exact, solve_co, A_exact, A_co, prep = prepare_solvers(
        B1=B1, B2=B2, ridge=args.ridge, solver=args.solver, cg_rtol=args.cg_rtol, cg_maxiter=args.cg_maxiter
    )
    meta["solver_prep"] = prep
    meta["ridge"] = float(args.ridge)
    print("Solver prep:", prep)

    # For mapping edge energies to nodes
    absB1 = np.abs(B1).tocsr()
    deg = absB1 @ np.ones(n_edges, float)
    deg = np.maximum(deg, 1.0)

    # Global energy fraction summary across fluxes
    energy_rows = []

    for fluxname, f in fluxes.items():
        print("\n============================================================")
        print(f"[Run] Hodge decomposition for: {fluxname}")
        print(f"Flux stats: mean={np.mean(f):.3e}, std={np.std(f):.3e}, min={np.min(f):.3e}, max={np.max(f):.3e}")

        t0 = time.perf_counter()
        f_exact, f_co, f_harm, energies, info_e, info_c, diag = hodge_decompose(
            f=f, B1=B1, B2=B2, solve_exact=solve_exact, solve_co=solve_co
        )
        t1 = time.perf_counter()

        energies["time_decompose_sec"] = float(t1 - t0)
        energies["solver_exact_method"] = info_e.method
        energies["solver_exact_ok"] = info_e.ok
        energies["solver_exact_info"] = info_e.info
        energies["solver_exact_iters"] = info_e.iters
        energies["solver_coexact_method"] = info_c.method
        energies["solver_coexact_ok"] = info_c.ok
        energies["solver_coexact_info"] = info_c.info
        energies["solver_coexact_iters"] = info_c.iters

        print("Energy fractions:",
              f"exact={energies['frac_exact']:.6f}, coexact={energies['frac_coexact']:.6f}, harmonic={energies['frac_harmonic']:.3e}")
        print("Mean |div| / |curl| diagnostics:", diag)
        print("Solver exact:", info_e)
        print("Solver coexact:", info_c)
        print(f"Timing: decompose={t1 - t0:.3f} sec")

        energy_rows.append({
            "flux": fluxname,
            "frac_exact": energies["frac_exact"],
            "frac_coexact": energies["frac_coexact"],
            "frac_harmonic": energies["frac_harmonic"],
            "E_total": energies["E_total"],
            "time_decompose_sec": energies["time_decompose_sec"],
        })

        # Edge outputs (required)
        edge_out = edges[["edge_id", "tail", "head", "length"]].copy()
        edge_out["flux_total"] = f
        edge_out["flux_exact"] = f_exact
        edge_out["flux_coexact"] = f_co
        edge_out["flux_harmonic"] = f_harm
        edge_out.to_csv(outdir / f"step7_edges_hodge_{fluxname}.csv", index=False)

        # Node mapping: component energies (degree-normalized incident squared flux)
        node_energy_exact = (absB1 @ (f_exact * f_exact)) / deg
        node_energy_co = (absB1 @ (f_co * f_co)) / deg
        node_energy_harm = (absB1 @ (f_harm * f_harm)) / deg
        node_energy_total = node_energy_exact + node_energy_co + node_energy_harm

        exact_frac = node_energy_exact / np.maximum(node_energy_total, 1e-18)
        co_frac = node_energy_co / np.maximum(node_energy_total, 1e-18)
        harm_frac = node_energy_harm / np.maximum(node_energy_total, 1e-18)

        # Node outputs (required)
        node_out = pd.DataFrame({
            "node_id": nodes["node_id"].values,
            "barcode": nodes["barcode"].values,
            "x_fullres": nodes["x_fullres"].values,
            "y_fullres": nodes["y_fullres"].values,
            "array_row": nodes.get("array_row", pd.Series([np.nan] * n_nodes)).values,
            "array_col": nodes.get("array_col", pd.Series([np.nan] * n_nodes)).values,
            "region": regions,
            "tumor_score": scores["tumor_score"].values,
            "stroma_score": scores["stroma_score"].values,
            "immune_score": scores["immune_score"].values,
            "node_energy_total": node_energy_total,
            "node_energy_exact": node_energy_exact,
            "node_energy_coexact": node_energy_co,
            "node_energy_harmonic": node_energy_harm,
            "exact_fraction": exact_frac,
            "coexact_fraction": co_frac,
            "harmonic_fraction": harm_frac,
        })
        node_out.to_csv(outdir / f"step7_nodes_hodge_{fluxname}.csv", index=False)

        # Region summaries (required)
        reg_sum = node_out.groupby("region")[METRICS_FOR_TESTS].agg(["count", "median", "mean", "std"])
        reg_sum = flatten_agg_columns(reg_sum)
        reg_sum = reg_sum.reindex([r for r in REGION_ORDER if r in reg_sum.index])

        # Add median ratios vs tumor_core
        if "tumor_core" in reg_sum.index:
            for m in METRICS_FOR_TESTS:
                base = float(reg_sum.loc["tumor_core", f"{m}__median"])
                for r in reg_sum.index:
                    val = float(reg_sum.loc[r, f"{m}__median"])
                    reg_sum.loc[r, f"{m}__median_ratio_vs_tumor_core"] = (val / base) if base > 0 else np.nan

        reg_sum.to_csv(outdir / f"step7_region_summary_{fluxname}.csv")

        # MWU + permutation controls (requested tables)
        comp_tables = []
        perm_tables = []
        for m in METRICS_FOR_TESTS:
            vals = node_out[m].to_numpy(float)
            comp_tables.append(mwu_table(vals, regions, DEFAULT_COMPARISONS, flux=fluxname, metric=m))
            perm_tables.append(permutation_median_diff(vals, regions, DEFAULT_COMPARISONS, args.perm, args.seed, flux=fluxname, metric=m))

        comp_df = pd.concat(comp_tables, axis=0, ignore_index=True) if comp_tables else pd.DataFrame()
        perm_df = pd.concat(perm_tables, axis=0, ignore_index=True) if perm_tables else pd.DataFrame()

        comp_df.to_csv(outdir / f"step7_region_comparisons_{fluxname}.csv", index=False)
        perm_df.to_csv(outdir / f"step7_permutation_results_{fluxname}.csv", index=False)

        # Figures (required)
        plot_maps(node_out, tissue_img, scale, fluxname, outdir / f"step7_maps_{fluxname}.png")
        plot_boxplots(node_out, fluxname, outdir / f"step7_boxplots_{fluxname}.png")

        meta.setdefault("per_flux", {})[fluxname] = {"energies": energies, "diagnostics": diag}

    # Save global energy fractions (recommended)
    energy_df = pd.DataFrame(energy_rows)
    energy_df.to_csv(outdir / "step7_energy_fractions.csv", index=False)

    meta["run_finished"] = now_iso()
    with open(outdir / "step7_run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n[Run] Saved: step7_run_metadata.json and step7_energy_fractions.csv")
    print("[Run] Done.")


if __name__ == "__main__":
    main()
