from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.stats import mannwhitneyu


REGION_ORDER = [
    "tumor_enriched",
    "stroma_enriched",
    "immune_enriched",
    "interface_like",
    "other",
]


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def project_exact(B1: sparse.csr_matrix, f: np.ndarray, ridge: float = 1e-6):
    n_nodes = B1.shape[0]
    A = (B1 @ B1.T).tocsr() + ridge * sparse.eye(n_nodes, format="csr")
    rhs = B1 @ f
    alpha = spsolve(A, rhs)
    f_exact = B1.T @ alpha
    return np.asarray(alpha), np.asarray(f_exact).ravel()


def project_coexact(B2: sparse.csr_matrix, f_resid: np.ndarray, ridge: float = 1e-6):
    n_faces = B2.shape[1]
    if n_faces == 0:
        return np.zeros(0, dtype=float), np.zeros_like(f_resid)

    A = (B2.T @ B2).tocsr() + ridge * sparse.eye(n_faces, format="csr")
    rhs = B2.T @ f_resid
    beta = spsolve(A, rhs)
    f_coexact = B2 @ beta
    return np.asarray(beta), np.asarray(f_coexact).ravel()


def node_mean_incident_energy(B1_abs: sparse.csr_matrix, edge_component: np.ndarray) -> np.ndarray:
    deg = np.asarray(B1_abs @ np.ones_like(edge_component)).ravel()
    deg = np.maximum(deg, 1.0)
    node_energy = np.asarray(B1_abs @ (edge_component ** 2)).ravel() / deg
    return node_energy


def face_mapped_energy(B2_abs: sparse.csr_matrix, edge_component: np.ndarray) -> np.ndarray:
    if B2_abs.shape[1] == 0:
        return np.zeros(0, dtype=float)
    counts = np.asarray(B2_abs.T @ np.ones_like(edge_component)).ravel()
    counts = np.maximum(counts, 1.0)
    face_energy = np.asarray(B2_abs.T @ (edge_component ** 2)).ravel() / counts
    return face_energy


def safe_mean(x):
    return float(np.mean(x)) if len(x) else np.nan


def safe_median(x):
    return float(np.median(x)) if len(x) else np.nan


def compute_ratio(a, b, eps=1e-18):
    return safe_median(a) / max(safe_median(b), eps)


def mwu_pvalue(x, y, alternative="greater"):
    if len(x) == 0 or len(y) == 0:
        return np.nan, np.nan
    stat, p = mannwhitneyu(x, y, alternative=alternative)
    return float(stat), float(p)


def permutation_pvalue(x, y, n_perm=1000, alternative="greater", seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    pooled = np.concatenate([x, y])
    n_x = len(x)
    obs = np.median(x) - np.median(y)

    perm_stats = np.empty(n_perm, dtype=float)
    for k in range(n_perm):
        perm = rng.permutation(len(pooled))
        px = pooled[perm[:n_x]]
        py = pooled[perm[n_x:]]
        perm_stats[k] = np.median(px) - np.median(py)

    if alternative == "greater":
        p = (1.0 + np.sum(perm_stats >= obs)) / (n_perm + 1.0)
    elif alternative == "less":
        p = (1.0 + np.sum(perm_stats <= obs)) / (n_perm + 1.0)
    else:
        p = (1.0 + np.sum(np.abs(perm_stats) >= abs(obs))) / (n_perm + 1.0)

    return float(obs), float(p)


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


def save_region_boxplots(node_df: pd.DataFrame, sample_id: str, target_flux: str, outpath: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    specs = [
        ("node_energy_coexact", "Absolute coexact energy by region"),
        ("frac_coexact", "Coexact fraction by region"),
    ]

    for ax, (col, title) in zip(axes, specs):
        data = [
            node_df.loc[node_df["region_step2"] == reg, col].to_numpy(dtype=float)
            for reg in REGION_ORDER
        ]
        ax.boxplot(data, tick_labels=REGION_ORDER, showfliers=False)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)

    plt.suptitle(f"{sample_id}: GNN flux region diagnostics — {target_flux}", y=0.98)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 15 TNBC: downstream Hodge/curl/interface analysis on learned GNN flux."
    )
    parser.add_argument("--sample_id", required=True)
    parser.add_argument(
        "--target_flux",
        required=True,
        choices=["flux_tumor_immune", "flux_tumor_stroma", "flux_immune_stroma"],
    )
    parser.add_argument("--statsdir", default="stats")
    parser.add_argument("--figdir", default="visium_figures")
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--n_perm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hotspot_q", type=float, default=0.95)
    args = parser.parse_args()

    sample_id = args.sample_id
    target_flux = args.target_flux
    statsdir = Path(args.statsdir)
    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)

    nodes_csv = require_file(statsdir / f"{sample_id}_step3_nodes.csv")
    faces_csv = require_file(statsdir / f"{sample_id}_step3_faces.csv")
    edges_csv = require_file(statsdir / f"{sample_id}_step3_edges.csv")
    gnn_csv = require_file(statsdir / f"{sample_id}_step14_gnn_learned_flux_{target_flux}.csv")
    B1_path = require_file(statsdir / f"{sample_id}_step3_B1.npz")
    B2_path = require_file(statsdir / f"{sample_id}_step3_B2.npz")

    nodes_df = pd.read_csv(nodes_csv)
    faces_df = pd.read_csv(faces_csv)
    edges_df = pd.read_csv(edges_csv)
    gnn_df = pd.read_csv(gnn_csv)

    B1 = sparse.load_npz(B1_path).tocsr()
    B2 = sparse.load_npz(B2_path).tocsr()
    B1_abs = abs(B1)
    B2_abs = abs(B2)

    if "flux_gnn_unscaled" not in gnn_df.columns:
        raise KeyError("Expected 'flux_gnn_unscaled' in step14 learned flux file.")

    f = gnn_df["flux_gnn_unscaled"].to_numpy(dtype=float)

    print("=" * 80)
    print("STEP 15: downstream analysis on learned GNN flux")
    print("=" * 80)
    print(f"sample_id        : {sample_id}")
    print(f"target_flux      : {target_flux}")
    print(f"n_nodes          : {len(nodes_df)}")
    print(f"n_edges          : {len(edges_df)}")
    print(f"n_faces          : {len(faces_df)}")

    # Hodge decomposition
    alpha, f_exact = project_exact(B1, f, ridge=args.ridge)
    f_resid = f - f_exact
    beta, f_coexact = project_coexact(B2, f_resid, ridge=args.ridge)
    f_harmonic = f - f_exact - f_coexact

    E_total = float(np.sum(f ** 2))
    E_exact = float(np.sum(f_exact ** 2))
    E_coexact = float(np.sum(f_coexact ** 2))
    E_harm = float(np.sum(f_harmonic ** 2))

    global_eps = max(E_total * 1e-6, 1e-12)
    frac_exact = E_exact / (E_total + global_eps)
    frac_coexact = E_coexact / (E_total + global_eps)
    frac_harm = E_harm / (E_total + global_eps)

    div_total = np.asarray(B1 @ f).ravel()
    curl_total = np.abs(np.asarray(B2.T @ f).ravel())

    print("\nGlobal energies")
    print("-" * 80)
    print(f"E_total          : {E_total:.6e}")
    print(f"E_exact          : {E_exact:.6e}   frac={frac_exact:.4f}")
    print(f"E_coexact        : {E_coexact:.6e}   frac={frac_coexact:.4f}")
    print(f"E_harmonic       : {E_harm:.6e}   frac={frac_harm:.4f}")
    print(f"mean |div|       : {np.mean(np.abs(div_total)):.6e}")
    print(f"mean |curl|      : {np.mean(curl_total):.6e}")

    # Edge output
    edge_out = edges_df.copy()
    edge_out["flux_gnn_total"] = f
    edge_out["flux_gnn_exact"] = f_exact
    edge_out["flux_gnn_coexact"] = f_coexact
    edge_out["flux_gnn_harmonic"] = f_harmonic
    edge_out_csv = statsdir / f"{sample_id}_step15_edges_hodge_gnn_{target_flux}.csv"
    edge_out.to_csv(edge_out_csv, index=False)

    # Node mapped energies
    node_out = nodes_df.copy()
    node_out["node_energy_total"] = node_mean_incident_energy(B1_abs, f)
    node_out["node_energy_exact"] = node_mean_incident_energy(B1_abs, f_exact)
    node_out["node_energy_coexact"] = node_mean_incident_energy(B1_abs, f_coexact)
    node_out["node_energy_harmonic"] = node_mean_incident_energy(B1_abs, f_harmonic)

    node_total = node_out["node_energy_total"].to_numpy(dtype=float)
    positive_node_total = node_total[node_total > 0]
    node_eps = max(np.median(positive_node_total) * 0.01, 1e-12) if len(positive_node_total) else 1e-12
    denom = node_total + node_eps

    node_out["frac_exact"] = node_out["node_energy_exact"].to_numpy(dtype=float) / denom
    node_out["frac_coexact"] = node_out["node_energy_coexact"].to_numpy(dtype=float) / denom
    node_out["frac_harmonic"] = node_out["node_energy_harmonic"].to_numpy(dtype=float) / denom

    node_out_csv = statsdir / f"{sample_id}_step15_nodes_hodge_gnn_{target_flux}.csv"
    node_out.to_csv(node_out_csv, index=False)

    # Face curl
    faces_with_regions = assign_face_regions(nodes_df, faces_df)
    x_cent = []
    y_cent = []
    coords = nodes_df[["x_fullres", "y_fullres"]].to_numpy(dtype=float)
    for _, row in faces_df.iterrows():
        tri = coords[[int(row["i"]), int(row["j"]), int(row["k"])]]
        x_cent.append(np.mean(tri[:, 0]))
        y_cent.append(np.mean(tri[:, 1]))

    face_out = faces_with_regions.copy()
    face_out["x_centroid"] = x_cent
    face_out["y_centroid"] = y_cent
    face_out["curl_total_abs"] = np.abs(np.asarray(B2.T @ f).ravel())
    face_out["curl_coexact_abs"] = np.abs(np.asarray(B2.T @ f_coexact).ravel())
    face_out["face_energy_total"] = face_mapped_energy(B2_abs, f)
    face_out["face_energy_coexact"] = face_mapped_energy(B2_abs, f_coexact)
    face_out_csv = statsdir / f"{sample_id}_step15_face_curl_gnn_{target_flux}.csv"
    face_out.to_csv(face_out_csv, index=False)

    # Region enrichment on node coexact
    tests = []
    primary_pairs = [
        ("immune_enriched", "tumor_enriched"),
        ("immune_enriched", "stroma_enriched"),
        ("immune_enriched", "other"),
        ("interface_like", "tumor_enriched"),
        ("interface_like", "stroma_enriched"),
        ("interface_like", "other"),
    ]

    for metric in ["node_energy_coexact", "frac_coexact"]:
        for ra, rb in primary_pairs:
            xa = node_out.loc[node_out["region_step2"] == ra, metric].to_numpy(dtype=float)
            xb = node_out.loc[node_out["region_step2"] == rb, metric].to_numpy(dtype=float)
            mwu_stat, mwu_p = mwu_pvalue(xa, xb, alternative="greater")
            obs_diff, perm_p = permutation_pvalue(
                xa, xb, n_perm=args.n_perm, alternative="greater", seed=args.seed
            )
            tests.append({
                "metric": metric,
                "region_a": ra,
                "region_b": rb,
                "median_ratio_a_over_b": compute_ratio(xa, xb),
                "median_diff_a_minus_b": obs_diff,
                "mwu_stat": mwu_stat,
                "mwu_p": mwu_p,
                "perm_p": perm_p,
            })

    tests_df = pd.DataFrame(tests)
    tests_csv = statsdir / f"{sample_id}_step15_region_tests_gnn_{target_flux}.csv"
    tests_df.to_csv(tests_csv, index=False)

    # Interface/immune hotspot enrichment
    hotspot_rows = []
    for region_name in ["immune_enriched", "interface_like"]:
        mask = (face_out["face_region"] == region_name).to_numpy()
        hot_stats = hotspot_enrichment(face_out["curl_coexact_abs"].to_numpy(dtype=float), mask, q=args.hotspot_q)
        hotspot_rows.append({
            "sample_id": sample_id,
            "target_flux": target_flux,
            "region_name": region_name,
            "hotspot_q": args.hotspot_q,
            **hot_stats,
        })

    hotspot_df = pd.DataFrame(hotspot_rows)
    hotspot_csv = statsdir / f"{sample_id}_step15_hotspot_enrichment_gnn_{target_flux}.csv"
    hotspot_df.to_csv(hotspot_csv, index=False)

    # Summary
    summary = pd.DataFrame([{
        "sample_id": sample_id,
        "target_flux": target_flux,
        "E_total": E_total,
        "E_exact": E_exact,
        "E_coexact": E_coexact,
        "E_harmonic": E_harm,
        "frac_exact": frac_exact,
        "frac_coexact": frac_coexact,
        "frac_harmonic": frac_harm,
        "mean_abs_div": float(np.mean(np.abs(div_total))),
        "mean_abs_curl": float(np.mean(curl_total)),
    }])
    summary_csv = statsdir / f"{sample_id}_step15_gnn_operator_summary_{target_flux}.csv"
    summary.to_csv(summary_csv, index=False)

    # Figure
    fig_png = figdir / f"{sample_id}_step15_gnn_region_boxplots_{target_flux}.png"
    save_region_boxplots(node_out, sample_id, target_flux, fig_png)

    print("\nSaved outputs")
    print("-" * 80)
    print(f"Saved: {edge_out_csv}")
    print(f"Saved: {node_out_csv}")
    print(f"Saved: {face_out_csv}")
    print(f"Saved: {tests_csv}")
    print(f"Saved: {hotspot_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {fig_png}")

    print("\nPrimary region tests")
    print("-" * 80)
    print(tests_df[["metric", "region_a", "region_b", "median_ratio_a_over_b", "mwu_p", "perm_p"]])

    print("\nHotspot enrichment")
    print("-" * 80)
    print(hotspot_df[["region_name", "enrichment_ratio", "frac_hot_in_region", "frac_region"]])


if __name__ == "__main__":
    main()
