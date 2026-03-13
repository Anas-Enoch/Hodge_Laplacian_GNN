from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import sparse
from scipy.sparse.linalg import spsolve


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def load_sample_image_and_scale(sample_dir: Path) -> tuple[np.ndarray, float]:
    img_files = sorted(sample_dir.glob("*tissue_hires_image.png"))
    scale_files = sorted(sample_dir.glob("*scalefactors_json.json"))

    if len(img_files) != 1:
        raise RuntimeError(
            f"Expected exactly one hires image in {sample_dir}, found {[p.name for p in img_files]}"
        )
    if len(scale_files) != 1:
        raise RuntimeError(
            f"Expected exactly one scalefactors json in {sample_dir}, found {[p.name for p in scale_files]}"
        )

    import json
    image = np.array(Image.open(img_files[0]))
    with open(scale_files[0], "r", encoding="utf-8") as f:
        scalefactors = json.load(f)

    hires_scale = float(scalefactors.get("tissue_hires_scalef", 1.0))
    return image, hires_scale


def project_exact(B1: sparse.csr_matrix, f: np.ndarray, ridge: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact component: solve alpha from (B1 B1^T + ridge I) alpha = B1 f
    then f_exact = B1^T alpha
    """
    n_nodes = B1.shape[0]
    A = (B1 @ B1.T).tocsr() + ridge * sparse.eye(n_nodes, format="csr")
    rhs = B1 @ f
    alpha = spsolve(A, rhs)
    f_exact = B1.T @ alpha
    return np.asarray(alpha), np.asarray(f_exact).ravel()


def project_coexact(B2: sparse.csr_matrix, f_resid: np.ndarray, ridge: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """
    Coexact component: solve beta from (B2^T B2 + ridge I) beta = B2^T f_resid
    then f_coexact = B2 beta
    """
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


def save_maps(
    node_df: pd.DataFrame,
    image: np.ndarray,
    hires_scale: float,
    sample_id: str,
    flux_name: str,
    outpath: Path,
) -> None:
    x = node_df["x_fullres"].to_numpy(dtype=float) * hires_scale
    y = node_df["y_fullres"].to_numpy(dtype=float) * hires_scale

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    panels = [
        ("node_energy_total", "Total node energy"),
        ("node_energy_exact", "Exact / gradient energy"),
        ("node_energy_coexact", "Coexact / rotational energy"),
        ("node_energy_harmonic", "Harmonic energy"),
    ]

    for ax, (col, title) in zip(axes.ravel(), panels):
        ax.imshow(image)
        vals = node_df[col].to_numpy(dtype=float)
        vmax = np.quantile(vals, 0.99) if np.any(np.isfinite(vals)) else None
        sca = ax.scatter(
            x,
            y,
            c=vals,
            s=14,
            alpha=0.90,
            vmin=0,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.invert_yaxis()
        ax.axis("off")
        plt.colorbar(sca, ax=ax, fraction=0.035, pad=0.02)

    plt.suptitle(f"{sample_id}: Step 6 Hodge maps — {flux_name}", y=0.98)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close(fig)


def save_boxplots(node_df: pd.DataFrame, sample_id: str, flux_name: str, outpath: Path) -> None:
    region_order = [
        "tumor_enriched",
        "stroma_enriched",
        "immune_enriched",
        "interface_like",
        "other",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    specs = [
        ("node_energy_total", "Total node energy"),
        ("frac_exact", "Exact fraction"),
        ("frac_coexact", "Coexact fraction"),
        ("frac_harmonic", "Harmonic fraction"),
    ]

    for ax, (col, title) in zip(axes.ravel(), specs):
        data = [
            node_df.loc[node_df["region_step2"] == reg, col].to_numpy(dtype=float)
            for reg in region_order
        ]
        ax.boxplot(data, tick_labels=region_order, showfliers=False)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)

    plt.suptitle(f"{sample_id}: Step 6 region boxplots — {flux_name}", y=0.98)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 6 TNBC: Hodge decomposition of edge flux proxies."
    )
    parser.add_argument("--sample_id", required=True)
    parser.add_argument("--sample_dir", required=True)
    parser.add_argument(
        "--flux_col",
        required=True,
        choices=["flux_tumor_immune", "flux_tumor_stroma", "flux_immune_stroma"],
        help="Which edge flux proxy to decompose.",
    )
    parser.add_argument("--statsdir", default="stats")
    parser.add_argument("--figdir", default="visium_figures")
    parser.add_argument("--ridge", type=float, default=1e-6)
    args = parser.parse_args()

    sample_id = args.sample_id
    sample_dir = Path(args.sample_dir).resolve()
    statsdir = Path(args.statsdir)
    figdir = Path(args.figdir)
    ridge = float(args.ridge)

    nodes_csv = require_file(statsdir / f"{sample_id}_step3_nodes.csv")
    edges_csv = require_file(statsdir / f"{sample_id}_step4_edge_fluxes.csv")
    faces_csv = require_file(statsdir / f"{sample_id}_step3_faces.csv")
    B1_path = require_file(statsdir / f"{sample_id}_step3_B1.npz")
    B2_path = require_file(statsdir / f"{sample_id}_step3_B2.npz")

    node_df = pd.read_csv(nodes_csv)
    edge_df = pd.read_csv(edges_csv)
    face_df = pd.read_csv(faces_csv)

    B1 = sparse.load_npz(B1_path).tocsr()
    B2 = sparse.load_npz(B2_path).tocsr()
    B1_abs = abs(B1)
    B2_abs = abs(B2)

    flux_name = args.flux_col
    f = edge_df[flux_name].to_numpy(dtype=float)

    print("=" * 72)
    print(f"STEP 6: TNBC Hodge decomposition for {sample_id}")
    print("=" * 72)
    print(f"Flux column : {flux_name}")
    print(f"n_nodes     : {len(node_df)}")
    print(f"n_edges     : {len(edge_df)}")
    print(f"n_faces     : {len(face_df)}")
    print(f"B1 shape    : {B1.shape}")
    print(f"B2 shape    : {B2.shape}")

    # Exact
    alpha, f_exact = project_exact(B1, f, ridge=ridge)

    # Coexact from residual
    f_resid_1 = f - f_exact
    beta, f_coexact = project_coexact(B2, f_resid_1, ridge=ridge)

    # Harmonic residual
    f_harmonic = f - f_exact - f_coexact

    # Diagnostics
    div_total = np.asarray(B1 @ f).ravel()
    div_exact = np.asarray(B1 @ f_exact).ravel()
    div_coexact = np.asarray(B1 @ f_coexact).ravel()
    div_harm = np.asarray(B1 @ f_harmonic).ravel()

    curl_total = np.asarray(B2.T @ f).ravel() if B2.shape[1] > 0 else np.zeros(0)
    curl_exact = np.asarray(B2.T @ f_exact).ravel() if B2.shape[1] > 0 else np.zeros(0)
    curl_coexact = np.asarray(B2.T @ f_coexact).ravel() if B2.shape[1] > 0 else np.zeros(0)
    curl_harm = np.asarray(B2.T @ f_harmonic).ravel() if B2.shape[1] > 0 else np.zeros(0)

    # Global energies
    E_total = float(np.sum(f ** 2))
    E_exact = float(np.sum(f_exact ** 2))
    E_coexact = float(np.sum(f_coexact ** 2))
    E_harm = float(np.sum(f_harmonic ** 2))
    # Regularized denominator to avoid unstable fractions in very low-energy regimes
    global_eps = max(E_total * 1e-6, 1e-12)
    
    frac_exact = E_exact / (E_total + global_eps)
    frac_coexact = E_coexact / (E_total + global_eps)
    frac_harm = E_harm / (E_total + global_eps)

    print("\nGlobal energies")
    print("-" * 72)
    print(f"E_total    : {E_total:.6e}")
    print(f"E_exact    : {E_exact:.6e}   frac={frac_exact:.4f}")
    print(f"E_coexact  : {E_coexact:.6e}   frac={frac_coexact:.4f}")
    print(f"E_harmonic : {E_harm:.6e}   frac={frac_harm:.4f}")

    print("\nDiagnostic means")
    print("-" * 72)
    print(f"mean |div total|    : {np.mean(np.abs(div_total)):.6e}")
    print(f"mean |div exact|    : {np.mean(np.abs(div_exact)):.6e}")
    print(f"mean |div coexact|  : {np.mean(np.abs(div_coexact)):.6e}")
    print(f"mean |div harmonic| : {np.mean(np.abs(div_harm)):.6e}")
    if len(curl_total) > 0:
        print(f"mean |curl total|    : {np.mean(np.abs(curl_total)):.6e}")
        print(f"mean |curl exact|    : {np.mean(np.abs(curl_exact)):.6e}")
        print(f"mean |curl coexact|  : {np.mean(np.abs(curl_coexact)):.6e}")
        print(f"mean |curl harmonic| : {np.mean(np.abs(curl_harm)):.6e}")

    # Edge-level outputs
    edge_out = edge_df.copy()
    edge_out["flux_total"] = f
    edge_out["flux_exact"] = f_exact
    edge_out["flux_coexact"] = f_coexact
    edge_out["flux_harmonic"] = f_harmonic

    edge_out_csv = statsdir / f"{sample_id}_step6_edges_hodge_{flux_name}.csv"
    edge_out.to_csv(edge_out_csv, index=False)

    # Node-level mapped energies
    node_out = node_df.copy()
    node_out["node_energy_total"] = node_mean_incident_energy(B1_abs, f)
    node_out["node_energy_exact"] = node_mean_incident_energy(B1_abs, f_exact)
    node_out["node_energy_coexact"] = node_mean_incident_energy(B1_abs, f_coexact)
    node_out["node_energy_harmonic"] = node_mean_incident_energy(B1_abs, f_harmonic)

    
    node_total = node_out["node_energy_total"].to_numpy(dtype=float)

    # Use a data-adaptive epsilon based on the median nonzero node energy
    positive_node_total = node_total[node_total > 0]
    if len(positive_node_total) > 0:
       node_eps = max(np.median(positive_node_total) * 0.01, 1e-12)
    else:
       node_eps = 1e-12

    denom = node_total + node_eps

    node_out["frac_exact"] = node_out["node_energy_exact"].to_numpy(dtype=float) / denom
    node_out["frac_coexact"] = node_out["node_energy_coexact"].to_numpy(dtype=float) / denom
    node_out["frac_harmonic"] = node_out["node_energy_harmonic"].to_numpy(dtype=float) / denom

    node_out_csv = statsdir / f"{sample_id}_step6_nodes_hodge_{flux_name}.csv"
    node_out.to_csv(node_out_csv, index=False)

    # Face-level mapped energies
    face_out = face_df.copy()
    face_out["face_energy_total"] = face_mapped_energy(B2_abs, f)
    face_out["face_energy_exact"] = face_mapped_energy(B2_abs, f_exact)
    face_out["face_energy_coexact"] = face_mapped_energy(B2_abs, f_coexact)
    face_out["face_energy_harmonic"] = face_mapped_energy(B2_abs, f_harmonic)

    face_out_csv = statsdir / f"{sample_id}_step6_faces_hodge_{flux_name}.csv"
    face_out.to_csv(face_out_csv, index=False)

    # Region summary
    region_summary = (
        node_out.groupby("region_step2")[
            [
                "node_energy_total",
                "node_energy_exact",
                "node_energy_coexact",
                "node_energy_harmonic",
                "frac_exact",
                "frac_coexact",
                "frac_harmonic",
            ]
        ]
        .agg(["mean", "median", "std", "count"])
    )
    region_summary_csv = statsdir / f"{sample_id}_step6_region_summary_{flux_name}.csv"
    region_summary.to_csv(region_summary_csv)

    # Global energy summary row
    energy_summary = pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "flux_name": flux_name,
                "E_total": E_total,
                "E_exact": E_exact,
                "E_coexact": E_coexact,
                "E_harmonic": E_harm,
                "frac_exact": frac_exact,
                "frac_coexact": frac_coexact,
                "frac_harmonic": frac_harm,
                "mean_abs_div_total": float(np.mean(np.abs(div_total))),
                "mean_abs_div_exact": float(np.mean(np.abs(div_exact))),
                "mean_abs_div_coexact": float(np.mean(np.abs(div_coexact))),
                "mean_abs_div_harmonic": float(np.mean(np.abs(div_harm))),
                "mean_abs_curl_total": float(np.mean(np.abs(curl_total))) if len(curl_total) > 0 else np.nan,
                "mean_abs_curl_exact": float(np.mean(np.abs(curl_exact))) if len(curl_exact) > 0 else np.nan,
                "mean_abs_curl_coexact": float(np.mean(np.abs(curl_coexact))) if len(curl_coexact) > 0 else np.nan,
                "mean_abs_curl_harmonic": float(np.mean(np.abs(curl_harm))) if len(curl_harm) > 0 else np.nan,
            }
        ]
    )
    energy_summary_csv = statsdir / f"{sample_id}_step6_energy_summary_{flux_name}.csv"
    energy_summary.to_csv(energy_summary_csv, index=False)

    # Figures
    image, hires_scale = load_sample_image_and_scale(sample_dir)

    maps_png = figdir / f"{sample_id}_step6_hodge_maps_{flux_name}.png"
    save_maps(node_out, image, hires_scale, sample_id, flux_name, maps_png)

    boxplots_png = figdir / f"{sample_id}_step6_hodge_boxplots_{flux_name}.png"
    save_boxplots(node_out, sample_id, flux_name, boxplots_png)

    print(f"\nSaved: {edge_out_csv}")
    print(f"Saved: {node_out_csv}")
    print(f"Saved: {face_out_csv}")
    print(f"Saved: {region_summary_csv}")
    print(f"Saved: {energy_summary_csv}")
    print(f"Saved: {maps_png}")
    print(f"Saved: {boxplots_png}")
    print("\nStep 6 completed successfully.")


if __name__ == "__main__":
    main()
