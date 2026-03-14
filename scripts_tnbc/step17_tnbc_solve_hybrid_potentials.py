from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve


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


def solve_phi(B1: sparse.csr_matrix, f: np.ndarray, ridge: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve discrete node potential phi from:
        (B1 B1^T + ridge I) phi = - B1 f
    and reconstruct:
        f_exact = - B1^T phi
    """
    n_nodes = B1.shape[0]
    A = (B1 @ B1.T).tocsr() + ridge * sparse.eye(n_nodes, format="csr")
    rhs = -(B1 @ f)
    phi = spsolve(A, rhs)
    f_exact = -(B1.T @ phi)
    return np.asarray(phi).ravel(), np.asarray(f_exact).ravel()


def solve_psi(
    B2: sparse.csr_matrix,
    f_resid: np.ndarray,
    face_mask: np.ndarray | None = None,
    ridge: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve discrete stream function psi from:
        (B2^T B2 + ridge I) psi = B2^T f_resid

    If face_mask is provided, solve only on masked faces:
        psi_masked = D_mask psi
        f_coexact_masked = B2 psi_masked
    """
    n_faces = B2.shape[1]
    if n_faces == 0:
        return np.zeros(0, dtype=float), np.zeros_like(f_resid)

    if face_mask is None:
        A = (B2.T @ B2).tocsr() + ridge * sparse.eye(n_faces, format="csr")
        rhs = B2.T @ f_resid
        psi = spsolve(A, rhs)
        f_coexact = B2 @ psi
        return np.asarray(psi).ravel(), np.asarray(f_coexact).ravel()

    face_mask = np.asarray(face_mask, dtype=float).ravel()
    if len(face_mask) != n_faces:
        raise ValueError("face_mask length must equal number of faces.")

    Dm = sparse.diags(face_mask, format="csr")
    A = (Dm @ (B2.T @ B2) @ Dm).tocsr() + ridge * sparse.eye(n_faces, format="csr")
    rhs = Dm @ (B2.T @ f_resid)
    psi = spsolve(A, rhs)
    psi = face_mask * np.asarray(psi).ravel()
    f_coexact = B2 @ psi
    return np.asarray(psi).ravel(), np.asarray(f_coexact).ravel()


def node_mean_incident_energy(B1_abs: sparse.csr_matrix, edge_component: np.ndarray) -> np.ndarray:
    deg = np.asarray(B1_abs @ np.ones_like(edge_component)).ravel()
    deg = np.maximum(deg, 1.0)
    node_energy = np.asarray(B1_abs @ (edge_component ** 2)).ravel() / deg
    return node_energy


def face_energy(face_component: np.ndarray) -> np.ndarray:
    return np.asarray(face_component ** 2).ravel()


def triangle_centroids(node_df: pd.DataFrame, face_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    coords = node_df[["x_fullres", "y_fullres"]].to_numpy(dtype=float)
    tri = face_df[["i", "j", "k"]].to_numpy(dtype=int)
    tri_pts = coords[tri]
    centroids = tri_pts.mean(axis=1)
    return centroids[:, 0], centroids[:, 1]


def save_panel_figure(
    node_df: pd.DataFrame,
    face_df: pd.DataFrame,
    phi: np.ndarray,
    psi: np.ndarray,
    sample_id: str,
    flux_source: str,
    outpath: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    x_nodes = node_df["x_fullres"].to_numpy(dtype=float)
    y_nodes = node_df["y_fullres"].to_numpy(dtype=float)

    x_face, y_face = triangle_centroids(node_df, face_df)

    vmax_phi = np.quantile(np.abs(phi), 0.99) if len(phi) else None
    sc1 = axes[0].scatter(
        x_nodes,
        y_nodes,
        c=phi,
        s=10,
        alpha=0.9,
        vmin=-vmax_phi if vmax_phi is not None else None,
        vmax=vmax_phi,
    )
    axes[0].set_title(r"Node potential $\phi_d$")
    axes[0].invert_yaxis()
    axes[0].axis("off")
    plt.colorbar(sc1, ax=axes[0], fraction=0.04, pad=0.02)

    vmax_psi = np.quantile(np.abs(psi), 0.99) if len(psi) else None
    sc2 = axes[1].scatter(
        x_face,
        y_face,
        c=psi,
        s=10,
        alpha=0.9,
        vmin=-vmax_psi if vmax_psi is not None else None,
        vmax=vmax_psi,
    )
    axes[1].set_title(r"Face stream function $\psi_d$")
    axes[1].invert_yaxis()
    axes[1].axis("off")
    plt.colorbar(sc2, ax=axes[1], fraction=0.04, pad=0.02)

    plt.suptitle(f"{sample_id}: solved hybrid potentials — {flux_source}", y=0.98)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 17 TNBC: solve discrete gradient-plus-stream potentials for proxy or GNN flux."
    )
    parser.add_argument("--sample_id", required=True)
    parser.add_argument(
        "--target_flux",
        required=True,
        choices=["flux_tumor_immune", "flux_tumor_stroma", "flux_immune_stroma"],
    )
    parser.add_argument(
        "--flux_source",
        required=True,
        choices=["proxy", "gnn"],
        help="Use proxy edge flux (step4/step6 family) or learned GNN flux (step14/step15 family).",
    )
    parser.add_argument("--statsdir", default="stats")
    parser.add_argument("--figdir", default="visium_figures")
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument(
        "--interface_only",
        action="store_true",
        help="Solve stream function only on faces labeled interface_like.",
    )
    args = parser.parse_args()

    sample_id = args.sample_id
    target_flux = args.target_flux
    flux_source = args.flux_source
    statsdir = Path(args.statsdir)
    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)

    nodes_csv = require_file(statsdir / f"{sample_id}_step3_nodes.csv")
    faces_csv = require_file(statsdir / f"{sample_id}_step3_faces.csv")
    B1_path = require_file(statsdir / f"{sample_id}_step3_B1.npz")
    B2_path = require_file(statsdir / f"{sample_id}_step3_B2.npz")

    node_df = pd.read_csv(nodes_csv)
    face_df = pd.read_csv(faces_csv)
    B1 = sparse.load_npz(B1_path).tocsr()
    B2 = sparse.load_npz(B2_path).tocsr()
    B1_abs = abs(B1)

    if flux_source == "proxy":
        flux_csv = require_file(statsdir / f"{sample_id}_step4_edge_fluxes.csv")
        flux_df = pd.read_csv(flux_csv)
        if target_flux not in flux_df.columns:
            raise KeyError(f"Expected column {target_flux} in {flux_csv}")
        f = flux_df[target_flux].to_numpy(dtype=float)
        out_prefix = f"{sample_id}_step17_proxy_hybrid_{target_flux}"
    else:
        flux_csv = require_file(statsdir / f"{sample_id}_step14_gnn_learned_flux_{target_flux}.csv")
        flux_df = pd.read_csv(flux_csv)
        if "flux_gnn_unscaled" not in flux_df.columns:
            raise KeyError(f"Expected column flux_gnn_unscaled in {flux_csv}")
        f = flux_df["flux_gnn_unscaled"].to_numpy(dtype=float)
        out_prefix = f"{sample_id}_step17_gnn_hybrid_{target_flux}"

    print("=" * 80)
    print("STEP 17: solve hybrid potentials")
    print("=" * 80)
    print(f"sample_id        : {sample_id}")
    print(f"target_flux      : {target_flux}")
    print(f"flux_source      : {flux_source}")
    print(f"interface_only   : {args.interface_only}")
    print(f"n_nodes          : {len(node_df)}")
    print(f"n_faces          : {len(face_df)}")
    print(f"n_edges          : {len(f)}")

    phi, f_exact = solve_phi(B1, f, ridge=args.ridge)

    f_resid = f - f_exact
    face_regions_df = assign_face_regions(node_df, face_df)

    face_mask = None
    if args.interface_only:
        face_mask = (face_regions_df["face_region"].to_numpy() == "interface_like").astype(float)

    psi, f_stream = solve_psi(B2, f_resid, face_mask=face_mask, ridge=args.ridge)

    f_fit = f_exact + f_stream
    f_residual = f - f_fit

    E_total = float(np.sum(f**2))
    E_exact = float(np.sum(f_exact**2))
    E_stream = float(np.sum(f_stream**2))
    E_residual = float(np.sum(f_residual**2))

    frac_exact = E_exact / max(E_total, 1e-18)
    frac_stream = E_stream / max(E_total, 1e-18)
    frac_residual = E_residual / max(E_total, 1e-18)

    phi_l2 = float(np.linalg.norm(phi))
    psi_l2 = float(np.linalg.norm(psi))

    node_out = node_df.copy()
    node_out["phi_d"] = phi
    node_out["node_energy_exact_from_phi"] = node_mean_incident_energy(B1_abs, f_exact)
    node_out_csv = statsdir / f"{out_prefix}_nodes.csv"
    node_out.to_csv(node_out_csv, index=False)

    face_out = face_regions_df.copy()
    x_face, y_face = triangle_centroids(node_df, face_df)
    face_out["x_centroid"] = x_face
    face_out["y_centroid"] = y_face
    face_out["psi_d"] = psi
    face_out["face_energy_stream_from_psi"] = face_energy(psi)
    face_out_csv = statsdir / f"{out_prefix}_faces.csv"
    face_out.to_csv(face_out_csv, index=False)

    edge_out = flux_df.copy()
    edge_out["f_total"] = f
    edge_out["f_exact_from_phi"] = f_exact
    edge_out["f_stream_from_psi"] = f_stream
    edge_out["f_fit_hybrid"] = f_fit
    edge_out["f_residual_after_hybrid"] = f_residual
    edge_out_csv = statsdir / f"{out_prefix}_edges.csv"
    edge_out.to_csv(edge_out_csv, index=False)

    summary = pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "target_flux": target_flux,
                "flux_source": flux_source,
                "interface_only": bool(args.interface_only),
                "E_total": E_total,
                "E_exact_from_phi": E_exact,
                "E_stream_from_psi": E_stream,
                "E_residual_after_hybrid": E_residual,
                "frac_exact_from_phi": frac_exact,
                "frac_stream_from_psi": frac_stream,
                "frac_residual_after_hybrid": frac_residual,
                "phi_l2": phi_l2,
                "psi_l2": psi_l2,
                "mean_abs_residual_after_hybrid": float(np.mean(np.abs(f_residual))),
            }
        ]
    )
    summary_csv = statsdir / f"{out_prefix}_summary.csv"
    summary.to_csv(summary_csv, index=False)

    fig_png = figdir / f"{out_prefix}.png"
    save_panel_figure(
        node_df=node_df,
        face_df=face_df,
        phi=phi,
        psi=psi,
        sample_id=sample_id,
        flux_source=f"{flux_source} / {target_flux}",
        outpath=fig_png,
    )

    print("\nHybrid model summary")
    print("-" * 80)
    print(summary.to_string(index=False))
    print("-" * 80)
    print(f"Saved: {node_out_csv}")
    print(f"Saved: {face_out_csv}")
    print(f"Saved: {edge_out_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {fig_png}")


if __name__ == "__main__":
    main()
