from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import sparse
from scipy.sparse.linalg import eigsh


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

    image = np.array(Image.open(img_files[0]))
    with open(scale_files[0], "r", encoding="utf-8") as f:
        scalefactors = json.load(f)

    hires_scale = float(scalefactors.get("tissue_hires_scalef", 1.0))
    return image, hires_scale


def triangle_centroids(node_df: pd.DataFrame, face_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    coords = node_df[["x_fullres", "y_fullres"]].to_numpy(dtype=float)
    tri = face_df[["i", "j", "k"]].to_numpy(dtype=int)
    tri_pts = coords[tri]
    centroids = tri_pts.mean(axis=1)
    return centroids[:, 0], centroids[:, 1]


def robust_smallest_positive_eigs(
    L1: sparse.csr_matrix,
    k: int = 64,
    tol_zero: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a low-frequency spectral basis of L1.
    Returns positive eigenvalues/eigenvectors only.
    """
    n = L1.shape[0]
    k_eff = min(max(8, k), n - 2) if n > 2 else 1

    vals, vecs = eigsh(L1, k=k_eff, which="SM")
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]

    mask = vals > tol_zero
    vals_pos = vals[mask]
    vecs_pos = vecs[:, mask]

    if len(vals_pos) == 0:
        raise RuntimeError("No positive eigenvalues found for L1; cannot build Lie-structured null.")

    return vals_pos, vecs_pos


def choose_s_from_gap(delta: float, lam_min_pos: float) -> float:
    """
    Monotone-in-gap spectral exponent.
    """
    return float(np.log2(1.0 + delta / max(lam_min_pos, 1e-12)))


def spectral_resolvent_diag_weights(
    vals_pos: np.ndarray,
    vecs_pos: np.ndarray,
    eps_reg: float,
    s_exp: float,
) -> np.ndarray:
    """
    w_e = diag((L1 + eps I)^(-s)) using truncated spectral expansion.
    """
    coeff = (vals_pos + eps_reg) ** (-s_exp)
    # diag(V diag(c) V^T) = sum_m c_m * V[:,m]^2
    w = np.sum((vecs_pos ** 2) * coeff[None, :], axis=1)
    return np.asarray(w).ravel()


def edge_midpoint_regions(edge_df: pd.DataFrame) -> np.ndarray:
    """
    Crude region label for each edge from its endpoints.
    Priority:
      same region -> that region
      if one endpoint interface_like -> interface_like
      else mixed_tail__head
    """
    out = []
    for _, row in edge_df.iterrows():
        r1 = row["tail_region"]
        r2 = row["head_region"]
        if r1 == r2:
            out.append(r1)
        elif r1 == "interface_like" or r2 == "interface_like":
            out.append("interface_like")
        else:
            out.append(f"mixed__{r1}__{r2}")
    return np.array(out, dtype=object)


def build_lie_surrogate(
    f_real: np.ndarray,
    weights_norm: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Local operator-weighted perturbation.

    We do not globally shuffle edges.
    Instead, we perturb each edge locally using a weighted signed phase mixture:
        f_null = f * cos(theta) + eta * sin(theta)
    where eta is a sign-flipped matched-magnitude companion field.
    """
    theta = rng.uniform(-np.pi * weights_norm, np.pi * weights_norm)
    eta = np.abs(f_real) * rng.choice([-1.0, 1.0], size=len(f_real))
    f_null = f_real * np.cos(theta) + eta * np.sin(theta)
    return f_null


def top_quantile_mean(x: np.ndarray, q: float = 0.95) -> float:
    thr = np.quantile(x, q)
    return float(np.mean(x[x >= thr]))


def edge_level_coexact_hotspot_stat(
    edge_df: pd.DataFrame,
    f_real: np.ndarray,
    f_null: np.ndarray,
    region_name: str = "interface_like",
    q: float = 0.95,
) -> tuple[float, float]:
    """
    Region-localized edge hotspot statistic on absolute perturbed magnitude.
    This is a pragmatic surrogate for region-focused coexact hotspot strength.
    """
    edge_regions = edge_level_regions = edge_midpoint_regions(edge_df)
    mask = edge_regions == region_name

    x_real = np.abs(f_real[mask]) if np.any(mask) else np.abs(f_real)
    x_null = np.abs(f_null[mask]) if np.any(mask) else np.abs(f_null)

    return top_quantile_mean(x_real, q=q), top_quantile_mean(x_null, q=q)


def face_curl_stats(B2: sparse.csr_matrix, f: np.ndarray) -> dict[str, float]:
    curl = np.abs(np.asarray(B2.T @ f).ravel())
    return {
        "mean_curl": float(np.mean(curl)),
        "median_curl": float(np.median(curl)),
        "top95_mean_curl": top_quantile_mean(curl, q=0.95),
        "top99_mean_curl": top_quantile_mean(curl, q=0.99),
    }


def plot_null_hist(
    null_vals: np.ndarray,
    real_val: float,
    title: str,
    xlabel: str,
    outpath: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(null_vals, bins=30, alpha=0.85)
    ax.axvline(real_val, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_hotspot_overlay(
    image: np.ndarray,
    hires_scale: float,
    x_face: np.ndarray,
    y_face: np.ndarray,
    curl_real_abs: np.ndarray,
    sample_id: str,
    flux_name: str,
    outpath: Path,
) -> None:

    # threshold = top 1% curl
    thr99 = np.quantile(curl_real_abs, 0.99)
    mask = curl_real_abs >= thr99

    fig, ax = plt.subplots(figsize=(7, 6))

    # tissue background
    ax.imshow(image)

    # faint curl field (optional but nice)
    ax.scatter(
        x_face * hires_scale,
        y_face * hires_scale,
        c=curl_real_abs,
        s=6,
        cmap="viridis",
        alpha=0.25,
    )

    # strong hotspot markers
    ax.scatter(
        x_face[mask] * hires_scale,
        y_face[mask] * hires_scale,
        c="red",
        s=70,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.95,
    )

    ax.set_title(f"{sample_id}: top 1% curl hotspots — {flux_name}")
    ax.invert_yaxis()
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 11 TNBC: Lie-structured spatial null with operator-derived edge weights."
    )
    parser.add_argument("--sample_id", required=True)
    parser.add_argument("--sample_dir", required=True)
    parser.add_argument(
        "--flux_name",
        required=True,
        choices=["flux_tumor_immune", "flux_tumor_stroma", "flux_immune_stroma"],
    )
    parser.add_argument("--statsdir", default="stats")
    parser.add_argument("--figdir", default="visium_figures")
    parser.add_argument("--n_null", type=int, default=500)
    parser.add_argument("--n_eigs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--interface_region",
        default="interface_like",
        help="Region name used for edge-localized hotspot summary.",
    )
    args = parser.parse_args()

    sample_id = args.sample_id
    flux_name = args.flux_name
    sample_dir = Path(args.sample_dir).resolve()
    statsdir = Path(args.statsdir)
    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)

    nodes_csv = require_file(statsdir / f"{sample_id}_step3_nodes.csv")
    faces_csv = require_file(statsdir / f"{sample_id}_step3_faces.csv")
    edges_hodge_csv = require_file(statsdir / f"{sample_id}_step6_edges_hodge_{flux_name}.csv")
    edges_csv = require_file(statsdir / f"{sample_id}_step3_edges.csv")
    L1_path = require_file(statsdir / f"{sample_id}_step3_L1_edge_hodge.npz")
    B2_path = require_file(statsdir / f"{sample_id}_step3_B2.npz")

    node_df = pd.read_csv(nodes_csv)
    face_df = pd.read_csv(faces_csv)
    edge_df_hodge = pd.read_csv(edges_hodge_csv)
    edge_df = pd.read_csv(edges_csv)

    # Ensure region metadata is present in edge table used for hotspot summaries
    if "tail_region" not in edge_df.columns or "head_region" not in edge_df.columns:
        raise KeyError("step3_edges.csv must contain tail_region and head_region for the Lie-step summaries.")

    L1 = sparse.load_npz(L1_path).tocsr()
    B2 = sparse.load_npz(B2_path).tocsr()

    image, hires_scale = load_sample_image_and_scale(sample_dir)
    x_face, y_face = triangle_centroids(node_df, face_df)

    # Use the total flux field from Step 6
    f_real = edge_df_hodge["flux_total"].to_numpy(dtype=float)

    vals_pos, vecs_pos = robust_smallest_positive_eigs(L1, k=args.n_eigs)
    lam_min_pos = float(vals_pos[0])
    delta = float(vals_pos[1] - vals_pos[0]) if len(vals_pos) > 1 else lam_min_pos

    eps_reg = max(0.05 * lam_min_pos, 1e-10)
    s_exp = choose_s_from_gap(delta, lam_min_pos)

    weights = spectral_resolvent_diag_weights(vals_pos, vecs_pos, eps_reg=eps_reg, s_exp=s_exp)
    weights_norm = weights / max(np.max(weights), 1e-18)

    rng = np.random.default_rng(args.seed)

    real_stats = face_curl_stats(B2, f_real)
    curl_real_abs = np.abs(np.asarray(B2.T @ f_real).ravel())

    real_hot_if, _ = edge_level_coexact_hotspot_stat(
        edge_df=edge_df,
        f_real=f_real,
        f_null=f_real,
        region_name=args.interface_region,
        q=0.95,
    )

    null_rows = []
    for k in range(args.n_null):
        f_null = build_lie_surrogate(f_real, weights_norm, rng)
        s = face_curl_stats(B2, f_null)
        _, null_hot_if = edge_level_coexact_hotspot_stat(
            edge_df=edge_df,
            f_real=f_real,
            f_null=f_null,
            region_name=args.interface_region,
            q=0.95,
        )
        s["edge_hotspot_top95_" + args.interface_region] = null_hot_if
        s["replicate"] = k
        null_rows.append(s)

    null_df = pd.DataFrame(null_rows)

    # Empirical one-sided p-values: real > null
    p_mean = (1.0 + np.sum(null_df["mean_curl"].to_numpy() >= real_stats["mean_curl"])) / (args.n_null + 1.0)
    p_top95 = (1.0 + np.sum(null_df["top95_mean_curl"].to_numpy() >= real_stats["top95_mean_curl"])) / (args.n_null + 1.0)
    p_top99 = (1.0 + np.sum(null_df["top99_mean_curl"].to_numpy() >= real_stats["top99_mean_curl"])) / (args.n_null + 1.0)
    p_hot_if = (
        1.0
        + np.sum(null_df["edge_hotspot_top95_" + args.interface_region].to_numpy() >= real_hot_if)
    ) / (args.n_null + 1.0)

    summary = pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "flux_name": flux_name,
                "n_null": args.n_null,
                "n_eigs": args.n_eigs,
                "lambda_min_pos": lam_min_pos,
                "spectral_gap_delta": delta,
                "eps_reg": eps_reg,
                "s_exp": s_exp,
                "real_mean_curl": real_stats["mean_curl"],
                "real_median_curl": real_stats["median_curl"],
                "real_top95_mean_curl": real_stats["top95_mean_curl"],
                "real_top99_mean_curl": real_stats["top99_mean_curl"],
                f"real_edge_hotspot_top95_{args.interface_region}": real_hot_if,
                "null_mean_of_mean_curl": float(null_df["mean_curl"].mean()),
                "null_sd_of_mean_curl": float(null_df["mean_curl"].std()),
                "null_mean_of_top95_mean_curl": float(null_df["top95_mean_curl"].mean()),
                "null_mean_of_top99_mean_curl": float(null_df["top99_mean_curl"].mean()),
                f"null_mean_edge_hotspot_top95_{args.interface_region}": float(
                    null_df["edge_hotspot_top95_" + args.interface_region].mean()
                ),
                "p_mean_curl": float(p_mean),
                "p_top95_mean_curl": float(p_top95),
                "p_top99_mean_curl": float(p_top99),
                f"p_edge_hotspot_top95_{args.interface_region}": float(p_hot_if),
            }
        ]
    )

    weights_df = pd.DataFrame(
        {
            "edge_id": np.arange(len(weights_norm), dtype=int),
            "weight_zeta": weights,
            "weight_norm": weights_norm,
        }
    )

    out_null_csv = statsdir / f"{sample_id}_step11_lie_null_distribution_{flux_name}.csv"
    out_summary_csv = statsdir / f"{sample_id}_step11_lie_null_summary_{flux_name}.csv"
    out_weights_csv = statsdir / f"{sample_id}_step11_lie_weights_{flux_name}.csv"

    null_df.to_csv(out_null_csv, index=False)
    summary.to_csv(out_summary_csv, index=False)
    weights_df.to_csv(out_weights_csv, index=False)

    hist_mean_png = figdir / f"{sample_id}_step11_lie_null_hist_meancurl_{flux_name}.png"
    hist_top95_png = figdir / f"{sample_id}_step11_lie_null_hist_top95curl_{flux_name}.png"
    hist_hot_png = figdir / f"{sample_id}_step11_lie_null_hist_hotspot_{flux_name}.png"
    hotspot_png = figdir / f"{sample_id}_step11_lie_hotspots_{flux_name}.png"

    plot_null_hist(
        null_vals=null_df["mean_curl"].to_numpy(),
        real_val=real_stats["mean_curl"],
        title=f"{sample_id}: Lie-null mean curl — {flux_name}",
        xlabel="null mean curl",
        outpath=hist_mean_png,
    )

    plot_null_hist(
        null_vals=null_df["top95_mean_curl"].to_numpy(),
        real_val=real_stats["top95_mean_curl"],
        title=f"{sample_id}: Lie-null top-5% curl mean — {flux_name}",
        xlabel="null top-5% curl mean",
        outpath=hist_top95_png,
    )

    plot_null_hist(
        null_vals=null_df["edge_hotspot_top95_" + args.interface_region].to_numpy(),
        real_val=real_hot_if,
        title=f"{sample_id}: Lie-null {args.interface_region} edge-hotspot statistic — {flux_name}",
        xlabel=f"null top-5% |f| in {args.interface_region}",
        outpath=hist_hot_png,
    )

    plot_hotspot_overlay(
        image=image,
        hires_scale=hires_scale,
        x_face=x_face,
        y_face=y_face,
        curl_real_abs=curl_real_abs,
        sample_id=sample_id,
        flux_name=flux_name,
        outpath=hotspot_png,
    )

    print("=" * 72)
    print(f"STEP 11: Lie-structured null for {sample_id}")
    print("=" * 72)
    print(f"Flux                : {flux_name}")
    print(f"n_null              : {args.n_null}")
    print(f"n_eigs              : {args.n_eigs}")
    print(f"lambda_min_pos      : {lam_min_pos:.6e}")
    print(f"spectral_gap_delta  : {delta:.6e}")
    print(f"eps_reg             : {eps_reg:.6e}")
    print(f"s_exp               : {s_exp:.6f}")
    print("-" * 72)
    print(f"Real mean curl      : {real_stats['mean_curl']:.6e}")
    print(f"Null mean(mean curl): {null_df['mean_curl'].mean():.6e}")
    print(f"p_mean_curl         : {p_mean:.6f}")
    print(f"Real top95 curl     : {real_stats['top95_mean_curl']:.6e}")
    print(f"Null mean(top95)    : {null_df['top95_mean_curl'].mean():.6e}")
    print(f"p_top95_mean_curl   : {p_top95:.6f}")
    print(f"Real top99 curl     : {real_stats['top99_mean_curl']:.6e}")
    print(f"Null mean(top99)    : {null_df['top99_mean_curl'].mean():.6e}")
    print(f"p_top99_mean_curl   : {p_top99:.6f}")
    print(f"Real {args.interface_region} hotspot stat : {real_hot_if:.6e}")
    print(
        f"Null mean hotspot stat                  : "
        f"{null_df['edge_hotspot_top95_' + args.interface_region].mean():.6e}"
    )
    print(f"p_hotspot            : {p_hot_if:.6f}")
    print("-" * 72)
    print(f"Saved: {out_null_csv}")
    print(f"Saved: {out_summary_csv}")
    print(f"Saved: {out_weights_csv}")
    print(f"Saved: {hist_mean_png}")
    print(f"Saved: {hist_top95_png}")
    print(f"Saved: {hist_hot_png}")
    print(f"Saved: {hotspot_png}")


if __name__ == "__main__":
    main()
