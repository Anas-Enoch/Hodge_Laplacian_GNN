from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


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


def subsample_edges_by_quantile(
    edge_df: pd.DataFrame,
    flux_col: str,
    q: float = 0.92,
    max_edges: int = 700,
    seed: int = 0,
) -> pd.DataFrame:
    vals = np.abs(edge_df[flux_col].to_numpy(dtype=float))
    thr = np.quantile(vals, q)
    sub = edge_df.loc[vals >= thr].copy()

    if len(sub) > max_edges:
        rng = np.random.default_rng(seed)
        keep = rng.choice(sub.index.to_numpy(), size=max_edges, replace=False)
        sub = sub.loc[np.sort(keep)].copy()

    return sub


def build_edge_midpoints_and_vectors(
    edge_df: pd.DataFrame,
    flux_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_tail = edge_df["x_tail"].to_numpy(dtype=float)
    y_tail = edge_df["y_tail"].to_numpy(dtype=float)
    x_head = edge_df["x_head"].to_numpy(dtype=float)
    y_head = edge_df["y_head"].to_numpy(dtype=float)

    xm = 0.5 * (x_tail + x_head)
    ym = 0.5 * (y_tail + y_head)

    dx = x_head - x_tail
    dy = y_head - y_tail

    length = np.sqrt(dx**2 + dy**2)
    length = np.where(length <= 1e-12, 1.0, length)

    ux = dx / length
    uy = dy / length

    flux = edge_df[flux_col].to_numpy(dtype=float)

    vx = flux * ux
    vy = flux * uy

    return xm, ym, vx, vy


def highlight_interface_spots(
    ax,
    region_df: pd.DataFrame,
    hires_scale: float,
    label: str = "interface_like",
) -> None:
    sub = region_df.loc[region_df["region_step2"] == label]
    if sub.empty:
        return

    xs = sub["x_fullres"].to_numpy(dtype=float) * hires_scale
    ys = sub["y_fullres"].to_numpy(dtype=float) * hires_scale

    ax.scatter(
        xs,
        ys,
        s=8,
        alpha=0.18,
    )


def panel_observed_proxy(
    ax,
    image: np.ndarray,
    hires_scale: float,
    proxy_edges: pd.DataFrame,
    region_df: pd.DataFrame,
    flux_col: str,
) -> None:
    ax.imshow(image)
    highlight_interface_spots(ax, region_df, hires_scale)

    xm, ym, vx, vy = build_edge_midpoints_and_vectors(proxy_edges, flux_col)

    ax.quiver(
        xm * hires_scale,
        ym * hires_scale,
        vx,
        vy,
        angles="xy",
        scale_units="xy",
        scale=0.015,
        width=0.0022,
        alpha=0.9,
    )
    ax.set_title("A  Observed proxy transport")
    ax.invert_yaxis()
    ax.axis("off")


def panel_gnn_learned(
    ax,
    image: np.ndarray,
    hires_scale: float,
    gnn_edges: pd.DataFrame,
    region_df: pd.DataFrame,
    flux_col: str,
) -> None:
    ax.imshow(image)
    highlight_interface_spots(ax, region_df, hires_scale)

    xm, ym, vx, vy = build_edge_midpoints_and_vectors(gnn_edges, flux_col)

    ax.quiver(
        xm * hires_scale,
        ym * hires_scale,
        vx,
        vy,
        angles="xy",
        scale_units="xy",
        scale=0.015,
        width=0.0022,
        alpha=0.9,
    )
    ax.set_title("B  PDE-constrained learned transport")
    ax.invert_yaxis()
    ax.axis("off")


def panel_curl_hotspots(
    ax,
    image: np.ndarray,
    hires_scale: float,
    face_curl_df: pd.DataFrame,
    region_df: pd.DataFrame,
    hotspot_q: float = 0.99,
) -> None:
    ax.imshow(image)
    highlight_interface_spots(ax, region_df, hires_scale)

    vals = face_curl_df["curl_coexact_abs"].to_numpy(dtype=float)
    thr = np.quantile(vals, hotspot_q)
    mask = vals >= thr

    ax.scatter(
        face_curl_df.loc[mask, "x_centroid"].to_numpy(dtype=float) * hires_scale,
        face_curl_df.loc[mask, "y_centroid"].to_numpy(dtype=float) * hires_scale,
        c="red",
        s=70,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.95,
    )
    ax.set_title("C  Top 1% curl hotspots")
    ax.invert_yaxis()
    ax.axis("off")


def panel_energy_collapse(
    ax,
    proxy_summary: pd.DataFrame,
    gnn_summary: pd.DataFrame,
) -> None:
    proxy_exact = float(proxy_summary["frac_exact"].iloc[0])
    proxy_coexact = float(proxy_summary["frac_coexact"].iloc[0])

    gnn_exact = float(gnn_summary["frac_exact"].iloc[0])
    gnn_coexact = float(gnn_summary["frac_coexact"].iloc[0])

    labels = ["Exact", "Coexact"]
    x = np.arange(len(labels))
    w = 0.36

    ax.bar(x - w / 2, [proxy_exact, proxy_coexact], width=w, label="Proxy")
    ax.bar(x + w / 2, [gnn_exact, gnn_coexact], width=w, label="PDE-GNN")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylim(0, 1.05)
    ax.set_title("D  Coexact collapse")
    ax.legend(frameon=False, fontsize=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 16 TNBC: 4-panel figure showing observed transport, learned transport, curl hotspots, and coexact collapse."
    )
    parser.add_argument("--sample_id", required=True)
    parser.add_argument(
        "--target_flux",
        required=True,
        choices=["flux_tumor_immune", "flux_tumor_stroma", "flux_immune_stroma"],
    )
    parser.add_argument("--sample_dir", required=True)
    parser.add_argument("--statsdir", default="stats")
    parser.add_argument("--figdir", default="visium_figures")
    parser.add_argument("--proxy_quantile", type=float, default=0.92)
    parser.add_argument("--gnn_quantile", type=float, default=0.92)
    parser.add_argument("--hotspot_quantile", type=float, default=0.99)
    parser.add_argument("--max_edges", type=int, default=700)
    args = parser.parse_args()

    sample_id = args.sample_id
    target_flux = args.target_flux
    statsdir = Path(args.statsdir)
    figdir = Path(args.figdir)
    sample_dir = Path(args.sample_dir).resolve()
    figdir.mkdir(parents=True, exist_ok=True)

    image, hires_scale = load_sample_image_and_scale(sample_dir)

    region_csv = require_file(statsdir / f"{sample_id}_step2_region_assignments.csv")
    proxy_edges_csv = require_file(statsdir / f"{sample_id}_step4_edge_fluxes.csv")
    gnn_edges_csv = require_file(statsdir / f"{sample_id}_step14_gnn_learned_flux_{target_flux}.csv")
    edges_geo_csv = require_file(statsdir / f"{sample_id}_step3_edges.csv")
    face_curl_csv = require_file(statsdir / f"{sample_id}_step15_face_curl_gnn_{target_flux}.csv")
    proxy_summary_csv = require_file(statsdir / f"{sample_id}_step6_energy_summary_{target_flux}.csv")
    gnn_summary_csv = require_file(statsdir / f"{sample_id}_step15_gnn_operator_summary_{target_flux}.csv")

    region_df = pd.read_csv(region_csv)
    proxy_edges = pd.read_csv(proxy_edges_csv)
    gnn_edges = pd.read_csv(gnn_edges_csv)
    edges_geo = pd.read_csv(edges_geo_csv)
    face_curl_df = pd.read_csv(face_curl_csv)
    proxy_summary = pd.read_csv(proxy_summary_csv)
    gnn_summary = pd.read_csv(gnn_summary_csv)

    # Attach geometry to GNN edge file
    geo_cols = [
        "edge_id",
        "x_tail",
        "y_tail",
        "x_head",
        "y_head",
        "tail_region",
        "head_region",
    ]
    missing_geo = [c for c in geo_cols if c not in edges_geo.columns]
    if missing_geo:
        raise KeyError(f"Missing geometry columns in step3_edges.csv: {missing_geo}")

    gnn_plot_df = gnn_edges.merge(edges_geo[geo_cols], on="edge_id", how="left")
    if gnn_plot_df["x_tail"].isna().any():
        raise RuntimeError("Failed to merge GNN edge predictions with edge geometry by edge_id.")

    # Proxy edge file already contains geometry in the current pipeline
    needed_proxy_cols = ["x_tail", "y_tail", "x_head", "y_head", target_flux]
    missing_proxy = [c for c in needed_proxy_cols if c not in proxy_edges.columns]
    if missing_proxy:
        raise KeyError(
            f"Proxy edge file is missing required columns for plotting: {missing_proxy}"
        )

    if "flux_gnn_unscaled" not in gnn_plot_df.columns:
        raise KeyError("Expected 'flux_gnn_unscaled' in step14 learned flux file.")

    proxy_sub = subsample_edges_by_quantile(
        proxy_edges,
        flux_col=target_flux,
        q=args.proxy_quantile,
        max_edges=args.max_edges,
        seed=0,
    )

    gnn_sub = subsample_edges_by_quantile(
        gnn_plot_df,
        flux_col="flux_gnn_unscaled",
        q=args.gnn_quantile,
        max_edges=args.max_edges,
        seed=1,
    )

    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.65])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])

    panel_observed_proxy(
        ax=ax1,
        image=image,
        hires_scale=hires_scale,
        proxy_edges=proxy_sub,
        region_df=region_df,
        flux_col=target_flux,
    )

    panel_gnn_learned(
        ax=ax2,
        image=image,
        hires_scale=hires_scale,
        gnn_edges=gnn_sub,
        region_df=region_df,
        flux_col="flux_gnn_unscaled",
    )

    panel_curl_hotspots(
        ax=ax3,
        image=image,
        hires_scale=hires_scale,
        face_curl_df=face_curl_df,
        region_df=region_df,
        hotspot_q=args.hotspot_quantile,
    )

    panel_energy_collapse(
        ax=ax4,
        proxy_summary=proxy_summary,
        gnn_summary=gnn_summary,
    )

    plt.suptitle(
        f"{sample_id}: observed transport, conservation-constrained learning, and coexact collapse — {target_flux}",
        y=0.98,
    )
    plt.tight_layout()

    out_png = figdir / f"{sample_id}_step16_transport_equation_figure_{target_flux}.png"
    plt.savefig(out_png, dpi=300)
    plt.close(fig)

    print("=" * 80)
    print("STEP 16: transport equation figure")
    print("=" * 80)
    print(f"sample_id         : {sample_id}")
    print(f"target_flux       : {target_flux}")
    print(f"proxy_edges_shown : {len(proxy_sub)}")
    print(f"gnn_edges_shown   : {len(gnn_sub)}")
    print(f"hotspot_quantile  : {args.hotspot_quantile}")
    print("-" * 80)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
