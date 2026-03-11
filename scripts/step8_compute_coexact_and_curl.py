#!/usr/bin/env python3
"""
Step 8: Coexact flux analysis, including fixes and LaTeX table output.
"""

import argparse, json, platform, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
from PIL import Image

def require_file(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {p.resolve()}")

def load_spatial_image(spatial_dir):
    require_file(spatial_dir/"scalefactors_json.json")
    require_file(spatial_dir/"tissue_hires_image.png")
    sf = json.load(open(spatial_dir/"scalefactors_json.json"))
    img = np.array(Image.open(spatial_dir/"tissue_hires_image.png"))
    return img, float(sf["tissue_hires_scalef"])

def build_edge_lookup_and_neighbors(edges_df):
    """
    Build edge lookup and adjacency list. Size by max node index to avoid errors.
    """
    edge_lookup = {}
    max_idx = int(max(edges_df["tail"].max(), edges_df["head"].max()) + 1)
    neighbors = [set() for _ in range(max_idx)]
    for idx, row in edges_df.iterrows():
        i, j = int(row["tail"]), int(row["head"])
        a, b = (i, j) if i < j else (j, i)
        edge_lookup[(a, b)] = int(idx)
        neighbors[a].add(b)
        neighbors[b].add(a)
    return edge_lookup, neighbors

def mannwhitney_test(x, y):
    stat, p = mannwhitneyu(x, y, alternative="two-sided")
    return float(stat), float(p)

def permutation_test(values, regions, ra, rb, n_perm, seed):
    idx_a = np.where(regions == ra)[0]
    idx_b = np.where(regions == rb)[0]
    if len(idx_a) < 1 or len(idx_b) < 1:
        return np.nan
    med_diff = abs(np.median(values[idx_a]) - np.median(values[idx_b]))
    rng = np.random.default_rng(seed)
    ge = 0
    combined = values.copy()
    for _ in range(n_perm):
        rng.shuffle(combined)
        if abs(np.median(combined[idx_a]) - np.median(combined[idx_b])) >= med_diff:
            ge += 1
    return (ge + 1) / (n_perm + 1)  # add-one smoothing

def scatter_map(img, scale, x, y, vals, title, fname):
    plt.figure(figsize=(8,8))
    plt.imshow(img)
    sc = plt.scatter(x*scale, y*scale, c=vals, cmap='viridis', s=10, alpha=0.8)
    plt.title(title)
    plt.axis('off')
    plt.colorbar(sc, ax=plt.gca())
    plt.savefig(fname, dpi=200)
    plt.close()

def boxplot_by_region(df, column, title, fname):
    regions = ["tumor_core","invasive_margin","stroma","immune_rich","mixed_unassigned"]
    data = [df[df["region"]==r][column].values for r in regions]
    plt.figure(figsize=(7,5))
    plt.boxplot(data, labels=regions, showfliers=False)
    plt.title(title)
    plt.xticks(rotation=30)
    plt.ylabel(column)
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Compute coexact node energy, curl, and region stats (with LaTeX output).")
    parser.add_argument("--perm", type=int, default=1000, help="Number of permutations")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--outdir", type=str, default=".", help="Output directory")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    # Required input files
    required_files = [
        "step7_energy_fractions.csv",
        "step7_nodes_hodge_immune_tumor_wedge.csv",
        "step7_edges_hodge_immune_tumor_wedge.csv",
        "step7_nodes_hodge_stroma_tumor_wedge.csv",
        "step7_edges_hodge_stroma_tumor_wedge.csv",
        "step7_nodes_hodge_immune_stroma_wedge.csv",
        "step7_edges_hodge_immune_stroma_wedge.csv",
        "step3_nodes.csv", "step3_edges.csv", "step3_B1_incidence.npz",
        "step2_region_assignments.csv",
        "spatial/scalefactors_json.json", "spatial/tissue_hires_image.png"
    ]
    for rf in required_files:
        require_file(rf)

    nodes_df = pd.read_csv("step3_nodes.csv")
    edges_df = pd.read_csv("step3_edges.csv")
    B1 = sparse.load_npz("step3_B1_incidence.npz").tocsr()
    n_nodes, n_edges = nodes_df.shape[0], edges_df.shape[0]

    img, scale = load_spatial_image(Path("spatial"))

    # Build triangles (B2) if needed
    tri_file = outdir/"step8_triangles.csv"
    if not tri_file.exists():
        edge_lookup, neighbors = build_edge_lookup_and_neighbors(edges_df)
        triangles = []
        for i in range(n_nodes):
            for j in neighbors[i]:
                if j <= i: continue
                common = neighbors[i].intersection(neighbors[j])
                for k in common:
                    if k <= j: continue
                    triangles.append((i, j, k))
        if triangles:
            pd.DataFrame(triangles, columns=["i","j","k"]).to_csv(tri_file, index=False)
            row, col, data = [], [], []
            for fid,(i,j,k) in enumerate(triangles):
                e1 = edge_lookup[(min(i,j), max(i,j))]
                e2 = edge_lookup[(min(i,k), max(i,k))]
                e3 = edge_lookup[(min(j,k), max(j,k))]
                row += [e1, e2, e3]
                col += [fid]*3
                data += [1, -1, 1]
            B2 = sparse.csr_matrix((data, (row, col)), shape=(n_edges, len(triangles)))
            sparse.save_npz(outdir/"step8_B2_faces.npz", B2)
        else:
            B2 = sparse.csr_matrix(([], ([], [])), shape=(n_edges, 0))
    else:
        triangles = pd.read_csv(tri_file)[["i","j","k"]].values.tolist()
        B2 = sparse.load_npz(outdir/"step8_B2_faces.npz")

    fluxes = ["immune_tumor_wedge", "stroma_tumor_wedge", "immune_stroma_wedge"]
    combined_rows = []
    energy_fracs = pd.read_csv("step7_energy_fractions.csv")

    for flux in fluxes:
        node_file = f"step7_nodes_hodge_{flux}.csv"
        edge_file = f"step7_edges_hodge_{flux}.csv"
        require_file(node_file); require_file(edge_file)

        node_df = pd.read_csv(node_file)
        edge_df = pd.read_csv(edge_file)
        if "region" not in node_df.columns:
            reg2 = pd.read_csv("step2_region_assignments.csv", index_col="barcode")["region_step2"]
            node_df["region"] = node_df["barcode"].map(reg2).fillna("mixed_unassigned")

        absB1 = np.abs(B1)
        degree = absB1 @ np.ones(n_edges)
        degree = np.maximum(degree, 1.0)
        if "node_energy_total" not in node_df:
            f_total = edge_df["flux_total"].to_numpy(float)
            node_df["node_energy_total"] = (absB1 @ (f_total**2)) / degree
        if "node_energy_coexact" not in node_df:
            f_co = edge_df["flux_coexact"].to_numpy(float)
            node_df["node_energy_coexact"] = (absB1 @ (f_co**2)) / degree
        node_df["coexact_fraction"] = node_df["node_energy_coexact"] / node_df["node_energy_total"].clip(lower=1e-18)

        node_df[["node_id","barcode","x_fullres","y_fullres","region",
                 "node_energy_coexact","node_energy_total","coexact_fraction"]]\
            .to_csv(outdir/f"step8_node_abs_coexact_{flux}.csv", index=False)

        sign_can = np.where(edges_df["tail"] < edges_df["head"], 1.0, -1.0)
        f_total_oriented = edge_df["flux_total"].to_numpy(float)
        f_can = sign_can * f_total_oriented
        if B2.shape[1] > 0:
            curl_vals = np.abs(B2.T @ f_can)
        else:
            curl_vals = np.array([])
        node_curl = np.zeros(n_nodes); node_ct = np.zeros(n_nodes)
        for fid,(i,j,k) in enumerate(triangles):
            c = curl_vals[fid] if fid < len(curl_vals) else 0.0
            for v in (i,j,k):
                node_curl[v] += c
                node_ct[v] += 1
        node_df["node_mean_curl"] = node_curl / np.maximum(node_ct, 1.0)

        faces_df = pd.DataFrame(triangles, columns=["i","j","k"])
        faces_df["face_id"] = np.arange(len(triangles))
        faces_df["curl"] = curl_vals
        faces_df["mapped_node_mean_curl"] = (
            node_df.loc[faces_df["i"],"node_mean_curl"].values +
            node_df.loc[faces_df["j"],"node_mean_curl"].values +
            node_df.loc[faces_df["k"],"node_mean_curl"].values
        ) / 3.0
        faces_df[["face_id","i","j","k","curl","mapped_node_mean_curl"]]\
            .to_csv(outdir/f"step8_face_curl_{flux}.csv", index=False)

        medians = node_df.groupby("region")[["node_energy_total","node_energy_coexact","coexact_fraction"]].median()
        if "tumor_core" not in medians.index:
            raise ValueError(f"{flux}: missing tumor_core region for summary")
        tc = medians.loc["tumor_core"]
        row = {"flux": flux}
        erow = energy_fracs[energy_fracs["flux"]==flux].iloc[0]
        row.update({
            "global_frac_exact": float(erow["frac_exact"]),
            "global_frac_coexact": float(erow["frac_coexact"]),
            "global_frac_harmonic": float(erow["frac_harmonic"])
        })
        for region in medians.index:
            row[f"node_energy_total_median_{region}"] = float(medians.loc[region,"node_energy_total"])
            row[f"node_energy_coexact_median_{region}"] = float(medians.loc[region,"node_energy_coexact"])
            row[f"coexact_fraction_median_{region}"] = float(medians.loc[region,"coexact_fraction"])
            row[f"node_energy_coexact_ratio_vs_tc_{region}"] = (
                float(medians.loc[region,"node_energy_coexact"] / tc["node_energy_coexact"])
                if tc["node_energy_coexact"] != 0 else np.nan
            )
        combined_rows.append(row)

        tests = []
        for metric in ["node_energy_coexact","node_mean_curl","coexact_fraction"]:
            vals = node_df[metric].values; regs = node_df["region"].values
            for ra, rb in [("tumor_core","invasive_margin"), ("tumor_core","stroma"), ("tumor_core","immune_rich")]:
                xa = vals[regs==ra]; xb = vals[regs==rb]
                if len(xa)<10 or len(xb)<10: continue
                stat, pval = mannwhitney_test(xa, xb)
                perm_p = permutation_test(vals, regs, ra, rb, args.perm, args.seed)
                tests.append({
                    "flux": flux, "metric": metric,
                    "region_a": ra, "region_b": rb,
                    "n_a": len(xa), "n_b": len(xb),
                    "median_a": float(np.median(xa)), "median_b": float(np.median(xb)),
                    "u_stat": stat, "u_p_value": pval,
                    "perm_p_value": perm_p
                })
        pd.DataFrame(tests).to_csv(outdir/f"step8_region_tests_{flux}.csv", index=False)

        scatter_map(img, scale, node_df["x_fullres"].to_numpy(float), node_df["y_fullres"].to_numpy(float),
                    node_df["node_energy_coexact"].to_numpy(float),
                    f"Node coexact energy ({flux})", outdir/f"step8_map_node_abs_coexact_{flux}.png")
        scatter_map(img, scale, node_df["x_fullres"].to_numpy(float), node_df["y_fullres"].to_numpy(float),
                    node_df["node_mean_curl"].to_numpy(float),
                    f"Node mean curl ({flux})", outdir/f"step8_map_node_mean_curl_{flux}.png")
        boxplot_by_region(node_df, "node_energy_coexact",
                          f"Coexact energy by region ({flux})", outdir/f"step8_boxplots_abs_coexact_{flux}.png")
        boxplot_by_region(node_df, "node_mean_curl",
                          f"Mean curl by region ({flux})", outdir/f"step8_boxplots_node_mean_curl_{flux}.png")

    # Write combined summary CSV
    pd.DataFrame(combined_rows).to_csv(outdir/"combined_step7_summary.csv", index=False)

    # LaTeX table formatting
    combined_df = pd.read_csv(outdir/"combined_step7_summary.csv")
    required_cols = ["flux","global_frac_exact","global_frac_coexact","global_frac_harmonic"]
    for col in required_cols:
        if col not in combined_df.columns:
            raise KeyError(f"Missing required column: {col}")
    float_cols = combined_df.select_dtypes(include=["float"]).columns
    for col in float_cols:
        combined_df[col] = combined_df[col].apply(lambda x: f"{x:.3e}")
    caption = ("Combined summary of global energy fractions and regional medians "
               "(flux: immune–tumor (IT), stroma–tumor (ST), immune–stroma (IS))")
    label = "tab:combined_summary"
    tex_str = combined_df.to_latex(index=False, caption=caption, label=label, column_format="lcccccc")
    with open(outdir/"combined_step7_summary.tex", "w") as texf:
        texf.write(tex_str)
    print("LaTeX table written to combined_step7_summary.tex (include with \input{})".format("{"+"combined_step7_summary.tex"+"}"))

    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "python": sys.version, "platform": platform.platform(),
        "fluxes": fluxes, "n_nodes": n_nodes, "n_edges": n_edges
    }
    with open(outdir/"step8_run_metadata.json","w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
