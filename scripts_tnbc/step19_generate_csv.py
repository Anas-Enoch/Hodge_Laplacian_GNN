from pathlib import Path
import argparse
import numpy as np
import pandas as pd


# =========================
# CLI
# =========================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Step 19 CSV generator.\n\n"
            "Builds two node-level CSV files required by step19_coexact_bio_correlation.py:\n"
            "  1. {sample_id}_step6_nodes_hodge_{flux_tag}.csv\n"
            "  2. {sample_id}_{flux_tag}_nodes_for_gnn.csv\n\n"
            "Example:\n"
            "  python step19_generate_csv.py \\\n"
            "    --sample-id GSM_6433619 \\\n"
            "    --flux-tag flux_tumor_immune_region_interface_weighted"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--sample-id",
        required=True,
        help="GSM sample ID (e.g. GSM_6433619).",
    )
    p.add_argument(
        "--flux-tag",
        default="flux_tumor_immune_region_interface_weighted",
        help=(
            "Flux column tag used in Step 6 output files. "
            "Must match the --flux-col used in step6_tnbc_hodge_decomposition.py. "
            "(default: flux_tumor_immune_region_interface_weighted)"
        ),
    )
    p.add_argument(
        "--statsdir",
        default="stats/CSV_GSM",
        help="Directory containing Step 3/4/6 CSVs (default: stats/CSV_GSM).",
    )
    p.add_argument(
        "--gnndir",
        default="stats/gnn_data",
        help="Output directory for nodes_for_gnn CSV (default: stats/gnn_data).",
    )
    return p.parse_args()


args = _parse_args()

sample_id = args.sample_id
flux_tag  = args.flux_tag

csv_gsm  = Path(args.statsdir)
gnn_data = Path(args.gnndir)
csv_gsm.mkdir(parents=True, exist_ok=True)
gnn_data.mkdir(parents=True, exist_ok=True)


# =========================
# PATHS
# =========================

edge_file  = csv_gsm  / f"{sample_id}_step6_edges_hodge_{flux_tag}.csv"
nodes_file = csv_gsm  / f"{sample_id}_step3_nodes.csv"
resid_file = csv_gsm  / f"{sample_id}_step4_node_residualized_fields.csv"

out_nodes_hodge  = csv_gsm  / f"{sample_id}_step6_nodes_hodge_{flux_tag}.csv"
out_nodes_for_gnn = gnn_data / f"{sample_id}_{flux_tag}_nodes_for_gnn.csv"

print(f"Sample:    {sample_id}")
print(f"Flux tag:  {flux_tag}")
print(f"Edges:     {edge_file}")
print(f"Nodes:     {nodes_file}")
print(f"Residuals: {resid_file}")
print()

for f in [edge_file, nodes_file, resid_file]:
    if not f.exists():
        raise FileNotFoundError(f"Required input not found: {f}")


# =========================
# LOAD
# =========================

edges = pd.read_csv(edge_file)
nodes = pd.read_csv(nodes_file)
resid = pd.read_csv(resid_file)


# =========================
# VALIDATE COLUMNS
# =========================

required_edge_cols = [
    "tail", "head",
    flux_tag,           # raw flux column — uses the actual flux_tag
    "flux_exact",
    "flux_coexact",
    "flux_harmonic",
]
missing = [c for c in required_edge_cols if c not in edges.columns]
if missing:
    raise ValueError(
        f"Missing edge columns: {missing}\n"
        f"Available: {edges.columns.tolist()}\n"
        f"Check that --flux-tag matches the column name in {edge_file.name}"
    )

required_node_cols = [
    "node_id", "barcode", "x_fullres", "y_fullres",
    "tumor_score", "stroma_score", "immune_score",
    "interface_score", "region_step2",
]
missing = [c for c in required_node_cols if c not in nodes.columns]
if missing:
    raise ValueError(f"Missing node columns: {missing}")

required_resid_cols = [
    "node_id", "barcode", "x_fullres", "y_fullres",
    "tumor_score", "stroma_score", "immune_score",
    "tumor_residual", "stroma_residual", "immune_residual",
    "region_step2",
]
missing = [c for c in required_resid_cols if c not in resid.columns]
if missing:
    raise ValueError(f"Missing residual columns: {missing}")


# =========================
# BUILD step6 PROXY NODE HODGE FILE
# =========================

rows = []
for node_id, node_row in nodes.set_index("node_id").iterrows():
    sub = edges[(edges["tail"] == node_id) | (edges["head"] == node_id)].copy()

    if len(sub) == 0:
        node_energy_total    = 0.0
        node_energy_exact    = 0.0
        node_energy_coexact  = 0.0
        node_energy_harmonic = 0.0
    else:
        node_energy_total    = float(np.mean(sub[flux_tag].to_numpy() ** 2))
        node_energy_exact    = float(np.mean(sub["flux_exact"].to_numpy() ** 2))
        node_energy_coexact  = float(np.mean(sub["flux_coexact"].to_numpy() ** 2))
        node_energy_harmonic = float(np.mean(sub["flux_harmonic"].to_numpy() ** 2))

    denom = node_energy_total if node_energy_total > 0 else np.nan
    frac_exact    = node_energy_exact    / denom if np.isfinite(denom) else np.nan
    frac_coexact  = node_energy_coexact  / denom if np.isfinite(denom) else np.nan
    frac_harmonic = node_energy_harmonic / denom if np.isfinite(denom) else np.nan

    rows.append({
        "node_id":             int(node_id),
        "barcode":             node_row["barcode"],
        "x_fullres":           node_row["x_fullres"],
        "y_fullres":           node_row["y_fullres"],
        "tumor_score":         node_row["tumor_score"],
        "stroma_score":        node_row["stroma_score"],
        "immune_score":        node_row["immune_score"],
        "interface_score":     node_row["interface_score"],
        "region_step2":        node_row["region_step2"],
        "node_energy_total":   node_energy_total,
        "node_energy_exact":   node_energy_exact,
        "node_energy_coexact": node_energy_coexact,
        "node_energy_harmonic":node_energy_harmonic,
        "frac_exact":          frac_exact,
        "frac_coexact":        frac_coexact,
        "frac_harmonic":       frac_harmonic,
    })

nodes_hodge = pd.DataFrame(rows)
nodes_hodge.to_csv(out_nodes_hodge, index=False)


# =========================
# BUILD nodes_for_gnn FILE
# =========================

nodes_for_gnn = resid[required_resid_cols].copy()
nodes_for_gnn.to_csv(out_nodes_for_gnn, index=False)


# =========================
# SUMMARY
# =========================

print(f"Saved: {out_nodes_hodge}  ({len(nodes_hodge)} rows)")
print(f"Saved: {out_nodes_for_gnn}  ({len(nodes_for_gnn)} rows)")
print()
print(nodes_hodge[["node_id", "region_step2",
                    "node_energy_coexact", "frac_coexact"]].head().to_string(index=False))


