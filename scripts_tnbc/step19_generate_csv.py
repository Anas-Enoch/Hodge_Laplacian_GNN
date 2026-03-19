from pathlib import Path
import numpy as np
import pandas as pd

sample_id = "GSM_6433619"

# -----------------------------
# Paths
# -----------------------------
root_stats = Path("stats")
csv_gsm = root_stats / "CSV_GSM"
gnn_data = root_stats / "gnn_data"
csv_gsm.mkdir(parents=True, exist_ok=True)
gnn_data.mkdir(parents=True, exist_ok=True)

edge_file = csv_gsm / f"{sample_id}_step6_edges_hodge_flux_tumor_immune.csv"
nodes_file = csv_gsm / f"{sample_id}_step3_nodes.csv"
resid_file = csv_gsm / f"{sample_id}_step4_node_residualized_fields.csv"

out_nodes_hodge = csv_gsm / f"{sample_id}_step6_nodes_hodge_flux_tumor_immune.csv"
out_nodes_for_gnn = gnn_data / f"{sample_id}_flux_tumor_immune_nodes_for_gnn.csv"

# -----------------------------
# Load inputs
# -----------------------------
edges = pd.read_csv(edge_file)
nodes = pd.read_csv(nodes_file)
resid = pd.read_csv(resid_file)

# -----------------------------
# Build step6 proxy node hodge file for tumor_immune
# -----------------------------
required_edge_cols = [
    "tail", "head",
    "flux_tumor_immune",
    "flux_exact",
    "flux_coexact",
    "flux_harmonic",
]
missing = [c for c in required_edge_cols if c not in edges.columns]
if missing:
    raise ValueError(f"Missing edge columns: {missing}")

required_node_cols = [
    "node_id", "barcode", "x_fullres", "y_fullres",
    "tumor_score", "stroma_score", "immune_score",
    "interface_score", "region_step2"
]
missing = [c for c in required_node_cols if c not in nodes.columns]
if missing:
    raise ValueError(f"Missing node columns: {missing}")

rows = []
for node_id, node_row in nodes.set_index("node_id").iterrows():
    sub = edges[(edges["tail"] == node_id) | (edges["head"] == node_id)].copy()

    if len(sub) == 0:
        node_energy_total = 0.0
        node_energy_exact = 0.0
        node_energy_coexact = 0.0
        node_energy_harmonic = 0.0
    else:
        node_energy_total = float(np.mean(sub["flux_tumor_immune"].to_numpy() ** 2))
        node_energy_exact = float(np.mean(sub["flux_exact"].to_numpy() ** 2))
        node_energy_coexact = float(np.mean(sub["flux_coexact"].to_numpy() ** 2))
        node_energy_harmonic = float(np.mean(sub["flux_harmonic"].to_numpy() ** 2))

    denom = node_energy_total if node_energy_total > 0 else np.nan
    frac_exact = node_energy_exact / denom if np.isfinite(denom) else np.nan
    frac_coexact = node_energy_coexact / denom if np.isfinite(denom) else np.nan
    frac_harmonic = node_energy_harmonic / denom if np.isfinite(denom) else np.nan

    rows.append({
        "node_id": int(node_id),
        "barcode": node_row["barcode"],
        "x_fullres": node_row["x_fullres"],
        "y_fullres": node_row["y_fullres"],
        "tumor_score": node_row["tumor_score"],
        "stroma_score": node_row["stroma_score"],
        "immune_score": node_row["immune_score"],
        "interface_score": node_row["interface_score"],
        "region_step2": node_row["region_step2"],
        "node_energy_total": node_energy_total,
        "node_energy_exact": node_energy_exact,
        "node_energy_coexact": node_energy_coexact,
        "node_energy_harmonic": node_energy_harmonic,
        "frac_exact": frac_exact,
        "frac_coexact": frac_coexact,
        "frac_harmonic": frac_harmonic,
    })

nodes_hodge = pd.DataFrame(rows)
nodes_hodge.to_csv(out_nodes_hodge, index=False)

# -----------------------------
# Build nodes_for_gnn file
# -----------------------------
required_resid_cols = [
    "node_id", "barcode", "x_fullres", "y_fullres",
    "tumor_score", "stroma_score", "immune_score",
    "tumor_residual", "stroma_residual", "immune_residual",
    "region_step2",
]
missing = [c for c in required_resid_cols if c not in resid.columns]
if missing:
    raise ValueError(f"Missing residual columns: {missing}")

nodes_for_gnn = resid[required_resid_cols].copy()
nodes_for_gnn.to_csv(out_nodes_for_gnn, index=False)

print(f"Saved: {out_nodes_hodge}")
print(f"Saved: {out_nodes_for_gnn}")
print(nodes_hodge.head().to_string(index=False))
print(nodes_for_gnn.head().to_string(index=False))

