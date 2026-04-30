#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

SRC = Path("stats/CSV_GSM")
OUT = Path("results_tnbc_rebuild")
OUT.mkdir(exist_ok=True)

edge_files = sorted(SRC.glob("GSM_*_step6_edges_hodge_flux_tumor_immune_region_interface_weighted.csv"))

for edge_path in edge_files:
    sid = edge_path.name.split("_step6_edges_hodge")[0]
    node_path = SRC / f"{sid}_step6_nodes_hodge_flux_tumor_immune_region_interface_weighted.csv"

    if not node_path.exists():
        print(f"[skip] {sid}: missing node file")
        continue

    edges = pd.read_csv(edge_path)
    nodes = pd.read_csv(node_path)

    edges_out = pd.DataFrame({
        "i": edges["tail"].astype(int),
        "j": edges["head"].astype(int),
        "flux_coexact": edges["flux_coexact"].astype(float),
    })

    spots_out = pd.DataFrame({
        "node_id": nodes["node_id"].astype(int),
        "x": nodes["x_fullres"].astype(float),
        "y": nodes["y_fullres"].astype(float),
        "region": nodes["region_step2"].astype(str),
        "tumor_score": nodes["tumor_score"].astype(float),
        "immune_score": nodes["immune_score"].astype(float),
        "stroma_score": nodes["stroma_score"].astype(float),
        "coexact_energy": nodes["node_energy_coexact"].astype(float),
    })

    edges_out.to_csv(OUT / f"{sid}_edges_hodge.csv", index=False)
    spots_out.to_csv(OUT / f"{sid}_spots_coexact_energy.csv", index=False)

    print(f"[done] {sid}")

print(f"\nConverted {len(list(OUT.glob('*_edges_hodge.csv')))} samples → {OUT}")
