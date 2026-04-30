#!/usr/bin/env python3

"""
Normalize TNBC region labels to modern schema required by Step10 and KTS.

Mapping:
interface_like → interface
tumor_enriched → tumor_core
stroma_enriched → stroma
immune_enriched → immune
other → other

Adds:
- region_original column (for traceability)
"""

from pathlib import Path
import pandas as pd

SRC = Path("results_tnbc_rebuild")

mapping = {
    "interface_like": "interface",
    "tumor_enriched": "tumor_core",
    "stroma_enriched": "stroma",
    "immune_enriched": "immune",
    "other": "other",
}

files = sorted(SRC.glob("*_spots_coexact_energy.csv"))

if not files:
    print("[error] no spot files found in results_tnbc_rebuild/")
    exit(1)

for f in files:
    df = pd.read_csv(f)

    if "region" not in df.columns:
        print(f"[skip] {f.name}: no 'region' column")
        continue

    df["region_original"] = df["region"]

    df["region"] = (
        df["region"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
        .fillna(df["region"])
    )

    df.to_csv(f, index=False)
    print(f"[done] {f.name}")

print("\n[completed] region normalization finished.")
