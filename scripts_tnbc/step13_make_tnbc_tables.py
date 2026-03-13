from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


def find_samples(statsdir: Path) -> list[str]:
    sample_ids = set()
    for p in statsdir.glob("GSM_*_step2_region_assignments.csv"):
        sample_ids.add(p.name.split("_step2_region_assignments.csv")[0])
    return sorted(sample_ids)


def write_tex_table(df: pd.DataFrame, outpath: Path, caption: str, label: str) -> None:
    tex = df.to_latex(
        index=False,
        escape=False,
        longtable=False,
        caption=caption,
        label=label,
    )
    outpath.write_text(tex, encoding="utf-8")


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def make_samples_overview(sample_ids: Iterable[str], statsdir: Path) -> pd.DataFrame:
    rows = []

    for sample_id in sample_ids:
        path = statsdir / f"{sample_id}_step2_region_assignments.csv"
        df = safe_read_csv(path)
        if df is None:
            continue

        counts = df["region_step2"].value_counts()
        rows.append(
            {
                "sample_id": sample_id,
                "n_spots": int(len(df)),
                "n_tumor_enriched": int(counts.get("tumor_enriched", 0)),
                "n_stroma_enriched": int(counts.get("stroma_enriched", 0)),
                "n_immune_enriched": int(counts.get("immune_enriched", 0)),
                "n_interface_like": int(counts.get("interface_like", 0)),
                "n_other": int(counts.get("other", 0)),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("sample_id").reset_index(drop=True)
    return out


def make_hodge_energy_summary(sample_ids: Iterable[str], statsdir: Path) -> pd.DataFrame:
    rows = []

    for sample_id in sample_ids:
        for flux_name in ["flux_tumor_immune", "flux_tumor_stroma", "flux_immune_stroma"]:
            path = statsdir / f"{sample_id}_step6_energy_summary_{flux_name}.csv"
            df = safe_read_csv(path)
            if df is None or df.empty:
                continue

            row = df.iloc[0].to_dict()
            rows.append(
                {
                    "sample_id": sample_id,
                    "flux_name": flux_name,
                    "E_total": row.get("E_total"),
                    "E_exact": row.get("E_exact"),
                    "E_coexact": row.get("E_coexact"),
                    "E_harmonic": row.get("E_harmonic"),
                    "frac_exact": row.get("frac_exact"),
                    "frac_coexact": row.get("frac_coexact"),
                    "frac_harmonic": row.get("frac_harmonic"),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["sample_id", "flux_name"]).reset_index(drop=True)
    return out


def make_region_enrichment(sample_ids: Iterable[str], statsdir: Path) -> pd.DataFrame:
    rows = []

    for sample_id in sample_ids:
        for flux_name in ["flux_tumor_immune", "flux_tumor_stroma", "flux_immune_stroma"]:
            tests_path = statsdir / f"{sample_id}_step7_region_tests_{flux_name}.csv"
            tests_df = safe_read_csv(tests_path)
            if tests_df is None or tests_df.empty:
                continue

            keep = tests_df[
                tests_df["region_a"].isin(["immune_enriched", "interface_like"])
            ].copy()

            for _, r in keep.iterrows():
                rows.append(
                    {
                        "sample_id": sample_id,
                        "flux_name": flux_name,
                        "metric": r.get("metric"),
                        "region_a": r.get("region_a"),
                        "region_b": r.get("region_b"),
                        "median_a": r.get("median_a"),
                        "median_b": r.get("median_b"),
                        "median_ratio_a_over_b": r.get("median_ratio_a_over_b"),
                        "mwu_p": r.get("mwu_p"),
                        "perm_p": r.get("perm_p"),
                    }
                )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["sample_id", "flux_name", "metric", "region_a", "region_b"]).reset_index(drop=True)
    return out


def make_lie_null_validation(sample_ids: Iterable[str], statsdir: Path) -> pd.DataFrame:
    rows = []

    for sample_id in sample_ids:
        for flux_name in ["flux_tumor_immune", "flux_tumor_stroma", "flux_immune_stroma"]:
            s11 = safe_read_csv(statsdir / f"{sample_id}_step11_lie_null_summary_{flux_name}.csv")
            s12 = safe_read_csv(statsdir / f"{sample_id}_step12_region_hotspot_lie_test_{flux_name}.csv")

            if s11 is None or s11.empty:
                continue

            row11 = s11.iloc[0].to_dict()

            if s12 is not None and not s12.empty:
                interface_row = s12.loc[s12["region_name"] == "interface_like"]
                immune_row = s12.loc[s12["region_name"] == "immune_enriched"]
                interface_row = interface_row.iloc[0].to_dict() if not interface_row.empty else {}
                immune_row = immune_row.iloc[0].to_dict() if not immune_row.empty else {}
            else:
                interface_row = {}
                immune_row = {}

            rows.append(
                {
                    "sample_id": sample_id,
                    "flux_name": flux_name,
                    "real_mean_curl": row11.get("real_mean_curl"),
                    "null_mean_of_mean_curl": row11.get("null_mean_of_mean_curl"),
                    "p_mean_curl": row11.get("p_mean_curl"),
                    "real_top95_mean_curl": row11.get("real_top95_mean_curl"),
                    "null_mean_of_top95_mean_curl": row11.get("null_mean_of_top95_mean_curl"),
                    "p_top95_mean_curl": row11.get("p_top95_mean_curl"),
                    "real_top99_mean_curl": row11.get("real_top99_mean_curl"),
                    "null_mean_of_top99_mean_curl": row11.get("null_mean_of_top99_mean_curl"),
                    "p_top99_mean_curl": row11.get("p_top99_mean_curl"),
                    "interface_enrichment_ratio": interface_row.get("enrichment_ratio"),
                    "interface_empirical_p": interface_row.get("empirical_p"),
                    "immune_enrichment_ratio": immune_row.get("enrichment_ratio"),
                    "immune_empirical_p": immune_row.get("empirical_p"),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["sample_id", "flux_name"]).reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 13 TNBC: build manuscript-ready TNBC cohort summary tables."
    )
    parser.add_argument("--statsdir", default="stats/CSV_GSM")
    parser.add_argument("--outdir", default="tables/TNBC_GSE210616_table")
    args = parser.parse_args()

    statsdir = Path(args.statsdir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sample_ids = find_samples(statsdir)
    print(f"Found samples: {sample_ids}")

    # Table S1
    df1 = make_samples_overview(sample_ids, statsdir)
    csv1 = outdir / "tnbc_samples_overview.csv"
    tex1 = outdir / "tnbc_samples_overview.tex"
    df1.to_csv(csv1, index=False)
    write_tex_table(
        df1,
        tex1,
        caption="Table S1: Spatial transcriptomics TNBC cohort characteristics.",
        label="tab:tnbc_samples_overview",
    )

    # Table S2
    df2 = make_hodge_energy_summary(sample_ids, statsdir)
    csv2 = outdir / "tnbc_hodge_energy_summary.csv"
    tex2 = outdir / "tnbc_hodge_energy_summary.tex"
    df2.to_csv(csv2, index=False)
    write_tex_table(
        df2,
        tex2,
        caption="Table S2: Hodge decomposition of signaling flux fields.",
        label="tab:tnbc_hodge_energy_summary",
    )

    # Table S3
    df3 = make_region_enrichment(sample_ids, statsdir)
    csv3 = outdir / "tnbc_region_enrichment.csv"
    tex3 = outdir / "tnbc_region_enrichment.tex"
    df3.to_csv(csv3, index=False)
    write_tex_table(
        df3,
        tex3,
        caption="Table S3: Region-level enrichment of rotational signaling energy.",
        label="tab:tnbc_region_enrichment",
    )

    # Table S4
    df4 = make_lie_null_validation(sample_ids, statsdir)
    csv4 = outdir / "tnbc_lie_null_validation.csv"
    tex4 = outdir / "tnbc_lie_null_validation.tex"
    df4.to_csv(csv4, index=False)
    write_tex_table(
        df4,
        tex4,
        caption="Table S4: Structured null validation of curl signal.",
        label="tab:tnbc_lie_null_validation",
    )

    print("\nSaved:")
    print(csv1)
    print(tex1)
    print(csv2)
    print(tex2)
    print(csv3)
    print(tex3)
    print(csv4)
    print(tex4)


if __name__ == "__main__":
    main()

