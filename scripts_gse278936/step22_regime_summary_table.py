#!/usr/bin/env python3
"""
step22_regime_summary_table.py — Manuscript-Ready Two-Regime Model Table
=========================================================================
Generates a structured CSV summarising the two-regime model result:

  Bulk regime     → gradient-compatible, statistical, near-equilibrium
  Interface regime → non-integrable, constraint-dominated, non-equilibrium
  KTS bias         → directional dynamical endpoint (exhaustion attractor)
  KS-like score    → exploratory instability proxy

SAFETY STATEMENT:
-----------------
The regime labels and the Maxwell-Boltzmann / Euler-Bernoulli references
are operator-level analogies, not mechanistic claims. The table describes
operator signatures, not physical processes.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


# ── Table definition ──────────────────────────────────────────────────────
TWO_REGIME_TABLE = [
    {
        "analysis_layer":   "Coexact energy enrichment",
        "regime":           "Interface",
        "metric":           "R_coexact = interface / tumor-core median coexact energy",
        "TNBC_result":      "40/40 sections; median R = 19.5×; sign test p < 1e-12",
        "GSE278936_result": "21/23 sections; sign test p < 1e-5",
        "operator_interpretation": (
            "Interface concentrates non-gradient interaction intensity; "
            "consistent with constraint-dominated regime"
        ),
        "analogy_note": (
            "Analogous to a curvature-driven boundary regime; "
            "this is an operator analogy, not a physical law"
        ),
    },
    {
        "analysis_layer":   "Gradient energy enrichment",
        "regime":           "Both (but weaker at interface)",
        "metric":           "R_exact = interface / tumor-core median exact energy",
        "TNBC_result":      "Coexact/exact differential > 1.0 in 38/40 (median 2.54×)",
        "GSE278936_result": "Not separately reported; coexact dominance confirmed",
        "operator_interpretation": (
            "Exact component also elevated at interface but substantially "
            "weaker than coexact; non-gradient structure adds independent information"
        ),
        "analogy_note": "Gradient-dominated bulk = near-equilibrium baseline",
    },
    {
        "analysis_layer":   "Graph curvature |Lu|",
        "regime":           "Interface (higher) vs Bulk (lower)",
        "metric":           "Median |Lu| at interface vs bulk-like nodes",
        "TNBC_result":      "Tested in two-regime permutation (Step 17)",
        "GSE278936_result": "Tested in two-regime permutation (Step 17)",
        "operator_interpretation": (
            "Interface nodes show higher Laplacian curvature of the "
            "coexact field, consistent with a constrained boundary zone"
        ),
        "analogy_note": (
            "Graph curvature is an operator-level measure; "
            "not a claim about mechanical curvature of tissue"
        ),
    },
    {
        "analysis_layer":   "CDIS (Constraint-Dominated Interface Score)",
        "regime":           "Interface (enriched vs bulk)",
        "metric":           "z(coexact/exact) + z(|L²u|) + z(nonlin_grad)",
        "TNBC_result":      "Testable from rebuild outputs (Step 19)",
        "GSE278936_result": "Testable from Step 19 outputs",
        "operator_interpretation": (
            "Combined operator score quantifies degree of non-gradient, "
            "constraint-dominated activity per node"
        ),
        "analogy_note": "Composite score; component weights are equal (z-score additive)",
    },
    {
        "analysis_layer":   "KS-like instability proxy",
        "regime":           "Interface (elevated)",
        "metric":           "Median |−Lu − L²u − |∇u|²| at interface vs bulk",
        "TNBC_result":      "Elevated (exploratory; not primary result)",
        "GSE278936_result": "Median interface/tumor fold ≈ 8.96 (23/23 fold>1)",
        "operator_interpretation": (
            "Interface region coexact field has operator properties consistent "
            "with nonlinear spatial instability; exploratory, not confirmatory"
        ),
        "analogy_note": (
            "KS analogy is an operator analogy only. "
            "Does not claim the biological system solves the KS PDE."
        ),
    },
    {
        "analysis_layer":   "KTS transition bias",
        "regime":           "Exhaustion-directed (all compartments)",
        "metric":           "Bias ratio toward IMMUNE_EXHAUSTED, permutation null",
        "TNBC_result":      (
            "IMMUNE_ACTIVE→IE: bias 5.68 (16/28 sig); "
            "STROMA→IE: bias 2.41 (17/29 sig); "
            "TUMOR→IE: not enriched (bias 0.42)"
        ),
        "GSE278936_result": (
            "Near-universal attractor: "
            "bias 1.91–3.63×, 21–23/23 sections"
        ),
        "operator_interpretation": (
            "Spatially disordered coexact structure is directionally biased "
            "toward immune exhaustion; pathway-specific in TNBC, near-universal in GSE278936"
        ),
        "analogy_note": (
            "KTS bias is the primary dynamical signal; "
            "exhaustion is a conserved endpoint despite spatial disorder"
        ),
    },
    {
        "analysis_layer":   "Bulk equilibrium null",
        "regime":           "Bulk (near-equilibrium reference)",
        "metric":           "Bulk-matched random node sets vs interface (Step 18)",
        "TNBC_result":      "Bulk coexact and KS-like metrics < interface (Step 18)",
        "GSE278936_result": "Same test structure applicable",
        "operator_interpretation": (
            "Bulk-like nodes behave as a near-equilibrium null distribution "
            "for the operator metrics measured at interface nodes"
        ),
        "analogy_note": (
            "Maxwell-Boltzmann analogy: bulk = diffuse statistical regime. "
            "This is an operator analogy, not a physical claim."
        ),
    },
]

GLOBAL_SAFETY_NOTE = (
    "SAFETY: All regime labels, physics analogies (Maxwell-Boltzmann, "
    "Euler-Bernoulli, Kuramoto-Sivashinsky), and therapeutic principles "
    "derived from this table are interpretive operator analogies and "
    "formal hypotheses. They do not claim that tumor tissue literally "
    "follows these physical equations and do not constitute clinical "
    "recommendations."
)


def main():
    ap = argparse.ArgumentParser(description="Step 22: Two-regime model summary table")
    ap.add_argument("--outdir", type=Path, default=Path("results_gse278936"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(TWO_REGIME_TABLE)
    df["global_safety_note"] = GLOBAL_SAFETY_NOTE
    df["regeneration_note"] = (
        "This table contains manuscript-level summary entries. "
        "TNBC and GSE278936 result fields should be regenerated from "
        "current cohort_two_regime_test.csv, cohort_constraint_score_summary.csv, "
        "and cohort_bulk_vs_interface_null.csv before final submission. "
        "CDIS entries are valid only when cdis_formula_used (Step 19 output) "
        "is reported alongside the numeric values."
    )

    out = args.outdir / "table_two_regime_model.csv"
    df.to_csv(out, index=False)

    print("=== TWO-REGIME MODEL SUMMARY TABLE ===")
    for _, row in df.iterrows():
        print(f"\n  Layer: {row['analysis_layer']}")
        print(f"    Regime: {row['regime']}")
        print(f"    TNBC:   {row['TNBC_result'][:70]}…" if len(row['TNBC_result']) > 70
              else f"    TNBC:   {row['TNBC_result']}")

    print(f"\n[done] {out}")
    print(f"\n{GLOBAL_SAFETY_NOTE}")


if __name__ == "__main__":
    main()
