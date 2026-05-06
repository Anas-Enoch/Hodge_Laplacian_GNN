#!/usr/bin/env python3
"""
step20_interface_targeting_principles.py — Therapeutic Hypothesis Table
========================================================================
Converts operator signatures from Steps 15–19 into a structured table of
formal therapeutic hypotheses. This script does NOT generate clinical
recommendations and must NOT be used for clinical decision-making.

SAFETY STATEMENT (MANDATORY):
------------------------------
These analyses generate formal operator-derived hypotheses ONLY.
They are not therapeutic guidance and should not be interpreted as
treatment recommendations under any circumstances. The physics analogies
(Maxwell–Boltzmann / Euler–Bernoulli) are interpretive operator analogies
and do not claim that tumor tissue literally follows these equations.
Each hypothesis requires orthogonal experimental validation in appropriate
pre-clinical or clinical study designs before any clinical inference can
be drawn. Evidence tiers classify the epistemic distance from the data:
Tier 2 = operator inference; Tier 3 = speculative extrapolation.

Output
------
  interface_targeting_principles.csv
  Columns: principle_id, principle_name, supporting_metric, evidence_tier,
           operator_prediction, required_validation_experiment, safety_note
"""

import argparse
from pathlib import Path
import pandas as pd


PRINCIPLES = [
    {
        "principle_id": 1,
        "principle_name": "Interrupt coexact feedback loops",
        "operator_mechanism": (
            "Coexact component captures closed-loop (non-gradient) structure "
            "in the interaction field. High coexact energy at the interface "
            "is consistent with mutual-inhibition or feedback cycles between "
            "tumor and immune programs."
        ),
        "supporting_metric": "coexact_exact_ratio at interface > tumor bulk",
        "evidence_tier": "Tier 2 — Operator inference",
        "evidence_note": (
            "Supported by coexact enrichment 40/40 TNBC (p<1e-12) and "
            "21/23 GSE278936 (p<1e-5). Molecular identity of loops not established."
        ),
        "candidate_targets": (
            "Checkpoint cycle inhibitors; CXCL9/10 gradient disruptors; "
            "cytokine feedback antagonists"
        ),
        "operator_prediction": (
            "Effective loop disruption reduces coexact_exact_ratio at interface "
            "relative to tumor bulk without reducing exact enrichment."
        ),
        "required_validation_experiment": (
            "Paired pre/post-treatment spatial transcriptomics: measure "
            "Delta(coexact_exact_ratio) at interface. Hypothesis: ratio "
            "decreases after loop-disrupting intervention."
        ),
        "safety_note": (
            "Hypothesis only. Molecular identity of the feedback loops is not "
            "established by the present operator analysis."
        ),
    },
    {
        "principle_id": 2,
        "principle_name": "Target spatial geometry of the interface",
        "operator_mechanism": (
            "Coexact enrichment is a geometric property of the field at the "
            "boundary — it reflects spatial opposition between programs, not "
            "just their co-presence. ECM architecture and CAF organization may "
            "constrain the spatial pattern that generates coexact structure."
        ),
        "supporting_metric": (
            "CDIS (Constraint-Dominated Interface Score) and graph_curvature "
            "enriched at interface relative to bulk"
        ),
        "evidence_tier": "Tier 3 — Geometric inference (speculative)",
        "evidence_note": (
            "CDIS enrichment is testable (Step 19). Whether ECM drives "
            "coexact structure is not established."
        ),
        "candidate_targets": (
            "FAP inhibitors (CAF targeting); LOX inhibitors (collagen crosslinking); "
            "integrin blockers (matrix remodeling)"
        ),
        "operator_prediction": (
            "Structural interventions reduce graph_curvature and CDIS at the interface "
            "by flattening the spatial opposition between programs."
        ),
        "required_validation_experiment": (
            "Post-treatment spatial transcriptomics after CAF/ECM targeting: "
            "measure Delta(CDIS) and Delta(graph_curvature) at interface. "
            "Concurrent histological assessment of collagen architecture required."
        ),
        "safety_note": (
            "Causal link between ECM geometry and coexact structure is speculative. "
            "This is a geometric inference, not a mechanistic claim."
        ),
    },
    {
        "principle_id": 3,
        "principle_name": "Normalize vasculature / metabolic heterogeneity to reduce non-integrability",
        "operator_mechanism": (
            "Coexact structure arises where the gradient model fails. Reducing "
            "metabolic or vascular heterogeneity may push the interaction field "
            "toward gradient-compatibility, reducing the structural basis for "
            "coexact enrichment."
        ),
        "supporting_metric": (
            "Coexact/exact differential at interface; Step 18 bulk equilibrium null "
            "(interface >> bulk in ks_like and coexact_energy)"
        ),
        "evidence_tier": "Tier 3 — Systemic inference (speculative)",
        "evidence_note": (
            "Measurable criterion specified (Delta coexact_exact_ratio pre/post). "
            "Mechanism linking vasculature to coexact structure is not established."
        ),
        "candidate_targets": (
            "Anti-VEGF normalization (bevacizumab); oxidative phosphorylation "
            "inhibitors; hypoxia-targeting agents"
        ),
        "operator_prediction": (
            "Heterogeneity-reducing interventions decrease coexact/exact ratio "
            "toward 1.0 (gradient-compatible) at the interface. Falsifiable by "
            "paired pre/post spatial transcriptomics."
        ),
        "required_validation_experiment": (
            "Paired spatial transcriptomics pre/post anti-VEGF: measure "
            "Delta(coexact_exact_ratio) and Delta(CDIS). Compare to non-interface "
            "regions as internal control."
        ),
        "safety_note": (
            "The prediction is diffuse: many interventions would reduce spatial "
            "heterogeneity. The falsifiable criterion is the magnitude and "
            "interface-specificity of the operator change."
        ),
    },
    {
        "principle_id": 4,
        "principle_name": "Exploit interface as spatially stable delivery target",
        "operator_mechanism": (
            "The interface is spatially stable (coexact enrichment observed "
            "40/40 TNBC sections independently of clinical variables), "
            "biologically specific (not driven by generic heterogeneity), "
            "and carries high signal contrast vs tumor bulk."
        ),
        "supporting_metric": (
            "Coexact enrichment ratio R (median 19.5x in TNBC); "
            "clinical independence (race, chemotherapy, RFS all p>0.18)"
        ),
        "evidence_tier": "Tier 2 — Spatial inference (more grounded)",
        "evidence_note": (
            "Spatial stability and biological specificity are empirically "
            "established. Delivery feasibility requires independent validation."
        ),
        "candidate_targets": (
            "Interface marker-targeted antibody-drug conjugates; "
            "bispecific T-cell engagers localized at boundary; "
            "nanoparticle systems with interface microenvironment activation"
        ),
        "operator_prediction": (
            "Interface-targeted delivery achieves higher local accumulation "
            "than tumor-core-targeted delivery, given the documented spatial "
            "contrast (coexact R > 4.6 in all valid sections)."
        ),
        "required_validation_experiment": (
            "Biodistribution study measuring interface vs tumor-core accumulation "
            "of interface-targeted vs bulk-targeted drug vehicles. "
            "Surrogate imaging markers of the interface zone required."
        ),
        "safety_note": (
            "Interface identifiability requires spatial transcriptomics or imaging "
            "surrogates. Current workflow is ex vivo. In vivo interface targeting "
            "requires independent feasibility assessment."
        ),
    },
]


def main():
    ap = argparse.ArgumentParser(
        description="Step 20: Therapeutic hypothesis table"
    )
    ap.add_argument("--outdir", type=Path, default=Path("results_gse278936"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(PRINCIPLES)

    # Add global safety note
    df["global_safety_note"] = (
        "These analyses generate formal operator-derived hypotheses, not clinical "
        "recommendations. The physics analogies are interpretive operator analogies "
        "and do not claim that tumor tissue literally follows Maxwell-Boltzmann or "
        "Euler-Bernoulli dynamics. All hypotheses require orthogonal experimental "
        "validation before any clinical inference can be drawn."
    )

    out = args.outdir / "interface_targeting_principles.csv"
    df.to_csv(out, index=False)

    print("=== INTERFACE TARGETING PRINCIPLES ===")
    for _, row in df.iterrows():
        print(f"\nPrinciple {row['principle_id']}: {row['principle_name']}")
        print(f"  Evidence tier:      {row['evidence_tier']}")
        print(f"  Supporting metric:  {row['supporting_metric']}")
        print(f"  Required validation: {row['required_validation_experiment'][:80]}…")

    print(f"\n[done] {out}")
    print("\nSAFETY: All principles are formal hypotheses requiring validation.")


if __name__ == "__main__":
    main()
