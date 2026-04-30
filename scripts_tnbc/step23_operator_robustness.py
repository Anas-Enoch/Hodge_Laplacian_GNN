"""
Step 23 — Operator Robustness: Antisymmetric Interaction Operator Comparison
=============================================================================

Core claim: coexact interface enrichment is not a numerical artifact of the
wedge construction. If it were, it would collapse as magnitude / rank / threshold
information is progressively destroyed.

Five operators tested (monotone information-destruction chain):
────────────────────────────────────────────────────────────────
Operator            Magnitude  Order    Sign      Robustness level
─────────────────   ─────────  ───────  ────────  ────────────────
proxy_wedge         ✓✓✓        ✓        ✓         baseline
normalized_wedge    ✓✓         ✓        ✓         scale-robust
rank_antisym        ✗          ✓✓✓      ✓         monotonic-robust
thresholded_antisym ✗          ✗        ✓✓(bin)   coarse-robust
sign_only           ✗          ✗        ✓✓✓       extreme-robust

Falsification logic:
  IF coexact enrichment collapses at sign_only → artifact of magnitude/rank
  IF coexact enrichment survives sign_only → topological/geometric invariant
  not explainable by numerical properties of the antisymmetric construction.

Construction (for all operators):
  Raw: f_ij = a_i * b_j - a_j * b_i  (proxy wedge, sign = interface_weighted)
  Then each operator applies a transformation T to f:
    normalized:    T(f_ij) = f_ij / (|a_i| + |a_j| + ε)
    rank:          T(f_ij) = sign(f_ij) * rank(|f_ij|) / n_edges
    thresholded:   T(f_ij) = f_ij * 1[|f_ij| > median(|f|)]
    sign_only:     T(f_ij) = sign(f_ij)

Per operator, per section:
  - Hodge decomposition of the transformed flux
  - Interface coexact enrichment ratio (vs tumor core)
  - Permutation p-value (1000 label shuffles)

Outputs:
  *_step23_operator_robustness_{flux_tag}.csv   per section × operator summary
  cohort_step23_operator_robustness_{flux_tag}.csv   cohort sign tests per operator

Circularity constraint (hard):
  Operator transformations are applied to the raw proxy flux only.
  Region labels are NEVER used to construct or transform the flux.

Usage:
  # Per sample
  python scripts_tnbc/step23_operator_robustness.py \\
    --mode sample --sample-id GSM_6433619 \\
    --flux-tag flux_tumor_immune_region_interface_weighted \\
    --statsdir stats/CSV_GSM --outdir stats/CSV_GSM

  # Cohort
  python scripts_tnbc/step23_operator_robustness.py \\
    --mode cohort \\
    --flux-tag flux_tumor_immune_region_interface_weighted \\
    --statsdir stats/CSV_GSM --outdir stats/CSV_GSM
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.stats import binomtest


# =============================================================================
# CONSTANTS
# =============================================================================

INTERFACE_LABEL = "interface_like"
TUMOR_LABELS    = {"tumor_enriched", "tumor_core"}

OPERATORS = ["proxy_wedge", "normalized_wedge", "rank_antisym",
             "thresholded_antisym", "sign_only"]

OPERATOR_DESCRIPTIONS = {
    "proxy_wedge":        "Baseline: raw wedge flux with interface weighting (Steps 4–6 output)",
    "normalized_wedge":   "Scale-robust: wedge flux divided by local sum of absolute program scores",
    "rank_antisym":       "Monotonic-robust: magnitude replaced by rank; preserves order only",
    "thresholded_antisym":"Coarse-robust: magnitude zeroed below median; binary above/below",
    "sign_only":          "Extreme-robust: only sign of antisymmetric interaction retained",
}


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode",      choices=["sample", "cohort"], default="sample")
    p.add_argument("--sample-id", default=None)
    p.add_argument("--flux-tag",
                   default="flux_tumor_immune_region_interface_weighted")
    p.add_argument("--statsdir",  default="stats/CSV_GSM")
    p.add_argument("--outdir",    default="stats/CSV_GSM")
    p.add_argument("--n-perm",    type=int, default=1000,
                   help="Number of label permutations for null (default 1000).")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--min-nodes", type=int, default=10,
                   help="Minimum nodes in each region for valid enrichment.")
    return p.parse_args()


_ARGS     = _parse_args()
FLUX_TAG  = _ARGS.flux_tag
STATS_DIR = Path(_ARGS.statsdir)
OUT_DIR   = Path(_ARGS.outdir)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# OPERATOR TRANSFORMATIONS
# =============================================================================

def apply_operator(
    flux_values: np.ndarray,
    score_a: np.ndarray,      # tumor score at each node
    score_b: np.ndarray,      # immune score at each node
    tail: np.ndarray,         # edge tail node indices
    head: np.ndarray,         # edge head node indices
    operator: str,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Apply the specified antisymmetric operator transformation to the raw proxy flux.

    All operators preserve the antisymmetric sign structure (f_ij = -f_ji) and
    the direction of the interaction (positive = a dominates, negative = b dominates).
    They differ in how much of the magnitude / rank information they retain.

    Parameters
    ----------
    flux_values : raw proxy wedge flux per edge (from Step 6 edge CSV)
    score_a, score_b : residualized program scores per node (a=tumor, b=immune)
    tail, head : node indices for each edge
    operator : one of OPERATORS
    """
    f = flux_values.copy()
    n_edges = len(f)

    if operator == "proxy_wedge":
        # Identity — use the Step 6 output directly
        return f

    elif operator == "normalized_wedge":
        # Divide by local sum of absolute program magnitudes.
        # This removes the absolute scale of the scores while preserving
        # the antisymmetric interaction structure.
        # Normalization: f_ij / (|a_i| + |a_j| + |b_i| + |b_j| + ε)
        a_tail = score_a[tail]; a_head = score_a[head]
        b_tail = score_b[tail]; b_head = score_b[head]
        denom = np.abs(a_tail) + np.abs(a_head) + np.abs(b_tail) + np.abs(b_head) + eps
        return f / denom

    elif operator == "rank_antisym":
        # Replace magnitude with rank.
        # All magnitude information is destroyed; only the order of interactions
        # is preserved. If coexact survives this, it cannot be due to score scaling.
        #
        # Construction: rank |f_ij| from 1 to n_edges, then restore sign.
        # Ties broken by first occurrence (arbitrary but fixed by seed).
        abs_f   = np.abs(f)
        ranks   = np.zeros(n_edges)
        order   = np.argsort(abs_f)            # ascending: weakest → strongest
        ranks[order] = np.arange(1, n_edges + 1, dtype=float)
        return np.sign(f) * ranks / n_edges   # normalize ranks to [0, 1]

    elif operator == "thresholded_antisym":
        # Zero out all edges below the median absolute flux.
        # Only the top 50% of interactions survive; magnitude still present for them.
        # If coexact survives this, it is robust to removing weak interactions.
        threshold = np.median(np.abs(f))
        mask = np.abs(f) >= threshold
        return f * mask.astype(float)

    elif operator == "sign_only":
        # Retain only the sign of the antisymmetric interaction.
        # All magnitude, rank, and threshold information is destroyed.
        # Only the direction of the biological program dominance is preserved.
        # Surviving coexact enrichment here = geometric invariant.
        return np.sign(f)

    else:
        raise ValueError(f"Unknown operator: {operator}")


# =============================================================================
# HODGE DECOMPOSITION (self-contained, no dependency on step6 output)
# =============================================================================

def hodge_decompose(
    f: np.ndarray,
    B1: sp.csr_matrix,
    n_nodes: int,
) -> dict:
    """
    Minimal lower-Hodge decomposition: f = f_exact + f_coexact.

    f_exact   = B1^T (B1 B1^T)^+ B1 f   [projection onto Im(B1^T)]
    f_coexact = f - f_exact              [cycle-space residual]

    Uses sparse LSQR for numerical stability at scale.
    """
    n_edges = len(f)

    # Exact component: solve B1 x = B1 f, then f_exact = B1^T x
    # Equivalently: f_exact = B1^T α where α = (B1 B1^T)^+ B1 f
    # Use LSQR: find α minimizing ||B1^T α - f||, then f_exact = B1^T α
    rhs = B1 @ f                            # n_nodes vector (divergence of f)
    # Solve B1 B1^T α = rhs   (node Laplacian system)
    L0  = (B1 @ B1.T).tocsr()
    result = spla.lsqr(L0, rhs, atol=1e-10, btol=1e-10, iter_lim=5000)
    alpha   = result[0]                     # n_nodes
    f_exact = B1.T @ alpha                  # n_edges

    f_coexact = f - f_exact

    # Node-level energies
    node_coex  = np.zeros(n_nodes)
    node_exact = np.zeros(n_nodes)
    tail       = np.array(B1.nonzero()[1])  # this is the column indices of B1

    # Build tail/head from B1 structure
    B1_coo = B1.tocoo()
    # For each edge e (column), tail = node with -1, head = node with +1
    edge_tail = np.zeros(n_edges, dtype=int)
    edge_head = np.zeros(n_edges, dtype=int)
    for row, col, val in zip(B1_coo.row, B1_coo.col, B1_coo.data):
        if val < 0:
            edge_tail[col] = row
        else:
            edge_head[col] = row

    counts = np.zeros(n_nodes)
    for e in range(n_edges):
        t, h = edge_tail[e], edge_head[e]
        for nid in [t, h]:
            node_coex[nid]  += f_coexact[e] ** 2
            node_exact[nid] += f_exact[e]   ** 2
            counts[nid]     += 1

    counts = np.maximum(counts, 1)
    node_coex  /= counts
    node_exact /= counts

    E_coexact = float(np.sum(f_coexact ** 2))
    E_exact   = float(np.sum(f_exact   ** 2))
    E_total   = E_coexact + E_exact

    return {
        "f_coexact":   f_coexact,
        "f_exact":     f_exact,
        "node_coex":   node_coex,
        "node_exact":  node_exact,
        "E_coexact":   E_coexact,
        "E_exact":     E_exact,
        "E_total":     E_total,
        "frac_coexact": E_coexact / E_total if E_total > 0 else np.nan,
    }


# =============================================================================
# ENRICHMENT + PERMUTATION TEST
# =============================================================================

def enrichment_and_perm(
    node_coex: np.ndarray,
    region: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
    min_nodes: int,
) -> dict:
    """
    Compute interface vs. tumor-core coexact enrichment ratio and permutation p-value.

    Observed ratio: mean(coex[interface]) / mean(coex[tumor])
    Null: shuffle region labels 1000 times, recompute ratio each time.
    """
    iface_mask = region == INTERFACE_LABEL
    tumor_mask = np.isin(region, list(TUMOR_LABELS))

    n_iface = int(iface_mask.sum())
    n_tumor = int(tumor_mask.sum())

    if n_iface < min_nodes or n_tumor < min_nodes:
        return {
            "enrichment_ratio": np.nan,
            "perm_p":           np.nan,
            "n_interface":      n_iface,
            "n_tumor":          n_tumor,
            "note":             "low_sample_size",
        }

    mu_iface = float(node_coex[iface_mask].mean())
    mu_tumor = float(node_coex[tumor_mask].mean())

    if mu_tumor < 1e-12:
        return {
            "enrichment_ratio": np.nan,
            "perm_p":           np.nan,
            "n_interface":      n_iface,
            "n_tumor":          n_tumor,
            "note":             "zero_tumor_baseline",
        }

    observed = mu_iface / mu_tumor

    # Permutation null: shuffle region labels, recompute ratio
    null_ratios  = np.zeros(n_perm)
    region_arr   = np.array(region, dtype=str)   # ensure plain numpy, not pandas StringArray
    region_perm  = region_arr.copy()
    for k in range(n_perm):
        rng.shuffle(region_perm)
        mi = float(node_coex[region_perm == INTERFACE_LABEL].mean()) \
            if (region_perm == INTERFACE_LABEL).sum() > 0 else 0.0
        mt = float(node_coex[np.isin(region_perm, list(TUMOR_LABELS))].mean()) \
            if np.isin(region_perm, list(TUMOR_LABELS)).sum() > 0 else 1e-12
        null_ratios[k] = mi / mt if mt > 1e-12 else 0.0

    perm_p = float((np.sum(null_ratios >= observed) + 1) / (n_perm + 1))

    return {
        "enrichment_ratio": float(observed),
        "perm_p":           perm_p,
        "n_interface":      n_iface,
        "n_tumor":          n_tumor,
        "note":             "ok",
        "null_mean":        float(null_ratios.mean()),
        "null_std":         float(null_ratios.std()),
    }


# =============================================================================
# BUILD B1 FROM EDGE CSV
# =============================================================================

def build_B1(edges: pd.DataFrame, n_nodes: int) -> sp.csr_matrix:
    n_edges = len(edges)
    tail = edges["tail"].to_numpy(dtype=int)
    head = edges["head"].to_numpy(dtype=int)
    rows = np.concatenate([head, tail])
    cols = np.concatenate([np.arange(n_edges), np.arange(n_edges)])
    data = np.concatenate([np.ones(n_edges), -np.ones(n_edges)])
    return sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_edges))


# =============================================================================
# SAMPLE MODE
# =============================================================================

def run_sample(
    sample_id: str, flux_tag: str,
    stats_dir: Path, out_dir: Path,
    n_perm: int, seed: int, min_nodes: int,
) -> Optional[list[dict]]:

    print(f"\n=== {sample_id} ===")
    rng = np.random.default_rng(seed)

    edge_file = stats_dir / f"{sample_id}_step6_edges_hodge_{flux_tag}.csv"
    node_file = stats_dir / f"{sample_id}_step6_nodes_hodge_{flux_tag}.csv"
    # Also need raw marker scores for score-based normalizations
    score_file = stats_dir / f"{sample_id}_step1_scores.csv"

    if not edge_file.exists() or not node_file.exists():
        print(f"  [skip] Missing step6 files.")
        return None

    edges  = pd.read_csv(edge_file)
    nodes  = pd.read_csv(node_file)
    n_nodes = len(nodes)
    n_edges = len(edges)

    if n_edges < 30:
        print(f"  [skip] Too few edges ({n_edges}).")
        return None

    if flux_tag not in edges.columns:
        print(f"  [skip] flux_tag '{flux_tag}' not in edge CSV.")
        return None

    print(f"  n_nodes={n_nodes}, n_edges={n_edges}")

    # Raw proxy flux (Step 6 output)
    flux_raw = edges[flux_tag].fillna(0.0).to_numpy(dtype=float)

    # Region labels
    nodes_s = nodes.sort_values("node_id").reset_index(drop=True)
    region  = nodes_s["region_step2"].to_numpy()

    # Tail / head arrays
    tail = edges["tail"].to_numpy(dtype=int)
    head = edges["head"].to_numpy(dtype=int)

    # Node-level program scores for normalized_wedge
    # Try step1 scores file; fall back to tumor_score / immune_score in node CSV
    score_a = np.ones(n_nodes)   # tumor score
    score_b = np.ones(n_nodes)   # immune score
    if score_file.exists():
        try:
            sc = pd.read_csv(score_file).sort_values("node_id").reset_index(drop=True)
            if "tumor_score" in sc.columns:
                score_a = sc["tumor_score"].fillna(0.0).to_numpy(dtype=float)
            if "immune_score" in sc.columns:
                score_b = sc["immune_score"].fillna(0.0).to_numpy(dtype=float)
        except Exception:
            pass
    elif "tumor_score" in nodes_s.columns and "immune_score" in nodes_s.columns:
        score_a = nodes_s["tumor_score"].fillna(0.0).to_numpy(dtype=float)
        score_b = nodes_s["immune_score"].fillna(0.0).to_numpy(dtype=float)

    # Build incidence matrix
    B1 = build_B1(edges, n_nodes)

    rows = []
    for op in OPERATORS:
        print(f"  [{op}]", end="", flush=True)

        # 1. Apply operator transformation
        f_transformed = apply_operator(flux_raw, score_a, score_b,
                                       tail, head, op)

        # 2. Skip if all-zero (can happen with sign_only on zero-flux edges)
        if np.allclose(f_transformed, 0):
            print(" [all-zero, skip]")
            rows.append({
                "sample_id":        sample_id,
                "operator":         op,
                "enrichment_ratio": np.nan,
                "perm_p":           np.nan,
                "frac_coexact":     np.nan,
                "E_coexact":        np.nan,
                "n_interface":      0,
                "n_tumor":          0,
                "note":             "all_zero_flux",
            })
            continue

        # 3. Hodge decomposition
        hodge = hodge_decompose(f_transformed, B1, n_nodes)

        # 4. Interface enrichment + permutation test
        enrich = enrichment_and_perm(
            hodge["node_coex"], region, n_perm, rng, min_nodes
        )

        row = {
            "sample_id":        sample_id,
            "operator":         op,
            "enrichment_ratio": enrich["enrichment_ratio"],
            "enrichment_gt1":   float(enrich["enrichment_ratio"] > 1.0)
                                if not np.isnan(enrich.get("enrichment_ratio", np.nan)) else np.nan,
            "perm_p":           enrich["perm_p"],
            "frac_coexact":     hodge["frac_coexact"],
            "E_coexact":        hodge["E_coexact"],
            "E_total":          hodge["E_total"],
            "n_interface":      enrich["n_interface"],
            "n_tumor":          enrich["n_tumor"],
            "note":             enrich.get("note", "ok"),
        }
        rows.append(row)

        status = f" R={enrich['enrichment_ratio']:.3f}  p={enrich['perm_p']:.3f}  frac_coex={hodge['frac_coexact']:.3f}"
        print(status)

    if rows:
        out_path = out_dir / f"{sample_id}_step23_operator_robustness_{flux_tag}.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"  Saved → {out_path.name}")

    return rows


# =============================================================================
# COHORT MODE
# =============================================================================

def run_cohort(flux_tag: str, stats_dir: Path, out_dir: Path) -> None:

    files = sorted(stats_dir.glob(f"*_step23_operator_robustness_{flux_tag}.csv"))
    if not files:
        print("[cohort] No per-sample files found. Run --mode sample first.")
        return

    print(f"[cohort] Found {len(files)} per-sample files.")
    all_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # Save full cohort CSV
    all_df.to_csv(out_dir / f"cohort_step23_operator_robustness_{flux_tag}.csv", index=False)

    print()
    print("=== STEP 23 OPERATOR ROBUSTNESS — COHORT SUMMARY ===")
    print()
    print(f"{'Operator':<24} {'Med R':>6} {'N>1/N':>7} {'Sign p':>10} "
          f"{'Med frac_coex':>14} {'Med p_perm':>10}  Description")
    print("-" * 100)

    summary_rows = []
    for op in OPERATORS:
        sub = all_df[all_df["operator"] == op].dropna(subset=["enrichment_ratio"])
        n = len(sub)
        if n == 0:
            print(f"  {op:<24} no valid sections")
            continue

        R_vals   = sub["enrichment_ratio"].to_numpy()
        p_vals   = sub["perm_p"].to_numpy()
        fc_vals  = sub["frac_coexact"].dropna().to_numpy()
        n_gt1    = int(np.sum(R_vals > 1.0))
        med_R    = float(np.median(R_vals))
        med_p    = float(np.median(p_vals))
        med_fc   = float(np.median(fc_vals)) if len(fc_vals) > 0 else np.nan
        sign_p   = binomtest(n_gt1, n, p=0.5, alternative="greater").pvalue

        print(f"  {op:<24} {med_R:>6.3f} {n_gt1:>3}/{n:<3} {float(sign_p):>10.5f} "
              f"{med_fc:>14.4f} {med_p:>10.4f}  {OPERATOR_DESCRIPTIONS[op][:50]}")

        summary_rows.append({
            "operator":        op,
            "description":     OPERATOR_DESCRIPTIONS[op],
            "n_sections":      n,
            "n_enriched":      n_gt1,
            "median_R":        round(med_R, 4),
            "sign_test_p":     round(float(sign_p), 6),
            "median_frac_coexact": round(med_fc, 4) if not np.isnan(med_fc) else np.nan,
            "median_perm_p":   round(med_p, 4),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / f"cohort_step23_summary_{flux_tag}.csv", index=False)

    # ── Robustness collapse analysis ────────────────────────────────────────────
    print()
    print("=== ROBUSTNESS COLLAPSE ANALYSIS ===")
    print()
    print("Interpretation guide:")
    print("  If enrichment survives sign_only  → coexact is a topological invariant,")
    print("  not explainable by magnitude/rank properties of the wedge construction.")
    print("  If enrichment collapses at rank or threshold → artifact of score scaling.")
    print()

    for op in OPERATORS:
        sub = all_df[(all_df["operator"] == op)].dropna(subset=["enrichment_ratio"])
        if len(sub) == 0:
            continue
        n     = len(sub)
        n_gt1 = int(np.sum(sub["enrichment_ratio"] > 1.0))
        sign_p = binomtest(n_gt1, n, p=0.5, alternative="greater").pvalue
        status = "✓ SURVIVES" if sign_p < 0.05 else "✗ COLLAPSES"
        print(f"  {status}  {op:<24}  {n_gt1}/{n}  p={float(sign_p):.5f}")

    print()
    print("[cohort] Done.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if _ARGS.mode == "sample":
    if _ARGS.sample_id is None:
        import sys
        print("Error: --sample-id required for --mode sample.")
        sys.exit(1)
    run_sample(
        sample_id = _ARGS.sample_id,
        flux_tag  = FLUX_TAG,
        stats_dir = STATS_DIR,
        out_dir   = OUT_DIR,
        n_perm    = _ARGS.n_perm,
        seed      = _ARGS.seed,
        min_nodes = _ARGS.min_nodes,
    )

elif _ARGS.mode == "cohort":
    run_cohort(flux_tag=FLUX_TAG, stats_dir=STATS_DIR, out_dir=OUT_DIR)
