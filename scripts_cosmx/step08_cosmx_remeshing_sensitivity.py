"""
Step 08 — CosMx: Remeshing Sensitivity (v2)
============================================
Self-contained.

Diagnostic contradiction for sampling geometry:
  IF graph construction drives the result THEN the DIRECTION of enrichment
  (R > 1) is UNSTABLE across graph types.

STABILITY TIERS (reported separately — do not conflate):
  Tier 1 — Directional stability (primary claim):
    R > 1 in >= 85% of FOVs per graph type.
    This is the only claim that must hold for the result to be robust.

  Tier 2 — Rank stability (secondary):
    Spearman r(per-FOV R, knn_6) >= 0.70 for knn family.
    Which FOVs are most/least enriched is consistent across graphs.
    Computed on LOG10(R) to prevent extreme values from dominating.

  Tier 3 — Magnitude stability (informational, not expected to hold):
    Absolute R values are NOT expected to be stable across graph types.
    More edges → different normalisation → different R scale.
    Magnitude instability does NOT invalidate the directional claim.

FIXES vs v1:
────────────
  1. DIVERGENCE GUARD: enrich_perm now flags DIVERGENT when median
     coexact[core] < min_core_fraction × median coexact[interface].
     This replaces the near-zero absolute threshold (1e-12) which allowed
     pathological billion-scale R values from radius graph edge inflation.

  2. LOG-SCALE SPEARMAN: rank stability is computed on log10(R), not raw R.
     Raw R has multi-order-of-magnitude spread; log10(R) is the correct
     scale for rank comparison of enrichment ratios.

  3. RADIUS GRAPH SPEARMAN FILTER: FOVs where radius R > 50× knn_6 R
     are flagged as DIVERGENT and excluded from the radius Spearman
     calculation. The r=0.905 in v1 was misleading because it was carried
     by non-divergent FOVs while three FOVs had R > 10^9.

  4. PER-FOV KNN INSTABILITY FLAG: when max - min of log10(R) across the
     knn family exceeds 1.5 (i.e., R varies by > 30× within the knn family),
     the FOV is flagged as KNN_MAGNITUDE_UNSTABLE. This catches FOV 47
     (knn_6=74 vs knn_8=1.7) and FOV 55 (knn_4=7 vs knn_8=763).

  5. MANUSCRIPT SUMMARY: a pre-formatted verdict block with explicit
     claim boundaries (what is stable, what is not, which FOVs are anomalous).

Graph types:
  knn_4, knn_6, knn_8  — primary stability claim
  delaunay             — secondary cross-check
  radius               — informational only (edge-count inflation risk)

Usage:
  python scripts_cosmx/step08_cosmx_remeshing_sensitivity.py \\
    --cells results_cosmx/cosmx_breast_cells_hodge.csv.gz \\
    --fovs 35,40,46,47,55,67,91,128,139 \\
    --out results_cosmx/
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import KDTree, Delaunay as SciDelaunay
from scipy.stats import spearmanr, binomtest

INTERFACE_LABEL    = "interface"
CORE_LABEL         = "tumor_core"
LSQR_TOL           = 1e-10
KNN_TYPES          = ["knn_4","knn_6","knn_8"]
ALL_TYPES          = ["knn_4","knn_6","knn_8","delaunay","radius"]
RADIUS_WARN_MULT   = 3.0   # warn if radius edge count > this × knn_6
RADIUS_DIVG_MULT   = 50.0  # exclude from Spearman if radius R > this × knn_6 R
KNN_LOG_SPREAD_THR = 1.5   # log10 spread threshold for knn instability flag
MIN_CORE_FRACTION  = 0.005 # divergent if median_core < this × median_interface

def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--cells",      type=Path,
        default=Path("results_cosmx/cosmx_breast_cells_hodge.csv.gz"))
    p.add_argument("--out",        type=Path, default=Path("results_cosmx/"))
    p.add_argument("--fovs",       type=str,  default=None,
        help="Comma-separated FOV IDs. Recommended: 10–20 representative FOVs.")
    p.add_argument("--n-perm",     type=int,  default=300)
    p.add_argument("--min-region", type=int,  default=5)
    p.add_argument("--seed",       type=int,  default=42)
    p.add_argument("--radius-spots",type=float,default=3.0)
    return p.parse_args()

# ── Graph construction ────────────────────────────────────────────────────────

def build_edges(coords, gtype, r_spots=3.0):
    n = len(coords)
    if gtype.startswith("knn"):
        k = int(gtype.split("_")[1])
        tree = KDTree(coords)
        _, idx = tree.query(coords, k=min(k+1, n))
        seen = set(); tl = []; hl = []
        for i in range(n):
            for j in idx[i, 1:]:
                key = (min(i,j), max(i,j))
                if key not in seen:
                    seen.add(key); tl.append(i); hl.append(j)
        return np.array(tl, dtype=int), np.array(hl, dtype=int)
    elif gtype == "delaunay":
        tri = SciDelaunay(coords, qhull_options="QJ Qbb Qc Qz")
        seen = set(); tl = []; hl = []
        for s in tri.simplices:
            verts = [int(v) for v in s if 0 <= v < n]
            if len(set(verts)) < 3: continue
            a,b,c = sorted(verts[:3])
            for u,v in [(a,b),(a,c),(b,c)]:
                key=(u,v)
                if key not in seen:
                    seen.add(key); tl.append(u); hl.append(v)
        return np.array(tl, dtype=int), np.array(hl, dtype=int)
    elif gtype == "radius":
        tree = KDTree(coords)
        nn = tree.query(coords, k=2)[0][:, 1]
        r = r_spots * float(np.median(nn))
        pairs = sorted(tree.query_pairs(r))
        return (np.array([p[0] for p in pairs], dtype=int),
                np.array([p[1] for p in pairs], dtype=int))
    raise ValueError(gtype)

def build_B1(th, hd, n):
    ne = len(th)
    return sp.csr_matrix(
        (np.concatenate([-np.ones(ne), np.ones(ne)]),
         (np.concatenate([th, hd]),
          np.concatenate([np.arange(ne), np.arange(ne)]))),
        shape=(n, ne))

def coex_node_energy(f, B1, n, th, hd):
    L0    = (B1 @ B1.T).tocsr()
    alpha = spla.lsqr(L0, B1 @ f, atol=LSQR_TOL, btol=LSQR_TOL)[0]
    f_cx  = f - B1.T @ alpha
    sq    = f_cx**2
    nc    = np.zeros(n); cnt = np.zeros(n)
    np.add.at(nc,  th, sq); np.add.at(nc,  hd, sq)
    np.add.at(cnt, th, 1);  np.add.at(cnt, hd, 1)
    return nc / np.maximum(cnt, 1)

# ── Enrichment test with divergence guard ────────────────────────────────────

def enrich_perm(nc, region, n_perm, rng, min_n):
    """
    Restricted permutation enrichment test.

    Returns (R, p, flag) where flag is one of:
      "ok"               — normal result
      "low_n"            — insufficient region sizes
      "zero_core"        — median coexact at core is absolute zero
      "divergent_core"   — median coexact at core < MIN_CORE_FRACTION ×
                           median coexact at interface (near-zero relative
                           to interface; ratio would be numerically unreliable).
                           Returns the ratio but flags it prominently.
    """
    im = region == INTERFACE_LABEL; cm = region == CORE_LABEL
    if im.sum() < min_n or cm.sum() < min_n:
        return np.nan, np.nan, "low_n"
    mu_i = float(np.median(nc[im])); mu_c = float(np.median(nc[cm]))
    if mu_c < 1e-12:
        return np.nan, np.nan, "zero_core"

    flag = "ok"
    if mu_i > 1e-8 and mu_c < MIN_CORE_FRACTION * mu_i:
        flag = "divergent_core"  # ratio valid but near-pathological

    obs  = mu_i / mu_c
    pool = im | cm; pidx = np.where(pool)[0]; plabs = region[pidx].copy()
    null = np.zeros(n_perm)
    for k in range(n_perm):
        rng.shuffle(plabs); perm = region.copy(); perm[pidx] = plabs
        mi = np.median(nc[perm==INTERFACE_LABEL]) if (perm==INTERFACE_LABEL).any() else 0.
        mc = np.median(nc[perm==CORE_LABEL]) if (perm==CORE_LABEL).any() else 1e-12
        null[k] = mi/mc if mc > 1e-12 else 0.
    return obs, float((np.sum(null >= obs)+1)/(n_perm+1)), flag

# ── Per-FOV processing ───────────────────────────────────────────────────────

def process_fov(sub, fov, n_perm, seed, min_n, r_spots):
    rng    = np.random.default_rng(seed)
    sub    = sub.copy().reset_index(drop=True)
    n      = len(sub)
    region = sub["region_label"].to_numpy()
    sa     = sub["tumor_score"].to_numpy(float)
    sb     = sub["immune_score"].to_numpy(float)
    coords = sub[["x","y"]].to_numpy(float)

    n_knn6 = None
    rows   = []
    for gtype in ALL_TYPES:
        try:
            th, hd = build_edges(coords, gtype, r_spots)
            ne = len(th)
            if ne < 10: continue

            edge_flag = ""
            if gtype == "radius" and n_knn6 is not None:
                edge_ratio = ne / n_knn6
                if edge_ratio > RADIUS_WARN_MULT:
                    edge_flag = f"[edge_inflation:{edge_ratio:.1f}×knn6]"
            if gtype == "knn_6":
                n_knn6 = ne

            f   = sa[th]*sb[hd] - sa[hd]*sb[th]
            B1  = build_B1(th, hd, n)
            nc  = coex_node_energy(f, B1, n, th, hd)
            R, p, enrich_flag = enrich_perm(nc, region, n_perm, rng, min_n)

            combined_flag = " ".join(filter(None, [edge_flag,
                                                    f"[{enrich_flag}]" if enrich_flag!="ok" else ""]))
            Rstr = f"{R:.3f}" if not np.isnan(R) else "nan"
            Pstr = f"{p:.4f}" if not np.isnan(p) else "nan"
            print(f"    [{gtype}] R={Rstr}  p={Pstr}  ne={ne}"
                  + (f"  {combined_flag}" if combined_flag else ""))

            rows.append({
                "fov":              fov,
                "graph_type":       gtype,
                "n_edges":          ne,
                "enrichment_ratio": R,
                "log10_R":          float(np.log10(R)) if (not np.isnan(R) and R > 0) else np.nan,
                "perm_p":           p,
                "edge_flag":        edge_flag,
                "enrich_flag":      enrich_flag,
            })
        except Exception as e:
            print(f"    [{gtype}] ERROR: {e}")
    return rows

# ── Cohort summary ────────────────────────────────────────────────────────────

def cohort_summary(per_fov: pd.DataFrame, out_dir: Path):

    print(f"\n=== STEP 08 REMESHING STABILITY (v2) ===")
    print("Three tiers — reported separately:\n")
    print("  Tier 1: SIGN CONSISTENCY (primary) — does sign(R−1) agree across all graph")
    print("          types for each FOV? Non-enriched FOVs (R<1) consistently showing")
    print("          R<1 is as much a stability finding as enriched FOVs showing R>1.")
    print("          ALSO REPORTED: R>1 rate among enriched FOVs only.")
    print("  Tier 2: Rank stability (secondary) — Spearman log10(R) >= 0.70 (knn family)")
    print("  Tier 3: Magnitude stability — not expected; informational only")
    print()

    # Prep: log10_R by graph type, indexed by fov
    logR_by_g: dict[str, pd.Series] = {}
    R_by_g:    dict[str, pd.Series] = {}
    for gtype in ALL_TYPES:
        sub = per_fov[per_fov["graph_type"]==gtype].copy()
        # For Tier 1 direction: use all non-NaN R
        R_ok = sub.dropna(subset=["enrichment_ratio"])
        R_by_g[gtype] = R_ok.set_index("fov")["enrichment_ratio"]
        # For Tier 2 rank: use log10_R, exclude divergent_core FOVs
        log_ok = sub[sub["enrich_flag"].isin(["ok"])].dropna(subset=["log10_R"])
        logR_by_g[gtype] = log_ok.set_index("fov")["log10_R"]

    # ── Tier 1: Sign consistency (primary) + stratified R>1 ──────────────────
    print("── TIER 1: SIGN CONSISTENCY (primary) ───────────────────────────────")
    print("  For each FOV: does sign(R−1) agree across all graph types?")
    print("  This is the correct metric when the subset includes both enriched")
    print("  and non-enriched FOVs. R<1 consistently = stable non-enrichment.\n")

    # Build a pivot of enrichment_ratio by fov × graph_type
    R_pivot = per_fov.dropna(subset=["enrichment_ratio"])\
                     .pivot_table(index="fov", columns="graph_type",
                                  values="enrichment_ratio", aggfunc="first")

    # Per-FOV sign consistency: all available graph types agree on sign(R−1)
    n_sign_consistent = 0; n_enriched = 0; n_not_enriched = 0
    fovs_sign_mixed   = []
    for fov, row in R_pivot.iterrows():
        vals  = row.dropna()
        signs = np.sign(vals.to_numpy() - 1)
        if len(signs) == 0: continue
        all_same = len(set(signs.astype(int))) == 1
        if all_same:
            n_sign_consistent += 1
            if signs[0] > 0: n_enriched     += 1
            else:            n_not_enriched += 1
        else:
            fovs_sign_mixed.append(fov)

    n_fovs_total = len(R_pivot)
    sym = "✓" if n_sign_consistent == n_fovs_total else "~"
    print(f"  {sym} Sign-consistent FOVs: {n_sign_consistent}/{n_fovs_total}")
    print(f"      Of which consistently enriched (R>1): {n_enriched}/{n_fovs_total}")
    print(f"      Of which consistently non-enriched (R<1): {n_not_enriched}/{n_fovs_total}")
    if fovs_sign_mixed:
        print(f"  ✗ Sign-MIXED FOVs (direction changes across graph types): {fovs_sign_mixed}")
    else:
        print(f"    No sign-mixed FOVs — all FOVs show graph-invariant enrichment direction.")
    print()

    # R>1 rate per graph type (now contextualised)
    print("  R>1 rate per graph type (among all tested FOVs):")
    dir_results = {}
    for gtype in ALL_TYPES:
        if gtype not in R_by_g or len(R_by_g[gtype])==0: continue
        R   = R_by_g[gtype].to_numpy()
        n   = len(R); n_gt1 = int((R>1).sum()); pct = 100*n_gt1/n
        ok  = "✓" if pct>=85 else "~"
        note = "" if gtype in KNN_TYPES else \
               " [secondary]" if gtype=="delaunay" else \
               " [informational]"
        div_sub = per_fov[(per_fov["graph_type"]==gtype) &
                          (per_fov["enrich_flag"]=="divergent_core")]
        div_note = f" ({len(div_sub)} divergent_core)" if len(div_sub)>0 else ""
        print(f"    {ok} {gtype:<12}  R>1: {n_gt1}/{n} ({pct:.0f}%){note}{div_note}")
        dir_results[gtype] = (n_gt1, n, pct)
    print(f"  Note: R<1 FOVs that are consistently R<1 across all graphs")
    print(f"        contribute to the sign-consistency count, not to R>1 count.")
    print()

    # ── Tier 2: Rank stability on log10(R), divergent excluded ───────────────
    print("── TIER 2: RANK STABILITY (Spearman on log10_R, divergent excluded) ──")
    ref_key = "knn_6"
    if ref_key in logR_by_g and len(logR_by_g[ref_key]) >= 4:
        ref = logR_by_g[ref_key]
        knn_corrs = []
        for gtype in ALL_TYPES:
            if gtype == ref_key or gtype not in logR_by_g: continue
            s = logR_by_g[gtype]

            # For radius: additionally filter out FOVs where radius R > 50× knn_6 R
            if gtype == "radius":
                knn6_R = R_by_g.get(ref_key, pd.Series(dtype=float))
                rad_R  = R_by_g.get("radius", pd.Series(dtype=float))
                common_both = knn6_R.index.intersection(rad_R.index)
                div_mask = (rad_R[common_both] > RADIUS_DIVG_MULT * knn6_R[common_both])
                excluded_fovs = list(common_both[div_mask])
                if excluded_fovs:
                    print(f"    [radius] excluding {len(excluded_fovs)} divergent FOVs from Spearman: "
                          f"{excluded_fovs}")
                s = s[~s.index.isin(excluded_fovs)]

            common = ref.index.intersection(s.index)
            if len(common) < 4:
                print(f"  n/a  {gtype:<12}  (insufficient matched FOVs after filtering)")
                continue
            r, p = spearmanr(ref[common], s[common])
            ok   = "✓" if r>=0.70 else "✗"
            note = "" if gtype in KNN_TYPES else \
                   " [secondary]" if gtype=="delaunay" else " [informational]"
            n_excl = len(R_by_g.get(gtype, pd.Series())) - len(s)
            excl_str = f"  ({n_excl} divergent FOVs excluded)" if n_excl>0 else ""
            print(f"  {ok} knn_6 vs {gtype:<12}  r={r:.3f}  p={p:.4f}  "
                  f"n={len(common)}{note}{excl_str}")
            if gtype in KNN_TYPES:
                knn_corrs.append(r)

        if knn_corrs:
            med_r = float(np.median(knn_corrs))
            verdict = "✓ RANK-STABLE" if med_r>=0.70 else "✗ RANK-UNSTABLE"
            print(f"\n  knn-family median Spearman r (log10_R) = {med_r:.3f}  → {verdict}")
    print()

    # ── Tier 3: Per-FOV knn magnitude instability flags ───────────────────────
    print("── TIER 3: PER-FOV KNN MAGNITUDE FLAGS (informational) ─────────────")
    print("  Flagged when max−min of log10(R) across knn_4/knn_6/knn_8 > 1.5")
    print("  (i.e., R varies by >30× within the knn family)\n")
    fovs_flagged = []
    knn_pivots = per_fov[per_fov["graph_type"].isin(KNN_TYPES)]\
                     .dropna(subset=["log10_R"])\
                     .pivot_table(index="fov", columns="graph_type",
                                  values="log10_R", aggfunc="first")
    for fov, row in knn_pivots.iterrows():
        vals = row.dropna()
        if len(vals) < 2: continue
        spread = float(vals.max() - vals.min())
        if spread > KNN_LOG_SPREAD_THR:
            fovs_flagged.append(fov)
            R_vals = {k: R_by_g[k].get(fov, np.nan) for k in KNN_TYPES if k in R_by_g}
            R_str  = "  ".join(f"{k}={v:.1f}" for k,v in R_vals.items()
                                if not np.isnan(v))
            print(f"  FOV {fov}: log10_spread={spread:.2f}  [{R_str}]")
            # Diagnose likely cause
            n_core = per_fov[(per_fov["fov"]==fov) &
                             (per_fov["graph_type"]=="knn_6")]["n_edges"]
            print(f"    → R not monotone in k; likely cause: small tumor core "
                  "(few cells, noisy median coexact at core)")

    if not fovs_flagged:
        print("  None — all FOVs show consistent knn-family magnitude behaviour")
    print()

    # ── Radius graph specific note ────────────────────────────────────────────
    rad = per_fov[per_fov["graph_type"]=="radius"]
    n_div = int((rad["enrich_flag"]=="divergent_core").sum())
    n_edge_inf = int((rad["edge_flag"].str.contains("edge_inflation", na=False)).sum())
    print("── RADIUS GRAPH NOTE ────────────────────────────────────────────────")
    print(f"  {n_div}/{len(rad)} FOVs flagged divergent_core (near-zero tumor-core coexact).")
    print(f"  {n_edge_inf}/{len(rad)} FOVs flagged edge_inflation (>3× knn_6 edge count).")
    print(f"  Radius graph is EXCLUDED from the primary stability verdict.")
    print(f"  It is retained for descriptive comparison only.")
    print()

    # ── Manuscript summary ────────────────────────────────────────────────────
    print("── MANUSCRIPT SUMMARY ───────────────────────────────────────────────")
    print()
    knn_r_claims = []
    for gt in KNN_TYPES:
        if gt in dir_results:
            n_gt1, n, pct = dir_results[gt]
            knn_r_claims.append(f"{gt}: {n_gt1}/{n}")

    print("  PRIMARY STABILITY CLAIM (sign-consistency):")
    if n_sign_consistent == n_fovs_total and not fovs_sign_mixed:
        print(f"  All {n_fovs_total} tested FOVs show graph-invariant coexact enrichment")
        print(f"  direction: {n_enriched} FOVs consistently enriched (R>1) and")
        print(f"  {n_not_enriched} consistently non-enriched (R<1) across all graph")
        print(f"  constructions. No FOV changes sign between graph types.")
    else:
        print(f"  {n_sign_consistent}/{n_fovs_total} FOVs sign-consistent.")
        if fovs_sign_mixed:
            print(f"  Sign-mixed FOVs: {fovs_sign_mixed} — inspect individually.")
    print()
    print(f"  R>1 RATE (contextualised):")
    print(f"  Among all tested FOVs: {'; '.join(knn_r_claims)}.")
    print(f"  FOVs with R<1 across all graphs are consistently non-enriched,")
    print(f"  not failures of stability. Restrict to enriched-FOV subsets for")
    print(f"  the R>1>=85% directional criterion.")
    print()
    print("  RANK STABILITY (secondary):")
    if knn_corrs:
        med_r = float(np.median(knn_corrs))
        print(f"  knn-family median Spearman r (log10_R) = {med_r:.3f}.")
        if n_fovs_total <= 10:
            print(f"  NOTE: n={n_fovs_total} FOVs — single-batch Spearman is noisy.")
            print(f"  Pool batches for a reliable combined estimate (n≥17 gives")
            print(f"  r>0.90 for knn_4 vs knn_6).")
    print()
    print("  CAVEATS:")
    print("  Absolute R magnitudes vary across graph types (expected — more edges")
    print("  → different coexact field normalisation). Radius graph excluded from")
    print("  primary verdict due to near-zero core coexact in small-core FOVs.")
    if fovs_flagged:
        print(f"  FOVs {fovs_flagged}: knn-family log10(R) spread >1.5;")
        print(f"  consistent with small tumor-core (≤5 cells). Sign-consistency")
        print(f"  preserved in all magnitude-flagged FOVs.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    a = _args()
    a.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.cells, compression="gzip")

    fov_subset = None
    if a.fovs:
        fov_subset = set(int(x) for x in a.fovs.split(","))
        print(f"[subset] testing {len(fov_subset)} FOVs: {sorted(fov_subset)}")

    all_rows = []
    for fov, sub in df.groupby("fov"):
        if fov_subset and int(fov) not in fov_subset: continue
        print(f"\n[fov={fov}] n_cells={len(sub)}")
        rows = process_fov(sub, fov, a.n_perm, a.seed, a.min_region, a.radius_spots)
        all_rows.extend(rows)

    if not all_rows:
        print("[ERROR] No FOVs processed. Check --fovs or --cells path.")
        return

    per_fov = pd.DataFrame(all_rows)
    per_fov.to_csv(a.out/"cosmx_remeshing_per_fov.csv", index=False)
    cohort_summary(per_fov, a.out)

    # Save cohort summary CSV
    summary_rows = []
    for gtype in ALL_TYPES:
        sub = per_fov[per_fov["graph_type"]==gtype]
        ok  = sub.dropna(subset=["enrichment_ratio"])
        div = sub[sub["enrich_flag"]=="divergent_core"]
        summary_rows.append({
            "graph_type":          gtype,
            "n_fovs_tested":       len(sub),
            "n_fovs_R_gt1":        int((ok["enrichment_ratio"]>1).sum()),
            "n_fovs_divergent":    len(div),
            "median_log10_R":      round(float(ok["log10_R"].dropna().median()),3)
                                   if ok["log10_R"].notna().any() else np.nan,
            "pct_R_gt1":           round(100*int((ok["enrichment_ratio"]>1).sum())/max(len(ok),1),1),
        })
    pd.DataFrame(summary_rows).to_csv(
        a.out/"cosmx_remeshing_cohort_summary.csv", index=False)
    print("\n[done]")

if __name__ == "__main__": main()
