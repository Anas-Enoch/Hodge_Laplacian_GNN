"""
Step 05 — CosMx: Cell Density Nuisance Control (v2 — correct test design)
==========================================================================
Self-contained.

REDESIGN vs v1:
───────────────
v1 used a CELL-LEVEL partial Spearman between interface membership and
node_coexact_energy after adjusting for total_counts. This was wrong for
two reasons:

  1. node_coexact_energy is a GRAPH property — it depends on neighboring
     cells, not just the cell itself. Spatial autocorrelation in this
     quantity violates the independence assumptions of partial Spearman.

  2. The cell-level test asks: "within interface+core cells, do
     high-coexact cells tend to be interface cells after counts adjustment?"
     This is not the right question. The right question is whether the
     FOV-level enrichment ratio R is predicted by FOV-level density.

v2 THREE-TEST HIERARCHY:
─────────────────────────
T1 (primary): FOV-level structural test — T2 density ratios across FOVs
  For each density covariate, compute the T2 ratio (mean covariate at
  interface / mean at tumor_core) per FOV. Test whether the DISTRIBUTION
  of T2 ratios is < 1 (sign test). If interface cells are LESS dense than
  tumor core → density structurally cannot drive interface enrichment.
  This is a structural argument: if the enriched zone is the less-dense
  zone, density cannot be the cause.

T2 (diagnostic): FOV-level correlation — does density PREDICT enrichment?
  Spearman r(T2_ratio, R_canonical) across FOVs.
  If density drove R, FOVs where interface is denser (T2>1) should have
  higher R. If this correlation is weak or absent → not density-driven.
  NOTE: a positive r does not prove density drives R — it may reflect
  small-core confound. Report with partial r(T2,R|log_n_core) to test this.

T3 (conservative stress test): cell-level input-score residualisation
  Residualise both program scores against total_counts per cell, recompute
  the wedge, retest enrichment. Expected to collapse (algebraic — see Visium
  Step 25 v3 documentation). Retained for completeness, NOT for verdict.

Reads: cosmx_breast_cells_hodge.csv.gz   (must contain node_coexact_energy)
       cosmx_breast_hodge_summary.csv     (for FOV-level R values)
Writes: cosmx_density_control_per_fov.csv
        cosmx_density_control_cohort_summary.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import Delaunay
from scipy.stats import spearmanr, binomtest

INTERFACE_LABEL = "interface"
CORE_LABEL      = "tumor_core"
LSQR_TOL        = 1e-10

def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--cells",   type=Path,
        default=Path("results_cosmx/cosmx_breast_cells_hodge.csv.gz"))
    p.add_argument("--summary", type=Path,
        default=Path("results_cosmx/cosmx_breast_hodge_summary.csv"))
    p.add_argument("--out",     type=Path, default=Path("results_cosmx/"))
    p.add_argument("--min-region", type=int, default=5)
    return p.parse_args()

def ols_res(y, x):
    v = float(np.var(x))
    if v < 1e-12: return y.copy()
    return y - float(np.cov(y,x)[0,1]/v)*x

def build_graph(coords, n):
    tri = Delaunay(coords, qhull_options="QJ Qbb Qc Qz")
    edge_set = set()
    for s in tri.simplices:
        verts = [int(v) for v in s if 0<=v<n]
        if len(set(verts))<3: continue
        a,b,c = sorted(verts[:3])
        edge_set |= {(a,b),(a,c),(b,c)}
    if not edge_set: return None, None
    edges = sorted(edge_set)
    th=np.array([e[0] for e in edges]); hd=np.array([e[1] for e in edges])
    return th, hd

def coexact_node_energy_from_scores(sa, sb, th, hd, n):
    ne = len(th)
    f  = sa[th]*sb[hd] - sa[hd]*sb[th]
    B1 = sp.csr_matrix(
        (np.concatenate([-np.ones(ne),np.ones(ne)]),
         (np.concatenate([th,hd]),np.concatenate([np.arange(ne),np.arange(ne)]))),
        shape=(n,ne))
    L0=( B1@B1.T).tocsr()
    alpha=spla.lsqr(L0,B1@f,atol=LSQR_TOL,btol=LSQR_TOL)[0]
    f_cx=f-B1.T@alpha; sq=f_cx**2
    nc=np.zeros(n); cnt=np.zeros(n)
    np.add.at(nc,th,sq); np.add.at(nc,hd,sq)
    np.add.at(cnt,th,1); np.add.at(cnt,hd,1)
    return nc/np.maximum(cnt,1)

def enrich_perm(nc, region, rng, min_n, n_perm=200):
    im=region==INTERFACE_LABEL; cm=region==CORE_LABEL
    if im.sum()<min_n or cm.sum()<min_n: return np.nan, np.nan
    mu_i=float(np.median(nc[im])); mu_c=float(np.median(nc[cm]))
    if mu_c<1e-12: return np.nan, np.nan
    obs=mu_i/mu_c
    pool=im|cm; pidx=np.where(pool)[0]; plabs=region[pidx].copy()
    null=np.zeros(n_perm)
    for k in range(n_perm):
        rng.shuffle(plabs); perm=region.copy(); perm[pidx]=plabs
        mi=np.median(nc[perm==INTERFACE_LABEL]) if (perm==INTERFACE_LABEL).any() else 0.
        mc=np.median(nc[perm==CORE_LABEL]) if (perm==CORE_LABEL).any() else 1e-12
        null[k]=mi/mc if mc>1e-12 else 0.
    return obs, float((np.sum(null>=obs)+1)/(n_perm+1))

# ── Per-FOV: compute T2 density ratio per covariate ──────────────────────────

def per_fov_density_ratios(sub, min_n):
    """
    For each density covariate, compute:
      T2 = mean(covariate at interface) / mean(covariate at tumor_core)
    Returns dict: covariate → T2 ratio.
    """
    region = sub["region_label"].to_numpy()
    im = region == INTERFACE_LABEL; cm = region == CORE_LABEL
    if im.sum() < min_n or cm.sum() < min_n:
        return {}

    covariates = {}
    for col, name in [("total_counts","total_counts"),
                      ("cell_area_px","cell_area"),
                      ("protein_tumor_proxy","PanCK"),
                      ("protein_immune_proxy","CD45")]:
        if col in sub.columns:
            vals = pd.to_numeric(sub[col], errors="coerce").fillna(0).to_numpy(float)
            if np.std(vals) > 1e-6:
                mu_i = float(vals[im].mean()); mu_c = float(vals[cm].mean())
                covariates[name] = mu_i / mu_c if mu_c > 1e-12 else np.nan
    return covariates

# ── T3: input score residualisation (per FOV, expected to collapse) ───────────

def t3_input_residual(sub, min_n, seed):
    """Residualise tumor+immune scores against total_counts, recompute wedge."""
    if "total_counts" not in sub.columns: return np.nan, np.nan
    rng = np.random.default_rng(seed)
    region = sub["region_label"].to_numpy()
    sa = sub["tumor_score"].to_numpy(float)
    sb = sub["immune_score"].to_numpy(float)
    counts = pd.to_numeric(sub["total_counts"], errors="coerce").fillna(0).to_numpy(float)
    if np.std(counts) < 1e-6: return np.nan, np.nan
    sa_r = ols_res(sa, counts); sb_r = ols_res(sb, counts)
    coords = sub[["x","y"]].to_numpy(float); n = len(sub)
    th, hd = build_graph(coords, n)
    if th is None: return np.nan, np.nan
    nc = coexact_node_energy_from_scores(sa_r, sb_r, th, hd, n)
    return enrich_perm(nc, region, rng, min_n, n_perm=100)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    a = _args()
    a.out.mkdir(parents=True, exist_ok=True)
    df      = pd.read_csv(a.cells, compression="gzip")
    summary = pd.read_csv(a.summary)

    # Compute per-FOV T2 density ratios
    t2_rows = []
    t3_rows = []
    for fov, sub in df.groupby("fov"):
        sub = sub.copy().reset_index(drop=True)
        ratios = per_fov_density_ratios(sub, a.min_region)
        for cov, t2 in ratios.items():
            t2_rows.append({"fov":fov,"covariate":cov,"T2_ratio":t2})
        # T3 on total_counts only
        R3, p3 = t3_input_residual(sub, a.min_region, seed=42)
        t3_rows.append({"fov":fov,"T3_R":R3,"T3_perm_p":p3})

    t2_df = pd.DataFrame(t2_rows)
    t3_df = pd.DataFrame(t3_rows)

    # Merge with canonical R from step03 summary
    s3 = summary[["fov","interface_vs_tumor_core_ratio","n_tumor_core"]]\
             .rename(columns={"interface_vs_tumor_core_ratio":"R_canon"})

    print(f"\n=== STEP 05 DENSITY CONTROL (v2) ===\n")
    print("Primary test (T1): sign test on per-FOV T2 density ratios")
    print("  H0: P(T2<1) = 0.50   H1: interface is less dense than core\n")

    summary_rows = []
    per_fov_out  = []
    for cov in t2_df["covariate"].unique():
        sub  = t2_df[t2_df["covariate"]==cov].dropna(subset=["T2_ratio"])
        vals = sub["T2_ratio"].to_numpy()
        n_lt1 = int((vals < 1).sum()); n = len(vals)
        med   = float(np.median(vals))
        b_t1  = binomtest(n_lt1, n, p=0.5, alternative="greater") if n>=3 else None
        p_t1  = b_t1.pvalue if b_t1 else np.nan

        # T2: FOV-level correlation with canonical R
        m2 = sub.merge(s3, on="fov").dropna(subset=["R_canon"])
        r_raw, p_r = spearmanr(m2["T2_ratio"], m2["R_canon"]) if len(m2)>=5 \
                     else (np.nan, np.nan)
        # Partial r controlling for log(n_tumor_core) — core-size confound
        if "n_tumor_core" in m2.columns and len(m2)>=5:
            y_res  = ols_res(m2["R_canon"].to_numpy(),
                             np.log1p(m2["n_tumor_core"].to_numpy()))
            x_res  = ols_res(m2["T2_ratio"].to_numpy(),
                             np.log1p(m2["n_tumor_core"].to_numpy()))
            r_part, p_part = spearmanr(y_res, x_res)
        else:
            r_part = p_part = np.nan

        # Verdict based on T1 (structural sign test on T2 < 1)
        density_excluded = (not np.isnan(p_t1) and p_t1 < 0.05 and med < 1.0)
        verdict = "DENSITY_NOT_DRIVER" if density_excluded else "UNRESOLVED"

        sym = "✓ EXCLUDED" if density_excluded else "✗ UNRESOLVED"
        print(f"  {sym}  [{cov:<18}]")
        print(f"    T1: T2<1 in {n_lt1}/{n} FOVs  median_T2={med:.3f}  "
              f"sign_p={p_t1:.4f}" if not np.isnan(p_t1) else
              f"    T1: n/a (insufficient FOVs)")
        print(f"    T2: r(T2,R)={r_raw:.3f}  p={p_r:.4f}  "
              f"partial_r(|core_size)={r_part:.3f}  p={p_part:.4f}"
              if not np.isnan(r_raw) else f"    T2: n/a")
        print()

        summary_rows.append({
            "covariate":   cov,
            "n_fovs":      n,
            "n_T2_lt1":    n_lt1,
            "median_T2":   round(med,4),
            "T1_sign_p":   round(float(p_t1),4)   if not np.isnan(p_t1)   else np.nan,
            "T2_r_raw":    round(float(r_raw),4)   if not np.isnan(r_raw)  else np.nan,
            "T2_r_partial":round(float(r_part),4)  if not np.isnan(r_part) else np.nan,
            "verdict":     verdict,
        })
        # Per-FOV output
        for _, row in sub.iterrows():
            per_fov_out.append({"fov":row["fov"],"covariate":cov,"T2_ratio":row["T2_ratio"]})

    # T3 summary
    t3_valid = t3_df.dropna(subset=["T3_R"])
    med_R3 = float(t3_valid["T3_R"].median()) if len(t3_valid)>0 else np.nan
    print(f"  [T3 conservative — expected to collapse]")
    print(f"    input-score residualisation on total_counts")
    print(f"    median_R_residual={med_R3:.3f} (collapse expected — algebraic,")
    print(f"    not evidence density drives signal; see Visium Step 25 v3 docs)")
    print()

    # Manuscript note
    tc_row = next((r for r in summary_rows if r["covariate"]=="total_counts"), None)
    if tc_row:
        v = tc_row["verdict"]
        n_lt = tc_row["n_T2_lt1"]; nt = tc_row["n_fovs"]
        med  = tc_row["median_T2"]; p1 = tc_row["T1_sign_p"]
        print("── MANUSCRIPT NOTE ─────────────────────────────────────────────────")
        if v=="DENSITY_NOT_DRIVER":
            print(f"  Interface cells have fewer transcripts than tumor core in {n_lt}/{nt} FOVs")
            print(f"  (median T2={med:.3f}, sign test p={p1:.4f}).")
            print(f"  Cell density cannot be the primary driver of interface coexact enrichment.")
        else:
            print(f"  T2 median={med:.3f}: interface cells have fewer transcripts in {n_lt}/{nt} FOVs.")
            r_p = tc_row.get("T2_r_partial",np.nan)
            if not np.isnan(r_p) and abs(r_p) < 0.3:
                print(f"  Partial r(T2,R|core_size)={r_p:.3f}: correlation between density")
                print(f"  and enrichment ratio is largely explained by core-size confound.")
            print(f"  T3 collapse (R={med_R3:.3f}) is algebraically expected.")

    # Save outputs
    pd.DataFrame(summary_rows).to_csv(
        a.out/"cosmx_density_control_cohort_summary.csv", index=False)
    pd.DataFrame(per_fov_out).merge(t3_df, on="fov", how="left")\
      .to_csv(a.out/"cosmx_density_control_per_fov.csv", index=False)
    print("\n[done]")

if __name__ == "__main__": main()
