"""
Step 06 — CosMx: Operator Robustness (Step 23 equivalent)
===========================================================
Self-contained.

Five antisymmetric operators applied in decreasing information order:
  proxy_wedge         full magnitude + rank + sign  (baseline)
  normalized_wedge    removes absolute scale
  rank_antisym        rank only + sign
  thresholded_antisym strong magnitude + sign (top 50%)
  sign_only           sign only  ← extreme-robust test

If coexact enrichment survives sign_only → geometric invariant, not artifact.

Reads: cosmx_breast_cells_hodge.csv.gz
Writes: cosmx_operator_robustness_per_fov.csv
        cosmx_operator_robustness_cohort_summary.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import Delaunay
from scipy.stats import binomtest

INTERFACE_LABEL = "interface"
CORE_LABEL      = "tumor_core"
LSQR_TOL        = 1e-10
OPERATORS       = ["proxy_wedge","normalized_wedge","rank_antisym",
                   "thresholded_antisym","sign_only"]

def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--cells",    type=Path,
        default=Path("results_cosmx/cosmx_breast_cells_hodge.csv.gz"))
    p.add_argument("--out",      type=Path, default=Path("results_cosmx/"))
    p.add_argument("--n-perm",   type=int,  default=500)
    p.add_argument("--min-region",type=int, default=5)
    p.add_argument("--seed",     type=int,  default=42)
    return p.parse_args()

def build_graph(coords):
    n = len(coords)
    tri = Delaunay(coords, qhull_options="QJ Qbb Qc Qz")
    edge_set = set()
    for s in tri.simplices:
        verts = [int(v) for v in s if 0<=v<n]
        if len(set(verts))<3: continue
        a,b,c = sorted(verts[:3])
        edge_set |= {(a,b),(a,c),(b,c)}
    if not edge_set: return None, None, None
    edges = sorted(edge_set); ne = len(edges)
    th = np.array([e[0] for e in edges]); hd = np.array([e[1] for e in edges])
    B1 = sp.csr_matrix(
        (np.concatenate([-np.ones(ne),np.ones(ne)]),
         (np.concatenate([th,hd]),np.concatenate([np.arange(ne),np.arange(ne)]))),
        shape=(n,ne))
    return B1, th, hd

def coex_node_energy(f, B1, n, th, hd):
    L0 = (B1@B1.T).tocsr()
    alpha = spla.lsqr(L0, B1@f, atol=LSQR_TOL, btol=LSQR_TOL)[0]
    f_cx = f - B1.T@alpha
    sq = f_cx**2; nc = np.zeros(n); cnt = np.zeros(n)
    np.add.at(nc,th,sq); np.add.at(nc,hd,sq)
    np.add.at(cnt,th,1); np.add.at(cnt,hd,1)
    return nc/np.maximum(cnt,1)

def enrich_perm(nc, region, n_perm, rng, min_n):
    im = region==INTERFACE_LABEL; cm = region==CORE_LABEL
    if im.sum()<min_n or cm.sum()<min_n: return np.nan, np.nan
    mu_i=float(np.median(nc[im])); mu_c=float(np.median(nc[cm]))
    if mu_c<1e-12: return np.nan, np.nan
    obs = mu_i/mu_c
    pool=im|cm; pidx=np.where(pool)[0]; plabs=region[pidx].copy()
    null=np.zeros(n_perm)
    for k in range(n_perm):
        rng.shuffle(plabs); perm=region.copy(); perm[pidx]=plabs
        mi = np.median(nc[perm==INTERFACE_LABEL]) if (perm==INTERFACE_LABEL).any() else 0.
        mc = np.median(nc[perm==CORE_LABEL]) if (perm==CORE_LABEL).any() else 1e-12
        null[k]=mi/mc if mc>1e-12 else 0.
    return obs, float((np.sum(null>=obs)+1)/(n_perm+1))

def apply_operator(f_raw, sa, sb, th, hd, op, eps=1e-8):
    ne = len(f_raw)
    if op=="proxy_wedge":
        return f_raw.copy()
    elif op=="normalized_wedge":
        denom = np.abs(sa[th])+np.abs(sa[hd])+np.abs(sb[th])+np.abs(sb[hd])+eps
        return f_raw/denom
    elif op=="rank_antisym":
        ranks = np.zeros(ne)
        ranks[np.argsort(np.abs(f_raw))] = np.arange(1,ne+1,dtype=float)
        return np.sign(f_raw)*ranks/ne
    elif op=="thresholded_antisym":
        thr = np.median(np.abs(f_raw))
        return f_raw*(np.abs(f_raw)>=thr).astype(float)
    elif op=="sign_only":
        return np.sign(f_raw)
    raise ValueError(op)

def process_fov(sub, n_perm, seed, min_n):
    rng = np.random.default_rng(seed)
    sub = sub.copy().reset_index(drop=True)
    n = len(sub)
    region = sub["region_label"].to_numpy()
    sa = sub["tumor_score"].to_numpy(float)
    sb = sub["immune_score"].to_numpy(float)
    B1, th, hd = build_graph(sub[["x","y"]].to_numpy(float))
    if B1 is None: return []

    f_raw = sa[th]*sb[hd] - sa[hd]*sb[th]
    rows = []
    for op in OPERATORS:
        f_t = apply_operator(f_raw, sa, sb, th, hd, op)
        if np.allclose(f_t,0):
            rows.append({"operator":op,"R":np.nan,"perm_p":np.nan,
                         "frac_coexact":np.nan,"verdict":"zero_flux"})
            continue
        nc = coex_node_energy(f_t, B1, n, th, hd)
        R, p = enrich_perm(nc, region, n_perm, rng, min_n)
        L0=(B1@B1.T).tocsr()
        alpha=spla.lsqr(L0,B1@f_t,atol=LSQR_TOL,btol=LSQR_TOL)[0]
        f_cx=f_t-B1.T@alpha
        e_tot=float(np.sum(f_t**2)); e_cx=float(np.sum(f_cx**2))
        frac=e_cx/e_tot if e_tot>0 else np.nan
        verdict = ("PASS" if not np.isnan(R) and R>1 else "FAIL") \
                  if not np.isnan(R) else "inconclusive"
        rows.append({"operator":op,"R":R,"perm_p":p,
                     "frac_coexact":round(frac,4),"verdict":verdict})
    return rows

def main():
    a = _args()
    a.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.cells, compression="gzip")

    all_rows = []
    for fov, sub in df.groupby("fov"):
        rows = process_fov(sub, a.n_perm, a.seed, a.min_region)
        for r in rows: r["fov"]=fov
        all_rows.extend(rows)
        parts = " ".join(f"{r['operator'][:4]}:{r['R']:.2f}"
                         if not np.isnan(r['R']) else f"{r['operator'][:4]}:nan"
                         for r in rows)
        print(f"[fov={fov}] {parts}")

    per_fov = pd.DataFrame(all_rows)
    per_fov.to_csv(a.out/"cosmx_operator_robustness_per_fov.csv", index=False)

    print(f"\n=== STEP 06 OPERATOR ROBUSTNESS COHORT ===\n")
    summary = []
    for op in OPERATORS:
        sub = per_fov[(per_fov["operator"]==op)].dropna(subset=["R"])
        if len(sub)==0: continue
        R = sub["R"].to_numpy()
        n_gt1 = int((R>1).sum()); n=len(R)
        med_R = float(np.median(R))
        med_fc= float(sub["frac_coexact"].dropna().median()) if sub["frac_coexact"].notna().any() else np.nan
        b = binomtest(n_gt1,n,p=0.5,alternative="greater")
        sym = "✓ SURVIVES" if b.pvalue<0.05 else "✗ COLLAPSES"
        print(f"  {sym}  {op:<24}  med_R={med_R:.3f}  "
              f"R>1: {n_gt1}/{n}  sign_p={b.pvalue:.4f}  frac_coex={med_fc:.3f}")
        summary.append({"operator":op,"n_sections":n,"n_enriched":n_gt1,
            "median_R":round(med_R,3),"sign_test_p":round(b.pvalue,5),
            "median_frac_coexact":round(med_fc,4) if not np.isnan(med_fc) else np.nan,
            "verdict":"SURVIVES" if b.pvalue<0.05 else "COLLAPSES"})

    pd.DataFrame(summary).to_csv(
        a.out/"cosmx_operator_robustness_cohort_summary.csv", index=False)
    print("\n[done]")

if __name__ == "__main__": main()
