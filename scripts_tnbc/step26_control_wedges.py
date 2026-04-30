"""
Step 26 — Control Antisymmetric Operators
==========================================
Self-contained. No external dependencies beyond numpy/pandas/scipy.

Diagnostic contradiction for TISSUE ANISOTROPY / GENERIC ANTISYMMETRY:
  IF anisotropy drives the result THEN control operators from non-biological
  program pairs ALSO show interface enrichment.
  IF controls yield R ≈ 1 THEN anisotropy is ruled out.

Five control operators:
  tumor_x_housekeeping   tumor × synthetic random HK scores
  immune_x_housekeeping  immune × synthetic random HK scores
  stroma_x_epithelial    stroma × synthetic random epithelial scores
  tumor_x_stroma         same-axis control
  shuffled_AB            10 random matched-size score pairs (reports mean ± SD)

Pass criterion (pre-specified):
  median R(control) < 3.0  AND  Wilcoxon vs canonical p < 0.01

Usage:
  python scripts_tnbc/step26_control_wedges.py \
    --mode sample --sample-id GSM_6433619 \
    --flux-tag flux_tumor_immune_region_interface_weighted \
    --statsdir stats/CSV_GSM --outdir stats/CSV_GSM

  python scripts_tnbc/step26_control_wedges.py \
    --mode cohort --statsdir stats/CSV_GSM --outdir stats/CSV_GSM
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.stats import mannwhitneyu

INTERFACE_LABEL   = "interface_like"
TUMOR_LABELS      = {"tumor_enriched", "tumor_core"}
N_RANDOM_DRAWS    = 10
CONTROL_R_MAX     = 3.0   # pre-specified: controls must be below this
CONTROL_OPS       = ["tumor_x_housekeeping","immune_x_housekeeping",
                     "stroma_x_epithelial","tumor_x_stroma","shuffled_AB"]

def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",      choices=["sample","cohort"], default="sample")
    p.add_argument("--sample-id", default=None)
    p.add_argument("--flux-tag",  default="flux_tumor_immune_region_interface_weighted")
    p.add_argument("--statsdir",  default="stats/CSV_GSM")
    p.add_argument("--outdir",    default="stats/CSV_GSM")
    p.add_argument("--n-perm",    type=int, default=500)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--min-nodes", type=int, default=10)
    return p.parse_args()

def build_B1(edges, n_nodes):
    ne = len(edges)
    tail = edges["tail"].to_numpy(dtype=int)
    head = edges["head"].to_numpy(dtype=int)
    return sp.csr_matrix(
        (np.concatenate([np.ones(ne),-np.ones(ne)]),
         (np.concatenate([head,tail]),np.concatenate([np.arange(ne),np.arange(ne)]))),
        shape=(n_nodes,ne))

def hodge_node_coex(f, B1, n_nodes):
    ne = len(f)
    L0    = (B1 @ B1.T).tocsr()
    alpha = spla.lsqr(L0, B1@f, atol=1e-10, btol=1e-10, iter_lim=5000)[0]
    f_cx  = f - B1.T @ alpha
    B1c   = B1.tocoo()
    et = np.zeros(ne,dtype=int); eh = np.zeros(ne,dtype=int)
    for r,c,v in zip(B1c.row,B1c.col,B1c.data):
        if v < 0: et[c]=r
        else:     eh[c]=r
    nc = np.zeros(n_nodes); cnt = np.zeros(n_nodes)
    for e in range(ne):
        nc[et[e]] += f_cx[e]**2; cnt[et[e]] += 1
        nc[eh[e]] += f_cx[e]**2; cnt[eh[e]] += 1
    return nc / np.maximum(cnt, 1)

def enrich_perm(nc, region, n_perm, rng, min_n):
    im = region == INTERFACE_LABEL
    tm = np.isin(region, list(TUMOR_LABELS))
    ni, nt = int(im.sum()), int(tm.sum())
    if ni < min_n or nt < min_n: return dict(R=np.nan, p=np.nan, note="low_sample_size")
    mu_i = float(nc[im].mean()); mu_t = float(nc[tm].mean())
    if mu_t < 1e-12: return dict(R=np.nan, p=np.nan, note="zero_tumor")
    obs = mu_i / mu_t
    ra  = np.array(region,dtype=str); rp = ra.copy()
    null = np.zeros(n_perm)
    for k in range(n_perm):
        rng.shuffle(rp)
        mi = nc[rp==INTERFACE_LABEL].mean() if (rp==INTERFACE_LABEL).any() else 0.0
        mt = nc[np.isin(rp,list(TUMOR_LABELS))].mean() \
             if np.isin(rp,list(TUMOR_LABELS)).any() else 1e-12
        null[k] = mi/mt if mt > 1e-12 else 0.0
    return dict(R=obs, p=float((np.sum(null>=obs)+1)/(n_perm+1)), note="ok")

def wedge(a, b, tail, head):
    return np.array([a[tail[e]]*b[head[e]] - a[head[e]]*b[tail[e]]
                     for e in range(len(tail))])

def zscore(v):
    s = np.std(v)
    return (v - np.mean(v)) / (s if s > 1e-8 else 1.0)

def run_sample(args):
    sid = args.sample_id
    print(f"\n=== {sid} ===")
    rng = np.random.default_rng(args.seed)
    rng2 = np.random.default_rng(args.seed + 9999)
    sd  = Path(args.statsdir); od = Path(args.outdir); od.mkdir(parents=True,exist_ok=True)
    ef  = sd / f"{sid}_step6_edges_hodge_{args.flux_tag}.csv"
    nf  = sd / f"{sid}_step6_nodes_hodge_{args.flux_tag}.csv"
    if not ef.exists() or not nf.exists():
        print("  [skip] missing step6 files"); return None
    edges = pd.read_csv(ef); nodes = pd.read_csv(nf)
    n_nodes = len(nodes)
    if args.flux_tag not in edges.columns:
        print("  [skip] flux_tag missing"); return None
    ns     = nodes.sort_values("node_id").reset_index(drop=True)
    region = ns["region_step2"].to_numpy()
    B1     = build_B1(edges, n_nodes)
    tail   = edges["tail"].to_numpy(dtype=int)
    head   = edges["head"].to_numpy(dtype=int)
    # canonical
    f_raw  = edges[args.flux_tag].fillna(0).to_numpy(float)
    nc0    = hodge_node_coex(f_raw, B1, n_nodes)
    e0     = enrich_perm(nc0, region, args.n_perm, rng, args.min_nodes)
    R0     = e0["R"]
    print(f"  [canonical_TxI]       R={R0:.3f}  p={e0['p']:.3f}")
    # program scores
    sa = zscore(ns["tumor_score"].fillna(0).to_numpy(float)) \
         if "tumor_score" in ns.columns else np.zeros(n_nodes)
    sb = zscore(ns["immune_score"].fillna(0).to_numpy(float)) \
         if "immune_score" in ns.columns else np.zeros(n_nodes)
    sc = zscore(ns["stroma_score"].fillna(0).to_numpy(float)) \
         if "stroma_score" in ns.columns else np.zeros(n_nodes)
    # synthetic controls — random normal, independent of biology
    hk  = zscore(rng2.standard_normal(n_nodes))
    epi = zscore(rng2.standard_normal(n_nodes))
    rows = [{"sample_id":sid,"operator":"canonical_TxI","n_draws":1,
             "R_mean":R0,"R_std":np.nan,"perm_p":e0["p"],"note":e0["note"]}]
    ctrl_defs = {
        "tumor_x_housekeeping":  (sa, hk),
        "immune_x_housekeeping": (sb, hk),
        "stroma_x_epithelial":   (sc, epi),
        "tumor_x_stroma":        (sa, sc),
    }
    for op_name, (a, b) in ctrl_defs.items():
        f_ctrl = wedge(a, b, tail, head)
        if np.allclose(f_ctrl, 0):
            rows.append({"sample_id":sid,"operator":op_name,"n_draws":1,
                "R_mean":np.nan,"R_std":np.nan,"perm_p":np.nan,"note":"zero_flux"})
            continue
        nc  = hodge_node_coex(f_ctrl, B1, n_nodes)
        e   = enrich_perm(nc, region, args.n_perm, rng, args.min_nodes)
        R   = e["R"]
        print(f"  [{op_name:<28}] R={R:.3f}  p={e['p']:.3f}")
        rows.append({"sample_id":sid,"operator":op_name,"n_draws":1,
            "R_mean":round(R,4) if not np.isnan(R) else np.nan,
            "R_std":np.nan,"perm_p":e["p"],"note":e["note"]})
    # shuffled_AB: N_RANDOM_DRAWS random pairs
    R_shuf = []
    for _ in range(N_RANDOM_DRAWS):
        a_r = zscore(rng2.standard_normal(n_nodes))
        b_r = zscore(rng2.standard_normal(n_nodes))
        f_r = wedge(a_r, b_r, tail, head)
        nc  = hodge_node_coex(f_r, B1, n_nodes)
        e   = enrich_perm(nc, region, args.n_perm, rng, args.min_nodes)
        if not np.isnan(e["R"]): R_shuf.append(e["R"])
    if R_shuf:
        R_arr = np.array(R_shuf)
        print(f"  [shuffled_AB ×{N_RANDOM_DRAWS}]             "
              f"R={R_arr.mean():.3f} ± {R_arr.std():.3f}  "
              f"range=[{R_arr.min():.3f},{R_arr.max():.3f}]")
        rows.append({"sample_id":sid,"operator":"shuffled_AB","n_draws":len(R_shuf),
            "R_mean":round(float(R_arr.mean()),4),"R_std":round(float(R_arr.std()),4),
            "perm_p":np.nan,"note":"mean_of_draws"})
    if rows:
        out = od / f"{sid}_step26_control_wedges_{args.flux_tag}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  Saved → {out.name}")
    return rows

def run_cohort(args):
    sd = Path(args.statsdir); od = Path(args.outdir); od.mkdir(parents=True,exist_ok=True)
    files = sorted(sd.glob(f"*_step26_control_wedges_{args.flux_tag}.csv"))
    if not files: print("[cohort] No files."); return
    print(f"[cohort] {len(files)} files.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(od / f"cohort_step26_control_wedges_{args.flux_tag}.csv", index=False)
    print(f"\n=== STEP 26 COHORT SUMMARY ===")
    print(f"Pass: median R(control) < {CONTROL_R_MAX}  AND  Wilcoxon p < 0.01 vs canonical\n")
    canon = df[df["operator"]=="canonical_TxI"]["R_mean"].dropna().to_numpy()
    print(f"  [canonical_TxI]  median_R={np.median(canon):.2f}  n={len(canon)}")
    summary = []
    for op in CONTROL_OPS:
        sub = df[df["operator"]==op]["R_mean"].dropna()
        if len(sub) == 0: continue
        R = sub.to_numpy(); med = float(np.median(R))
        wil_p = np.nan
        if len(canon) >= 3 and len(R) >= 3:
            _, wil_p = mannwhitneyu(canon, R, alternative="greater")
        ok = med < CONTROL_R_MAX and (np.isnan(wil_p) or wil_p < 0.01)
        sym = "✓ EXCLUDED" if ok else "✗ UNRESOLVED"
        p_str = f"  Wilcoxon p={wil_p:.4f}" if not np.isnan(wil_p) else ""
        print(f"  {sym}  [{op:<28}]  median_R={med:.3f}{p_str}")
        summary.append({"operator":op,"n_sections":len(R),"median_R":round(med,3),
            "wilcoxon_p":round(float(wil_p),4) if not np.isnan(wil_p) else np.nan,
            "verdict":"ANISOTROPY_EXCLUDED" if ok else "UNRESOLVED"})
    pd.DataFrame(summary).to_csv(
        od / f"cohort_step26_summary_{args.flux_tag}.csv", index=False)
    print("\n[cohort] Done.")

if __name__ == "__main__":
    a = _args()
    if a.mode == "sample":
        if not a.sample_id: print("--sample-id required"); sys.exit(1)
        run_sample(a)
    else:
        run_cohort(a)
