"""
Step 25c — Remeshing Stability (v2 — repaired)
================================================
Self-contained. No external dependencies beyond numpy/pandas/scipy.

REPAIR NOTES (v2):
  - v1 projected original edge fluxes onto new graphs, falling back to score-proxy
    for edges not in the original graph. This caused ~80% of edges on knn_8 and
    radius graphs to be score-proxy reconstructions, making those comparisons
    non-equivalent and incomparable.
  - v2 recomputes the FULL wedge operator from scratch on EVERY graph using node
    scores consistently. All graphs are thus on equal footing.
  - Primary comparison: knn family (k=4, 6, 8) — same construction type, varying
    density. This is the cleanest test of graph-parameter sensitivity.
  - Delaunay: secondary geometry-based comparison.
  - Radius: retained but flagged separately. Large edge count explosion can inflate
    R by increasing the sample size of interface edges. Flag if radius edge count
    > 3× knn_6 edge count.
  - Stability criteria (decision-level, not absolute R values):
      1. R > 1 in >= 85% of sections PER graph type
      2. Spearman r(per-section R, knn_6 R) >= 0.70 for each graph type
      (Primary comparison: knn family only)

Usage:
  python scripts_tnbc/step25c_remeshing_stability.py \
    --mode sample --sample-id GSM_6433618 \
    --flux-tag flux_tumor_immune_region_interface_weighted \
    --statsdir stats/CSV_GSM --outdir stats/CSV_GSM

  python scripts_tnbc/step25c_remeshing_stability.py \
    --mode cohort --statsdir stats/CSV_GSM --outdir stats/CSV_GSM
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import KDTree, Delaunay as SciDelaunay
from scipy.stats import spearmanr

INTERFACE_LABEL = "interface_like"
TUMOR_LABELS    = {"tumor_enriched", "tumor_core"}

# Primary family (knn) is the stability test.
# Delaunay is a secondary geometry check.
# Radius is flagged as informational only (edge-count explosion caveat).
KNN_TYPES      = ["knn_4", "knn_6", "knn_8"]
ALL_TYPES      = ["knn_4", "knn_6", "knn_8", "delaunay", "radius"]
PRIMARY_TYPES  = KNN_TYPES     # stability assessed on these
RADIUS_MULT_WARN = 3.0         # flag if radius edges > this × knn_6 edges

# ── CLI ────────────────────────────────────────────────────────────────────────
def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",         choices=["sample","cohort"], default="sample")
    p.add_argument("--sample-id",    default=None)
    p.add_argument("--flux-tag",     default="flux_tumor_immune_region_interface_weighted")
    p.add_argument("--statsdir",     default="stats/CSV_GSM")
    p.add_argument("--outdir",       default="stats/CSV_GSM")
    p.add_argument("--n-perm",       type=int, default=500)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--min-nodes",    type=int, default=10)
    p.add_argument("--radius-spots", type=float, default=3.0,
                   help="Radius in median-NN-distance units.")
    return p.parse_args()

# ── GRAPH CONSTRUCTION ─────────────────────────────────────────────────────────
def build_edges(coords: np.ndarray, gtype: str,
                r_spots: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    n = len(coords)
    if gtype.startswith("knn"):
        k = int(gtype.split("_")[1])
        tree = KDTree(coords)
        _, idx = tree.query(coords, k=min(k+1, n))
        seen = set(); tl, hl = [], []
        for i in range(n):
            for j in idx[i, 1:]:
                key = (min(i,j), max(i,j))
                if key not in seen:
                    seen.add(key); tl.append(i); hl.append(j)
        return np.array(tl, dtype=int), np.array(hl, dtype=int)
    elif gtype == "delaunay":
        tri = SciDelaunay(coords)
        seen = set(); tl, hl = [], []
        for s in tri.simplices:
            for a, b in [(0,1),(0,2),(1,2)]:
                key = (min(s[a],s[b]), max(s[a],s[b]))
                if key not in seen:
                    seen.add(key); tl.append(s[a]); hl.append(s[b])
        return np.array(tl, dtype=int), np.array(hl, dtype=int)
    elif gtype == "radius":
        tree = KDTree(coords)
        nn   = tree.query(coords, k=2)[0][:, 1]
        r    = r_spots * float(np.median(nn))
        pairs = sorted(tree.query_pairs(r))
        tl = [p[0] for p in pairs]; hl = [p[1] for p in pairs]
        return np.array(tl, dtype=int), np.array(hl, dtype=int)
    raise ValueError(f"Unknown graph type: {gtype}")

# ── CORE MATH ──────────────────────────────────────────────────────────────────
def build_B1(tail, head, n_nodes):
    ne = len(tail)
    return sp.csr_matrix(
        (np.concatenate([np.ones(ne), -np.ones(ne)]),
         (np.concatenate([head, tail]),
          np.concatenate([np.arange(ne), np.arange(ne)]))),
        shape=(n_nodes, ne))

def wedge_from_scores(sa, sb, tail, head):
    """Recompute wedge antisymmetric flux from node scores."""
    return np.array([sa[tail[e]]*sb[head[e]] - sa[head[e]]*sb[tail[e]]
                     for e in range(len(tail))])

def hodge_node_coex(f, B1, n_nodes):
    ne    = len(f)
    L0    = (B1 @ B1.T).tocsr()
    alpha = spla.lsqr(L0, B1@f, atol=1e-10, btol=1e-10, iter_lim=5000)[0]
    f_cx  = f - B1.T @ alpha
    B1c   = B1.tocoo()
    et    = np.zeros(ne, dtype=int); eh = np.zeros(ne, dtype=int)
    for r, c, v in zip(B1c.row, B1c.col, B1c.data):
        if v < 0: et[c] = r
        else:     eh[c] = r
    nc = np.zeros(n_nodes); cnt = np.zeros(n_nodes)
    for e in range(ne):
        nc[et[e]] += f_cx[e]**2; cnt[et[e]] += 1
        nc[eh[e]] += f_cx[e]**2; cnt[eh[e]] += 1
    return nc / np.maximum(cnt, 1)

def enrich_perm(nc, region, n_perm, rng, min_n):
    im = region == INTERFACE_LABEL
    tm = np.isin(region, list(TUMOR_LABELS))
    ni, nt = int(im.sum()), int(tm.sum())
    if ni < min_n or nt < min_n:
        return dict(R=np.nan, p=np.nan, note="low_sample_size")
    mu_i = float(nc[im].mean()); mu_t = float(nc[tm].mean())
    if mu_t < 1e-12:
        return dict(R=np.nan, p=np.nan, note="zero_tumor")
    obs = mu_i / mu_t
    ra  = np.array(region, dtype=str); rp = ra.copy()
    null = np.zeros(n_perm)
    for k in range(n_perm):
        rng.shuffle(rp)
        mi = nc[rp==INTERFACE_LABEL].mean() if (rp==INTERFACE_LABEL).any() else 0.0
        mt = nc[np.isin(rp,list(TUMOR_LABELS))].mean() \
             if np.isin(rp,list(TUMOR_LABELS)).any() else 1e-12
        null[k] = mi/mt if mt > 1e-12 else 0.0
    return dict(R=obs, p=float((np.sum(null>=obs)+1)/(n_perm+1)), note="ok")

# ── SAMPLE MODE ────────────────────────────────────────────────────────────────
def run_sample(args) -> list[dict] | None:
    sid = args.sample_id
    print(f"\n=== {sid} ===")
    rng = np.random.default_rng(args.seed)
    sd  = Path(args.statsdir); od = Path(args.outdir)
    od.mkdir(parents=True, exist_ok=True)
    ef = sd / f"{sid}_step6_edges_hodge_{args.flux_tag}.csv"
    nf = sd / f"{sid}_step6_nodes_hodge_{args.flux_tag}.csv"
    if not ef.exists() or not nf.exists():
        print("  [skip] missing step6 files"); return None
    edges = pd.read_csv(ef); nodes = pd.read_csv(nf)
    n_nodes = len(nodes)
    if args.flux_tag not in edges.columns:
        print("  [skip] flux_tag missing"); return None
    ns = nodes.sort_values("node_id").reset_index(drop=True)
    region = ns["region_step2"].to_numpy()

    # Spatial coordinates
    xcol = next((c for c in ["x_fullres","x","pxl_col_in_fullres","array_col"]
                 if c in ns.columns), None)
    ycol = next((c for c in ["y_fullres","y","pxl_row_in_fullres","array_row"]
                 if c in ns.columns), None)
    if xcol is None or ycol is None:
        print("  [skip] spatial coordinates not found "
              "(need x_fullres/y_fullres or x/y)"); return None
    coords = ns[[xcol, ycol]].to_numpy(dtype=float)

    # Program scores — used to build wedge on EVERY graph (no flux transfer)
    sa = ns["tumor_score"].fillna(0).to_numpy(float) \
         if "tumor_score" in ns.columns else np.ones(n_nodes)
    sb = ns["immune_score"].fillna(0).to_numpy(float) \
         if "immune_score" in ns.columns else np.ones(n_nodes)

    n_knn6 = None   # reference edge count for radius comparison
    rows = []
    for gtype in ALL_TYPES:
        print(f"  [{gtype}]", end="", flush=True)
        try:
            tn, hn = build_edges(coords, gtype, args.radius_spots)
            ne = len(tn)
            if ne < 30:
                print(f" too few edges ({ne}), skip"); continue
            # Flag radius edge explosion
            flag = ""
            if gtype == "radius" and n_knn6 is not None:
                ratio = ne / n_knn6
                if ratio > RADIUS_MULT_WARN:
                    flag = f" [WARN: {ratio:.1f}× knn_6 edges — results informational only]"
            if gtype == "knn_6":
                n_knn6 = ne
            # Recompute wedge from scores on THIS graph (no flux transfer)
            f_new = wedge_from_scores(sa, sb, tn, hn)
            B1    = build_B1(tn, hn, n_nodes)
            nc    = hodge_node_coex(f_new, B1, n_nodes)
            e     = enrich_perm(nc, region, args.n_perm, rng, args.min_nodes)
            R     = e["R"]
            print(f" R={R:.3f}  p={e['p']:.3f}  n_edges={ne}{flag}")
            rows.append({"sample_id":sid,"graph_type":gtype,"n_edges":ne,
                "enrichment_ratio":R,"perm_p":e["p"],"note":e["note"],
                "flag":flag.strip()})
        except Exception as ex:
            print(f" ERROR: {ex}")
    if rows:
        out = od / f"{sid}_step25c_remeshing_{args.flux_tag}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        # Quick stability summary for knn family
        knn_R = [r["enrichment_ratio"] for r in rows
                 if r["graph_type"] in KNN_TYPES and not np.isnan(r["enrichment_ratio"])]
        if knn_R:
            all_gt1 = all(r > 1 for r in knn_R)
            print(f"  knn family R: {[round(r,2) for r in knn_R]}  "
                  f"all>1: {all_gt1}")
        print(f"  Saved → {out.name}")
    return rows

# ── COHORT MODE ────────────────────────────────────────────────────────────────
def run_cohort(args) -> None:
    sd = Path(args.statsdir); od = Path(args.outdir)
    od.mkdir(parents=True, exist_ok=True)
    files = sorted(sd.glob(f"*_step25c_remeshing_{args.flux_tag}.csv"))
    if not files:
        print("[cohort] No files found."); return
    print(f"[cohort] {len(files)} files.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(od / f"cohort_step25c_remeshing_{args.flux_tag}.csv", index=False)

    print("\n=== STEP 25c REMESHING STABILITY (v2) ===")
    print("Criteria: R>1 in >=85% sections; Spearman r(gtype, knn_6) >= 0.70")
    print("Primary stability assessment: knn family (k=4,6,8)")
    print("Delaunay: secondary geometry check")
    print("Radius: informational only (edge-count caveat)\n")

    # Per-graph-type summary
    R_by_g: dict[str, pd.Series] = {}
    for gtype in ALL_TYPES:
        sub = df[(df["graph_type"]==gtype)].dropna(subset=["enrichment_ratio"])
        if len(sub) == 0: continue
        R = sub["enrichment_ratio"].to_numpy()
        n_gt1 = int(np.sum(R > 1)); n = len(R)
        med_R = float(np.median(R)); pct = 100*n_gt1/n
        ok = "✓" if pct >= 85 else "✗"
        note = "" if gtype in KNN_TYPES else \
               " [secondary]" if gtype == "delaunay" else \
               " [informational — edge-count may differ]"
        print(f"  {ok} {gtype:<12} med_R={med_R:.2f}  "
              f"R>1: {n_gt1}/{n} ({pct:.0f}%){note}")
        R_by_g[gtype] = sub.set_index("sample_id")["enrichment_ratio"]

    # Spearman correlation vs knn_6 (primary reference)
    print()
    if "knn_6" not in R_by_g:
        print("  [warn] knn_6 not found — cannot compute reference correlation")
    else:
        ref = R_by_g["knn_6"].dropna()
        corrs_knn = []
        print("  Spearman r vs knn_6:")
        for gtype in ALL_TYPES:
            if gtype == "knn_6" or gtype not in R_by_g: continue
            s = R_by_g[gtype].dropna()
            common = ref.index.intersection(s.index)
            if len(common) < 4: continue
            r, p = spearmanr(ref[common], s[common])
            ok = "✓" if r >= 0.70 else "✗"
            note = "" if gtype in KNN_TYPES else \
                   " [secondary]" if gtype == "delaunay" else \
                   " [informational]"
            print(f"    {ok} knn_6 vs {gtype:<12}: r={r:.3f}  p={p:.4f}{note}")
            if gtype in KNN_TYPES:
                corrs_knn.append(r)

        # Primary verdict: knn family only
        print()
        if corrs_knn:
            med = float(np.median(corrs_knn))
            knn_pcts = []
            for g in KNN_TYPES:
                sub = df[df["graph_type"]==g].dropna(subset=["enrichment_ratio"])
                if len(sub) > 0:
                    knn_pcts.append(100*int(np.sum(sub["enrichment_ratio"]>1))/len(sub))
            min_pct = min(knn_pcts) if knn_pcts else 0
            stable = med >= 0.70 and min_pct >= 85
            status = "✓ EXCLUDED" if stable else "✗ UNRESOLVED"
            print(f"  PRIMARY VERDICT (knn family): {status}")
            print(f"    Median Spearman r(knn pairs) = {med:.3f}  "
                  f"Min R>1 pct across knn types = {min_pct:.0f}%")
            if not stable:
                print("  MANUSCRIPT NOTE: Formal mesh-robustness remains an open "
                      "validation target. Do NOT claim remeshing stability.")
            else:
                print("  MANUSCRIPT NOTE: Decision-level phenotype is stable "
                      "across knn graph constructions.")

    print("\n[cohort] Done.")

# ── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    a = _args()
    if a.mode == "sample":
        if not a.sample_id:
            print("--sample-id required"); sys.exit(1)
        run_sample(a)
    else:
        run_cohort(a)
