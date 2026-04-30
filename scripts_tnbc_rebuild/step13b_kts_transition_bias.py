#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


TARGET_TRANSITIONS = [
    ("MIXED", "IMMUNE_EXHAUSTED"),
    ("MIXED", "STROMA"),
    ("MIXED", "TUMOR"),
    ("IMMUNE_ACTIVE", "IMMUNE_EXHAUSTED"),
    ("TUMOR", "STROMA"),
    ("TUMOR", "IMMUNE_EXHAUSTED"),
    ("STROMA", "IMMUNE_EXHAUSTED"),
]


def weighted_transition_matrix(states, edges):
    state_list = sorted(pd.unique(states))
    idx = {s: i for i, s in enumerate(state_list)}

    M = np.zeros((len(state_list), len(state_list)))

    for _, e in edges.iterrows():
        i = int(e["i"])
        j = int(e["j"])
        w = abs(float(e["flux_coexact"]))

        si = states[i]
        sj = states[j]

        M[idx[si], idx[sj]] += w
        M[idx[sj], idx[si]] += w

    row_sum = M.sum(axis=1, keepdims=True)
    P = M / np.maximum(row_sum, 1e-12)

    return state_list, P


def get_prob(P, state_list, source, target):
    if source not in state_list or target not in state_list:
        return np.nan

    return float(P[state_list.index(source), state_list.index(target)])


def process_sample(sid, statsdir, n_perm, seed):
    rng = np.random.default_rng(seed)

    nodes = pd.read_csv(statsdir / f"{sid}_kts_states.csv")
    edges = pd.read_csv(statsdir / f"{sid}_edges_hodge.csv")

    if "state" not in nodes.columns:
        raise ValueError(f"{sid}: missing state column")

    if not {"i", "j", "flux_coexact"} <= set(edges.columns):
        raise ValueError(f"{sid}: edges missing i/j/flux_coexact")

    states = nodes["state"].astype(str).to_numpy()
    state_list, P_obs = weighted_transition_matrix(states, edges)

    rows = []

    for source, target in TARGET_TRANSITIONS:
        obs = get_prob(P_obs, state_list, source, target)

        if np.isnan(obs):
            rows.append({
                "sample": sid,
                "source": source,
                "target": target,
                "status": "state_absent",
                "P_obs": np.nan,
                "P_null_median": np.nan,
                "bias_ratio": np.nan,
                "p_enriched": np.nan,
            })
            continue

        null = []

        for _ in range(n_perm):
            shuffled = states.copy()
            rng.shuffle(shuffled)

            null_state_list, P_null = weighted_transition_matrix(shuffled, edges)
            p = get_prob(P_null, null_state_list, source, target)

            if not np.isnan(p):
                null.append(p)

        null = np.asarray(null)

        if len(null) == 0:
            p_enriched = np.nan
            null_median = np.nan
            bias = np.nan
        else:
            null_median = float(np.median(null))
            bias = obs / max(null_median, 1e-12)
            p_enriched = (np.sum(null >= obs) + 1) / (len(null) + 1)

        rows.append({
            "sample": sid,
            "source": source,
            "target": target,
            "status": "ok",
            "P_obs": obs,
            "P_null_median": null_median,
            "bias_ratio": bias,
            "p_enriched": p_enriched,
        })

    out = pd.DataFrame(rows)
    out.to_csv(statsdir / f"{sid}_kts_transition_bias.csv", index=False)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--statsdir", type=Path, default=Path("results_gse278936"))
    parser.add_argument("--out", type=Path, default=Path("results_gse278936/kts_transition_bias_summary.csv"))
    parser.add_argument("--n-perm", type=int, default=300)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    sample_ids = sorted([
        p.name.replace("_kts_states.csv", "")
        for p in args.statsdir.glob("*_kts_states.csv")
    ])

    if not sample_ids:
        raise ValueError("No *_kts_states.csv found. Run Step11 first.")

    all_rows = []

    for sid in sample_ids:
        df = process_sample(
            sid=sid,
            statsdir=args.statsdir,
            n_perm=args.n_perm,
            seed=args.seed + sum(ord(c) for c in sid),
        )
        all_rows.append(df)
        print(f"[done] {sid}")

    summary = pd.concat(all_rows, ignore_index=True)
    summary.to_csv(args.out, index=False)

    ok = summary[summary["status"] == "ok"].copy()

    grouped = (
        ok.groupby(["source", "target"])
        .agg(
            n=("sample", "count"),
            median_P_obs=("P_obs", "median"),
            median_bias_ratio=("bias_ratio", "median"),
            n_p_lt_005=("p_enriched", lambda x: int((x < 0.05).sum())),
            median_p=("p_enriched", "median"),
        )
        .reset_index()
        .sort_values(["n_p_lt_005", "median_bias_ratio"], ascending=False)
    )

    grouped.to_csv(args.statsdir / "kts_transition_bias_grouped_summary.csv", index=False)

    print("\n=== STEP13b KTS TRANSITION BIAS SUMMARY ===")
    print(grouped.to_string(index=False))
    print(f"\n[done] {args.out}")
    print(f"[done] {args.statsdir / 'kts_transition_bias_grouped_summary.csv'}")


if __name__ == "__main__":
    main()
