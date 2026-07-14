#!/usr/bin/env python3
"""
entropy_vs_sum.py — Does Shannon entropy detect structure a Part-score summary misses?
======================================================================================
Standalone companion to ppmi_entropy_analysis.py, for the H1 genotype comparison
(GBA1-PD vs sporadic PD). For each MDS-UPDRS Part it runs TWO divergence tests on
identical data and identical permutations:

  1. Shannon entropy of the item-score distribution   (the manuscript's metric)
  2. Mean item score                                  (the first-moment / Part-sum analog)

Both use the same permutation test, seed, --min-n, month-snapping, and item-inclusion
rule, so any difference in what they detect is due to the metric, not data handling.
The entropy side is produced by calling ppmi_entropy_analysis.run_permutation_test
directly, so it reproduces the manuscript's entropy numbers exactly; the score side
mirrors the same permutation loop with the same seed, so both metrics see the same
label shuffles (a paired comparison).

Why: the "mean item score" is the first moment of the very distribution whose entropy
the paper reports. Because a Part has a fixed item count, it is monotonic with the Part
sum, so its permutation p-value is the Part-score / total-score result. If a Part
diverges under the score but not under entropy (or vice versa), the two metrics are
detecting different things.

Usage:
  python entropy_vs_sum.py --n-perms 5000 --min-n 15
  python entropy_vs_sum.py --data-dir /path/to/ppmi_csvs/ --n-perms 5000 --min-n 15

Output (all in results/):
  entropy_vs_sum_h1.json   paired per-Part results (entropy vs score: IAD, p, sigma)
  entropy_vs_sum_h1.txt    formatted table + interpretation
  entropy_vs_sum_h1.tex    LaTeX table fragment for the manuscript

Author: Anderson M. Rodriguez
ORCID:  0009-0007-5179-9341
"""

import os
import sys
import json
import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Reuse the main analysis module: loaders, group assignment, month-snapping,
# item detection, divergence metric, and the entropy permutation test itself.
import ppmi_entropy_analysis as pea

# Family of four Part-level tests, matching the manuscript.
BONF_ALPHA = 0.05 / 4


def compute_meanscore_by_group(updrs_data, group_a_ids, group_b_ids,
                               group_a_label, group_b_label, min_n=15):
    """First-moment analog of pea.compute_entropy_by_group.

    Identical data handling, month-snapping, min-n threshold, and
    'at least half the items' inclusion rule; the ONLY change is the per-item
    statistic -- the population MEAN score instead of the score distribution's
    Shannon entropy. Averaging the per-item means across a Part gives the Part's
    mean item score (the Part-score / sum analog) at each visit.
    """
    groups = {group_a_label: set(group_a_ids), group_b_label: set(group_b_ids)}
    group_traj = {g: {} for g in groups}
    group_info = {g: {"n_patients": len(ids)} for g, ids in groups.items()}

    for dname, dinfo in pea.DOMAINS.items():
        if dname not in updrs_data:
            continue
        df = updrs_data[dname].copy()
        items = pea.detect_item_columns(df, dinfo["prefix"])
        if not items:
            continue

        for col in items:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] > 4, col] = np.nan  # 101 = "unable to rate"

        df = pea.compute_months_from_baseline(df)
        all_months = sorted(df["Month"].unique())

        for glabel, gids in groups.items():
            gdf = df[df["PATNO"].isin(gids)]
            scores = {}
            for m in all_months:
                month_data = gdf[gdf["Month"] == m]
                item_means = []
                for col in items:
                    vals = month_data[col].dropna().values
                    if len(vals) < min_n:
                        continue
                    item_means.append(float(np.mean(vals)))
                if len(item_means) >= max(1, len(items) // 2):
                    scores[m] = float(np.mean(item_means))
            group_traj[glabel][dname] = scores

    month_sets = []
    for glabel in group_traj:
        for dname in group_traj[glabel]:
            month_sets.append(set(group_traj[glabel][dname].keys()))
    if not month_sets:
        return group_traj, [], group_info

    common = sorted(set.intersection(*month_sets))
    for glabel in group_traj:
        for dname in list(group_traj[glabel].keys()):
            s = group_traj[glabel][dname]
            group_traj[glabel][dname] = np.array([s[m] for m in common])
    return group_traj, [int(m) for m in common], group_info


def run_score_permtest(updrs_data, group_a_ids, group_b_ids,
                       group_a_label, group_b_label, n_perms, min_n, seed):
    """Mirror of pea.run_permutation_test using the mean-score statistic.

    Same seed and same all_ids ordering as the entropy run, so the sequence of
    participant-level label shuffles is identical -> the two metrics are tested
    on the same permutations. Uses the add-one (b+1)/(m+1) p-value, as the paper does.
    """
    rng = np.random.RandomState(seed)
    all_ids = np.concatenate([group_a_ids, group_b_ids])
    n_a = len(group_a_ids)

    print("\n  Computing observed mean-score trajectories...")
    obs_traj, months, info = compute_meanscore_by_group(
        updrs_data, group_a_ids, group_b_ids, group_a_label, group_b_label, min_n)

    common_domains = (set(obs_traj[group_a_label].keys()) &
                      set(obs_traj[group_b_label].keys()))
    domains = [d for d in pea.PART_ORDER if d in common_domains]
    if not domains or not months:
        print("  ERROR: no overlapping domains/months for the score metric.")
        return {}

    obs = pea.compute_divergence(
        obs_traj[group_a_label], obs_traj[group_b_label], months, domains)

    null = {k: [] for k in obs}
    print(f"  Running score permutation test ({n_perms} permutations)...")
    for i in range(n_perms):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"    permutation {i+1}/{n_perms}")
        shuf = rng.permutation(all_ids)
        perm_a, perm_b = shuf[:n_a], shuf[n_a:]
        try:
            pt, pm, _ = compute_meanscore_by_group(
                updrs_data, perm_a, perm_b, group_a_label, group_b_label, min_n)
            perm_domains = sorted(set(pt[group_a_label].keys()) &
                                  set(pt[group_b_label].keys()))
            if perm_domains and pm:
                m = pea.compute_divergence(
                    pt[group_a_label], pt[group_b_label], pm, perm_domains)
                for k in null:
                    if k in m:
                        null[k].append(m[k])
        except Exception:
            continue

    results = {}
    for k in sorted(obs.keys()):
        nv = np.array(null.get(k, []))
        o = obs[k]
        if len(nv) == 0:
            results[k] = {"observed": float(o), "null_mean": np.nan,
                          "null_std": np.nan, "p_value": np.nan, "n_perms": 0}
            continue
        if "IAD" in k or "peak_diff" in k:
            b = int(np.sum(nv >= o))
        else:
            b = int(np.sum(np.abs(nv) >= abs(o)))
        results[k] = {
            "observed": float(o), "null_mean": float(np.mean(nv)),
            "null_std": float(np.std(nv)),
            "p_value": float((b + 1) / (len(nv) + 1)), "n_perms": len(nv),
        }
    return results


def _sigma(r):
    """(observed - null_mean) / null_std, guarding against zero/NaN."""
    s = r.get("null_std", 0)
    if s and not np.isnan(s):
        return (r["observed"] - r["null_mean"]) / s
    return float("nan")


def main():
    ap = argparse.ArgumentParser(
        description="Entropy vs Part-score (sum) divergence comparison, H1 genotype.")
    ap.add_argument("--data-dir", default="data",
                    help="Folder containing PPMI CSV files")
    ap.add_argument("--output", default="results", help="Output directory")
    ap.add_argument("--n-perms", type=int, default=5000,
                    help="Permutations (default 5000 = the manuscript setting)")
    ap.add_argument("--min-n", type=int, default=15,
                    help="Minimum observations per item per visit")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)

    print("=" * 78)
    print("ENTROPY vs PART-SCORE (SUM) DIVERGENCE — H1 (GBA1-PD vs Sporadic PD)")
    print("=" * 78)

    data = pea.load_ppmi_data(args.data_dir)
    master = pea.build_participant_table(data)
    gba, sporadic, la, lb = pea.assign_groups_h1(master, data)
    if gba is None or len(gba) == 0 or len(sporadic) == 0:
        print("  ERROR: could not assign H1 groups.")
        sys.exit(1)

    print("\n=== ENTROPY (paper metric; via ppmi_entropy_analysis) ===")
    ent, _obs, _traj, _months, _domains, _info = pea.run_permutation_test(
        data["updrs"], gba, sporadic, la, lb, args.n_perms, args.min_n, args.seed)

    print("\n=== MEAN SCORE (Part-score / sum analog; identical shuffles) ===")
    score = run_score_permtest(
        data["updrs"], gba, sporadic, la, lb, args.n_perms, args.min_n, args.seed)

    if not ent or not score:
        print("  ERROR: one metric produced no results; aborting.")
        sys.exit(1)

    # ---- assemble paired per-Part comparison ----
    rows = []
    for d in pea.PART_ORDER:
        k = f"{d}_IAD"
        if k not in ent or k not in score:
            continue
        e, s = ent[k], score[k]
        e_sig = (not np.isnan(e["p_value"])) and e["p_value"] < BONF_ALPHA
        s_sig = (not np.isnan(s["p_value"])) and s["p_value"] < BONF_ALPHA
        if e_sig and s_sig:
            verdict = "both detect"
        elif e_sig and not s_sig:
            verdict = "ENTROPY ONLY"
        elif s_sig and not e_sig:
            verdict = "SCORE ONLY"
        else:
            verdict = "neither"
        rows.append({
            "part": d.replace("_", " "),
            "entropy_IAD": e["observed"], "entropy_p": e["p_value"],
            "entropy_sigma": _sigma(e),
            "score_IAD": s["observed"], "score_p": s["p_value"],
            "score_sigma": _sigma(s),
            "verdict": verdict,
        })

    tot = {}
    if "total_IAD" in ent and "total_IAD" in score:
        tot = {
            "entropy_p": ent["total_IAD"]["p_value"],
            "entropy_sigma": _sigma(ent["total_IAD"]),
            "score_p": score["total_IAD"]["p_value"],
            "score_sigma": _sigma(score["total_IAD"]),
        }

    # ---- console + .txt table ----
    L = []
    L.append("=" * 78)
    L.append(f"ENTROPY vs PART-SCORE (SUM) DIVERGENCE  —  H1: {la} vs {lb}")
    L.append("=" * 78)
    L.append(f"  n_perms={args.n_perms}  min_n={args.min_n}  seed={args.seed}  "
             f"Bonferroni alpha (4 tests)={BONF_ALPHA:.4f}")
    L.append("  Identical data and permutations; metrics differ only in the per-item statistic.")
    L.append("  Entropy IAD is in bits; Score IAD in score-points -- compare p and sigma")
    L.append("  across metrics, not raw IAD magnitude.")
    L.append("")
    hdr = ("  %-14s | %10s %9s %7s | %10s %9s %7s | %-12s"
           % ("Part", "Ent IAD", "Ent p", "Ent sig", "Score IAD", "Score p",
              "Scr sig", "Concordance"))
    L.append(hdr)
    L.append("  " + "-" * (len(hdr) - 2))
    for r in rows:
        L.append("  %-14s | %10.3f %9.4f %7.1f | %10.3f %9.4f %7.1f | %-12s"
                 % (r["part"], r["entropy_IAD"], r["entropy_p"], r["entropy_sigma"],
                    r["score_IAD"], r["score_p"], r["score_sigma"], r["verdict"]))
    if tot:
        L.append("  " + "-" * (len(hdr) - 2))
        L.append("  %-14s | %10s %9.4f %7.1f | %10s %9.4f %7.1f |"
                 % ("OMNIBUS/total", "", tot["entropy_p"], tot["entropy_sigma"],
                    "", tot["score_p"], tot["score_sigma"]))
    L.append("")

    ent_only = [r["part"] for r in rows if r["verdict"] == "ENTROPY ONLY"]
    score_only = [r["part"] for r in rows if r["verdict"] == "SCORE ONLY"]
    L.append(f"INTERPRETATION (at Bonferroni alpha = {BONF_ALPHA:.4f}):")
    if ent_only:
        L.append(f"  Entropy detects, score does NOT: {', '.join(ent_only)}")
        L.append("    -> entropy captures distributional structure the Part-score summary misses.")
    if score_only:
        L.append(f"  Score detects, entropy does NOT: {', '.join(score_only)}")
        L.append("    -> the group difference here is in central tendency, not distribution shape.")
    if not ent_only and not score_only:
        L.append("  Entropy and score reach the same verdict on every Part at this threshold.")
        L.append("  Note: the total-score dilution argument is separate -- it concerns summing")
        L.append("  ACROSS Parts, not per-Part detection; sigma still shows relative effect size.")
    L.append("=" * 78)
    txt = "\n".join(L)
    print("\n" + txt)

    with open(os.path.join(args.output, "entropy_vs_sum_h1.txt"), "w") as f:
        f.write(txt + "\n")

    # ---- JSON ----
    out = {
        "analysis": "h1",
        "comparison": "shannon_entropy_vs_mean_item_score",
        "n_perms": args.n_perms, "min_n": args.min_n, "seed": args.seed,
        "bonferroni_alpha_4tests": BONF_ALPHA,
        "note": ("Score = mean item score (first-moment / Part-sum analog). Same data, "
                 "inclusion rule, and permutation shuffles as the entropy test. Compare "
                 "p-values and sigma across metrics; IAD units differ (bits vs score points)."),
        "parts": rows, "omnibus": tot,
    }
    with open(os.path.join(args.output, "entropy_vs_sum_h1.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)

    # ---- LaTeX fragment for the manuscript ----
    tl = []
    tl.append("% Auto-generated by entropy_vs_sum.py")
    tl.append("\\begin{table}[H]")
    tl.append("\\centering")
    tl.append("\\caption{Per-Part divergence between GBA1-PD and sporadic PD under two "
              "metrics computed on identical data and identical permutations: Shannon "
              "entropy of the item-score distribution versus the mean item score (the "
              "first-moment, Part-score analog). Permutation $p$-values (%d permutations); "
              "$\\sigma$ is deviation from the permutation null.}" % args.n_perms)
    tl.append("\\label{tab:entropy_vs_score}")
    tl.append("\\begin{tabular}{lcccc}")
    tl.append("\\toprule")
    tl.append("Part & Entropy $p$ & Entropy $\\sigma$ & Score $p$ & Score $\\sigma$ \\\\")
    tl.append("\\midrule")
    for r in rows:
        tl.append("%s & $%.4f$ & $%.1f$ & $%.4f$ & $%.1f$ \\\\"
                  % (r["part"], r["entropy_p"], r["entropy_sigma"],
                     r["score_p"], r["score_sigma"]))
    tl.append("\\bottomrule")
    tl.append("\\end{tabular}")
    tl.append("\\end{table}")
    with open(os.path.join(args.output, "entropy_vs_sum_h1.tex"), "w") as f:
        f.write("\n".join(tl) + "\n")

    print(f"\n  Wrote entropy_vs_sum_h1.{{txt,json,tex}} to {args.output}/")


if __name__ == "__main__":
    main()
