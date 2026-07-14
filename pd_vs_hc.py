#!/usr/bin/env python3
"""
pd_vs_hc.py — Cross-validation: PD vs Healthy Controls entropy divergence.

Methodological cross-validation of the H1 genotype analysis. Runs the same
Shannon entropy permutation test on Diagnosed PD vs Healthy Controls from the
PPMI cohort. Prediction is essentially common sense:

  Part III (motor examination) SHOULD diverge — motor signs are diagnostic.
  Part IV  (motor complications) SHOULD be weak/absent — HCs take no levodopa.

This is the exact flip of the H1 genotype result (Part III null, Part IV strong).
If the method finds Part III divergence here and not for GBA1 vs sporadic, the
Part III null in H1 cannot be an artifact of the method; i.e., the method detects it
when it should.

Imports ppmi_entropy_analysis's loader, month-snapping, and permutation test
directly; the only new code is the PD-vs-HC group assignment.

Usage:
  python pd_vs_hc.py                                # defaults: 5000 perms, min_n=15
  python pd_vs_hc.py --n-perms 5000 --min-n 15      # explicit (manuscript settings)
  python pd_vs_hc.py --n-perms 500                   # fast preview

Output (all in results/):
  summary_pd_vs_hc.txt                 Human-readable results
  divergence_permtest_pd_vs_hc.json    Permutation test statistics
  entropy_trajectories_pd_vs_hc.json   Per-group, per-Part entropy trajectories
  fig1_entropy_trajectories_pd_vs_hc.png
  fig2_divergence_summary_pd_vs_hc.png

Author: Anderson M. Rodriguez
ORCID:  0009-0007-5179-9341
"""
import argparse
import json
import os
import sys

import numpy as np

from ppmi_entropy_analysis import (
    load_ppmi_data,
    build_participant_table,
    run_permutation_test,
    generate_figures,
    PART_ORDER,
)

# Bonferroni set dynamically in main() based on viable Parts; default 3
# (Parts I-III; Part IV excluded because HCs don't take levodopa)
BONF_ALPHA = 0.05 / 3
ANALYSIS_TAG = "pd_vs_hc"


def assign_groups_pd_vs_hc(master):
    """Diagnosed PD (all genotypes) vs Healthy Controls."""
    if "COHORT" not in master.columns:
        print("  ERROR: No COHORT column in master table.")
        return None, None, None, None

    cohort = master["COHORT"].astype(str)

    pd_ids = master.loc[
        cohort.str.contains("Parkinson", case=False, na=False),
        "PATNO"
    ].unique()

    hc_ids = master.loc[
        cohort.str.contains("Healthy", case=False, na=False),
        "PATNO"
    ].unique()

    # Fall back to "Control" if "Healthy" matches nothing
    if len(hc_ids) == 0:
        hc_ids = master.loc[
            cohort.str.contains("Control", case=False, na=False),
            "PATNO"
        ].unique()

    if len(pd_ids) == 0 or len(hc_ids) == 0:
        print(f"  ERROR: PD={len(pd_ids)}, HC={len(hc_ids)}. "
              f"Cohort values: {master['COHORT'].value_counts().to_dict()}")
        return None, None, None, None

    # Ensure no overlap
    overlap = set(pd_ids) & set(hc_ids)
    if overlap:
        print(f"  WARNING: {len(overlap)} participants in both groups; removing from HC.")
        hc_ids = np.array([p for p in hc_ids if p not in overlap])

    print(f"\n  PD vs HC Groups:")
    print(f"    Diagnosed PD:    {len(pd_ids)} participants")
    print(f"    Healthy Controls: {len(hc_ids)} participants")

    return pd_ids, hc_ids, "Diagnosed PD", "Healthy Control"


def sigma(r):
    """Standard deviations from null mean."""
    s = r.get("null_std", 0)
    if s and not np.isnan(s) and s > 0:
        return (r["observed"] - r["null_mean"]) / s
    return float("nan")


def write_summary(results, months, outdir):
    """Write human-readable summary to text file."""
    lines = []
    lines.append("=" * 70)
    lines.append("PPMI ENTROPY DIVERGENCE ANALYSIS — PD vs Healthy Controls")
    lines.append("=" * 70)
    lines.append(f"  Time points: {len(months)} ({min(months)}-{max(months)} months)")
    n_parts = len([d for d in PART_ORDER if f"{d}_IAD" in results])
    lines.append(f"  Bonferroni alpha ({n_parts} tests): {BONF_ALPHA:.4f}")
    lines.append("")
    lines.append("DOMAIN-LEVEL RESULTS:")
    lines.append("-" * 50)

    part_labels = {
        "Part_I": "MDS-UPDRS Part I (Non-Motor)",
        "Part_II": "MDS-UPDRS Part II (Motor ADL)",
        "Part_III": "MDS-UPDRS Part III (Motor Exam)",
        "Part_IV": "MDS-UPDRS Part IV (Motor Complications)",
    }

    for d in PART_ORDER:
        k = f"{d}_IAD"
        if k not in results:
            continue
        r = results[k]
        sig = sigma(r)
        stars = ""
        if r["p_value"] < 0.001:
            stars = " ***"
        elif r["p_value"] < BONF_ALPHA:
            stars = " **"
        elif r["p_value"] < 0.05:
            stars = " *"

        lines.append(f"  {part_labels.get(d, d)}:")
        lines.append(f"    IAD observed:  {r['observed']:.4f}")
        lines.append(f"    IAD null:      {r['null_mean']:.4f} +/- {r['null_std']:.4f}")
        lines.append(f"    p = {r['p_value']:.4f}{stars}")
        lines.append(f"    sigma: {sig:.1f}")
        lines.append("")

    if "total_IAD" in results:
        t = results["total_IAD"]
        tsig = sigma(t)
        lines.append("OMNIBUS:")
        lines.append("-" * 50)
        lines.append(f"  Total IAD:     {t['observed']:.4f}")
        lines.append(f"  Total null:    {t['null_mean']:.4f} +/- {t['null_std']:.4f}")
        lines.append(f"  p = {t['p_value']:.4f}")
        lines.append(f"  sigma from null: {tsig:.1f}")
        lines.append("")

    lines.append("CROSS-VALIDATION LOGIC:")
    lines.append("-" * 50)

    p3 = results.get("Part_III_IAD", {})
    p3_sig = p3.get("p_value", 1) < BONF_ALPHA if p3 else False
    p4_present = "Part_IV_IAD" in results

    if not p4_present:
        lines.append("  Part IV excluded: HCs do not receive levodopa and have")
        lines.append("  insufficient motor complications assessments (n < min_n).")
        lines.append("  Part IV is a PD-specific domain by instrument design.")
        lines.append("")

    if p3_sig:
        lines.append("  CONFIRMED: Part III (motor examination) diverges between")
        lines.append("  PD and Healthy Controls.")
        lines.append("")
        lines.append("  In H1, Part III is null (p = 0.339) for GBA1 vs sporadic PD.")
        lines.append("  The same framework detects Part III divergence when the groups")
        lines.append("  differ on motor examination (PD vs HC) and correctly finds no")
        lines.append("  difference when they do not (GBA1 vs sporadic). The Part III")
        lines.append("  null in H1 is not a method artifact.")
    else:
        lines.append("  WARNING: Part III did NOT diverge. This is unexpected")
        lines.append("  and would undermine the cross-validation argument.")

    txt = "\n".join(lines)
    path = os.path.join(outdir, f"summary_{ANALYSIS_TAG}.txt")
    with open(path, "w") as f:
        f.write(txt + "\n")
    print("\n" + txt)
    return path


def write_json_results(results, obs_traj, months, info, la, lb, outdir):
    """Write JSON outputs matching the main analysis format."""
    # Divergence permtest
    path_div = os.path.join(outdir, f"divergence_permtest_{ANALYSIS_TAG}.json")
    with open(path_div, "w") as f:
        json.dump(results, f, indent=2, default=float)

    # Entropy trajectories
    traj_out = {}
    for glabel in [la, lb]:
        if glabel in obs_traj:
            gdata = {"months": months}
            if glabel in info:
                gdata["n_patients"] = info[glabel]["n_patients"]
            domains = {}
            for d in PART_ORDER:
                if d in obs_traj[glabel]:
                    arr = obs_traj[glabel][d]
                    if hasattr(arr, "tolist"):
                        domains[d] = arr.tolist()
                    elif isinstance(arr, dict):
                        domains[d] = [arr.get(m, None) for m in months]
                    else:
                        domains[d] = list(arr)
            gdata["domains"] = domains
            traj_out[glabel] = gdata

    path_traj = os.path.join(outdir, f"entropy_trajectories_{ANALYSIS_TAG}.json")
    with open(path_traj, "w") as f:
        json.dump(traj_out, f, indent=2, default=float)

    return path_div, path_traj


def main():
    ap = argparse.ArgumentParser(
        description="Cross-validation: PD vs Healthy Controls entropy divergence.")
    ap.add_argument("--data-dir", default="data",
                    help="Folder containing PPMI CSV files")
    ap.add_argument("--output", default="results",
                    help="Output directory")
    ap.add_argument("--n-perms", type=int, default=5000,
                    help="Permutations (default 5000)")
    ap.add_argument("--min-n", type=int, default=15,
                    help="Minimum observations per item per visit")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 70)
    print("CROSS-VALIDATION: Diagnosed PD vs Healthy Controls")
    print("=" * 70)
    print("  Prediction: Part III diverges (motor exam = diagnostic),")
    print("              Part IV weak/absent (HCs take no levodopa).")
    print("  This is the exact flip of H1 (Part III null, Part IV strong).")
    print("=" * 70)

    data = load_ppmi_data(args.data_dir)
    master = build_participant_table(data)

    pd_ids, hc_ids, la, lb = assign_groups_pd_vs_hc(master)
    if pd_ids is None:
        print("  Group assignment failed.")
        sys.exit(1)

    # --- Diagnostic: check HC coverage on Parts I-III ---
    # Part IV is excluded by design: HCs do not receive levodopa, so motor
    # complications (dyskinesias, fluctuations, OFF episodes) are not assessed.
    # Only 19 of 440 HCs have any Part IV record, none meeting min_n.
    from ppmi_entropy_analysis import (
        compute_months_from_baseline, detect_item_columns, DOMAINS,
    )

    VIABLE_PARTS = ["Part_I", "Part_II", "Part_III"]

    print("\n  --- HC coverage diagnostic (Parts I-III; Part IV excluded by design) ---")
    hc_set = set(hc_ids)
    hc_has_data = False
    for dname in VIABLE_PARTS:
        dinfo = DOMAINS[dname]
        if dname not in data["updrs"]:
            print(f"    {dname}: no data loaded")
            continue
        df = data["updrs"][dname].copy()
        hc_rows = df[df["PATNO"].isin(hc_set)]
        n_hc_patnos = hc_rows["PATNO"].nunique()
        if n_hc_patnos == 0:
            print(f"    {dname}: 0 HC participants in file")
            continue
        hc_has_data = True
        hc_months = compute_months_from_baseline(hc_rows.copy())
        items = detect_item_columns(df, dinfo["prefix"])
        windows = sorted(hc_months["Month"].unique())
        counts = [hc_months[hc_months["Month"] == m]["PATNO"].nunique() for m in windows]
        above = sum(1 for c in counts if c >= args.min_n)
        print(f"    {dname}: {n_hc_patnos} HC participants, {len(items)} items, "
              f"{len(windows)} windows ({above} with n>={args.min_n})")

    if not hc_has_data:
        print("\n  Healthy Controls have no MDS-UPDRS records for Parts I-III.")
        sys.exit(1)

    # Filter UPDRS data to Parts I-III only
    updrs_filtered = {d: data["updrs"][d] for d in VIABLE_PARTS if d in data["updrs"]}
    print(f"  Running on: {', '.join(VIABLE_PARTS)}")
    print("  --- end diagnostic ---\n")

    # Temporarily narrow DOMAINS so the main module only iterates over Parts I-III.
    import ppmi_entropy_analysis as _pea
    _original_domains = _pea.DOMAINS
    _pea.DOMAINS = {k: v for k, v in _pea.DOMAINS.items() if k in VIABLE_PARTS}

    results, obs, obs_traj, months, domains, info = run_permutation_test(
        updrs_filtered, pd_ids, hc_ids, la, lb,
        args.n_perms, args.min_n, args.seed)

    _pea.DOMAINS = _original_domains  # restore

    if not results:
        print("  ERROR: Permutation test returned no results.")
        print("  Likely cause: HC windows with n >= min_n don't overlap across all Parts.")
        print("  Try: python pd_vs_hc.py --min-n 10  (if diagnostic shows marginal counts)")
        sys.exit(1)

    write_summary(results, months, args.output)
    write_json_results(results, obs_traj, months, info, la, lb, args.output)

    try:
        generate_figures(results, obs, obs_traj, months, domains, info,
                         la, lb, ANALYSIS_TAG, args.output)
    except Exception as e:
        print(f"  Figure generation failed (non-fatal): {e}")

    print(f"\n  All outputs written to {args.output}/")


if __name__ == "__main__":
    main()
