#!/usr/bin/env python3
"""
annual_only_sensitivity.py — robustness check against interim-visit sparsity.

Reruns the H1 genotype comparison (GBA1-PD vs Sporadic PD) on ANNUAL visits
only, dropping the sparse 6-month interim windows (months 6/18/30/42/54/66)
where small-sample entropy bias can distort the trajectory (e.g. the month-66
dip).

It does NOT modify the analysis pipeline: it imports ppmi_entropy_analysis's own
loader, group assignment, month-snapping, and permutation test, filters the
per-Part records to annual windows using the pipeline's OWN snapping, and calls
the SAME run_permutation_test. It also reruns the full (all-visit) data as a
self-check --- the FULL column should reproduce the published Table 1 exactly,
which demonstrates the annual filter is the only thing changing.

Run from the folder with ppmi_entropy_analysis.py and data/:
    python annual_only_sensitivity.py                       # 5000 perms, both columns
    python annual_only_sensitivity.py --n-perms 2000        # faster preview
    python annual_only_sensitivity.py --annual-only         # skip the full-data self-check

Interpretation: if Parts II and IV SURVIVE Bonferroni in the ANNUAL column, the
genotype divergence does not depend on the sparse interim windows.
"""
import argparse

import numpy as np

# Reuse the pipeline's exact logic (import only; its main() does not run).
from ppmi_entropy_analysis import (
    load_ppmi_data,
    build_participant_table,
    assign_groups_h1,
    compute_months_from_baseline,
    run_permutation_test,
    DOMAINS,
)

ANNUAL = {0, 12, 24, 36, 48, 60, 72, 84, 96, 108}
ALPHA_BONF = 0.05 / 4


def keep_annual(df):
    """Filter a per-Part dataframe to rows landing on annual windows.

    Uses the pipeline's OWN compute_months_from_baseline to decide which rows
    are annual (via a temporary row id for alignment), then returns the ORIGINAL
    rows (INFODT untouched) so the pipeline re-snaps them normally downstream.
    """
    d = df.copy()
    d["_rowid"] = np.arange(len(d))
    snapped = compute_months_from_baseline(d)          # exact pipeline snapping
    keep_ids = set(snapped.loc[snapped["Month"].isin(ANNUAL), "_rowid"])
    return d[d["_rowid"].isin(keep_ids)].drop(columns="_rowid")


def report(tag, results, months):
    print(f"\n{'=' * 66}\n  {tag}")
    print(f"  windows: {months}")
    print(f"{'=' * 66}")
    print(f"  {'Part':<5}{'IAD':>10}{'null_mu':>10}{'null_sd':>9}{'sigma':>7}{'p':>9}   Bonferroni")
    for p in ["I", "II", "III", "IV"]:
        r = results[f"Part_{p}_IAD"]
        sigma = (r["observed"] - r["null_mean"]) / r["null_std"]
        surv = "SURVIVES" if r["p_value"] < ALPHA_BONF else "no"
        print(f"  {p:<5}{r['observed']:>10.3f}{r['null_mean']:>10.3f}"
              f"{r['null_std']:>9.3f}{sigma:>7.2f}{r['p_value']:>9.4f}   {surv}")
    t = results["total_IAD"]
    sigma = (t["observed"] - t["null_mean"]) / t["null_std"]
    print(f"  {'Tot':<5}{t['observed']:>10.3f}{t['null_mean']:>10.3f}"
          f"{t['null_std']:>9.3f}{sigma:>7.2f}{t['p_value']:>9.4f}")


def main():
    ap = argparse.ArgumentParser(description="Annual-only sensitivity analysis (H1).")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n-perms", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--annual-only", action="store_true",
                    help="Skip the full-data self-check run.")
    args = ap.parse_args()

    data = load_ppmi_data(args.data_dir)
    master = build_participant_table(data)
    ga, gb, la, lb = assign_groups_h1(master, data)
    if ga is None or gb is None:
        print("Group assignment failed — check the genetic/cohort files.")
        return

    if not args.annual_only:
        res, _, _, months, _, _ = run_permutation_test(
            data["updrs"], ga, gb, la, lb, args.n_perms, 15, args.seed)
        report("FULL DATA (all visits — should match published Table 1)", res, months)

    data_annual = dict(data)
    data_annual["updrs"] = {d: keep_annual(df) for d, df in data["updrs"].items()}
    res_a, _, _, months_a, _, _ = run_permutation_test(
        data_annual["updrs"], ga, gb, la, lb, args.n_perms, 15, args.seed)
    report("ANNUAL VISITS ONLY (interim 6-month windows dropped)", res_a, months_a)

    print("\nIf Parts II and IV SURVIVE in the annual-only column, the genotype")
    print("divergence does not depend on the sparse interim windows (e.g. month 66).")


if __name__ == "__main__":
    main()
