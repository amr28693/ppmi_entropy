#!/usr/bin/env python3
"""
check_visit_counts.py — per-window participant counts for the PPMI entropy analysis.

This is a standalone script which does NOT modify the analysis pipeline; it imports ppmi_entropy_analysis's own loading, group-assignment, and month-snapping functions, so the 6-month windowing is identical to what the entropy computation used. Use it to confirm the completion-rate figures cited in the manuscript (e.g. the month-66 interim completion dip discussed in the manuscript) without touching the validated script.

Run from the same folder as ppmi_entropy_analysis.py and data/ dir:

    python check_visit_counts.py                     # H1: GBA1-PD vs Sporadic PD
    python check_visit_counts.py --data-dir data --months 60 66 72   # only these windows

"participants" = distinct PATNO with a record in that 6-month window (the
completion count the argument is about). "records" = raw row count; it exceeds
"participants" only where a person has multiple rows at one visit (e.g. Part III
ON/OFF states), so comparing the two columns also shows whether ON/OFF
duplication is inflating any Part III window.
"""
import argparse

import pandas as pd

# Reuse the pipeline's exact logic (import only — nothing here runs its main()).
from ppmi_entropy_analysis import (
    load_ppmi_data,
    build_participant_table,
    assign_groups_h1,
    compute_months_from_baseline,
    DOMAINS,
)


def group_counts(data, group_ids, group_label, only_months=None):
    """Print per-Part, per-window participant/record counts for one group."""
    print(f"\n{'=' * 58}")
    print(f"  {group_label}  (n = {len(set(group_ids))} participants total)")
    print(f"{'=' * 58}")

    for dname in DOMAINS:                      # Part_I .. Part_IV, canonical order
        if dname not in data["updrs"]:
            continue
        # Identical month derivation to compute_entropy_by_group()
        df = compute_months_from_baseline(data["updrs"][dname].copy())
        gdf = df[df["PATNO"].isin(group_ids)]

        months = sorted(int(m) for m in gdf["Month"].unique())
        if only_months:
            months = [m for m in months if m in only_months]
        if not months:
            continue

        print(f"\n  {dname}:")
        print(f"    {'window(mo)':>10}  {'participants':>12}  {'records':>8}")
        for m in months:
            md = gdf[gdf["Month"] == m]
            print(f"    {m:>10}  {md['PATNO'].nunique():>12}  {len(md):>8}")


def main():
    ap = argparse.ArgumentParser(
        description="Per-window participant counts for the PPMI entropy analysis.")
    ap.add_argument("--data-dir", default="data",
                    help="Folder with PPMI CSVs (same one the main script uses).")
    ap.add_argument("--months", type=int, nargs="*", default=None,
                    help="Optional: restrict output to these 6-month windows, e.g. --months 60 66 72.")
    args = ap.parse_args()

    data = load_ppmi_data(args.data_dir)
    master = build_participant_table(data)

    ga, gb, la, lb = assign_groups_h1(master, data)

    if ga is None or gb is None:
        print("Group assignment failed — check that the genetic/cohort files are present.")
        return

    only = set(args.months) if args.months else None
    group_counts(data, ga, la, only)
    group_counts(data, gb, lb, only)


if __name__ == "__main__":
    main()
